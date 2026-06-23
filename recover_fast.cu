// recover_fast.cu
// -----------------------------------------------------------------------------
// Version optimizada: aritmetica de campo en 64 bits (4 limbs) usando __int128,
// inversion por lotes Montgomery con tamano configurable, y perillas de tuning.
// Misma matematica y mismo self-test que recover.cu, pero ~varias veces mas rapida.
//
// Recuperacion de los 12 primeros hex de una clave privada ETH:
//   priv = (prefijo << 208) | sufijo
//
// Build (RTX 5090, CUDA >= 12.8):
//   nvcc -O3 -arch=sm_120 recover_fast.cu -o recover_fast
//
// Si tu compilador se queja de __int128 en device, avisame y cambio a PTX inline.
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

typedef unsigned long long u64;
typedef unsigned __int128   u128;

#define HD __host__ __device__ __forceinline__

// ===== Tuning (ajusta y recompila; reporta Mkeys/s de cada combinacion) =====
#ifndef BATCH
#define BATCH 64          // tamano de lote de inversion Montgomery (prueba 32/64/128)
#endif

// ============================ CAMPO secp256k1 (4x u64, LE) ===================
// p = 2^256 - 0x1000003D1.   2^256 ≡ C (mod p),  C = 0x1000003D1.
#define SECP_C  0x1000003D1ull

HD void f_copy(u64 r[4], const u64 a[4]){ r[0]=a[0]; r[1]=a[1]; r[2]=a[2]; r[3]=a[3]; }
HD void f_zero(u64 r[4]){ r[0]=r[1]=r[2]=r[3]=0; }
HD bool f_iszero(const u64 a[4]){ return (a[0]|a[1]|a[2]|a[3])==0; }
HD bool f_eq(const u64 a[4], const u64 b[4]){ return a[0]==b[0]&&a[1]==b[1]&&a[2]==b[2]&&a[3]==b[3]; }

HD void f_get_p(u64 p[4]){
  p[0]=0xFFFFFFFEFFFFFC2Full; p[1]=0xFFFFFFFFFFFFFFFFull;
  p[2]=0xFFFFFFFFFFFFFFFFull; p[3]=0xFFFFFFFFFFFFFFFFull;
}
HD bool f_geq(const u64 a[4], const u64 b[4]){
  for(int i=3;i>=0;i--){ if(a[i]>b[i]) return true; if(a[i]<b[i]) return false; }
  return true;
}
HD void f_sub_raw(u64 r[4], const u64 a[4], const u64 b[4]){ // r=a-b mod 2^256
  u128 borrow=0;
  for(int i=0;i<4;i++){ u128 d=(u128)a[i]-b[i]-borrow; r[i]=(u64)d; borrow=(d>>64)&1; }
}
HD void f_cond_sub_p(u64 r[4]){
  u64 p[4]; f_get_p(p);
  if(f_geq(r,p)) f_sub_raw(r,r,p);
  if(f_geq(r,p)) f_sub_raw(r,r,p);
}
HD void f_add_small(u64 r[4], u64 k){ u128 c=k; for(int i=0;i<4&&c;i++){ u128 s=(u128)r[i]+c; r[i]=(u64)s; c=s>>64; } }

HD void f_add(u64 r[4], const u64 a[4], const u64 b[4]){
  u128 carry=0;
  for(int i=0;i<4;i++){ u128 s=(u128)a[i]+b[i]+carry; r[i]=(u64)s; carry=s>>64; }
  if(carry) f_add_small(r, SECP_C);   // 2^256 ≡ C
  f_cond_sub_p(r);
}
HD void f_sub(u64 r[4], const u64 a[4], const u64 b[4]){
  u128 borrow=0;
  for(int i=0;i<4;i++){ u128 d=(u128)a[i]-b[i]-borrow; r[i]=(u64)d; borrow=(d>>64)&1; }
  if(borrow){ u64 p[4]; f_get_p(p); u128 c=0;
    for(int i=0;i<4;i++){ u128 s=(u128)r[i]+p[i]+c; r[i]=(u64)s; c=s>>64; } }
}

// Reduccion de 512 bits (8 limbs) -> 256 bits, plegando hi*C.
HD void f_reduce_wide(u64 prod[8], u64 r[4]){
  for(int it=0; it<4; it++){
    if((prod[4]|prod[5]|prod[6]|prod[7])==0) break;
    u64 hi[4]={prod[4],prod[5],prod[6],prod[7]};
    prod[4]=prod[5]=prod[6]=prod[7]=0;
    u128 carry=0;
    for(int k=0;k<8;k++){
      u128 cur=(u128)prod[k]+carry;
      if(k<4) cur += (u128)SECP_C * hi[k];   // low64 aqui, high64 via carry -> limb k+1
      prod[k]=(u64)cur; carry=cur>>64;
    }
  }
  r[0]=prod[0]; r[1]=prod[1]; r[2]=prod[2]; r[3]=prod[3];
  f_cond_sub_p(r);
}
HD void f_mul(u64 r[4], const u64 a[4], const u64 b[4]){
  u64 prod[8]={0,0,0,0,0,0,0,0};
  for(int i=0;i<4;i++){
    u128 carry=0;
    for(int j=0;j<4;j++){
      u128 t=(u128)a[i]*b[j] + prod[i+j] + carry;
      prod[i+j]=(u64)t; carry=t>>64;
    }
    int k=i+4;
    while(carry){ u128 t=(u128)prod[k]+carry; prod[k]=(u64)t; carry=t>>64; k++; }
  }
  f_reduce_wide(prod,r);
}
HD void f_sqr(u64 r[4], const u64 a[4]){ f_mul(r,a,a); }

HD void f_inv(u64 r[4], const u64 a[4]){
  u64 e[4]={0xFFFFFFFEFFFFFC2Dull,0xFFFFFFFFFFFFFFFFull,0xFFFFFFFFFFFFFFFFull,0xFFFFFFFFFFFFFFFFull}; // p-2
  u64 res[4]={1,0,0,0}, base[4]; f_copy(base,a);
  for(int bit=0; bit<256; bit++){
    if((e[bit>>6]>>(bit&63))&1ull) f_mul(res,res,base);
    f_sqr(base,base);
  }
  f_copy(r,res);
}

// =============================== CURVA =====================================
struct ECJ { u64 X[4], Y[4], Z[4]; };
HD void ecj_inf(ECJ* P){ f_zero(P->X); f_zero(P->Y); f_zero(P->Z); }
HD bool ecj_is_inf(const ECJ* P){ return f_iszero(P->Z); }

HD void f_Gx(u64 r[4]){ r[0]=0x59F2815B16F81798ull; r[1]=0x029BFCDB2DCE28D9ull; r[2]=0x55A06295CE870B07ull; r[3]=0x79BE667EF9DCBBACull; }
HD void f_Gy(u64 r[4]){ r[0]=0x9C47D08FFB10D4B8ull; r[1]=0xFD17B448A6855419ull; r[2]=0x5DA4FBFC0E1108A8ull; r[3]=0x483ADA7726A3C465ull; }

HD void ecj_dbl(const ECJ* P, ECJ* R){
  if(ecj_is_inf(P)){ ecj_inf(R); return; }
  u64 A[4],B[4],C[4],D[4],E[4],F[4],t1[4],t2[4],X3[4],Y3[4],Z3[4];
  f_sqr(A,P->X); f_sqr(B,P->Y); f_sqr(C,B);
  f_add(t1,P->X,B); f_sqr(t1,t1); f_sub(t1,t1,A); f_sub(t1,t1,C); f_add(D,t1,t1);
  f_add(E,A,A); f_add(E,E,A); f_sqr(F,E);
  f_add(t2,D,D); f_sub(X3,F,t2);
  f_sub(t1,D,X3); f_mul(t1,E,t1);
  f_add(t2,C,C); f_add(t2,t2,t2); f_add(t2,t2,t2); f_sub(Y3,t1,t2);
  f_mul(Z3,P->Y,P->Z); f_add(Z3,Z3,Z3);
  f_copy(R->X,X3); f_copy(R->Y,Y3); f_copy(R->Z,Z3);
}
HD void ecj_add_mixed(const ECJ* P, const u64 qx[4], const u64 qy[4], ECJ* R){
  if(ecj_is_inf(P)){ f_copy(R->X,qx); f_copy(R->Y,qy); f_zero(R->Z); R->Z[0]=1; return; }
  u64 Z1Z1[4],U2[4],S2[4],H[4],HH[4],I[4],J[4],r[4],V[4],t1[4],t2[4],X3[4],Y3[4],Z3[4];
  f_sqr(Z1Z1,P->Z); f_mul(U2,qx,Z1Z1);
  f_mul(S2,qy,P->Z); f_mul(S2,S2,Z1Z1);
  f_sub(H,U2,P->X); f_sub(r,S2,P->Y);
  if(f_iszero(H)){ if(f_iszero(r)){ ecj_dbl(P,R); return; } ecj_inf(R); return; }
  f_add(r,r,r); f_sqr(HH,H); f_add(I,HH,HH); f_add(I,I,I);
  f_mul(J,H,I); f_mul(V,P->X,I);
  f_sqr(X3,r); f_sub(X3,X3,J); f_add(t1,V,V); f_sub(X3,X3,t1);
  f_sub(t1,V,X3); f_mul(t1,r,t1); f_mul(t2,P->Y,J); f_add(t2,t2,t2); f_sub(Y3,t1,t2);
  f_mul(Z3,P->Z,H); f_add(Z3,Z3,Z3);
  f_copy(R->X,X3); f_copy(R->Y,Y3); f_copy(R->Z,Z3);
}
HD void ec_scalar_mul(const u64 k[4], const u64 bx[4], const u64 by[4], ECJ* R){
  ECJ acc; ecj_inf(&acc);
  int top=255; while(top>=0 && (((k[top>>6]>>(top&63))&1ull)==0)) top--;
  for(int bit=top; bit>=0; bit--){
    ECJ t; ecj_dbl(&acc,&t); acc=t;
    if((k[bit>>6]>>(bit&63))&1ull){ ECJ t2; ecj_add_mixed(&acc,bx,by,&t2); acc=t2; }
  }
  *R=acc;
}
HD void ecj_to_affine(const ECJ* P, u64 x[4], u64 y[4]){
  u64 zi[4],z2[4],z3[4]; f_inv(zi,P->Z); f_sqr(z2,zi); f_mul(x,P->X,z2); f_mul(z3,z2,zi); f_mul(y,P->Y,z3);
}

// ================================ KECCAK ===================================
HD u64 ROTL64(u64 x,int n){ return (x<<n)|(x>>(64-n)); }
HD void keccakf(u64 st[25]){
  const u64 RC[24]={0x0000000000000001ull,0x0000000000008082ull,0x800000000000808aull,0x8000000080008000ull,
    0x000000000000808bull,0x0000000080000001ull,0x8000000080008081ull,0x8000000000008009ull,
    0x000000000000008aull,0x0000000000000088ull,0x0000000080008009ull,0x000000008000000aull,
    0x000000008000808bull,0x800000000000008bull,0x8000000000008089ull,0x8000000000008003ull,
    0x8000000000008002ull,0x8000000000000080ull,0x000000000000800aull,0x800000008000000aull,
    0x8000000080008081ull,0x8000000000008080ull,0x0000000080000001ull,0x8000000080008008ull};
  const int rotc[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
  const int piln[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
  u64 t,bc[5];
  for(int r=0;r<24;r++){
    for(int i=0;i<5;i++) bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
    for(int i=0;i<5;i++){ t=bc[(i+4)%5]^ROTL64(bc[(i+1)%5],1); for(int j=0;j<25;j+=5) st[j+i]^=t; }
    t=st[1];
    for(int i=0;i<24;i++){ int j=piln[i]; bc[0]=st[j]; st[j]=ROTL64(t,rotc[i]); t=bc[0]; }
    for(int j=0;j<25;j+=5){ for(int i=0;i<5;i++) bc[i]=st[j+i]; for(int i=0;i<5;i++) st[j+i]^=(~bc[(i+1)%5])&bc[(i+2)%5]; }
    st[0]^=RC[r];
  }
}
HD void f_to_be32(const u64 a[4], uint8_t out[32]){
  for(int i=0;i<4;i++){ u64 w=a[3-i]; for(int b=0;b<8;b++) out[i*8+b]=(uint8_t)(w>>(56-8*b)); }
}
HD void pub_to_address(const u64 x[4], const u64 y[4], uint8_t addr[20]){
  uint8_t buf[64]; f_to_be32(x,buf); f_to_be32(y,buf+32);
  u64 st[25]; for(int i=0;i<25;i++) st[i]=0;
  uint8_t* sb=(uint8_t*)st; for(int i=0;i<64;i++) sb[i]^=buf[i]; sb[64]^=0x01; sb[135]^=0x80;
  keccakf(st);
  for(int i=0;i<20;i++) addr[i]=sb[12+i];
}

// ============================ CONSTANTES DEVICE ============================
__constant__ u64 c_Qx[4], c_Qy[4], c_Sx[4], c_Sy[4];
__constant__ uint8_t c_target[20];
__device__ int g_found=0;
__device__ u64 g_found_prefix=0;

// =============================== KERNEL ====================================
__global__ void search_kernel(u64 launch_base, unsigned run, u64 total){
  u64 gid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
  u64 p0=launch_base + gid*(u64)run;
  if(p0>=total) return;
  unsigned myrun=run; if(p0+myrun>total) myrun=(unsigned)(total-p0);

  u64 ks[4]={p0,0,0,0};                    // prefijo (<=48 bits) cabe en limb0
  ECJ P; ec_scalar_mul(ks,c_Qx,c_Qy,&P);
  ECJ tmp; ecj_add_mixed(&P,c_Sx,c_Sy,&tmp); P=tmp;

  unsigned done=0;
  while(done<myrun){
    if(g_found) return;
    unsigned B=(myrun-done<BATCH)?(myrun-done):BATCH;
    ECJ pts[BATCH];
    for(unsigned b=0;b<B;b++){ pts[b]=P; ECJ t; ecj_add_mixed(&P,c_Qx,c_Qy,&t); P=t; }
    u64 pref[BATCH][4];
    f_copy(pref[0],pts[0].Z);
    for(unsigned b=1;b<B;b++) f_mul(pref[b],pref[b-1],pts[b].Z);
    u64 inv[4]; f_inv(inv,pref[B-1]);
    for(int b=(int)B-1;b>=0;b--){
      u64 zi[4];
      if(b>0) f_mul(zi,inv,pref[b-1]); else f_copy(zi,inv);
      u64 ninv[4]; f_mul(ninv,inv,pts[b].Z); f_copy(inv,ninv);
      u64 z2[4],z3[4],x[4],y[4];
      f_sqr(z2,zi); f_mul(x,pts[b].X,z2); f_mul(z3,z2,zi); f_mul(y,pts[b].Y,z3);
      uint8_t addr[20]; pub_to_address(x,y,addr);
      bool match=true; for(int i=0;i<20;i++) if(addr[i]!=c_target[i]){match=false;break;}
      if(match){ if(atomicCAS(&g_found,0,1)==0) g_found_prefix=p0+done+(unsigned)b; }
    }
    done+=B;
  }
}

// =============================== SELF-TEST =================================
__global__ void selftest_kernel(uint8_t* a1,uint8_t* a2,u64* g1x,u64* g1y,uint8_t* ke){
  { u64 st[25]; for(int i=0;i<25;i++) st[i]=0; uint8_t* sb=(uint8_t*)st; sb[0]^=0x01; sb[135]^=0x80; keccakf(st);
    for(int i=0;i<32;i++) ke[i]=sb[i]; }
  u64 Gx[4],Gy[4]; f_Gx(Gx); f_Gy(Gy);
  { u64 k[4]={1,0,0,0}; ECJ R; ec_scalar_mul(k,Gx,Gy,&R); u64 x[4],y[4]; ecj_to_affine(&R,x,y);
    for(int i=0;i<4;i++){g1x[i]=x[i];g1y[i]=y[i];} uint8_t a[20]; pub_to_address(x,y,a); for(int i=0;i<20;i++) a1[i]=a[i]; }
  { u64 k[4]={2,0,0,0}; ECJ R; ec_scalar_mul(k,Gx,Gy,&R); u64 x[4],y[4]; ecj_to_affine(&R,x,y);
    uint8_t a[20]; pub_to_address(x,y,a); for(int i=0;i<20;i++) a2[i]=a[i]; }
}

// =============================== HOST UTILS ================================
static bool hexb(const char* h,uint8_t* o,int n){ if((int)strlen(h)!=n*2) return false;
  auto v=[](int c)->int{ if(c>='0'&&c<='9')return c-'0'; c=tolower(c); if(c>='a'&&c<='f')return c-'a'+10; return -1; };
  for(int i=0;i<n;i++){ int a=v(h[2*i]),b=v(h[2*i+1]); if(a<0||b<0)return false; o[i]=(uint8_t)((a<<4)|b);} return true; }
static void ph(const uint8_t* b,int n){ for(int i=0;i<n;i++) printf("%02x",b[i]); }
static bool suffix_to_scalar(const char* s52,u64 r[4]){ if((int)strlen(s52)!=52) return false;
  char full[65]; for(int i=0;i<12;i++) full[i]='0'; memcpy(full+12,s52,52); full[64]=0;
  uint8_t be[32]; if(!hexb(full,be,32)) return false;
  for(int i=0;i<4;i++){ u64 w=0; for(int b=0;b<8;b++) w=(w<<8)|be[(3-i)*8+b]; r[i]=w; } return true; }

int main(int argc,char**argv){
  const char* suffix=0; const char* addrhex=0; u64 start=0; bool st_only=false;
  int blocks=8192,threads=256; unsigned run=8192;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--suffix")&&i+1<argc) suffix=argv[++i];
    else if(!strcmp(argv[i],"--addr")&&i+1<argc) addrhex=argv[++i];
    else if(!strcmp(argv[i],"--start")&&i+1<argc) start=strtoull(argv[++i],0,16);
    else if(!strcmp(argv[i],"--blocks")&&i+1<argc) blocks=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--threads")&&i+1<argc) threads=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--run")&&i+1<argc) run=(unsigned)strtoul(argv[++i],0,10);
    else if(!strcmp(argv[i],"--selftest")) st_only=true;
  }
  uint8_t *d_a1,*d_a2,*d_ke; u64 *d_g1x,*d_g1y;
  CUDA_CHECK(cudaMalloc(&d_a1,20));CUDA_CHECK(cudaMalloc(&d_a2,20));CUDA_CHECK(cudaMalloc(&d_ke,32));
  CUDA_CHECK(cudaMalloc(&d_g1x,32));CUDA_CHECK(cudaMalloc(&d_g1y,32));
  selftest_kernel<<<1,1>>>(d_a1,d_a2,d_g1x,d_g1y,d_ke); CUDA_CHECK(cudaDeviceSynchronize());
  uint8_t a1[20],a2[20],ke[32]; u64 g1x[4],g1y[4];
  CUDA_CHECK(cudaMemcpy(a1,d_a1,20,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(a2,d_a2,20,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ke,d_ke,32,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g1x,d_g1x,32,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g1y,d_g1y,32,cudaMemcpyDeviceToHost));
  uint8_t ke_e[32]; hexb("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",ke_e,32);
  uint8_t a1e[20]; hexb("7e5f4552091a69125d5dfcb7b8c2659029395bdf",a1e,20);
  uint8_t a2e[20]; hexb("2b5ad5c4795c026514f8317c7a215e218dccd6cf",a2e,20);
  u64 Gx[4]={0x59F2815B16F81798ull,0x029BFCDB2DCE28D9ull,0x55A06295CE870B07ull,0x79BE667EF9DCBBACull};
  u64 Gy[4]={0x9C47D08FFB10D4B8ull,0xFD17B448A6855419ull,0x5DA4FBFC0E1108A8ull,0x483ADA7726A3C465ull};
  bool t_ke=!memcmp(ke,ke_e,32), t_g=!memcmp(g1x,Gx,32)&&!memcmp(g1y,Gy,32);
  bool t_a1=!memcmp(a1,a1e,20), t_a2=!memcmp(a2,a2e,20);
  printf("== SELF-TEST (BATCH=%d) ==\n",BATCH);
  printf("  keccak256(\"\") : %s\n",t_ke?"PASS":"FAIL");
  printf("  EC(1)==G      : %s\n",t_g?"PASS":"FAIL");
  printf("  addr(priv=1)  : 0x"); ph(a1,20); printf("  %s\n",t_a1?"PASS":"FAIL");
  printf("  addr(priv=2)  : 0x"); ph(a2,20); printf("  %s\n",t_a2?"PASS":"FAIL");
  bool ok=t_ke&&t_g&&t_a1&&t_a2;
  printf("  RESULTADO     : %s\n", ok?"TODO OK":"FALLO");
  if(!ok){ fprintf(stderr,"Self-test fallo: pasame esta salida.\n"); return 2; }
  if(st_only) return 0;
  if(!suffix||!addrhex){ fprintf(stderr,"Uso: ./recover_fast --suffix <52hex> --addr <40hex> [--start <hex>]\n"); return 1; }

  u64 s_scalar[4]; if(!suffix_to_scalar(suffix,s_scalar)){ fprintf(stderr,"sufijo invalido\n"); return 1; }
  uint8_t target[20]; { const char* p=addrhex; if(p[0]=='0'&&(p[1]=='x'||p[1]=='X'))p+=2;
    if(!hexb(p,target,20)){ fprintf(stderr,"direccion invalida\n"); return 1; } }
  u64 Qsc[4]={0,0,0,0x10000ull}; // 2^208
  ECJ Sj,Qj; ec_scalar_mul(s_scalar,Gx,Gy,&Sj); ec_scalar_mul(Qsc,Gx,Gy,&Qj);
  u64 Sx[4],Sy[4],Qx[4],Qy[4]; ecj_to_affine(&Sj,Sx,Sy); ecj_to_affine(&Qj,Qx,Qy);
  CUDA_CHECK(cudaMemcpyToSymbol(c_Sx,Sx,32));CUDA_CHECK(cudaMemcpyToSymbol(c_Sy,Sy,32));
  CUDA_CHECK(cudaMemcpyToSymbol(c_Qx,Qx,32));CUDA_CHECK(cudaMemcpyToSymbol(c_Qy,Qy,32));
  CUDA_CHECK(cudaMemcpyToSymbol(c_target,target,20));

  const u64 TOTAL=1ull<<48;
  u64 base=start; u64 per=(u64)blocks*threads*run;
  printf("\n== BUSQUEDA (BATCH=%d, %dx%d, run=%u) ==\n",BATCH,blocks,threads,run);
  printf("inicio 0x%012llx  | %llu prefijos/lanzamiento\n",base,per);
  int zero=0; CUDA_CHECK(cudaMemcpyToSymbol(g_found,&zero,sizeof(int)));
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  while(base<TOTAL){
    cudaEventRecord(e0);
    search_kernel<<<blocks,threads>>>(base,run,TOTAL); CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(e1); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    int found=0; CUDA_CHECK(cudaMemcpyFromSymbol(&found,g_found,sizeof(int)));
    if(found){ u64 pref=0; CUDA_CHECK(cudaMemcpyFromSymbol(&pref,g_found_prefix,sizeof(u64)));
      printf("\n*** ENCONTRADA ***\nprefijo: %012llx\nclave  : %012llx%s\n",pref,pref,suffix); return 0; }
    u64 cov=(base+per<=TOTAL)?per:(TOTAL-base); base+=cov;
    double rate=(ms>0)?(cov/(ms/1000.0)):0, eta=(rate>0)?((double)(TOTAL-base)/rate):0;
    printf("0x%012llx  %.4f%%  %.1f Mkeys/s  ETA %.2f h\r",base,(double)base/TOTAL*100.0,rate/1e6,eta/3600.0);
    fflush(stdout);
  }
  printf("\nEspacio agotado sin coincidencia.\n"); return 0;
}
