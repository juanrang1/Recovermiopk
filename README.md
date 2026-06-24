# Recuperación parcial de clave privada (Ethereum)

Herramienta en CUDA para recuperar los caracteres hexadecimales que faltan de una
clave privada de Ethereum de la que **sí conoces el resto y la dirección pública**.

Dada una clave a la que le faltan **N** hex (al inicio o al final), prueba todas las
combinaciones posibles en GPU: para cada candidata calcula la clave pública secp256k1,
su hash Keccak-256, deriva la dirección y la compara con la dirección objetivo.

> **No es un buscador de claves al azar.** Solo sirve cuando ya tienes la mayor parte
> de la clave y la dirección. El espacio de búsqueda es 16^N, así que N pequeño se
> recupera al instante y N grande puede ser inviable (ver tabla de tiempos abajo).

---

## Archivos del repo

| Archivo | Para qué sirve | ¿Necesario? |
|---|---|---|
| `recover_fast.cu` | Programa principal en CUDA (el motor de búsqueda). | **Sí, esencial** |
| `recuperar.sh` | Lanzador todo-en-uno: compila, corre el self-test, reparte entre todas las GPUs, estima el tiempo y avisa cuando encuentra la clave. **Punto de entrada recomendado.** | **Sí** |
| `eth_partial_recover.py` | Oráculo en Python (lento). Valida que tu sufijo + dirección son coherentes y confirma los vectores del self-test antes de un run largo. No busca el espacio real. | Útil (verificación) |
| `launch_multi.sh` | Lanzador multi-GPU **antiguo**. Quedó **reemplazado por `recuperar.sh`**, que hace lo mismo y más. | Redundante — se puede borrar |

**Recomendación de limpieza:** `launch_multi.sh` ya no aporta nada (todo lo que hace lo
hace mejor `recuperar.sh`). Puedes borrarlo para evitar confusión:
```
git rm launch_multi.sh && git commit -m "Eliminar lanzador antiguo, reemplazado por recuperar.sh"
```
`eth_partial_recover.py` sí vale la pena conservarlo: es tu red de seguridad para validar
los datos antes de gastar horas de GPU.

---

## Requisitos

- GPU NVIDIA con CUDA (probado en **RTX 5090**, `sm_120`, CUDA 13).
- Toolkit de CUDA (`nvcc`) y un `g++` funcional. Las plantillas **cuda-devel** o
  **PyTorch** de Vast.ai ya lo traen.
- Python solo para el oráculo: `pip install coincurve pycryptodome`

## Puesta en marcha (pod nuevo)

```bash
# 1) Comprobar entorno (los tres deben responder sin error)
nvidia-smi -L
nvcc --version
g++ --version

# 2) Traer el código
git clone https://github.com/juanrang1/Recovermiopk.git
cd Recovermiopk

# 3) Dependencias del oráculo (opcional)
pip install coincurve pycryptodome
```

---

## Uso rápido (recomendado: `recuperar.sh`)

El script compila solo, corre el self-test y reparte entre todas las GPUs.

```bash
chmod +x recuperar.sh

# Faltan los PRIMEROS 12 hex (lo más común). --known = los 52 hex que SÍ tienes.
./recuperar.sh --known <52hex> --addr 0x<40hex> --missing start --n 12 --batch 256

# Faltan los ÚLTIMOS 4 hex.  --known = los 60 hex iniciales.
./recuperar.sh --known <60hex> --addr 0x<40hex> --missing end --n 4 --batch 256

# Solo verificar que compila y el self-test pasa, sin buscar:
./recuperar.sh --known <hex> --addr 0x<...> --selftest
```

Opciones de `recuperar.sh`:

| Flag | Significado | Default |
|---|---|---|
| `--known <hex>` | La parte conocida (mide 64−N hex). | — |
| `--addr <0x...>` | Dirección Ethereum objetivo (40 hex). | — |
| `--missing start\|end` | Dónde faltan los caracteres. | `start` |
| `--n <N>` | Cuántos hex faltan (1–15). | `12` |
| `--gpus <num>` | Forzar nº de GPUs. | autodetecta |
| `--batch <num>` | Tamaño de lote (recompila si cambia). **Usa 256.** | `128` |
| `--blocks` / `--threads` | Config del grid CUDA. | `16384` / `256` |
| `--rate <Mkeys>` | Mkeys/s por GPU para estimar el tiempo. | `677` |
| `--selftest` | Solo compilar + self-test. | — |
| `--yes` | No pedir confirmación en runs largos. | — |

> **Nota:** el óptimo medido es `--batch 256` (~5 % más rápido que 128). El script
> recompila automáticamente si cambias el batch.

## Uso directo del binario (1 GPU / depuración)

```bash
# Compilar (ajusta sm_XXX a tu GPU; 5090 = sm_120)
nvcc -O3 -arch=sm_120 -DBATCH=256 recover_fast.cu -o recover_fast

# Self-test obligatorio (debe decir "TODO OK")
./recover_fast --selftest

# Buscar: faltan los primeros 12 hex
./recover_fast --suffix <52hex> --addr 0x<40hex> --prefixlen 12 --blocks 16384 --threads 256

# Buscar: faltan los últimos 4 hex
./recover_fast --missing end --prefix <60hex> --addr 0x<40hex> --prefixlen 4
```

Flags del binario: `--suffix` (parte conocida si faltan al inicio), `--prefix` (si faltan
al final), `--missing start|end`, `--addr`, `--prefixlen N`, `--start/--end` (límites de
rango para repartir entre GPUs manualmente), `--blocks`, `--threads`, `--selftest`.

---

## Velocidad por GPU

Rendimiento del kernel (una sola tarjeta):

| GPU | Mkeys/s | Estado |
|---|---|---|
| RTX 5090 | **~700–750** | **medido** |
| RTX 4090 | ~450–550 | estimado |
| RTX 3090 | ~250–300 | estimado |
| A100 | ~350–450 | estimado |

> Solo la **5090 está medida**. El resto son estimaciones orientativas: el cuello de
> botella es la aritmética entera de 64 bits (multiplicación modular secp256k1 + un
> Keccak por clave), que escala distinto al FP32. Para saber el número real de tu
> tarjeta, compílalo y míralo en primer plano unos segundos.

El número escala casi linealmente con el nº de GPUs: 8× RTX 5090 ≈ **5.6–6.0 Gkeys/s**.

---

## Tiempo de recuperación según los caracteres que faltan

Tiempos en el **peor caso** (recorrer todo el espacio 16^N). **En promedio la clave
aparece a la mitad de tiempo.** Calculado a 700 Mkeys/s por GPU (conservador).

| Hex faltantes (N) | Combinaciones (16^N) | 1× RTX 5090 | 8× RTX 5090 |
|---:|---:|---:|---:|
| 4 | 65 536 | instantáneo | instantáneo |
| 5 | 1 048 576 | ~1 ms | instantáneo |
| 6 | 16 777 216 | 24 ms | 3 ms |
| 7 | 268 435 456 | 0.4 s | 48 ms |
| 8 | 4 294 967 296 | 6.1 s | 0.8 s |
| 9 | 68 719 476 736 | 1.6 min | 12.3 s |
| 10 | 1 099 511 627 776 | 26 min | 3.3 min |
| 11 | 17 592 186 044 416 | 7.0 h | 52 min |
| 12 | 281 474 976 710 656 | 4.7 días | 14.0 h |
| 13 | 4 503 599 627 370 496 | 74 días | 9.3 días |
| 14 | 72 057 594 037 927 936 | 3.3 años | 149 días |
| 15 | 1 152 921 504 606 846 976 | 52 años | 6.5 años |

**Regla práctica:** cada hex extra que falta multiplica el tiempo **×16**.
Hasta N≈11–12 es razonable con varias GPUs; de N≥13 en adelante deja de ser práctico.

---

## ⚠️ Seguridad

Si recuperas una clave en un **servidor alquilado** (Vast.ai, etc.), considérala
**expuesta**: la clave completa estuvo en una máquina que no controlas. Antes de hacer
nada, **mueve los fondos a una billetera nueva desde un equipo de tu confianza**, nunca
desde el pod. Trata la clave recuperada como comprometida en cuanto aparezca.

## Cómo funciona (resumen)

Para cada candidata: `priv = prefijo · 2^desplazamiento + parte_conocida`. Se calcula el
punto público con suma incremental sobre secp256k1 (con inversión modular por lotes tipo
Montgomery para amortizar la división), luego Keccak-256 de la clave pública y se comparan
los últimos 20 bytes con la dirección objetivo. Toda la aritmética usa limbs de 64 bits.
El `--selftest` valida los vectores conocidos (Keccak vacío, `EC(1)=G`, direcciones de
`priv=1` y `priv=2`) antes de cualquier búsqueda.
