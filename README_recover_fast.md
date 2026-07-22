# recover_fast — Recuperar claves privadas de Ethereum por GPU

Herramienta CUDA para recuperar una clave privada de Ethereum (secp256k1, 64
caracteres hex) cuando faltan caracteres **contiguos**, en cualquier posición:
al **inicio**, en el **medio** o al **final**.

Funciona con dirección conocida (`--addr`) o, si no la sabes, buscando contra una
base de datos de direcciones (`--basedatos`). Toda la criptografía está validada
contra vectores conocidos (Hardhat).

Usa fuerza bruta acelerada: suma incremental de puntos + inversión por lotes
(Montgomery) + reducción rápida del primo de secp256k1. ~700 millones de claves/seg
en una RTX 5090.

---

## AVISO DE SEGURIDAD (leer primero)

Si corres esto en una **GPU alquilada**, el proveedor tiene acceso root. Cualquier
clave recuperada ahí queda **comprometida**.

- Cuando aparezca la clave, **NO** la importes ni uses en el pod.
- Desde **tu propia máquina** (teléfono/PC de confianza): importa la clave, crea
  una wallet **nueva**, y mueve **todos los fondos** ahí de inmediato.

---

## 1. Requisitos y compilación

Necesitas CUDA (probado con 13.0) y una GPU NVIDIA. Para RTX 5090 es `sm_120`.

```bash
# compilar (5090=sm_120, 4090=sm_89, 3090=sm_86)
nvcc -O3 -arch=sm_120 recover_fast.cu -o recover_fast

# confirmar que el motor cripto quedo bien
./recover_fast --selftest      # debe decir: TODO OK
```

---

## 2. Concepto: la clave y lo que falta

Una clave privada de Ethereum son **64 caracteres hexadecimales** (256 bits). El
programa rellena los caracteres que faltan con todas las combinaciones posibles,
deriva la dirección de cada candidata, y la compara (con `--addr` o contra la base).

Los caracteres faltantes deben ser **contiguos** (un bloque seguido). Le dices al
programa dónde está el hueco con `--missing` y las partes conocidas:

| Dónde faltan | Cómo se indica |
|---|---|
| Al **inicio** (default) | `--suffix <parte conocida>` |
| Al **final** | `--missing end --prefix <parte conocida>` |
| En el **medio** | `--missing middle --prefix <hex> --suffix <hex>` |

El programa calcula solo cuántos faltan: `64 − (caracteres conocidos)`.

> Importante: la clave debe estar en **hex**. Un WIF (Bitcoin) o Base58 (Solana)
> parcial NO se recupera así — los caracteres Base58 no forman un rango contiguo
> en la clave. Para esos casos el método es distinto (checksum del WIF).

---

## 3. Uso con dirección conocida (`--addr`)

Acepta dirección completa (40 hex) o parcial con `...` (p.ej. `0x6c14...6d50`).

```bash
# Faltan los PRIMEROS N caracteres (conoces el resto = sufijo)
./recover_fast --suffix <hex conocido> --addr 0x6c14...6d50

# Faltan los ULTIMOS N caracteres (conoces el prefijo)
./recover_fast --missing end --prefix <hex conocido> --addr 0x6c14...6d50

# Faltan en el MEDIO (conoces prefijo y sufijo)
./recover_fast --missing middle --prefix <hex> --suffix <hex> --addr 0x6c14...6d50
```

---

## 4. Uso SIN dirección, contra la base de datos (`--basedatos`)

Cuando no sabes la dirección, busca la clave cuya dirección esté en una lista de
direcciones de Ethereum con saldo. **No necesita `--addr`.**

```bash
# Faltan al inicio
./recover_fast --basedatos saldos.txt --suffix <hex conocido>

# Faltan al final
./recover_fast --missing end --basedatos saldos.txt --prefix <hex conocido>

# Faltan en el medio
./recover_fast --missing middle --basedatos saldos.txt --prefix <hex> --suffix <hex>
```

Cuando un candidato derive una dirección que esté en la base, se detiene y muestra
la clave completa + la dirección encontrada.

### 4.1 Preparar la base de datos `saldos.txt`

Misma que para `seed_search`. Una dirección por línea (40 hex, con o sin `0x`).

```bash
# descargar (~4.6 GB)
wget https://privatekeyfinder.io/assets/downloads/ethereum.tsv.gz

# descomprimir al vuelo y quedarse con la columna de direcciones -> saldos.txt (~1.3 GB)
gunzip -c ethereum.tsv.gz | cut -f1 > saldos.txt

# verificar
head saldos.txt        # direcciones de 40 hex, una por linea
wc -l saldos.txt       # cuantas hay (p.ej. ~174 millones)
```

**Cero falsos positivos:** el programa usa un filtro de Bloom (rechazo rápido) y
luego **verifica exacto** contra la base ordenada dentro de la GPU. Si encuentra
algo, es real (la probabilidad de una coincidencia falsa es ~10⁻²⁶).

---

## 5. Flags

| Flag | Qué hace |
|---|---|
| `--missing start\|end\|middle` | Dónde faltan los caracteres (default: `start`). |
| `--prefix <hex>` | Parte conocida al inicio (para `end` y `middle`). |
| `--suffix <hex>` | Parte conocida al final (para `start` y `middle`). |
| `--addr 0x...` | Dirección objetivo (completa o parcial `...`). |
| `--basedatos archivo` | Busca contra una lista de direcciones (sin `--addr`). |
| `--prefixlen N` | Nº de caracteres faltantes (default 12; en `middle` se calcula solo). |
| `--start / --end` | Sub-rango en hex (para reparto multi-GPU). |
| `--blocks / --threads / --run` | Ajustes de lanzamiento (normalmente no hace falta). |
| `--selftest` | Prueba el motor y sale. |

---

## 6. Viabilidad (cuánto tarda)

Cada carácter hex que falta multiplica el tiempo por 16. Velocidad ~700M claves/s
por RTX 5090.

| Faltan (hex) | Combinaciones | 1 GPU | 8 GPU |
|---:|---:|---:|---:|
| 8 | 2^32 | segundos | instantáneo |
| 10 | 2^40 | ~25 min | ~3 min |
| 11 | 2^44 | ~7 h | ~55 min |
| 12 | 2^48 | ~4 días | ~12 h |
| 13 | 2^52 | ~70 días | ~9 días |
| 15+ | 2^60+ | inviable | inviable |

**Techo práctico: ~12-13 caracteres** con una GPU. Funciona igual en inicio, medio
o final (la posición no cambia el tiempo, solo el número de caracteres faltantes).

Si te faltan **15+ caracteres contiguos al final** y tienes la **clave pública**
(no solo la dirección), Kangaroo (herramienta aparte) los recupera en √n — mucho
más lejos. Pero para inicio/medio y para lo demás, `recover_fast` es lo correcto.

---

## 7. Prueba antes de gastar GPU

Antes de una búsqueda larga, prueba con una clave de juguete tuya: toma una clave
conocida, quítale unos caracteres, y confirma que la recupera. Empieza con pocos
caracteres faltantes (4-5) para ver que todo el flujo funciona.

```bash
# ejemplo: faltan 4 al final de una clave conocida, con su direccion
./recover_fast --missing end --prefix <60 hex conocidos> --addr 0x<direccion> --prefixlen 4
```

---

## 8. Multi-GPU

Reparte el rango `[0, 16^N)` entre GPUs con `--start/--end` (en hex), una instancia
por GPU con `CUDA_VISIBLE_DEVICES`. Ejemplo con 2 GPUs y 11 caracteres faltantes
(16^11 = 0x100000000000):

```bash
CUDA_VISIBLE_DEVICES=0 ./recover_fast --missing end --basedatos saldos.txt \
  --prefix <hex> --start 0            --end 80000000000 &
CUDA_VISIBLE_DEVICES=1 ./recover_fast --missing end --basedatos saldos.txt \
  --prefix <hex> --start 80000000000  --end 100000000000 &
```

---

## 9. Resultado

Durante la búsqueda verás progreso con `% ... Mkeys/s ... ETA`. Cuando encuentra:

```
*** ENCONTRADA EN LA BASE *** direccion 0x<direccion>
*** ENCONTRADA ***
parte faltante (N hex, ...): <los caracteres>
clave privada  : <los 64 hex completos>
```

Si termina con "Espacio agotado sin coincidencia", ninguna combinación coincidió:
revisa la parte conocida, dónde están los caracteres faltantes, o (si usaste base
de datos) que la wallet esté en ella.

---

## Archivos del proyecto

| Archivo | Para qué |
|---|---|
| `recover_fast.cu` | Programa principal. |
| `saldos.txt` | Lista de direcciones (la preparas tú, ver sección 4.1). |
| `README.md` | Este archivo. |
