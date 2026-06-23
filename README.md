# Recuperacion parcial de clave privada Ethereum (primeros 12 hex)

Recupera los **12 primeros caracteres hex** (48 bits) de tu clave privada, dados
los **52 hex restantes** y la **direccion Ethereum** correspondiente.

    priv = (prefijo << 208) | sufijo
      prefijo = 12 hex desconocidos  (16^12 = 2^48 ~ 2.8e14 candidatos)
      sufijo  = 52 hex conocidos

## Requisitos

- GPU NVIDIA (RTX 5090 -> arquitectura Blackwell `sm_120`).
- **CUDA Toolkit >= 12.8** (necesario para `sm_120`). Comprueba con `nvcc --version`.
- Driver NVIDIA reciente. Verifica la GPU con `nvidia-smi`.

## 1) Compilar

    nvcc -O3 -arch=sm_120 recover.cu -o recover

Si tu CUDA aun no soporta Blackwell, actualiza el toolkit. Para otra GPU:
`sm_89` (Ada/4090), `sm_86` (Ampere/3090).

## 2) Self-test OBLIGATORIO antes de nada

    ./recover --selftest

Debe imprimir `TODO OK`. Valida keccak, la aritmetica de campo, la curva
secp256k1 y el calculo de direccion contra vectores conocidos. Si **algo falla**,
no ejecutes la busqueda: pasame la salida y lo corrijo (no pude compilarlo en mi
entorno, asi que este paso es la red de seguridad).

Opcional: confirma los mismos vectores con la referencia en Python:

    pip install coincurve pycryptodome
    python3 eth_partial_recover.py

## 3) Validar tu sufijo y direccion

    python3 eth_partial_recover.py validate <sufijo_52hex> <direccion>

Esto confirma que el formato es correcto (sufijo de 52 hex, direccion de 40 hex).

## 4) Lanzar la busqueda

    ./recover --suffix <sufijo_52hex> --addr <direccion_40hex>

Ejemplo de formato (valores ficticios):

    ./recover --suffix 0123456789abcdef0123456789abcdef0123456789abcdef0123 \
              --addr 0xabcdef0123456789abcdef0123456789abcdef01

Salida en vivo: prefijo actual, % completado, Mkeys/s y ETA en horas.

### Reanudar tras una interrupcion

Anota el ultimo prefijo impreso y reanuda con:

    ./recover --suffix ... --addr ... --start <prefijo_hex>

### Ajustes de rendimiento

    --blocks N    (def. 4096)
    --threads N   (def. 256)
    --run N       (prefijos por hilo por lanzamiento, def. 4096)

Sube `--blocks` para saturar la 5090. Si el SO mata el proceso por tiempos de
kernel largos, baja `--run` (lanzamientos mas cortos, refresco mas frecuente).

## Notas

- La direccion objetivo es **imprescindible**: es lo que permite identificar el
  candidato correcto entre ~2.8e14. Sin ella la recuperacion es imposible.
- Las claves con `priv >= n` (orden de la curva) son invalidas; el programa no
  las filtra explicitamente, pero la clave real es valida y se encontrara en su
  prefijo correcto.
- Throughput real depende del tuning; el `--selftest` garantiza la correccion,
  no la velocidad.
