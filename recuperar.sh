#!/usr/bin/env bash
# =============================================================================
# recuperar.sh - Lanzador completo para recover_fast.
#
# - Funciona con 1, 4, 8 o las GPUs que tengas (autodetecta y reparte).
# - N de caracteres faltantes configurable (1..15): 4, 6, 8, 10, 12, 15...
# - Faltan al INICIO (--missing start) o al FINAL (--missing end).
# - Compila si hace falta, corre el self-test, estima el tiempo,
#   y se detiene solo cuando encuentra la clave (mostrandola).
#
# Uso:
#   ./recuperar.sh --known <HEX> --addr <0x...> [opciones]
#
# Argumentos:
#   --known <hex>     La parte que SI tienes. Debe medir (64 - N) hex.
#                       modo start -> son los ULTIMOS  (64-N) hex
#                       modo end   -> son los PRIMEROS (64-N) hex
#   --addr <hex>      Direccion Ethereum objetivo (40 hex, con o sin 0x).
#   --missing start|end   Donde faltan los caracteres. Default: start.
#   --n <N>           Cuantos hex faltan (1..15). Default: 12.
#   --gpus <num>      Forzar numero de GPUs (default: autodetecta).
#   --batch <num>     Tamano de lote (default 256, optimo medido). Recompila si cambia.
#   --blocks <num>    Default 16384.   --threads <num>  Default 256.
#   --rate <Mkeys>    Mkeys/s por GPU para estimar tiempo (default 720, ~RTX 5090).
#   --selftest        Solo compilar y correr el self-test, sin buscar.
#   --yes             No pedir confirmacion antes de un run largo.
#
# Ejemplos:
#   ./recuperar.sh --known <52hex> --addr 0x... --missing start --n 12
#   ./recuperar.sh --known <60hex> --addr 0x... --missing end   --n 4
# =============================================================================
set -euo pipefail

KNOWN="" ADDR="" MISSING="start" N=12 GPUS="" BATCH=256 BLOCKS=16384 THREADS=256
RATE=720 SELFTEST_ONLY=0 ASSUME_YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --known) KNOWN="$2"; shift 2;;
    --addr) ADDR="$2"; shift 2;;
    --missing) MISSING="$2"; shift 2;;
    --n) N="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --blocks) BLOCKS="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --rate) RATE="$2"; shift 2;;
    --selftest) SELFTEST_ONLY=1; shift;;
    --yes) ASSUME_YES=1; shift;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "Argumento desconocido: $1"; exit 1;;
  esac
done

# --------- compilar si hace falta (o si cambio el BATCH) ----------
SRC="recover_fast.cu"; BIN="recover_fast"
if [ ! -f "$SRC" ]; then echo "No encuentro $SRC en esta carpeta."; exit 1; fi
NEED_BUILD=0
[ ! -x "$BIN" ] && NEED_BUILD=1
[ -f ".batch" ] && [ "$(cat .batch)" != "$BATCH" ] && NEED_BUILD=1
[ ! -f ".batch" ] && NEED_BUILD=1
if [ "$NEED_BUILD" -eq 1 ]; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  echo "Compilando para sm_${CAP} con BATCH=${BATCH}..."
  nvcc -O3 -arch=sm_${CAP} -DBATCH=${BATCH} "$SRC" -o "$BIN"
  echo "$BATCH" > .batch
fi

# --------- self-test obligatorio ----------
echo "== Self-test =="
if ! ./"$BIN" --selftest | tee /tmp/_st.txt | grep -q "TODO OK"; then
  echo "Self-test FALLO. No se ejecuta la busqueda."; exit 2
fi
[ "$SELFTEST_ONLY" -eq 1 ] && { echo "Self-test OK."; exit 0; }

# --------- validaciones ----------
[ -z "$KNOWN" ] && { echo "Falta --known <hex>"; exit 1; }
[ -z "$ADDR" ]  && { echo "Falta --addr <0x...>"; exit 1; }
[ "$MISSING" = "start" ] || [ "$MISSING" = "end" ] || { echo "--missing debe ser start o end"; exit 1; }
case "$N" in (*[!0-9]*) echo "--n debe ser numero"; exit 1;; esac
[ "$N" -ge 1 ] && [ "$N" -le 15 ] || { echo "--n debe estar entre 1 y 15"; exit 1; }
KLEN=$(( 64 - N ))
KNOWN_CLEAN="${KNOWN#0x}"
if [ "${#KNOWN_CLEAN}" -ne "$KLEN" ]; then
  echo "La parte conocida debe medir $KLEN hex (porque faltan $N). Tiene ${#KNOWN_CLEAN}."; exit 1
fi
if ! printf '%s' "$KNOWN_CLEAN" | grep -qiE '^[0-9a-f]+$'; then echo "--known no es hexadecimal valido"; exit 1; fi

case "$MISSING" in
  start) KARG=(--suffix "$KNOWN_CLEAN") ;;
  end)   KARG=(--missing end --prefix "$KNOWN_CLEAN") ;;
esac

# --------- GPUs y reparto ----------
if [ -z "$GPUS" ]; then GPUS=$(nvidia-smi -L | wc -l); fi
[ "$GPUS" -ge 1 ] || { echo "No detecto GPUs."; exit 1; }
TOTAL=$(( 16 ** N ))
CHUNK=$(( TOTAL / GPUS ))

# --------- estimacion de tiempo (mitad del espacio en promedio) ----------
ETA=$(awk -v t="$TOTAL" -v g="$GPUS" -v r="$RATE" 'BEGIN{
  rate=g*r*1e6; secs=(t/2)/rate;
  if(secs<90) printf "%.0f segundos", secs;
  else if(secs<5400) printf "%.1f minutos", secs/60;
  else if(secs<172800) printf "%.1f horas", secs/3600;
  else if(secs<31536000) printf "%.1f dias", secs/86400;
  else printf "%.1f ANOS", secs/31536000;
}')

echo
echo "== Plan =="
echo "  faltan        : $N hex al $([ "$MISSING" = end ] && echo FINAL || echo INICIO)"
echo "  conocido      : $KNOWN_CLEAN  (${KLEN} hex)"
echo "  direccion     : $ADDR"
echo "  espacio       : 16^$N = $TOTAL candidatos"
echo "  GPUs          : $GPUS  (BATCH=$BATCH, ${BLOCKS}x${THREADS})"
echo "  tiempo estimado (promedio, $RATE Mkeys/s por GPU): ~$ETA"
echo

# aviso si es inviable
if awk -v t="$TOTAL" -v g="$GPUS" -v r="$RATE" 'BEGIN{exit !((t/2)/(g*r*1e6) > 259200)}'; then
  echo "  AVISO: con esta N el tiempo es enorme (> 3 dias). Considera mas GPUs o una N menor."
fi

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "Empezar la busqueda? [s/N] " ans
  case "$ans" in s|S|y|Y) ;; *) echo "Cancelado."; exit 0;; esac
fi

# --------- lanzar una busqueda por GPU ----------
rm -f gpu*.log
for i in $(seq 0 $((GPUS-1))); do
  S=$(( i * CHUNK ))
  if [ "$i" -eq $((GPUS-1)) ]; then E=$TOTAL; else E=$(( (i+1) * CHUNK )); fi
  SH=$(printf '%x' "$S"); EH=$(printf '%x' "$E")
  echo "GPU $i : [$SH , $EH)"
  CUDA_VISIBLE_DEVICES=$i ./"$BIN" \
    "${KARG[@]}" --addr "$ADDR" --prefixlen "$N" \
    --start "$SH" --end "$EH" --blocks "$BLOCKS" --threads "$THREADS" \
    > "gpu${i}.log" 2>&1 &
done

echo
echo "Buscando en $GPUS GPU(s). Logs: gpu0.log ... gpu$((GPUS-1)).log"
echo "Progreso de una:  tail -1 gpu0.log"
echo "Esperando resultado (se detiene solo al encontrar la clave)..."
echo

# --------- vigilante: detiene todo y muestra la clave al encontrarla ----------
while :; do
  if grep -q "ENCONTRADA" gpu*.log 2>/dev/null; then
    echo
    echo "==================== CLAVE ENCONTRADA ===================="
    grep -A2 "ENCONTRADA" gpu*.log
    echo "========================================================="
    pkill -f "$BIN" 2>/dev/null || true
    echo
    echo "SEGURIDAD: mueve los fondos desde TU equipo de confianza a una wallet NUEVA."
    echo "Esta clave paso por un servidor alquilado: tratala como comprometida."
    break
  fi
  # si ya no queda ningun proceso, terminamos (espacio agotado)
  if ! pgrep -f "$BIN" >/dev/null 2>&1; then
    echo "Las busquedas terminaron sin encontrar la clave. Revisa --known / --addr / --missing / --n."
    break
  fi
  sleep 5
done
