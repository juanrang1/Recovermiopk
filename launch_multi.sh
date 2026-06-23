#!/usr/bin/env bash
# launch_multi.sh - reparte la busqueda entre todas las GPUs de la maquina.
# Uso:  ./launch_multi.sh <SUFIJO_52HEX> <DIRECCION_40HEX>
#
# Cada GPU cubre un trozo disjunto de 0..2^48 con --start/--end.
# Los logs van a gpu0.log, gpu1.log, ...  Cuando una encuentra la clave,
# imprime "*** ENCONTRADA ***" en su log (revisa con: grep -l ENCONTRADA *.log).

set -e
SUF="$1"; ADDR="$2"
if [ -z "$SUF" ] || [ -z "$ADDR" ]; then
  echo "Uso: ./launch_multi.sh <SUFIJO_52HEX> <DIRECCION_40HEX>"; exit 1
fi

N=$(nvidia-smi -L | wc -l)
echo "GPUs detectadas: $N"

# 2^48 = 281474976710656 ; chunk por GPU (la ultima absorbe el resto via --end=2^48)
TOTAL=281474976710656
CHUNK=$(( TOTAL / N ))

for i in $(seq 0 $((N-1))); do
  S=$(( i * CHUNK ))
  if [ "$i" -eq $((N-1)) ]; then E=$TOTAL; else E=$(( (i+1) * CHUNK )); fi
  SH=$(printf '%012x' "$S")
  EH=$(printf '%012x' "$E")
  echo "GPU $i : [$SH , $EH)"
  CUDA_VISIBLE_DEVICES=$i ./recover_fast \
    --suffix "$SUF" --addr "$ADDR" \
    --start "$SH" --end "$EH" \
    --blocks 16384 --threads 256 \
    > "gpu${i}.log" 2>&1 &
done

echo "Lanzadas $N busquedas en segundo plano."
echo "Ver progreso:        tail -f gpu0.log"
echo "Ver si ya aparecio:  grep -H ENCONTRADA gpu*.log"
echo "Parar todo:          pkill -f recover_fast"
wait
