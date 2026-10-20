#!/usr/bin/env bash
# Train + score every arm in order. Safe to re-run: finished runs are skipped,
# interrupted runs resume from last.pt, finished evals are skipped.
set -u
cd "$(dirname "$0")"
PY=".venv/bin/python -E -s"
JOBS=${JOBS:-"A_s0 B_s0 C_s0 A_s1 B_s1 C_s1 dose100_s0 dose250_s0 dose1000_s0 dose2000_s0"}
for job in $JOBS; do
  arm=${job%_s*}; seed=${job##*_s}; run=runs/$job
  if [ ! -f $run/DONE ]; then
    if [ -f $run/weights/last.pt ]; then $PY -u train.py --arm $arm --seed $seed --resume || exit 1
    else $PY -u train.py --arm $arm --seed $seed || exit 1; fi
  fi
  for split in day_test night_test dusk_test; do
    [ -f runs/eval/${job}__$split.json ] || $PY -u evaluate.py --weights $run/weights/last.pt --split $split || exit 1
  done
done
echo "QUEUE FINISHED"
