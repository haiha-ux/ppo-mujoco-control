#!/bin/bash
# Sequential grid. One process at a time, low priority.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=.venv/bin/python
wait_load() { while [ "$(cut -d. -f1 /proc/loadavg)" -gt 30 ]; do sleep 60; done; }
run() { t=""; [[ "$4" == "--tag" ]] && t=$5; n=$1_${t:-$2}_s$3; [ -f "logs/$n.time.json" ] && return; wait_load
  nice -n 19 ionice -c3 timeout 3000 $PY train.py --env $1 --impl $2 --seed $3 ${@:4}; }
for env in Hopper-v5 HalfCheetah-v5 Walker2d-v5; do
  for seed in 0 1 2; do for impl in ours sb3; do run $env $impl $seed; done; done
done
for seed in 0 1 2; do
  run Hopper-v5 ours $seed --tag noobsnorm --no-obs-norm
  run Hopper-v5 ours $seed --tag nogae --no-gae
done
echo ALLDONE
