#!/usr/bin/env bash
LAMBDAS=(0 0.1 0.3 0.5 1 2)          # 粗网格，自行增删

for L in "${LAMBDAS[@]}"; do
  LOGDIR=runs/lambda_${L}
  python main.py \
        --base configs/sc.yaml \
        --logdir ${LOGDIR} \
        optimizer.max_epoch=10 \              # 先跑 10 epoch
        loss.topo.enabled=$( [ "$L" = "0" ] && echo false || echo true ) \
        loss.topo.weight=${L}
done