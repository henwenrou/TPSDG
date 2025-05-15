#!/bin/bash
# run_grid.sh

lambdas=(0.0 0.1 0.2 0.5 1.0 2.0)
for lam in "${lambdas[@]}"; do
  echo "===== Starting experiment with lambda = $lam ====="
  python main.py \
    -b configs/sc.yaml \
    --loss.topo.weight $lam
done