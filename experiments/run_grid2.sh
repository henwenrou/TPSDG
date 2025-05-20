#!/usr/bin/env bash
set -e

# —— 全局环境变量，减少底层并行计算的不确定性 ——
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1

# —— 超参数循环 ——
for lam in 2.0 ; do
  echo "===== Starting experiment with lambda = $lam ====="
  python main-train.py \
    -b configs/sc-topo-temp.yaml \
    --seed=42 \
    loss.topo.weight=$lam
done