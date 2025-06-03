#!/usr/bin/env bash
# run_all.sh

configs=(sc-topo sc-notopo)

# 串行执行
for cfg in "${configs[@]}"; do
  echo ">>> Running configs/${cfg}.yaml"
  python main.py -b configs/${cfg}.yaml
done
