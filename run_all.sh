#!/usr/bin/env bash
# run_all.sh

configs=(lb-topo lb-notopo bl-topo bl-notopo)

# 串行执行
for cfg in "${configs[@]}"; do
  echo ">>> Running configs/${cfg}.yaml"
  python main.py -b configs/${cfg}.yaml
done
