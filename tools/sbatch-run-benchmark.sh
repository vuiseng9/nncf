#!/usr/bin/env bash

conda activate tpc-nncf
cd /data2/vchua/dev/tpc-nncf/nncf/tools
python3 benchmark_quantize_layers.py