#!/usr/bin/env bash

LOGDIR=/data1/vchua/clipopt/actdist/imgnet
DATA=/data/dataset/imagenet/ilsvrc2012/imgnet-train5k-val50k
NNCFCFG=/home/vchua/practical-q/nncf/examples/torch/classification/configs/quantization/rn18_imgnet_int8.json
# NNCFCFG=/home/vchua/practical-q/nncf/examples/torch/classification/configs/quantization/mbn2_imgnet_int8.json

mkdir -p $LOGDIR

# export CUDA_VISIBLE_DEVICES=0,1
cd /home/vchua/practical-q/nncf/examples/torch/classification/
nohup python main.py \
    --gpu-id 3 \
    -m actdist \
    --log-dir $LOGDIR \
    --config $NNCFCFG \
    --data $DATA 2>&1 | tee $LOGDIR/run.log &
    # --multiprocessing-distributed \
