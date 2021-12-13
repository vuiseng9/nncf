#!/usr/bin/env bash

# This is an example of single gpu training

RUNDIR=/workspace/resnet18-qaas-ft
DATA=/data/dataset/imagenet/ilsvrc2012/torchvision
mkdir -p $RUNDIR

export CUDA_VISIBLE_DEVICES=0
cd /workspace/nncf/examples/classification
python main.py \
    -m train \
    --gpu-id 0 \
    --log-dir $RUNDIR \
    --config /workspace/nncf/qaas/cfg/resnet18_generated_ft_cfg.json \
    --data $DATA
