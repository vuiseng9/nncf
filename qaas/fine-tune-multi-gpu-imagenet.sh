#!/usr/bin/env bash

# This is an example of multi-gpu training

RUNDIR=/workspace/resnet18-qaas-ft
DATA=/data/dataset/imagenet/ilsvrc2012/torchvision
mkdir -p $RUNDIR

export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /workspace/nncf/examples/classification
python main.py \
    -m train \
    --multiprocessing-distributed \
    --log-dir $RUNDIR \
    --config /workspace/nncf/qaas/cfg/resnet18_generated_ft_cfg.json \
    --data $DATA