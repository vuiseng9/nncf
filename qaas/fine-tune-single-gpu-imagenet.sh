#!/usr/bin/env bash

# This is an example of single gpu training

RUNDIR=/workspace/resnet18-qaas-ft
DATA=/data/dataset/imagenet/ilsvrc2012/torchvision

FTCFG=/workspace/nncf/qaas/cfg/mobilenet_v2_generated_ft_cfg.json
#FTCFG=/workspace/nncf/qaas/cfg/resnet18_generated_ft_cfg.json
#FTCFG=/workspace/nncf/qaas/cfg/resnet50_generated_ft_cfg.json
#FTCFG=/workspace/nncf/qaas/cfg/resnext101_32x8d_generated_ft_cfg.json

mkdir -p $RUNDIR

export CUDA_VISIBLE_DEVICES=0
cd /workspace/nncf/examples/torch/classification
python main.py \
    -m train \
    --gpu-id 0 \
    --log-dir $RUNDIR \
    --config $FTCFG \
    --data $DATA
