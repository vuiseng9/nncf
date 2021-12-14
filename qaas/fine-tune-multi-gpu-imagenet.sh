#!/usr/bin/env bash

# This is an example of multi-gpu training

RUNDIR=/workspace/resnet18-qaas-ft
DATA=/data/dataset/imagenet/ilsvrc2012/torchvision

FTCFG=/workspace/nncf/qaas/cfg/mobilenet_v2_generated_ft_cfg.json
#FTCFG=/workspace/nncf/qaas/cfg/resnet18_generated_ft_cfg.json
#FTCFG=/workspace/nncf/qaas/cfg/resnet50_generated_ft_cfg.json
#FTCFG=/workspace/nncf/qaas/cfg/resnext101_32x8d_generated_ft_cfg.json

# Some datapoints on memory requirements for fine-tuning job
#|------------------|------------|-------|---------------|
#| model            | batch_size | n_gpu | mem. per card |
#|------------------|------------|-------|---------------|
#| resnet50         | 256        | 4     | 12 GB         |
#| mobilenet_v2     | 200        | 2     | 10 GB         |
#| resnext101_32x8d | 56         | 4     | 10.5 GB       |
#|------------------|------------|-------|---------------|

mkdir -p $RUNDIR

export CUDA_VISIBLE_DEVICES=0,1.2,3
cd /workspace/nncf/examples/torch/classification
python main.py \
    -m train \
    --multiprocessing-distributed \
    --log-dir $RUNDIR \
    --config $FTCFG \
    --data $DATA
