#!/usr/bin/env bash

NNCFCFG=configs/full_precision/resnet18_cifar10.json
OUTDIR=/tmp/rn18-cifar10
DATADIR=/tmp/data

GPUID=0
BS=256
python3 main.py \
    -m train \
    --gpu-id $GPUID \
    -j 8 \
    -b $BS \
    --config $NNCFCFG \
    --log-dir $OUTDIR \
    --data $DATADIR

