#!/usr/bin/env bash

# conda activate

WORKDIR=$(pwd)
DATADIR=/data/dataset/imagenet/ilsvrc2012/torchvision/

function dump_quantize_kernel_io {
    # xml niter bs seqlen outdir verbose
    nncf_cfg=$1

    cd $WORKDIR
    python3 main.py \
        -m test \
        --gpu-id 0 \
        -j 16 \
        -b 64 \
        --log-dir /tmp/nncf-dump-qio/ \
        --dump_quantize_io \
        --config $nncf_cfg \
        --data $DATADIR
}

nncfcfg_list=$(ls qcfg-sweep/*)

for cfg in $nncfcfg_list;
do
    sleep 5
    echo "[Info] Running dump_quantize_kernel_io $cfg"
    dump_quantize_kernel_io $cfg
done