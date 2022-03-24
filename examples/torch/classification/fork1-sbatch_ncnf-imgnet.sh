#!/usr/bin/env bash


NNCFCFG=rn50-imgnet-const-8bit.json

OUTROOT=/nvme1/vchua/run/unstructured-rn50/
RUNID=rn50-85pc-sparse-8bit
SOURCECKPT=/nvme1/vchua/run/sparse-nn-pnc/rn50_85pc_sparse_FP32/85_sparse_resnet50_75136.pth.tar

NOW=$(date +"%Y%m%d_%H%M%S")
OUTDIR=$OUTROOT/$RUNID-$NOW

export CUDA_VISIBLE_DEVICES=0
CONDAROOT=/nvme1/vchua/miniconda3
CONDAENV=nm-sparsity

WORKDIR=/nvme1/vchua/dev/nm-sparsity/nncf/examples/torch/classification
DATADIR=/nvme1/datasets/ilsvrc2012/torchvision/

cmd="
python main.py \
    -m {train,test,export} \
    --gpu-id 0 \
    -j 8 \
    --weights $SOURCECKPT \
    --config $NNCFCFG \
    --data $DATADIR \
    --log-dir $OUTDIR
"
OUTDIR=$OUTROOT/$RUNID

if [[ $1 == "local" ]]; then
    echo "${cmd}" > $OUTDIR/cmd.log
    echo "### End of CMD ---" >> $OUTDIR/cmd.log
    cmd="nohup ${cmd}"
    eval $cmd &
    echo "logpath: $OUTDIR/output.log"
    # eval $cmd >> $OUTDIR/run.log 2>&1 &
    # echo "logpath: $OUTDIR/run.log"
else
    source $CONDAROOT/etc/profile.d/conda.sh
    conda activate ${CONDAENV}
    cd $WORKDIR
    eval $cmd
fi

# elif [[ $1 == "dryrun" ]]; then
#     echo "[INFO: dryrun, add --max_steps 25 to cli"
#     cmd="${cmd} --max_steps 25"
#     echo "${cmd}" > $OUTDIR/dryrun.log
#     echo "### End of CMD ---" >> $OUTDIR/dryrun.log
#     eval $cmd >> $OUTDIR/dryrun.log 2>&1 &
#     echo "logpath: $OUTDIR/dryrun.log"
