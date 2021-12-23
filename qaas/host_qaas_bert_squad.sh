# This current settings below correspond to containerized environment.

#!/usr/bin/env bash

export workload=bert-squad
export config=/workspace/nncf/qaas/cfg/bert_squad_qaas.json

# Pls revise path to dataset
# where subfolder 'train' is used as validation restapi and 'val' is used as test restapi
# ideally train should be a stratified sampled of full imagenet dataset
# val should be the original full imagenet dataset
export PYTHONPATH=/workspace/transformers/examples/pytorch/question-answering

export CUDA_VISIBLE_DEVICES=0

WORKDIR=/workspace/nncf

cd $WORKDIR
python qaas/manage.py \
        run \
        --no-reload \
        --host 0.0.0.0 


