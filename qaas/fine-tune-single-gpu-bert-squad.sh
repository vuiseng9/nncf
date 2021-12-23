#!/usr/bin/env bash

# This is an example of multi-gpu training

RUNDIR=/workspace/bert-base-uncased-squad-qaas-ft

FTCFG=/workspace/nncf/qaas/cfg/bert_squad_generated_ft_cfg.json

# Some datapoints on memory requirements for fine-tuning job
#|-------------------------|------------|-------|---------------|
#| model                   | batch_size | n_gpu | mem. per card |
#|-------------------------|------------|-------|---------------|
#| bert-base-uncased-squad | 8          | 1     | 10 GB         |
#|-------------------------|------------|-------|---------------|

mkdir -p $RUNDIR

export CUDA_VISIBLE_DEVICES=0

WORKDIR=/workspace/transformers/examples/pytorch/question-answering
cd $WORKDIR

python run_qa.py \
    --model_name_or_path vuiseng9/bert-base-uncased-squad \
    --dataset_name squad \
    --do_eval \
    --do_train \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --doc_stride 128 \
    --max_seq_length 384 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --evaluation_strategy steps \
    --eval_steps 2000 \
    --save_steps 1000000 \
    --logging_steps 1 \
    --overwrite_output_dir \
    --nncf_config $FTCFG \
    --output_dir $RUNDIR