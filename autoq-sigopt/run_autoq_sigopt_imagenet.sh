#!/usr/bin/env bash

# TODO: Modify and Review configuration below
WORKDIR=/path/to/nncf/autoq-sigopt/
DATA=/path/to/imagenet
RUN_ROOT=/tmp/autoq-sigopt-runs/
RUN_LABEL=resnet18_autoq
NNCF_CFG=/path/to/nncf/autoq-sigopt/cfg/resnet18_autoq.json
# NNCF_CFG=/path/to/nncf/autoq-sigopt/cfg/mobilenet_v2_autoq.json

# Pls signup an account here: https://app.sigopt.com/signup
# You can find the following information after signing in to sigopt at https://app.sigopt.com/tokens/info
SIGOPT_ID=<Your Client ID>
SIGOPT_TOKEN=<Your Client Token>

cd ${WORKDIR}
python image_classification.py \
    --gpu-id 0 \
    -m test \
    --config ${NNCF_CFG} \
    --data ${DATA}  \
    --log-dir ${RUN_ROOT}/${RUN_LABEL} \
    --sigopt-id ${SIGOPT_ID} \
    --sigopt-token ${SIGOPT_TOKEN}

