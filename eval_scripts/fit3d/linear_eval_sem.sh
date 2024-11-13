#!/bin/bash

model="dinov2_small" # ["dinov2_small", "dinov2_base", "dinov2_reg_small", "clip_base", "mae_base", "deit3_base"]
ngpu=4
dataset="scannetpp" # ["scannetpp", "scannet", "nyu", "ade20k", "voc2012"]
if [[ $model == *"small"* ]]; then
    vit="vits"
elif [[ $model == *"base"* ]]; then
    vit="vitb"
fi
config="evaluation/configs/${vit}_${dataset}_sem_linear_config.py"
workdir="work_dirs/segmentation_eval/${dataset}/${model}"
bash evaluation/scripts/dist_eval_segmentation.sh \
    ${config} ${ngpu} \
    --backbone-type $model \
    --work-dir ${workdir}
