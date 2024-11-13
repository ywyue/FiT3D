#!/bin/bash

model="dinov2_small" # ["dinov2_small", "dinov2_base", "dinov2_reg_small", "clip_base", "mae_base", "deit3_base"]
ngpu=8
dataset="scannetpp" # ["scannetpp", "scannet", "nyu", "kitti"]
if [[ $model == *"small"* ]]; then
    vit="vits"
elif [[ $model == *"base"* ]]; then
    vit="vitb"
fi
config="evaluation/baseline_configs/${vit}_${dataset}_depth_linear_config.py"
workdir="work_dirs/baseline_depth_eval/${dataset}/${model}"
bash evaluation/scripts/dist_eval_depth.sh \
    ${config} ${ngpu} \
    --backbone-type $model \
    --work-dir ${workdir} 