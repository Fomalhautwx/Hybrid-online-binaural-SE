#!/usr/bin/env bash
set -e
set -x

# 双掩码端到端 DC-GTCRN 训练启动脚本（公平对照：与双掩码级联同架构思想，但无 WPE/IVA 前级）
# 用法: bash train_dcg_dual.sh [job_name] [gpu_ids] [config]
#   job_name 默认 dcg_dual，保存到 /data/ssd0/jizhe.lu/exps/<job_name>
#   gpu_ids  默认 0（0,1 = 双卡）
#   config   默认 config_dcg_dual.yaml
job_name=${1:-dcg_dual}
GPUS=${2:-0}
config=${3:-/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/config_dcg_dual.yaml}

save_folder="/data/hdd0/jizhe.lu/exps/${job_name}"

DIST_BACKEND=gloo CUDA_VISIBLE_DEVICES="${GPUS}" /home/nis/jizhe.lu/anaconda3/envs/gtcrntest/bin/python3 \
    /home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/train_dcg.py \
    --cfg_filename "${config}" --save_folder "${save_folder}"
