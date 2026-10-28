#!/usr/bin/env bash
set -e
set -x

# 方案A（postfilter）重训启动脚本
# 用法: bash train_pfA.sh [job_name] [gpu_ids] [config]
#   job_name 默认 pfA，保存到 /data/ssd0/jizhe.lu/exps/<job_name>
#   gpu_ids  默认 0（当前机器只有 1 张卡；确认 GPU 空闲后再跑）
#   config   默认 config_pfA.yaml；其他实验配置可显式传入（如 config_pfA_mixmag.yaml）
job_name=${1:-pfA}
GPUS=${2:-0}
config=${3:-/home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/config_pfA.yaml}
export DIST_BACKEND=gloo

save_folder="/data/ssd0/jizhe.lu/exps/${job_name}"

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="${GPUS}" python3 /home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/train.py \
    --cfg_filename "${config}" --save_folder "${save_folder}"
