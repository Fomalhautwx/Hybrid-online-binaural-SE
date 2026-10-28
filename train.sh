#!/usr/bin/env bash
set -e
set -x

job_name=$1
n_gpu=$2
# export HOME=/tmp/"$user_name"

# 使用nlu的image, 需要安装2个关于语音的库包
# pip3 install soundfile -i https://pypi.tuna.tsinghua.edu.cn/simple --user
# pip3 install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple --user

save_folder="/data/ssd0/jizhe.lu/exps/${job_name}"
config="/home/nis/jizhe.lu/Code/paris-dev/config.yaml"
#config="/home/nis/jizhe.lu/Code/paris-dev/config_single.yaml"

# OMP_NUM_THREADS=4 torchrun --nproc_per_node="${n_gpu}" /home/nis/jizhe.lu/Code/DNS3_2Mix_noisy_reverb/train.py --cfg_filename "${config}" --save_folder "${save_folder}"
CUDA_VISIBLE_DEVICES=0,1 python3 /home/nis/jizhe.lu/Code/paris-dev/train.py --cfg_filename "${config}" --save_folder "${save_folder}"

# OMP_NUM_THREADS=4 torchrun --nproc_per_node="${n_gpu}" --master_port=29500 /home/nis/jizhe.lu/gtcrn/MultiChGTCRN/train.py --ckpt "${stage1_ckpt}" \
#   --train_scp "${train_scp}" \
#   --cv_scp "${cv_scp}" \
#   --pool_kernel "${pool_kernel}" \
#   --dropout "${dropout}" \
#   --is_online "${is_online}" \
#   --chunk_ms "${chunk_ms}" \
#   --batch_size "${batch_size}" \
#   --epochs "${epochs}" \
#   --use_cuda "${use_cuda}" \
#   --lr "${lr}" \
#   --num_workers "${num_workers}" \
#   --save_folder "${save_folder}" \
#   --continue_from "${continue_from}" \
#   --print_freq "${print_freq}"
