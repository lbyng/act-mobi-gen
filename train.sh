#!/bin/bash

export WANDB_MODE=offline

TASK_NAME="${1:-d1_mobile_manipulation}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="./ckpts/${TASK_NAME}_${TIMESTAMP}"

python imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --policy_class ACT \
    --batch_size 32 \
    --seed 0 \
    --num_steps 5000 \
    --lr 1e-4 \
    --chunk_size 100 \
    --kl_weight 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --validate_every 500 \
    --save_every 1000
