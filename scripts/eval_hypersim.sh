#!/usr/bin/env bash
set -e
set -x

export CUDA_VISIBLE_DEVICES=5
python evaluate.py \
    --base_data_dir path/to/basedata \
    --dataset_config config/dataset/data_hypersim_test.yaml \
    --alignment least_square_sqrt_disp \
    --output_dir output/hypersim/final \
    --checkpoint ckpt/eval \
    --processing_res 0 \
    --seed 1234 \
