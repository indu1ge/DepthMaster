#!/usr/bin/env bash
set -e
set -x

export CUDA_VISIBLE_DEVICES=5
python evaluate.py \
    --base_data_dir path/to/basedata \
    --dataset_config config/dataset/data_diode_all.yaml \
    --alignment least_square_sqrt_disp \
    --output_dir output/diode/final \
    --checkpoint ckpt/eval \
    --processing_res 640 \
    --seed 1234 \
