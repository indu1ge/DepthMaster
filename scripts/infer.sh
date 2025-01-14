#!/usr/bin/env bash
set -e
set -x

export CUDA_VISIBLE_DEVICES=5
python run.py \
    --checkpoint ckpt/eval \
    --processing_res 768 \
    --input_rgb_dir in_the_wild_example/input \
    --output_dir in_the_wild_example/output/final \