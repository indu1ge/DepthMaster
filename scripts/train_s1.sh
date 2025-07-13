BASE_DATA_DIR="path/to/basedata"
BASE_CKPT_DIR="path/to/sd2_ckpt"

export CUDA_VISIBLE_DEVICES=3
python train_s1.py --config config/train_s1.yaml \
    --base_data_dir $BASE_DATA_DIR \
    --base_ckpt_dir $BASE_CKPT_DIR \
    --output_dir log/stage1_bs8 \
    --no_wandb \