BASE_DATA_DIR="/zhdd/dataset"
BASE_CKPT_DIR="ori_ckpt"

export CUDA_VISIBLE_DEVICES=2
python train_s2.py --config config/train_s2.yaml \
    --base_data_dir $BASE_DATA_DIR \
    --base_ckpt_dir $BASE_CKPT_DIR \
    --output_dir log/stage2 \
    --no_wandb \