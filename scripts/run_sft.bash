export WANDB_BASE_URL="https://api.bandw.top"
# export CUDA_VISIBLE_DEVICES=0,1

accelerate launch utils/sft.py \
    --model_path /root/autodl-fs/models/Qwen2.5-Math-1.5B \
    --sft_jsonl /root/assignment5-alignment/data/math/train.jsonl \
    --val_jsonl /root/assignment5-alignment/data/math/test.jsonl \
    --out_dir ./out \
    --devices cuda:0 \
    --epochs 3 \
    --micro_batch_size 1 \
    --lr 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_every_steps 10 \
    --dataset_sizes -1
