#!/bin/bash

set -e
export PYTHONIOENCODING=utf-8
export HF_HOME="/root/.cache/huggingface"
export TRANSFORMERS_CACHE="/root/.cache/huggingface/transformers"
export WANDB_DISABLED=true

cd "/content/data_RAD/Fine_Tune"

echo "🔍 Model testi başlatılıyor..."

python3 "test_med.py" \
    --model_name_or_path "/content/data_RAD/checkpoints/llava-med-finetune" \
    --model_base "mistralai/Mistral-7B-Instruct-v0.2" \
    --vision_tower "openai/clip-vit-large-patch14-336"

echo "✅ Model testi tamamlandı!"
