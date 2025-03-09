#!/bin/bash

export PYTHONIOENCODING=utf-8
export HF_HOME="/c/Users/nezahat.korkmaz/.cache/huggingface"
export TRANSFORMERS_CACHE="/c/Users/nezahat.korkmaz/.cache/huggingface/transformers"
export WANDB_DISABLED=true

deepspeed /content/train_med.py \
    --lora_enable \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed /content/llava/scripts/zero3.json \
    --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
    --model_base mistralai/Mistral-7B-Instruct-v0.2 \
    --version v1 \
    --data_path /content/Turkish_MED_LLaVa_formatted_data_RAD/dataset/turkce_med_llava_veriseti.json \
    --image_folder /content/Turkish_MED_LLaVa_formatted_data_RAD/dataset/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --mm_use_im_patch_token \
    --image_aspect_ratio pad \
    --group_by_modality_length \
    --bf16 \
    --output_dir /content/checkpoints/llava-med-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 \
    --model_max_length 2048 \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --lazy_preprocess \
    --report_to wandb
