#!/bin/bash

set -e  # Hata oluşursa script'i durdur
set -x  # Komutları çalıştırırken göster

export PYTHONIOENCODING=utf-8
export HF_HOME="/root/.cache/huggingface"
export TRANSFORMERS_CACHE="/root/.cache/huggingface/transformers"
export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/content/data_RAD/llava  # LLaVA'nın yolu

# Çalışma dizinine gidiyoruz
echo "📂 Çalışma dizinine gidiliyor..."
cd "/content/data_RAD/Fine_Tune" || { echo "🚨 Dizine gidilemedi!"; exit 1; }

echo "🚀 Fine-tuning başlatılıyor..."
echo "🔎 Kullanılan Deepspeed Config: /content/data_RAD/llava/scripts/zero3.json"

# Eğitim başlatılıyor
deepspeed train_med.py \
    --model_name_or_path "/content/data_RAD/llava-med-v1.5-mistral-7b" \
    --model_base "mistralai/Mistral-7B-Instruct-v0.2" \
    --data_path "/content/data_RAD/dataset/turkce_med_llava_veriseti.json" \
    --image_folder "/content/data_RAD/dataset/images" \
    --output_dir "/content/data_RAD/checkpoints/llava-med-finetune" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --local_rank 0  2>&1 | tee fine_tune_log.txt  # Logları kaydet

if [ $? -eq 0 ]; then
    echo "✅ Fine-tuning başarıyla tamamlandı!"
    ls -lh /content/data_RAD/checkpoints/llava-med-finetune  # Checkpoint dosyalarını göster

else
    echo "🚨 Fine-tuning sırasında hata oluştu!"
    exit 1
fi
