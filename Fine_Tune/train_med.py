import argparse
import os
import torch
import deepspeed
import sys

# LLaVA'nın yolunu ekleyelim (llava/ içinde kodlar yoksa, bu satırı kaldırabilirsin)
sys.path.append("../llava")

from llava.model.builder import load_pretrained_model

def train(model, tokenizer, args):
    print("✅ Model eğitim için hazır!")
    print(f"📂 Veri Seti: {args.data_path}")
    print(f"📸 Görseller: {args.image_folder}")
    print(f"📍 Çıktı Klasörü: {args.output_dir}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_train_epochs):
        print(f"🚀 Epoch {epoch+1}/{args.num_train_epochs} başladı...")
        optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)  # Örnek loss değeri
        loss.backward()
        optimizer.step()
        print(f"🎯 Epoch {epoch+1} tamamlandı!")

def main():
    parser = argparse.ArgumentParser(description="Train Medical LLaVA using LoRA")
    
    # String, Integer ve Float Argümanları
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=True)  
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mm_projector_lr", type=float, default=2e-5)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu")
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--version", type=str, default="v1")

    # Boolean Parametreler
    parser.add_argument("--lora_enable", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true")
    parser.add_argument("--mm_use_im_patch_token", action="store_true")
    parser.add_argument("--group_by_modality_length", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--lazy_preprocess", action="store_true")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    args = parser.parse_args()

    # **Model Yükleme**
    model, tokenizer = load_pretrained_model(args.model_base, args.model_name_or_path)

    train(model, tokenizer, args)

if __name__ == "__main__":
    main()
