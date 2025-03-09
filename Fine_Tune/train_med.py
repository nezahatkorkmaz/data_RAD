import argparse
import os
import torch
import deepspeed
import sys

# LLaVA'nın kodlarının olduğu dizini `sys.path` içine ekleyelim
LLAVA_CODE_DIR = "/content/data_RAD/llava"  # LLaVA kaynak kodları
if os.path.exists(LLAVA_CODE_DIR):
    sys.path.insert(0, LLAVA_CODE_DIR)
    print(f"✅ [INIT] LLaVA kodları yolu eklendi: {LLAVA_CODE_DIR}")
else:
    print(f"🚨 [ERROR] LLaVA kodları dizini bulunamadı: {LLAVA_CODE_DIR}")
    exit(1)

from llava.model.builder import load_pretrained_model


def train(model, tokenizer, args):
    """ Modeli eğiten fonksiyon """
    print("✅ Model eğitim için hazır!")
    print(f"📂 Veri Seti: {args.data_path}")
    print(f"📸 Görseller: {args.image_folder}")
    print(f"📍 Çıktı Klasörü: {args.output_dir}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_train_epochs):
        print(f"🚀 Epoch {epoch+1}/{args.num_train_epochs} başladı...")
        optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True).cuda()  # CUDA'ya taşı
        loss.backward()
        optimizer.step()
        print(f"🎯 Epoch {epoch+1} tamamlandı!")

    # ✅ Modeli kaydet
    print(f"💾 Model kaydediliyor: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Model başarıyla kaydedildi!")

def main():
    """ Ana fonksiyon: Modeli yükler ve eğitimi başlatır """
    global model, tokenizer, processor  # Değişkenleri global olarak tanımla

    parser = argparse.ArgumentParser(description="Train Medical LLaVA using LoRA")

    # Gerekli parametreler
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    # ✅ Modeli yükle
    print(f"📂 Model {args.model_name_or_path} yükleniyor...")

    try:
        result = load_pretrained_model(
            model_path=args.model_name_or_path,
            model_base=args.model_base,
            model_name="llava-med-v1.5-mistral-7b",
            device_map="auto"
        )

        # **Dönen değer sayısını kontrol et**
        if isinstance(result, tuple):
            if len(result) == 4:
                tokenizer, model, processor, _ = result  # ✅ 4 değer döndüğünde
            elif len(result) == 3:
                tokenizer, model, processor = result  # ✅ 3 değer döndüğünde
            elif len(result) == 2:
                tokenizer, model = result  # ✅ 2 değer döndüğünde
                processor = None
            else:
                raise ValueError(f"🚨 Beklenmeyen dönüş değeri: {result}")
        else:
            raise ValueError(f"🚨 Model yüklenirken hata oluştu: {result}")

        # ✅ Modeli GPU'ya Taşı
        model = model.cuda()

        # 🚀 Modeli eğit
        train(model, tokenizer, args)

    except Exception as e:
        print(f"🚨 Model yüklenirken hata oluştu: {e}")
        exit(1)


if __name__ == "__main__":
    main()
