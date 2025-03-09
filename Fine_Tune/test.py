import argparse
import torch
import sys
from PIL import Image
from transformers import CLIPImageProcessor

sys.path.append("/content/data_RAD/llava")
from llava.model.builder import load_pretrained_model

def test(model, tokenizer, args):
    print("🧪 Model test ediliyor...")
    model.eval()

    test_image_path = "/content/data_RAD/dataset/images/sample.jpg"
    test_question = "Bu görüntüde ne görüyorsunuz?"

    print(f"📷 Kullanılan Görüntü: {test_image_path}")
    print(f"❓ Kullanılan Soru: {test_question}")

    try:
        image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
        image = Image.open(test_image_path).convert("RGB")
        image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to("cuda")

        inputs = tokenizer(test_question, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(**inputs, max_length=100)

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"📜 Modelin cevabı: {decoded_output}")

    except Exception as e:
        print(f"❌ Test sırasında hata oluştu: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test Fine-tuned LLaVA Model")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336")

    args = parser.parse_args()

    print("📦 Eğitilmiş model yükleniyor...")
    
    try:
        result = load_pretrained_model(
            model_path=args.model_name_or_path,
            model_base=args.model_base,
            model_name=args.model_name_or_path.split("/")[-1],
            device_map="auto"
        )

        if isinstance(result, tuple) and len(result) == 4:
            model, tokenizer, image_processor, model_max_length = result
        else:
            raise ValueError(f"Beklenmeyen dönüş değeri: {result}")

        print("✅ Model başarıyla yüklendi!")
    
    except Exception as e:
        print(f"❌ Model yükleme sırasında hata oluştu: {e}")
        return

    model.to("cuda")
    test(model, tokenizer, args)

if __name__ == "__main__":
    main()
