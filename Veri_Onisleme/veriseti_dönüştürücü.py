import json

# ✅ Hata almamak için yolun başına `r` koy veya `\\` kullan
with open(r"C:\Users\nezahat.korkmaz\Desktop\data_RAD\trainset.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

# ✅ Yeni formatta JSON oluştur
formatted_data = []

for entry in original_data:
    formatted_data.append({
        "id": str(entry["qid"]),  # "qid" → "id"
        "image": f"dataset/images/{entry['image_name']}",  # Görüntü yolu düzeltildi
        "conversations": [
            {"from": "human", "value": f"<image>\n{entry['question']}"},
            {"from": "gpt", "value": entry["answer"]}
        ]
    })

# ✅ Yeni JSON dosyasını kaydet
with open(r"C:\Users\nezahat.korkmaz\Desktop\data_RAD\trainset_llava.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print("✅ Veri başarıyla LLaVA formatına dönüştürüldü!")
