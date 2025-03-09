import json
import os

# ✅ Dosya yolları
trainset_path = r"C:\Users\nezahat.korkmaz\Desktop\data_RAD\trainset_llava.json"
testset_path = r"C:\Users\nezahat.korkmaz\Desktop\data_RAD\testset_llava.json"
imgid2idx_path = r"C:\Users\nezahat.korkmaz\Desktop\data_RAD\imgid2idx.json"

# ✅ JSON dosyalarını oku
with open(trainset_path, "r", encoding="utf-8") as f:
    trainset_data = json.load(f)

with open(testset_path, "r", encoding="utf-8") as f:
    testset_data = json.load(f)

with open(imgid2idx_path, "r", encoding="utf-8") as f:
    imgid2idx_data = json.load(f)

print("✅ Dosyalar başarıyla yüklendi!")

# ✅ "dataset/images/" kısmını temizleyerek sadece resim adlarını al
def normalize_filename(path):
    return os.path.basename(path)

# ✅ ID'leri imgid2idx.json'dan alarak güncelle
def update_ids(dataset, imgid2idx):
    for entry in dataset:
        image_name = normalize_filename(entry["image"])
        if image_name in imgid2idx:
            entry["id"] = str(imgid2idx[image_name])  # ID'yi güncelle
        else:
            print(f"❌ Uyarı: {image_name} için imgid2idx.json içinde ID bulunamadı!")
    return dataset

# ✅ Train ve Test setleri için ID güncelle
trainset_data = update_ids(trainset_data, imgid2idx_data)
testset_data = update_ids(testset_data, imgid2idx_data)

# ✅ İki veri setini birleştir
merged_data = trainset_data + testset_data

# ✅ Yeni JSON dosyasını kaydet
merged_output_path = r"C:\Users\nezahat.korkmaz\Desktop\data_RAD\turkce_med_llava_veriseti.json"
with open(merged_output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print("✅ Yeni veri seti başarıyla oluşturuldu: turkce_med_llava_veriseti.json")
