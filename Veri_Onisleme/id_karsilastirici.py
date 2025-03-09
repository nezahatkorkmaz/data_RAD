import json
import os

# ✅ Dosya yolları
veri_seti_path = r"C:\Users\nezahat.korkmaz\Desktop\data_RAD\turkce_med_llava_veriseti.json"
imgid2idx_path = r"C:\Users\nezahat.korkmaz\Desktop\data_RAD\imgid2idx.json"

# ✅ JSON dosyalarını oku
with open(veri_seti_path, "r", encoding="utf-8") as f:
    veri_seti = json.load(f)

with open(imgid2idx_path, "r", encoding="utf-8") as f:
    imgid2idx = json.load(f)

print("✅ Dosyalar başarıyla yüklendi!")

# ✅ "dataset/images/" kısmını temizleyerek sadece resim adlarını al
def normalize_filename(path):
    return os.path.basename(path)  # Yalnızca dosya adını al

# ✅ Eşleşmeleri kontrol et
eslesen_idler = {}
yanlis_eslesen_idler = {}
eksik_idler = set()

for entry in veri_seti:
    image_name = normalize_filename(entry["image"])  # Resim adını al
    if image_name in imgid2idx:
        llava_id = entry["id"]
        imgid2idx_id = str(imgid2idx[image_name])  # Sayıyı stringe çevir
        if llava_id == imgid2idx_id:
            eslesen_idler[image_name] = llava_id  # Doğru eşleşenleri kaydet
        else:
            yanlis_eslesen_idler[image_name] = (llava_id, imgid2idx_id)  # Yanlış eşleşenleri kaydet
    else:
        eksik_idler.add(image_name)  # Eğer imgid2idx içinde yoksa eksik olarak kaydet

# ✅ Sonuçları yazdır
print(f"✅ Eşleşen ID sayısı: {len(eslesen_idler)}")
print(f"❌ Yanlış eşleşen ID sayısı: {len(yanlis_eslesen_idler)}")
print(f"❌ Eksik resim sayısı: {len(eksik_idler)}")

# 🔴 Yanlış eşleşen ID'leri göster
if yanlis_eslesen_idler:
    print("\n🔴 Yanlış eşleşen ID'ler:")
    for image, ids in yanlis_eslesen_idler.items():
        print(f"{image}: Veri seti ID = {ids[0]}, imgid2idx ID = {ids[1]}")

# 🟡 Eksik ID'leri göster
if eksik_idler:
    print("\n🟡 imgid2idx.json içinde olmayan resimler:")
    print(eksik_idler)

print("\n✅ Karşılaştırma tamamlandı!")
