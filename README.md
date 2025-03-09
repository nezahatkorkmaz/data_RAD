# 📌 Türkçe Medikal LLaVA Veri Seti

Bu veri seti, **RAD-VQA** veri setinin **LLaVA formatına** uygun hale getirilmiş Türkçe versiyonudur. **Helsinki-NLP çeviri modeli** kullanılarak Türkçeye çevrilmiş olup; normalizasyon, veri ön işleme ve gerekli ID eşleştirmeleri yapılarak LLaVA formatına uyarlanmıştır.

## 📂 Veri Seti İçeriği
- **📁 `images/`**: Medikal görüntülerin bulunduğu klasör.
- **📜 `turkce_med_llava_veriseti.json`**: LLaVA formatına uygun hale getirilmiş veri seti.

## 🔄 Dönüştürme Süreci
1. **RAD-VQA** veri seti analiz edildi.
2. **Helsinki-NLP `opus-mt-en-tr` modeli** ile veri Türkçeye çevrildi.
3. **Mevcut format (`qid`, `image_name`, `question`, `answer`)**, LLaVA formatına dönüştürüldü.
4. **`imgid2idx.json`** dosyası ile ID eşleştirmeleri yapılarak, her görüntüye doğru ID atandı.
5. **Tüm veriler `turkce_med_llava_veriseti.json` içinde LLaVA formatına uygun hale getirildi.**

## 📌 Veri Formatı Karşılaştırması
### **Eski Format (RAD-VQA)**
```json
{
    "qid": 13,
    "image_name": "synpic29265.jpg",
    "image_organ": "GÖĞÜS",
    "answer": "Evet.",
    "answer_type": "EVET/HAYIR FORMATINDA",
    "question_type": "MEVCUTLUK",
    "question": "Sol tarafta hava sahası konsolidasyonu var mı?",
    "phrase_type": "serbest"
}
```

### **Yeni Format (LLaVA Uyumlu)**
```json
{
    "id": "215",
    "image": "dataset/images/synpic29265.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nBu göğüs MR'ı mı?"
        },
        {
            "from": "gpt",
            "value": "Hayır."
        }
    ]
}
```

## ✅ LLaVA Formatına Uygunluk
- **`image_name` → `image` olarak güncellendi** (Tam dosya yolu eklendi: `dataset/images/` klasörü).
- **`qid` yerine `id` alanı kullanıldı** ve `imgid2idx.json` ile eşleştirildi.
- **Soru-cevap yapısı `conversations` içine yerleştirildi**.
- **`<image>` etiketi eklendi**, böylece model görüntüyü de analiz edebilir.

## 🚀 Kullanım
Bu veri seti, **Türkçe medikal LLaVA modellerinin eğitimi için kullanılabilir.**

Örnek kullanım:
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="turkce_med_llava_veriseti.json")
print(dataset["train"][0])
```

📌 **Eğer eksik veya yanlış ID eşleşmeleri fark ederseniz, `imgid2idx.json` ile kontrol edebilirsiniz.**

---
📢 **Bu veri seti, Türkçe medikal görüntü işleme modelleri için büyük bir adım olup, araştırmalarda kullanılmak üzere optimize edilmiştir.** 🚀

