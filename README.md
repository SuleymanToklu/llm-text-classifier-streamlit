# 🏆 LLM Cevap Karşılaştırıcı

Bu proje, bir komuta (prompt) karşılık iki farklı Büyük Dil Modeli (LLM) tarafından üretilen cevapları karşılaştıran ve hangisinin daha 'başarılı' olduğunu tahmin eden bir web uygulamasıdır. Proje, Hugging Face `transformers` kütüphanesi kullanılarak fine-tune edilmiş bir `DistilBERT` modelini temel alır ve arayüzü `Streamlit` ile oluşturulmuştur.

Bu proje, [Kaggle'daki "LLM - Classification/Finetuning"](https://www.kaggle.com/competitions/llm-classification-finetuning) yarışmasının bir uygulaması olarak geliştirilmiştir.) 

## ✨ Özellikler

-   İki farklı LLM cevabını birbirine karşı analiz etme.
-   Bir komut ve iki cevabı girdi olarak alarak anlık tahmin yapma.
-   Hugging Face `transformers` ile fine-tune edilmiş `DistilBERT` modeli.
-   `Streamlit` ile oluşturulmuş interaktif ve kullanıcı dostu arayüz.
-   Modelin bellek kullanımını düşürmek için `Gradient Accumulation` tekniği ile eğitilmesi.

## 🛠️ Kullanılan Teknolojiler

-   **Backend & Modelleme:** Python, PyTorch, Hugging Face (Transformers, Datasets), Pandas, Scikit-learn
-   **Frontend:** Streamlit
-   **Geliştirme Ortamı:** GitHub Codespaces

## ⚙️ Kurulum ve Çalıştırma

Bu projeyi yerel makinenizde veya başka bir ortamda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Proje Dosyalarını Klonlayın:**
```bash
git clone [https://github.com/SuleymanTokluN/llm-text-classifier-streamlit.git](https://github.com/SuleymanToklu/llm-text-classifier-streamlit.git)
cd llm-text-classifier-streamlit
```

**2. Gerekli Kütüphaneleri Yükleyin:**
Projenin ihtiyaç duyduğu tüm kütüphaneler `requirements.txt` dosyasında listelenmiştir.
```bash
pip install -r requirements.txt
```

**3. Modeli Eğitin (İsteğe Bağlı):**
Bu repo, `.gitignore` dosyası aracılığıyla büyük model dosyalarını içermez. Uygulamanın çalışması için modeli eğitmeniz gerekmektedir.
```bash
# Önce Kaggle verilerini indirin (kaggle.json API anahtarınızın kurulu olması gerekir)
kaggle competitions download -c llm-classification-finetuning
unzip llm-classification-finetuning.zip

# Veriyi işleyin
python preprocess_data.py

# Modeli eğitin (Bu işlem CPU'da uzun sürebilir)
python train_model.py
```
Eğitim tamamlandığında, `llm_winner_classifier` adında bir klasör oluşacaktır.

**4. Streamlit Uygulamasını Başlatın:**
Modeliniz hazır olduğuna göre, web uygulamasını başlatabilirsiniz.
```bash
streamlit run app.py
```
Bu komuttan sonra tarayıcınızda uygulamanın çalıştığı yerel bir adres açılacaktır.

## 🎮 Nasıl Kullanılır?

Uygulama arayüzü oldukça basittir:
1.  **Bir komut belirleyin** (Örn: "Yapay zeka etiği nedir?").
2.  Bu komutu, karşılaştırmak istediğiniz **iki farklı yapay zeka modeline** sorun (Örn: ChatGPT ve Gemini).
3.  Aldığınız cevapları uygulamadaki **"Cevap A"** ve **"Cevap B"** kutucuklarına yapıştırın.
4.  **"Analiz Et ve Kazananı Bul!"** butonuna tıklayarak fine-tune ettiğimiz modelin tahminini görün.

## 📄 Lisans
Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakabilirsiniz.
