import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="LLM Karşılaştırıcı",
    page_icon="🏆",
    layout="wide"
)

MODEL_PATH = "./llm_winner_classifier"

@st.cache_resource
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer
    except OSError:
        return None, None

model, tokenizer = load_model()

st.sidebar.title("ℹ️ Nasıl Kullanılır?")
st.sidebar.markdown(
    """
    Bu uygulama, iki farklı yapay zeka modelinin cevaplarını karşılaştırmak için tasarlanmıştır.

    **Adımlar:**
    1.  **Bir komut (prompt) belirleyin.**
        *(Örn: "İstanbul'da gezilecek 5 yer önerir misin?*")

    2.  Bu komutu, sonuçlarını merak ettiğiniz **iki farklı yapay zeka modeline** sorun.
        *(Örn: ChatGPT, Gemini, Claude vb.)*

    3.  Aldığınız iki farklı cevabı aşağıdaki **'Cevap A'** ve **'Cevap B'** kutularına yapıştırın.

    4.  **'Analiz Et ve Kazananı Bul!'** butonuna tıklayarak modelimizin tahminini görün.
    """
)

st.sidebar.title("🤖 Model Hakkında")
st.sidebar.info(
    "Bu uygulama, `distilbert-base-uncased` modelinin Kaggle'daki bir veri seti ile "
    "fine-tune edilmiş halini kullanmaktadır. Amaç, iki metin arasındaki 'daha iyi' olanı belirlemektir."
)

st.title("🏆 LLM Cevap Karşılaştırıcı")
st.markdown("İki farklı yapay zeka cevabını analiz ederek hangisinin daha başarılı olduğunu tahmin edin.")
st.markdown("---")

if model is None or tokenizer is None:
    st.error(
        f"**Model yüklenemedi!** Lütfen `train_model.py` betiğini çalıştırdığınızdan ve '{MODEL_PATH}' "
        "klasörünün mevcut olduğundan emin olun."
    )
else:
    st.subheader("1. Adım: Karşılaştırma Komutunuz")
    prompt_text = st.text_area(
        "Karşılaştırma için temel alınacak komutu buraya girin:",
        height=100,
        placeholder="Örn: İklim değişikliğinin ana nedenleri nelerdir?"
    )

    st.subheader("2. Adım: Yapay Zeka Cevapları")
    col1, col2 = st.columns(2)
    with col1:
        response_a = st.text_area(
            "Cevap A",
            height=250,
            placeholder="İlk yapay zeka modelinden aldığınız cevabı buraya yapıştırın."
        )
    with col2:
        response_b = st.text_area(
            "Cevap B",
            height=250,
            placeholder="İkinci yapay zeka modelinden aldığınız cevabı buraya yapıştırın."
        )

    st.markdown("---")

    if st.button("Analiz Et ve Kazananı Bul!", type="primary", use_container_width=True):
        if all([prompt_text, response_a, response_b]):
            with st.spinner("🧠 Analiz yapılıyor..."):
                sep_token = tokenizer.sep_token
                input_text = prompt_text + sep_token + response_a + sep_token + response_b

                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

                with torch.no_grad():
                    logits = model(**inputs).logits

                predicted_class_id = torch.argmax(logits, dim=1).item()

                st.subheader("✨ Sonuç")
                if predicted_class_id == 0:
                    st.success("Kazanan: **Cevap A** daha başarılı görünüyor!", icon="🇦")
                elif predicted_class_id == 1:
                    st.success("Kazanan: **Cevap B** daha başarılı görünüyor!", icon="🇧")
                else:
                    st.info("Sonuç: İki cevap arasında belirgin bir fark bulunamadı (**Berabere**).", icon="🤝")
                
                st.balloons()
        else:
            st.warning("Lütfen karşılaştırma yapmak için tüm alanları doldurun.", icon="⚠️")