import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path

@st.cache_resource
def load_model():
    APP_DIR = Path(__file__).parent
    MODEL_PATH = APP_DIR / "llm_winner_classifier"
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

st.title("🤖 AI vs. Human Text Detector")
st.write("Bu uygulama, girdiğiniz metnin bir yapay zeka tarafından mı yoksa bir insan tarafından mı yazıldığını tahmin eder.")

try:
    model, tokenizer = load_model()
    user_input = st.text_area("Lütfen analiz edilecek metni buraya girin:", height=150)

    if st.button("Metni Analiz Et"):
        if user_input:
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            predicted_class_id = torch.argmax(logits, dim=1).item()
            
            if predicted_class_id == 1:
                st.success("Tahmin: Bu metin büyük ihtimalle **Yapay Zeka** tarafından üretilmiştir. 🤖")
            else:
                st.info("Tahmin: Bu metin büyük ihtimalle bir **İnsan** tarafından yazılmıştır. 🧑‍💻")
        else:
            st.warning("Lütfen analiz etmek için bir metin girin.")

except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")
    st.info("Lütfen 'llm_winner_classifier' klasörünün app.py ile aynı dizinde olduğundan emin olun.")
