import streamlit as st
import joblib
import re
import os

if not os.path.exists('spam_model.pkl') or not os.path.exists('spam_vectorizer.pkl'):
    st.error("Model not found! Please run 'train_model.py' first.")
    st.stop()

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('spam_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

st.title("📩 SMS Spam Detector")

message = st.text_area("Enter a message:", height=150)

if st.button("Predict"):
    if message.strip():
        cleaned = clean_text(message)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max()

        if pred == 1:
            st.error(f"Spam detected. Confidence: {prob:.2%}")
        else:
            st.success(f"Not spam. Confidence: {prob:.2%}")
    else:
        st.warning("Please enter a message.")
