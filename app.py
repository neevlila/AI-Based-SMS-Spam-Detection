import streamlit as st
import joblib
import re
import os
import pandas as pd
import random

# 1. Page Configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# 2. Load Model & Vectorizer (Cached)
@st.cache_resource
def load_model():
    if not os.path.exists('spam_model.pkl') or not os.path.exists('spam_vectorizer.pkl'):
        return None, None
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('spam_vectorizer.pkl')
    return model, vectorizer

# 3. Load Dataset for Random Examples (Cached)
@st.cache_data
def load_data():
    if os.path.exists('spam.csv'):
        try:
            # excessive error handling for encoding
            df = pd.read_csv('spam.csv', encoding='latin-1')
            spam_msgs = df[df['label'] == 'spam']['text'].tolist()
            ham_msgs = df[df['label'] == 'ham']['text'].tolist()
            return spam_msgs, ham_msgs
        except Exception as e:
            return [], []
    return [], []

model, vectorizer = load_model()
spam_messages, ham_messages = load_data()

# Stop if model is missing
if model is None or vectorizer is None:
    st.error("🚨 Model files not found! Please run 'train_model.py' first.")
    st.stop()

# 4. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# 5. Session State Logic
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

# Function to set random text from dataset
def set_random_text(msg_type):
    if msg_type == 'spam' and spam_messages:
        st.session_state['user_input'] = random.choice(spam_messages)
    elif msg_type == 'ham' and ham_messages:
        st.session_state['user_input'] = random.choice(ham_messages)
    else:
        # Fallback if CSV is missing
        if msg_type == 'spam':
            st.session_state['user_input'] = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt WORD to 81010 to claim No: 81010"
        else:
            st.session_state['user_input'] = "Hey, are we still meeting for lunch today? Let me know."

# 6. Professional Sidebar
with st.sidebar:    
    st.markdown("### 🕵️‍♂️ How it works")
    st.info(
        "This app uses **Logistic Regression** and **TF-IDF** "
        "to classify SMS messages as Spam or Safe."
    )
    
    st.markdown("### 📊 Model Details")
    st.markdown("""
    - **Algorithm:** Logistic Regression
    - **Vectorizer:** TF-IDF (Bi-grams)
    - **Accuracy:** ~98% (on test set)
    """)

    st.markdown("---")
    st.markdown("### 👥 Created By")
    st.markdown("""
    • Neev Lila  
    • Harsh Parekh  
    • Meet Gajjar  
    • Dhairya Kamdar  
    • Mitanshu Chauhan
    """)
    st.caption("© 2025 AI Spam Detection")

# 7. Main Interface
st.title("📩 AI-Based SMS Spam Detector")
st.markdown("""
Welcome! This tool helps you identify **Spam** or **Phishing** messages instantly. 
Type your message below or use the buttons to load a random example from our database.
""")

st.divider()

# Example Buttons
col1, col2 = st.columns(2)
with col1:
    st.button("⚠️ Load Random Spam", 
              on_click=set_random_text, 
              args=('spam',), 
              type="secondary",
              use_container_width=True)
with col2:
    st.button("✅ Load Random Safe Text", 
              on_click=set_random_text, 
              args=('ham',), 
              type="secondary",
              use_container_width=True)

# Main Input Area
# key='user_input' ensures this box updates when buttons are clicked
message = st.text_area("Message Content", height=150, key='user_input', placeholder="Paste the suspicious text here...")

# Predict Button
if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
    if message.strip():
        with st.spinner("Analyzing text patterns..."):
            cleaned = clean_text(message)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0].max()

        st.markdown("---")
        
        # Result Display
        if pred == 1:
            # Spam Result
            st.error("🚨 **SPAM DETECTED**")
            
            # Gauge / Meter visual
            st.progress(prob)
            st.caption(f"Confidence Score: {prob:.2%}")
            
            st.warning("⚠️ **Warning:** This message fits the pattern of known spam. Do not click links or reply.")
            
        else:
            # Safe Result
            st.success("✅ **SAFE MESSAGE**")
            
            # Gauge / Meter visual
            st.progress(prob)
            st.caption(f"Confidence Score: {prob:.2%}")
            
            st.info("This message looks like a normal conversation.")
            st.balloons()
            
    else:
        st.warning("⚠️ Please enter some text to analyze.")