<div align="center">

# 📩 AI-Based SMS Spam Detection

### Machine Learning Powered Real-Time SMS Classifier

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen?style=for-the-badge&logo=streamlit)](https://ai-based-sms-spam-detection.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Logistic%20Regression-orange?style=for-the-badge)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Live-success?style=for-the-badge)](https://ai-based-sms-spam-detection.onrender.com/)

[**🚀 View Live Demo**](https://ai-based-sms-spam-detection.onrender.com/)

</div>

---

## ⭐ Overview

This project delivers a comprehensive end-to-end machine learning pipeline designed to detect spam SMS messages with high accuracy. It involves text preprocessing, TF-IDF feature extraction, model training, and deployment via a user-friendly Streamlit web interface.

### ✨ Key Features
* **Text Cleaning:** Automated normalization of raw SMS text.
* **Feature Extraction:** Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) with bi-grams.
* **Model Performance:** Uses Logistic Regression (Best performing model) for classification.
* **Real-time Inference:** Instant prediction with confidence scores via the web app.

---

## 📂 Project Structure

```
AI-Based-SMS-Spam-Detection/
├── app.py                  # Streamlit application for inference
├── train_model.py          # Complete ML training pipeline
├── spam.csv                # Raw SMS Dataset
├── spam_model.pkl          # Serialized Trained Model
├── spam_vectorizer.pkl     # Serialized TF-IDF Vectorizer
├── requirements.txt        # Project dependencies
└── README.md               # Project Documentation

🔍 How It Works
The pipeline consists of four main stages:

1. Data Cleaning & Preprocessing
Raw SMS text undergoes rigorous normalization to ensure data quality:

Lowercasing: Converts all text to lowercase.

Noise Removal: Strips digits and punctuation.

Label Encoding: Maps 'Ham' and 'Spam' labels to integers.

2. Feature Extraction
We convert text data into numerical vectors using TF-IDF:

Configuration: Includes English stopwords and unigrams + bigrams.

Optimization: Features capped at 5,000 to balance performance and speed.

3. Model Training & Evaluation
Two primary algorithms were evaluated for this classification task:

Multinomial Naive Bayes

Logistic Regression (Selected for deployment due to superior metrics).

Metrics Tracked: Accuracy, Precision, Recall, and F1-Score.

4. Deployment
The application is built using Streamlit:

Loads the saved spam_model.pkl and spam_vectorizer.pkl.

Provides an input field for the user to type a message.

Returns a classification (Spam/Not Spam) along with a probability score.

🧪 Installation
Follow these steps to set up the project locally.

1. Clone the repository

git clone [https://github.com/neevlila/AI-Based-SMS-Spam-Detection](https://github.com/neevlila/AI-Based-SMS-Spam-Detection)
cd AI-Based-SMS-Spam-Detection

2. Install dependencies

pip install -r requirements.txt

🎯 Usage
Training the Model
If you wish to retrain the model or update the dataset:

python train_model.py
This will generate new spam_model.pkl and spam_vectorizer.pkl files.

Running the App
Launch the Streamlit interface locally:

python -m streamlit run app.py
📊 Tech Stack
Language: Python 3.11

Frontend: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn, Joblib

Visualization: Matplotlib, Seaborn

👥 Contributors
<table> <tr> <td align="center"><strong>Neev Lila</strong></td> <td align="center"><strong>Harsh Parekh</strong></td> <td align="center"><strong>Meet Gajjar</strong></td> <td align="center"><strong>Dhairya Kamdar</strong></td> <td align="center"><strong>Mitanshu Chauhan</strong></td> </tr> </table>