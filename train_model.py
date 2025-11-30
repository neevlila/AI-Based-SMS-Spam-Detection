import pandas as pd
import re 
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['text', 'label']]

df['label'] = df['label'].map({'ham': 0, 'spam': 1}).astype(int)

df = df.dropna()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

print(f"Dataset shape: {df.shape}")
print(df['label'].value_counts())

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    }

    cv_scores = cross_val_score(
        model, X_train_vec, y_train, cv=5, scoring='f1'
    )

    print(f"{name} - Test F1: {f1:.4f}, CV F1 Mean: {cv_scores.mean():.4f}")

    if name == 'Logistic Regression':
        print("\nClassification Report:\n",
              classification_report(y_test, predictions))

        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n", results_df.round(4))

best_model = models['Logistic Regression']
joblib.dump(best_model, 'spam_model.pkl')
joblib.dump(vectorizer, 'spam_vectorizer.pkl')

print("Model and vectorizer saved!")

new_msg = ["Free entry to win $1000 now! Click here."]
new_msg_vec = vectorizer.transform(new_msg)
prediction = best_model.predict(new_msg_vec)[0]

print(f"\nNew Message Prediction: {'Spam' if prediction == 1 else 'Not Spam'} "
      f"(Probability: {best_model.predict_proba(new_msg_vec)[0].max():.4f})")
