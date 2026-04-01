import os
import re
import pickle
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download("stopwords")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "amazon_reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

df = pd.read_csv(DATA_PATH)

print("Original columns:", df.columns)

df = df[["verified_reviews", "rating"]].dropna()

def convert_rating(r):
    if r >= 4:
        return "positive"
    elif r == 3:
        return "neutral"
    else:
        return "negative"

df["label"] = df["rating"].apply(convert_rating)

stop_words = set(stopwords.words("english")) - {"not", "no", "never"}

def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["verified_reviews"] = df["verified_reviews"].apply(clean)

print(df.head())
print(df["label"].value_counts())

df_balanced = df.groupby("label").sample(n=500, replace=True, random_state=42)

vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 3))

X = vectorizer.fit_transform(df_balanced["verified_reviews"])
y = df_balanced["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
print("Vocabulary size:", len(vectorizer.vocabulary_))

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")