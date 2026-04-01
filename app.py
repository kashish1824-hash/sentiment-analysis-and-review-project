import re
import pickle
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# Load model and vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english")) - {"not", "no", "never"}


def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def rule_based_override(text):
    text = text.lower().strip()

    rules = {
        "not bad": "neutral",
        "not too bad": "neutral",
        "not that bad": "neutral",
        "it is okay": "neutral",
        "its okay": "neutral",
        "average": "neutral",
        "could be better": "neutral",
        "not good": "negative",
        "very good": "positive",
        "excellent": "positive"
    }

    for phrase, label in rules.items():
        if phrase in text:
            return label
    return None


def predict_sentiment(review):
    rule = rule_based_override(review)
    if rule:
        return rule

    cleaned = clean(review)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    emoji = ""
    review = ""

    if request.method == "POST":
        review = request.form["review"]

        if review.strip():
            prediction = predict_sentiment(review)

            if prediction == "positive":
                emoji = "😊"
            elif prediction == "neutral":
                emoji = "😐"
            else:
                emoji = "😞"

    return render_template("index.html", prediction=prediction, emoji=emoji, review=review)


if __name__ == "__main__":
    app.run(debug=True)