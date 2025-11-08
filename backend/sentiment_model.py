import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_sentiment_model():
    df = pd.read_csv("sentiment_data/all-data.csv", encoding='latin1')
    df.columns = ["label", "text"]

    # Clean labels (Positive, Neutral, Negative)
    df["label"] = df["label"].str.lower().map({
        "positive": 1,
        "neutral": 0,
        "negative": -1
    })

    # Text preprocessing & feature extraction
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lightweight, high-accuracy classifier
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nSentiment Model Performance:\n")
    print(classification_report(y_test, y_pred))

    # Save model + vectorizer
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("\n✅ Sentiment model saved successfully.")

def predict_sentiment(text):
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    X = vectorizer.transform([text])
    return model.predict(X)[0]

if __name__ == "__main__":
    train_sentiment_model()
