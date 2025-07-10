import joblib
from utils import clean_text, extract_features, load_vectorizer
# from utils import clean_text, extract_features, load_vectorizer

import os

# Get the absolute path to the model directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'model', 'tfidf_vectorizer.pkl')

model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if model is None:
        model = joblib.load(MODEL_PATH)
    if vectorizer is None:
        vectorizer = load_vectorizer(VECTORIZER_PATH)
    return model, vectorizer

def predict_news(text):
    model, vectorizer = load_model()
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return pred, proba 