import re
import nltk
import string
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_download
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk_download('stopwords', quiet=True)
    nltk_download('wordnet', quiet=True)
    nltk_download('omw-1.4', quiet=True)  # Required for lemmatization
    nltk_download('punkt', quiet=True)      # Required for tokenization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"Error initializing NLTK: {e}")
    stop_words = set()  # Fallback to empty set if stopwords can't be loaded
    lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
        
    try:
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z ]', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        
        # Simple lemmatization with error handling
        def safe_lemmatize(word):
            try:
                return lemmatizer.lemmatize(word)
            except:
                return word
                
        tokens = [safe_lemmatize(word) for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in clean_text: {e}")
        return text  # Return original text if cleaning fails

def preprocess_texts(texts):
    return [clean_text(t) for t in texts]

def load_vectorizer(path='model/tfidf_vectorizer.pkl'):
    return joblib.load(path)

def extract_features(texts, vectorizer=None):
    if vectorizer is None:
        vectorizer = load_vectorizer()
    return vectorizer.transform(texts) 