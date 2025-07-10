# Fake News Detection Web App

<div align="center">
  <h1>Fake News Detection Web App</h1>
  <p>A full-stack, production-ready Fake News Detection system using Python, Machine Learning, and Streamlit.</p>
  
  <div>
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
    <img src="https://img.shields.io/badge/Scikit--Learn-1.0+-orange" alt="Scikit-Learn">
    <img src="https://img.shields.io/badge/Streamlit-1.22+-FF4B4B" alt="Streamlit">
  </div>
</div>

## 📸 Screenshots

<div align="center">
  <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
    <div style="flex: 1; min-width: 300px; max-width: 45%;">
      <img src="images/pic1.png" alt="Fake News Detector - Main Interface" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
      <p><em>Main Interface</em></p>
    </div>
    <div style="flex: 1; min-width: 300px; max-width: 45%;">
      <img src="images/pic2.png" alt="Fake News Detector - Analysis Results" style="width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
      <p><em>Analysis Results</em></p>
    </div>
  </div>
</div>

## 🚀 Features
- User authentication (Sign Up / Login)
- Input news via text, URL, or Twitter post
- Classifies news as Real or Fake
- Fetches related news/images for Real news
- Dashboard with user history and stats
- Clean UI with Bootstrap/Tailwind
- Latest trusted headlines on homepage

## 🛠️ Tech Stack
### Core Technologies
- **Python 3.8+**: Core programming language
- **Scikit-Learn**: Machine learning model for fake news detection
- **Streamlit**: Web application framework

### Data Processing
- **BeautifulSoup4**: Web scraping
- **NLTK**: Natural language processing
- **Pandas & NumPy**: Data manipulation

### APIs & Services
- **NewsAPI**: Fetching latest news
- **Tweepy**: Twitter integration
- **SQLAlchemy**: Database ORM
- **SQLite**: Lightweight database

## ⚡ Setup Instructions
1. **Clone the repo:**
   ```bash
   git clone <repo-url>
   cd fakenews
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
4. **Set up API keys:**
   - Create a `.env` file in the root directory:
     ```
     NEWS_API_KEY=your_newsapi_key
     TWITTER_BEARER_TOKEN=your_twitter_bearer_token
     ```
5. **Train the model:**
   - Run `model/train_model.ipynb` or use the provided model files.
6. **Run the app:**
   ```bash
   streamlit run app/main.py
   ```

## 🌐 Deployment
- **Heroku:**
  - Add `Procfile` with: `web: streamlit run app/main.py`
  - Set config vars for API keys
- **Render:**
  - Use a web service, point to `app/main.py`

## 📁 Project Structure
```
fakenews/
├── app/
│   ├── __init__.py
│   ├── auth.py
│   ├── dashboard.py
│   ├── main.py
│   ├── model.py
│   ├── news_fetcher.py
│   ├── utils.py
│   └── database.py
├── model/
│   ├── train_model.ipynb
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── .env
```

## 🙏 Credits
- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- [NewsAPI.org](https://newsapi.org/)
- [Tweepy](https://www.tweepy.org/)

---
**For any issues, open an issue or PR!** 