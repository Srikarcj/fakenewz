import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import tweepy
from datetime import datetime, timedelta

load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '4b9fdee9557648a4977a084b9ab89760')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')

# Common headers for API requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_news_from_url(url):
    """Fetch and extract main text content from a news article URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Try to find the main article content
        article = soup.find('article')
        if not article:
            # Common article content selectors
            selectors = [
                'article',
                'div.article-content',
                'div.article-body',
                'div.post-content',
                'div.entry-content',
                'main',
                'div[itemprop="articleBody"]',
                'div.story-body',
                'div.article-text'
            ]
            
            for selector in selectors:
                article = soup.select_one(selector)
                if article:
                    break
        
        # If still no article found, use the body
        if not article:
            article = soup
            
        # Remove unwanted elements
        for element in article(['script', 'style', 'nav', 'footer', 'aside', 'iframe', 'noscript']):
            element.decompose()
        
        # Extract metadata
        title = soup.find('title')
        title = title.get_text().strip() if title else ''
        
        description = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        description = description.get('content', '').strip() if description else ''
        
        # Try to find published time
        published_time = ''
        time_selectors = [
            'time[datetime]',
            'meta[property="article:published_time"]',
            'meta[name="publish-date"]',
            '.published-date',
            '.publish-time'
        ]
        
        for selector in time_selectors:
            time_elem = soup.select_one(selector)
            if time_elem:
                published_time = time_elem.get('datetime') or time_elem.get('content') or time_elem.get_text()
                break
        
        # Extract text from paragraphs
        paragraphs = article.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        content = ' '.join(content.split())  # Clean up whitespace
        
        return {
            'title': title,
            'description': description,
            'content': content,
            'published_time': published_time,
            'url': url
        }
    except Exception as e:
        print(f"Error fetching URL {url}: {str(e)}")
        return {'content': '', 'url': url, 'error': str(e)}

def fetch_latest_headlines(query='news', language='en', page_size=5):
    """Fetch latest news headlines using NewsAPI."""
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY not found in environment variables")
        return []
        
    try:
        url = 'https://newsapi.org/v2/top-headlines'
        params = {
            'apiKey': NEWS_API_KEY,
            'language': language,
            'pageSize': page_size,
            'q': query,
            'sortBy': 'publishedAt'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, dict):
            print(f"Unexpected API response format: {data}")
            return []
            
        if data.get('status') == 'ok' and data.get('totalResults', 0) > 0:
            articles = data.get('articles', [])
            if not isinstance(articles, list):
                return []
                
            result = []
            for article in articles:
                if not isinstance(article, dict) or not article.get('title'):
                    continue
                    
                # Handle different source formats
                source_name = 'Unknown'
                source = article.get('source')
                if isinstance(source, dict):
                    source_name = source.get('name', 'Unknown')
                elif isinstance(source, str):
                    source_name = source
                    
                result.append({
                    'title': article.get('title', 'No title'),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'urlToImage': article.get('urlToImage', ''),
                    'publishedAt': article.get('publishedAt', ''),
                    'source': source_name
                })
            return result
            
        return []
    except requests.exceptions.RequestException as e:
        print(f"Network error while fetching headlines: {str(e)}")
    except Exception as e:
        print(f"Error processing latest headlines: {str(e)}")
    return []

def fetch_related_news(query, language='en', page_size=3):
    """Fetch news articles related to the given query."""
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY not found in environment variables")
        return []
        
    try:
        # Clean and prepare the query
        if not query or not isinstance(query, str):
            return []
            
        # Use the first few meaningful words from the query
        query = ' '.join(str(query).strip().split()[:5])
        if not query:
            return []
            
        url = 'https://newsapi.org/v2/everything'
        params = {
            'apiKey': NEWS_API_KEY,
            'q': query,
            'language': language,
            'pageSize': page_size,
            'sortBy': 'relevancy'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, dict):
            print(f"Unexpected API response format: {data}")
            return []
            
        if data.get('status') == 'ok' and data.get('totalResults', 0) > 0:
            articles = data.get('articles', [])
            if not isinstance(articles, list):
                return []
                
            result = []
            for article in articles:
                if not isinstance(article, dict) or not article.get('title'):
                    continue
                    
                # Handle different source formats
                source_name = 'Unknown'
                source = article.get('source')
                if isinstance(source, dict):
                    source_name = source.get('name', 'Unknown')
                elif isinstance(source, str):
                    source_name = source
                    
                result.append({
                    'title': article.get('title', 'No title'),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'urlToImage': article.get('urlToImage', ''),
                    'publishedAt': article.get('publishedAt', ''),
                    'source': source_name
                })
            return result
            
        return []
    except requests.exceptions.RequestException as e:
        print(f"Network error while fetching related news: {str(e)}")
    except Exception as e:
        print(f"Error processing related news: {str(e)}")
    return []

def fetch_tweet_text(tweet_url_or_id):
    """Fetch tweet text by tweet ID or URL."""
    if not TWITTER_BEARER_TOKEN:
        print("Error: TWITTER_BEARER_TOKEN not found in environment variables")
        return ''
        
    try:
        client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
        tweet_id = tweet_url_or_id.split('/')[-1].split('?')[0]  # Extract ID from URL if needed
        tweet = client.get_tweet(tweet_id, tweet_fields=['text'])
        return tweet.data['text'] if tweet and tweet.data else ''
    except Exception as e:
        print(f"Error fetching tweet: {str(e)}")
        return ''
