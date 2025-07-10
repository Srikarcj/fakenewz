import requests
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse
import json
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsVerifier:
    def __init__(self, news_api_key: str):
        self.news_api_key = news_api_key
        self.news_api_base = 'https://newsapi.org/v2'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        self.trusted_domains = [
            'reuters.com', 'apnews.com', 'bbc.co.uk', 'npr.org',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com',
            'wsj.com', 'bloomberg.com', 'ap.org', 'afp.com', 'cnn.com',
            'abcnews.go.com', 'cbsnews.com', 'nbcnews.com', 'usatoday.com',
            'latimes.com', 'chicagotribune.com', 'time.com', 'newsweek.com',
            'economist.com', 'politico.com', 'thehill.com', 'axios.com',
            'forbes.com', 'fortune.com', 'businessinsider.com', 'espn.com',
            'cricbuzz.com', 'ndtv.com', 'thehindu.com', 'hindustantimes.com',
            'indianexpress.com', 'timesofindia.indiatimes.com', 'news18.com'
        ]

    def verify_news(self, text: str, source_url: str = None) -> Dict:
        """
        Verify the authenticity of a news article using multiple methods.
        
        Args:
            text: The text content of the news article
            source_url: Optional URL of the article
            
        Returns:
            dict: Verification results including confidence score and reasoning
        """
        results = {
            'verification_score': 0,
            'confidence': 0,
            'reasons': [],
            'source_analysis': {},
            'cross_verification': {}
        }
        
        try:
            # 1. Source Analysis
            if source_url:
                source_analysis = self.analyze_source(source_url)
                results['source_analysis'] = source_analysis
                results['verification_score'] += source_analysis.get('source_score', 0)
                results['reasons'].extend(source_analysis.get('reasons', []))
            
            # 2. Cross-verification with News API
            cross_verification = self.cross_verify_with_news_api(text, source_url)
            results['cross_verification'] = cross_verification
            results['verification_score'] += cross_verification.get('score', 0)
            results['reasons'].extend(cross_verification.get('reasons', []))
            
            # 3. Content Analysis
            content_analysis = self.analyze_content(text)
            results['content_analysis'] = content_analysis
            results['verification_score'] += content_analysis.get('score', 0)
            results['reasons'].extend(content_analysis.get('reasons', []))
            
            # Calculate final confidence score (0-100)
            results['verification_score'] = max(0, min(100, results['verification_score']))
            results['confidence'] = self.calculate_confidence(results)
            
            # Add final verdict - more balanced threshold
            results['is_trustworthy'] = results['verification_score'] >= 50
            
        except Exception as e:
            logger.error(f"Error during news verification: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def analyze_source(self, url: str) -> Dict:
        """Analyze the credibility of the news source."""
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '')
            
            result = {
                'domain': domain,
                'is_trusted': False,
                'source_score': 0,
                'reasons': []
            }
            
            # Check against trusted domains
            if any(trusted in domain for trusted in self.trusted_domains):
                result['is_trusted'] = True
                result['source_score'] = 50  # Increased from 40 to 50
                result['reasons'].append(f"Source domain '{domain}' is a trusted news outlet")
            else:
                # Give some credit to news-looking domains even if not in trusted list
                if any(keyword in domain for keyword in ['news', 'times', 'post', 'herald', 'tribune', 'guardian', 'daily']):
                    result['source_score'] = 15
                    result['reasons'].append(f"Source domain '{domain}' appears to be a news outlet")
                else:
                    result['reasons'].append(f"Source domain '{domain}' is not in the trusted list")
            
            # Check for suspicious domains
            if any(x in domain for x in ['.wordpress.', '.blogspot.', '.weebly.']):
                result['source_score'] -= 20
                result['reasons'].append("Source appears to be a personal blog")
            
            # Check for HTTPS
            if not url.startswith('https'):
                result['source_score'] -= 10
                result['reasons'].append("Connection is not secure (no HTTPS)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in source analysis: {str(e)}")
            return {'error': str(e), 'source_score': 0, 'reasons': ['Error analyzing source']}
    
    def cross_verify_with_news_api(self, text: str, source_url: str = None) -> Dict:
        """Cross-verify the news with News API using multiple search strategies."""
        try:
            if not self.news_api_key:
                return {'score': 0, 'reasons': ['News API key not configured'], 'matching_articles': 0}
                
            # Extract keywords for search
            keywords = self._extract_keywords(text)[:10]
            if not keywords:
                return {'score': 0, 'reasons': ['No keywords found for search'], 'matching_articles': 0}
            
            # Multiple search strategies for comprehensive verification
            all_articles = []
            search_queries = []
            
            # Strategy 1: Precise search with top keywords
            query1 = ' AND '.join(keywords[:3])
            search_queries.append(query1)
            
            # Strategy 2: Broader search
            if len(keywords) >= 4:
                query2 = ' OR '.join(keywords[:4])
                search_queries.append(query2)
            
            # Strategy 3: Entity-focused search (for names, places, events)
            entity_keywords = [kw for kw in keywords if len(kw) > 4 and kw.isalpha()]
            if entity_keywords:
                query3 = ' AND '.join(entity_keywords[:3])
                search_queries.append(query3)
            
            # Execute searches
            for query in search_queries:
                articles = self._execute_news_search(query)
                all_articles.extend(articles)
            
            # Remove duplicates
            seen_urls = set()
            unique_articles = []
            for article in all_articles:
                url = article.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_articles.append(article)
            
            return self._analyze_news_results(unique_articles, text)
            
        except Exception as e:
            logger.error(f"Error in News API verification: {str(e)}")
            return {'score': 0, 'reasons': ['Error during news verification'], 'matching_articles': 0}
    
    def _execute_news_search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Execute a single news search query."""
        try:
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'pageSize': max_results,
                'sortBy': 'relevancy',
                'language': 'en',
                'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Last 30 days
            }
            
            response = requests.get(
                f"{self.news_api_base}/everything",
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                logger.warning(f"News API returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error executing news search: {str(e)}")
            return []
    
    def _analyze_news_results(self, articles: List[Dict], original_text: str) -> Dict:
        """Analyze news search results for verification."""
        result = {
            'score': 0,
            'matching_articles': len(articles),
            'sources': [],
            'reasons': [],
            'trusted_sources': []
        }
        
        if not articles:
            result['reasons'].append("No matching articles found in recent news")
            return result
        
        # Trusted source keywords (comprehensive list)
        trusted_keywords = [
            # International news
            'reuters', 'associated press', 'ap news', 'bbc', 'npr', 'cnn',
            'new york times', 'washington post', 'guardian', 'wall street journal',
            'bloomberg', 'abc news', 'cbs news', 'nbc news', 'usa today',
            'time', 'newsweek', 'economist', 'politico', 'the hill',
            'axios', 'forbes', 'fortune', 'business insider', 
            # Indian news sources
            'ndtv', 'times of india', 'hindustan times', 'indian express', 
            'the hindu', 'news18', 'india today', 'firstpost', 'scroll.in',
            'economic times', 'business standard', 'live mint', 'zee news',
            'aaj tak', 'republic', 'news nation', 'mirror now',
            # Sports sources (very comprehensive)
            'espn', 'cricbuzz', 'cricinfo', 'espncricinfo', 'sports illustrated',
            'sky sports', 'bbc sport', 'goal.com', 'sporting news',
            'the athletic', 'bleacher report', 'sportskeeda', 'sportstar',
            'mid-day', 'mumbai mirror', 'deccan chronicle', 'telegraph',
            'cricket.com', 'wisden', 'sports today', 'times now sports',
            'news18 sports', 'india today sports', 'ndtv sports',
            # IPL and Cricket specific
            'iplt20.com', 'bcci.tv', 'ipl', 'royal challengers bangalore',
            'mumbai indians', 'chennai super kings', 'kolkata knight riders',
            # Regional Indian sources
            'punjab kesari', 'dainik bhaskar', 'hindustan', 'navbharat times',
            'anandabazar', 'eenadu', 'malayala manorama', 'mathrubhumi'
        ]
        
        trusted_matches = 0
        very_trusted_matches = 0
        
        # Very trusted sources get extra weight
        very_trusted = ['reuters', 'bbc', 'associated press', 'new york times', 'washington post']
        
        for article in articles:
            source_name = article.get('source', {}).get('name', '').lower()
            article_title = article.get('title', '').lower()
            article_content = article.get('description', '').lower()
            
            result['sources'].append(source_name)
            
            # Check if source is trusted
            is_trusted = any(keyword in source_name for keyword in trusted_keywords)
            is_very_trusted = any(keyword in source_name for keyword in very_trusted)
            
            if is_trusted:
                trusted_matches += 1
                result['trusted_sources'].append(source_name)
                
            if is_very_trusted:
                very_trusted_matches += 1
        
        # Balanced scoring - more generous for legitimate news
        if trusted_matches >= 3:
            # Multiple trusted sources - high confidence
            base_score = min(90, trusted_matches * 25)
            
            # Bonus for very trusted sources
            if very_trusted_matches > 0:
                base_score += very_trusted_matches * 10
                
            # Bonus for many articles
            if len(articles) >= 5:
                base_score += 10
            elif len(articles) >= 3:
                base_score += 5
                
            result['score'] = min(100, base_score)
            result['reasons'].append(f"Found {trusted_matches} trusted sources reporting similar news")
            
        elif trusted_matches >= 2:
            # Two trusted sources - good confidence
            base_score = 65
            if very_trusted_matches > 0:
                base_score += 15
            if len(articles) >= 3:
                base_score += 10
                
            result['score'] = min(100, base_score)
            result['reasons'].append(f"Found {trusted_matches} trusted sources reporting similar news")
            
        elif trusted_matches == 1:
            # Single trusted source - moderate score
            if very_trusted_matches > 0:
                result['score'] = 55  # Higher for very trusted sources
            else:
                result['score'] = 45  # Increased from 25/40
            result['reasons'].append("Found 1 trusted source reporting similar news")
            
        elif len(articles) > 0:
            # Some articles found but no clearly trusted sources - give some credit
            result['score'] = 25
            result['reasons'].append("Found some related articles but from less established sources")
            
        else:
            # No articles found
            result['score'] = 0
            result['reasons'].append("No matching articles found in recent news databases")
        
        return result
    
    def analyze_content(self, text: str) -> Dict:
        """Analyze the content for signs of misinformation."""
        result = {
            'score': 40,  # Start with neutral score
            'reasons': [],
            'warnings': []
        }
        
        text_lower = text.lower()
        
        # Check if this is sports content - be more lenient
        is_sports_content = any(word in text_lower for word in [
            'ipl', 'cricket', 'match', 'team', 'won', 'victory', 'championship',
            'tournament', 'game', 'score', 'runs', 'wickets', 'player', 'sport',
            'final', 'semi-final', 'qualifier', 'rcb', 'csk', 'mi', 'kkr',
            'sports', 'league', 'cup', 'trophy', 'stadium', 'innings'
        ])
        
        # Give bonus for sports content as it's often legitimate
        if is_sports_content:
            result['score'] += 15
            result['reasons'].append("Sports content - typically factual")
        
        # Check for obviously false claims about prominent figures
        false_claim_patterns = [
            # Death claims (but not sports defeats)
            (r'(prime minister|president|king|queen|celebrity|actor|politician).*(dead|died|death|killed|murdered)', 
             "Claims about death of prominent figures require careful verification"),
            # Extreme political claims
            (r'(resigned|quit|fired|arrested|banned|suspended).*(prime minister|president|ceo|minister)', 
             "Claims about major political/business changes require verification"),
            # Conspiracy theories
            (r'(government|media|doctors|scientists).*(hiding|covering up|lying|conspiracy)', 
             "Contains conspiracy theory language"),
            # Medical misinformation
            (r'(cure|treatment|vaccine).*(dangerous|deadly|kills|causes)', 
             "Contains potential medical misinformation")
        ]
        
        for pattern, warning in false_claim_patterns:
            if re.search(pattern, text_lower):
                # Lower penalty for sports content
                penalty = 10 if is_sports_content else 20
                result['score'] -= penalty
                result['warnings'].append(warning)
        
        try:
            # Check for sensational language (but be lenient with sports content)
            sensational_words = [
                'shocking', 'unbelievable', 'mind-blowing', 'you won\'t believe',
                'urgent', 'alert', 'warning', 'danger', 'must read',
                'go viral', 'will shock you', 'bombshell', 'exposed', 'secret',
                'hidden truth', 'they don\'t want you to know', 'conspiracy',
                'cover-up', 'scandal', 'miracle'
            ]
            
            # Some words are normal in sports context
            sports_normal_words = ['amazing', 'incredible', 'epic', 'huge', 'massive', 'insane', 'crazy', 'breaking']
            
            if is_sports_content:
                # Filter out sports-normal words
                sensational_count = sum(1 for word in sensational_words if word in text_lower and word not in sports_normal_words)
            else:
                sensational_count = sum(1 for word in sensational_words + sports_normal_words if word in text_lower)
            
            if sensational_count > 3:
                penalty = 10 if is_sports_content else 15
                result['score'] -= penalty
                result['warnings'].append(f"Contains {sensational_count} sensational words/phrases")
            elif sensational_count > 1:
                penalty = 3 if is_sports_content else 5
                result['score'] -= penalty
                result['warnings'].append(f"Contains {sensational_count} sensational words")
            
            # Check for fake news indicators
            fake_indicators = [
                'this will shock you', 'doctors hate this', 'secret revealed',
                'government hiding', 'media won\'t tell you', 'hoax', 'fake',
                'lies', 'propaganda', 'agenda', 'wake up', 'sheeple',
                'they want you to believe', 'the truth is'
            ]
            
            fake_count = sum(1 for indicator in fake_indicators if indicator in text.lower())
            if fake_count > 0:
                result['score'] -= fake_count * 10
                result['warnings'].append(f"Contains {fake_count} potential fake news indicators")
            
            # Check for excessive punctuation
            if text.count('!') > len(text) * 0.01:  # More than 1% exclamation marks
                result['score'] -= 10
                result['warnings'].append("Excessive use of exclamation marks")
            
            # Check for all caps
            words = text.split()
            all_caps = sum(1 for word in words if word.isupper() and len(word) > 2)
            if all_caps > len(words) * 0.05:  # More than 5% words in all caps
                result['score'] -= 10
                result['warnings'].append("Excessive use of ALL CAPS")
            
            # Check for emotional manipulation
            emotional_words = [
                'terrifying', 'devastating', 'outrageous', 'disgusting',
                'horrifying', 'appalling', 'sickening', 'disturbing'
            ]
            
            emotional_count = sum(1 for word in emotional_words if word in text.lower())
            if emotional_count > 2:
                result['score'] -= 10
                result['warnings'].append("Uses emotional manipulation language")
            
            # Check for balanced viewpoint
            balanced_terms = [
                ('however', 1), ('but', 1), ('although', 1),
                ('on the other hand', 1), ('while', 1), ('some argue', 1),
                ('according to experts', 2), ('researchers say', 2)
            ]
            
            balanced_score = sum(text.lower().count(term) * weight for term, weight in balanced_terms)
            if balanced_score >= 2:
                result['score'] += 15
                result['reasons'].append("Content presents balanced viewpoints")
            
            # Check for journalistic terms that indicate legitimate reporting
            journalistic_terms = [
                ('according to', 2), ('sources say', 2), ('reported', 1),
                ('investigation', 2), ('analysis', 1), ('interview', 1),
                ('statement', 1), ('official', 1), ('confirmed', 1),
                ('spokesperson', 2), ('press release', 2), ('study', 1),
                ('research shows', 2), ('data indicates', 2), ('survey found', 2)
            ]
            
            journalistic_score = sum(text.lower().count(term) * weight for term, weight in journalistic_terms)
            if journalistic_score >= 3:
                result['score'] += 20
                result['reasons'].append("Content contains journalistic language and sourcing")
            elif journalistic_score >= 1:
                result['score'] += 10
                result['reasons'].append("Content shows some journalistic elements")
            
            # Check for credible sources mentioned
            credible_sources = [
                'reuters', 'associated press', 'bbc', 'cnn', 'npr', 'new york times',
                'washington post', 'wall street journal', 'bloomberg', 'guardian',
                'abc news', 'cbs news', 'nbc news', 'usa today', 'time magazine',
                'newsweek', 'economist', 'harvard', 'stanford', 'mit', 'oxford',
                'cambridge', 'who', 'cdc', 'fda', 'nasa', 'university'
            ]
            
            credible_count = sum(1 for source in credible_sources if source in text.lower())
            if credible_count > 0:
                result['score'] += credible_count * 5
                result['reasons'].append(f"References {credible_count} credible sources")
            
            # Ensure score is within bounds
            result['score'] = max(0, min(100, result['score']))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in content analysis: {str(e)}")
            return {'score': 0, 'reasons': ['Error analyzing content']}
    
    def _extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """Extract top keywords from text."""
        try:
            # Simple word frequency counter (could be enhanced with NLP)
            words = re.findall(r'\b\w{3,}\b', text.lower())  # Reduced minimum length to 3
            stopwords = {'this', 'that', 'with', 'have', 'from', 'they', 'will', 'would', 'there',
                        'their', 'what', 'about', 'which', 'when', 'just', 'also', 'said', 'says', 'like',
                        'were', 'been', 'has', 'had', 'more', 'can', 'could', 'should', 'would', 'may',
                        'might', 'must', 'shall', 'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                        'any', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'how', 'man',
                        'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put',
                        'say', 'she', 'too', 'use'}
            
            word_count = {}
            for word in words:
                if word not in stopwords and word.isalpha() and len(word) > 2:
                    word_count[word] = word_count.get(word, 0) + 1
            
            # Sort by frequency and get top N
            sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, _ in sorted_words[:n]]
            
            # If we don't have enough keywords, try named entities (simple approach)
            if len(keywords) < 3:
                # Look for capitalized words (potential names, places, organizations)
                capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
                for word in capitalized:
                    if word.lower() not in [k.lower() for k in keywords] and len(keywords) < n:
                        keywords.append(word.lower())
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def calculate_confidence(self, results: Dict) -> float:
        """Calculate final confidence score (0-100)."""
        try:
            # Simple weighted average of different components
            source_score = results.get('source_analysis', {}).get('source_score', 0)
            cross_score = results.get('cross_verification', {}).get('score', 0)
            content_score = results.get('content_analysis', {}).get('score', 0)
            
            # Weights can be adjusted based on importance
            weights = {
                'source': 0.4,
                'cross': 0.4,
                'content': 0.2
            }
            
            confidence = (
                (source_score * weights['source']) +
                (cross_score * weights['cross']) +
                (content_score * weights['content'])
            )
            
            return min(100, max(0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
