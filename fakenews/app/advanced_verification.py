"""
Advanced News Verification System with Multiple AI Layers
"""
import re
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from textblob import TextBlob
import spacy
from collections import Counter
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedNewsVerifier:
    def __init__(self, news_api_key: str):
        self.news_api_key = news_api_key
        self.news_api_base = "https://newsapi.org/v2"
        self.headers = {'User-Agent': 'NewsVerifier/1.0'}
        
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features will be limited.")
            self.nlp = None
            
        # Comprehensive source credibility database
        self.source_credibility = self._load_source_credibility()
        
        # Fake news patterns database
        self.fake_patterns = self._load_fake_patterns()
        
    def _load_source_credibility(self) -> Dict[str, Dict]:
        """Load comprehensive source credibility database."""
        return {
            # Tier 1 - Highly Credible (Score: 90-100)
            'reuters': {'score': 95, 'bias': 'neutral', 'factual': 'very_high'},
            'associated press': {'score': 95, 'bias': 'neutral', 'factual': 'very_high'},
            'bbc': {'score': 90, 'bias': 'slight_left', 'factual': 'very_high'},
            'npr': {'score': 88, 'bias': 'slight_left', 'factual': 'very_high'},
            'wall street journal': {'score': 87, 'bias': 'slight_right', 'factual': 'high'},
            
            # Tier 2 - Credible (Score: 70-89)
            'cnn': {'score': 75, 'bias': 'left', 'factual': 'high'},
            'new york times': {'score': 82, 'bias': 'left', 'factual': 'high'},
            'washington post': {'score': 80, 'bias': 'left', 'factual': 'high'},
            'guardian': {'score': 78, 'bias': 'left', 'factual': 'high'},
            'bloomberg': {'score': 85, 'bias': 'slight_left', 'factual': 'high'},
            
            # Indian Sources - Tier 1
            'the hindu': {'score': 85, 'bias': 'neutral', 'factual': 'high'},
            'indian express': {'score': 82, 'bias': 'neutral', 'factual': 'high'},
            'hindustan times': {'score': 78, 'bias': 'slight_right', 'factual': 'high'},
            'times of india': {'score': 70, 'bias': 'neutral', 'factual': 'medium_high'},
            'ndtv': {'score': 75, 'bias': 'left', 'factual': 'high'},
            
            # Sports Sources
            'espn': {'score': 88, 'bias': 'neutral', 'factual': 'very_high'},
            'cricbuzz': {'score': 85, 'bias': 'neutral', 'factual': 'high'},
            'cricinfo': {'score': 90, 'bias': 'neutral', 'factual': 'very_high'},
            'sky sports': {'score': 85, 'bias': 'neutral', 'factual': 'high'},
            'bbc sport': {'score': 90, 'bias': 'neutral', 'factual': 'very_high'},
            
            # Questionable Sources (Score: 30-50)
            'daily mail': {'score': 45, 'bias': 'right', 'factual': 'mixed'},
            'fox news': {'score': 55, 'bias': 'right', 'factual': 'mixed'},
            'buzzfeed': {'score': 40, 'bias': 'left', 'factual': 'mixed'},
            
            # Unreliable Sources (Score: 0-30)
            'infowars': {'score': 10, 'bias': 'extreme_right', 'factual': 'very_low'},
            'breitbart': {'score': 25, 'bias': 'extreme_right', 'factual': 'low'},
        }
    
    def _load_fake_patterns(self) -> Dict[str, List]:
        """Load patterns commonly found in fake news."""
        return {
            'death_hoax_patterns': [
                r'(prime minister|president|celebrity|actor|politician|leader).*(dead|died|death|killed|passed away)',
                r'(death|died|killed).*(prime minister|president|celebrity|actor|politician|leader)',
                r'breaking.*(death|died|killed).*(prime minister|president|celebrity)'
            ],
            'conspiracy_patterns': [
                r'(government|media|they).*(hiding|covering up|don\'t want you to know)',
                r'(secret|hidden|conspiracy|cover.?up)',
                r'(wake up|sheeple|open your eyes)',
                r'(mainstream media|msm).*(lies|lying|fake)'
            ],
            'medical_misinformation': [
                r'(doctors|medical establishment).*(hiding|suppressing)',
                r'(cure|treatment).*(they don\'t want|big pharma)',
                r'(vaccine|vaccination).*(dangerous|deadly|kills)'
            ],
            'clickbait_patterns': [
                r'you won\'t believe',
                r'this will shock you',
                r'doctors hate this',
                r'[0-9]+ things .* don\'t want you to know'
            ]
        }
    
    def verify_news_advanced(self, text: str, source_url: str = None) -> Dict:
        """Advanced multi-layer news verification."""
        try:
            results = {
                'is_trustworthy': False,
                'confidence_score': 0,
                'verification_layers': {},
                'final_verdict': 'UNKNOWN',
                'reasoning': [],
                'warnings': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Layer 1: Content Analysis (Advanced NLP)
            content_analysis = self._advanced_content_analysis(text)
            results['verification_layers']['content_analysis'] = content_analysis
            
            # Layer 2: Source Verification
            source_analysis = self._advanced_source_analysis(source_url) if source_url else {'score': 50, 'reasons': ['No source URL provided']}
            results['verification_layers']['source_analysis'] = source_analysis
            
            # Layer 3: Cross-Reference Verification
            cross_reference = self._advanced_cross_reference(text)
            results['verification_layers']['cross_reference'] = cross_reference
            
            # Layer 4: Pattern Matching (Fake News Detection)
            pattern_analysis = self._pattern_analysis(text)
            results['verification_layers']['pattern_analysis'] = pattern_analysis
            
            # Layer 5: Semantic Consistency Check
            semantic_analysis = self._semantic_consistency_check(text)
            results['verification_layers']['semantic_analysis'] = semantic_analysis
            
            # Layer 6: Entity Verification
            entity_analysis = self._entity_verification(text)
            results['verification_layers']['entity_analysis'] = entity_analysis
            
            # Advanced Decision Engine
            final_decision = self._advanced_decision_engine(results['verification_layers'], text)
            results.update(final_decision)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced verification: {str(e)}")
            return {
                'is_trustworthy': False,
                'confidence_score': 0,
                'final_verdict': 'ERROR',
                'error': str(e),
                'reasoning': ['System error during verification']
            }
    
    def _advanced_content_analysis(self, text: str) -> Dict:
        """Advanced NLP-based content analysis."""
        analysis = {
            'score': 50,
            'features': {},
            'reasons': [],
            'warnings': []
        }
        
        try:
            # Text statistics
            word_count = len(text.split())
            sentence_count = len(text.split('.'))
            
            analysis['features']['word_count'] = word_count
            analysis['features']['sentence_count'] = sentence_count
            analysis['features']['avg_sentence_length'] = word_count / max(sentence_count, 1)
            
            # Sentiment analysis
            blob = TextBlob(text)
            sentiment = blob.sentiment
            analysis['features']['sentiment_polarity'] = sentiment.polarity
            analysis['features']['sentiment_subjectivity'] = sentiment.subjectivity
            
            # Extremely polarized sentiment can indicate bias
            if abs(sentiment.polarity) > 0.7:
                analysis['score'] -= 10
                analysis['warnings'].append(f"Highly polarized sentiment ({sentiment.polarity:.2f})")
            
            # High subjectivity can indicate opinion rather than fact
            if sentiment.subjectivity > 0.8:
                analysis['score'] -= 15
                analysis['warnings'].append(f"Highly subjective content ({sentiment.subjectivity:.2f})")
            
            # Check for balanced reporting indicators
            balanced_indicators = [
                'however', 'but', 'although', 'on the other hand', 'while',
                'according to', 'sources say', 'reported', 'allegedly'
            ]
            
            balanced_count = sum(1 for indicator in balanced_indicators if indicator.lower() in text.lower())
            if balanced_count >= 3:
                analysis['score'] += 20
                analysis['reasons'].append("Contains balanced reporting language")
            elif balanced_count >= 1:
                analysis['score'] += 10
                analysis['reasons'].append("Some balanced reporting elements")
            
            # Check for proper attribution
            attribution_patterns = [
                r'according to.*?sources?',
                r'[A-Z][a-z]+ (said|told|stated|confirmed)',
                r'in a statement',
                r'spokesperson.*?(said|confirmed)',
                r'official.*?(announced|confirmed)'
            ]
            
            attribution_count = sum(1 for pattern in attribution_patterns if re.search(pattern, text))
            if attribution_count >= 2:
                analysis['score'] += 15
                analysis['reasons'].append("Good source attribution")
            elif attribution_count >= 1:
                analysis['score'] += 8
                analysis['reasons'].append("Some source attribution")
            
            # NLP entity analysis if spaCy is available
            if self.nlp:
                doc = self.nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                analysis['features']['entities'] = entities
                
                # Check for credible organizations/people mentioned
                credible_orgs = ['reuters', 'bbc', 'cnn', 'government', 'university', 'research', 'study']
                org_mentions = sum(1 for ent_text, ent_label in entities 
                                 if ent_label in ['ORG', 'PERSON'] and 
                                 any(org in ent_text.lower() for org in credible_orgs))
                
                if org_mentions >= 2:
                    analysis['score'] += 12
                    analysis['reasons'].append("References credible organizations/people")
            
            # Ensure score is within bounds
            analysis['score'] = max(0, min(100, analysis['score']))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in content analysis: {str(e)}")
            return {'score': 0, 'error': str(e), 'reasons': ['Error in content analysis']}
    
    def _advanced_source_analysis(self, source_url: str) -> Dict:
        """Advanced source credibility analysis."""
        analysis = {
            'score': 50,
            'credibility_tier': 'unknown',
            'reasons': [],
            'warnings': []
        }
        
        if not source_url:
            return analysis
        
        try:
            # Extract domain
            domain = re.search(r'https?://(?:www\.)?([^/]+)', source_url.lower())
            if not domain:
                analysis['warnings'].append("Could not extract domain from URL")
                return analysis
            
            domain_name = domain.group(1)
            
            # Check against credibility database
            for source, info in self.source_credibility.items():
                if source in domain_name or any(part in domain_name for part in source.split()):
                    analysis['score'] = info['score']
                    analysis['credibility_tier'] = self._get_credibility_tier(info['score'])
                    analysis['bias'] = info.get('bias', 'unknown')
                    analysis['factual_rating'] = info.get('factual', 'unknown')
                    
                    if info['score'] >= 85:
                        analysis['reasons'].append(f"Highly credible source ({source})")
                    elif info['score'] >= 70:
                        analysis['reasons'].append(f"Credible source ({source})")
                    elif info['score'] >= 50:
                        analysis['reasons'].append(f"Moderately reliable source ({source})")
                    else:
                        analysis['warnings'].append(f"Low credibility source ({source})")
                    break
            else:
                # Unknown source - moderate score
                analysis['score'] = 45
                analysis['reasons'].append("Unknown source - moderate credibility assumed")
            
            # Check for suspicious domain patterns
            suspicious_patterns = [
                r'\.tk$', r'\.ml$', r'\.ga$',  # Free domains
                r'news\d+', r'breaking.*news',  # Generic news patterns
                r'real.*truth', r'truth.*news'  # "Truth" sites often unreliable
            ]
            
            if any(re.search(pattern, domain_name) for pattern in suspicious_patterns):
                analysis['score'] -= 20
                analysis['warnings'].append("Suspicious domain pattern detected")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in source analysis: {str(e)}")
            return {'score': 0, 'error': str(e), 'reasons': ['Error in source analysis']}
    
    def _get_credibility_tier(self, score: int) -> str:
        """Get credibility tier based on score."""
        if score >= 85:
            return 'very_high'
        elif score >= 70:
            return 'high'
        elif score >= 50:
            return 'medium'
        elif score >= 30:
            return 'low'
        else:
            return 'very_low'
    
    def _advanced_cross_reference(self, text: str) -> Dict:
        """Advanced cross-reference verification with multiple APIs."""
        analysis = {
            'score': 0,
            'matching_articles': 0,
            'trusted_sources': [],
            'contradicting_sources': [],
            'reasons': [],
            'search_strategies': []
        }
        
        try:
            # Extract key entities and keywords
            keywords = self._extract_advanced_keywords(text)
            entities = self._extract_entities(text) if self.nlp else []
            
            # Multiple search strategies
            search_strategies = [
                (' AND '.join(keywords[:3]), 'precise_keywords'),
                (' OR '.join(keywords[:5]), 'broad_keywords'),
                (' AND '.join([e[0] for e in entities[:3]]), 'entity_search') if entities else None,
                (f'"{keywords[0]}" AND "{keywords[1]}"', 'quoted_search') if len(keywords) >= 2 else None
            ]
            
            # Filter out None strategies
            search_strategies = [s for s in search_strategies if s is not None]
            
            all_articles = []
            for query, strategy_name in search_strategies:
                articles = self._search_news_api(query)
                all_articles.extend(articles)
                analysis['search_strategies'].append({
                    'strategy': strategy_name,
                    'query': query,
                    'results': len(articles)
                })
            
            # Remove duplicates
            unique_articles = self._deduplicate_articles(all_articles)
            analysis['matching_articles'] = len(unique_articles)
            
            if not unique_articles:
                analysis['reasons'].append("No matching articles found in recent news")
                return analysis
            
            # Analyze articles
            trusted_count = 0
            very_trusted_count = 0
            contradicting_count = 0
            
            for article in unique_articles:
                source_name = article.get('source', {}).get('name', '').lower()
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                
                # Check source credibility
                source_info = None
                for source, info in self.source_credibility.items():
                    if source in source_name:
                        source_info = info
                        break
                
                if source_info:
                    if source_info['score'] >= 85:
                        very_trusted_count += 1
                        analysis['trusted_sources'].append(source_name)
                    elif source_info['score'] >= 70:
                        trusted_count += 1
                        analysis['trusted_sources'].append(source_name)
                    elif source_info['score'] < 40:
                        contradicting_count += 1
                        analysis['contradicting_sources'].append(source_name)
                
                # Check for contradictory information
                if self._check_contradiction(text, title + ' ' + description):
                    contradicting_count += 1
            
            # Calculate score based on verification
            if very_trusted_count >= 2:
                analysis['score'] = min(95, 70 + very_trusted_count * 10)
                analysis['reasons'].append(f"Verified by {very_trusted_count} highly trusted sources")
            elif very_trusted_count >= 1 and trusted_count >= 1:
                analysis['score'] = min(90, 60 + (very_trusted_count + trusted_count) * 8)
                analysis['reasons'].append(f"Verified by {very_trusted_count + trusted_count} credible sources")
            elif trusted_count >= 2:
                analysis['score'] = min(80, 50 + trusted_count * 10)
                analysis['reasons'].append(f"Verified by {trusted_count} trusted sources")
            elif trusted_count >= 1:
                analysis['score'] = 45
                analysis['reasons'].append("Verified by 1 trusted source")
            else:
                analysis['score'] = 20 if len(unique_articles) > 0 else 0
                analysis['reasons'].append("Found related articles but from less established sources")
            
            # Penalty for contradicting sources
            if contradicting_count > 0:
                analysis['score'] -= contradicting_count * 15
                analysis['warnings'] = [f"Found {contradicting_count} contradicting or unreliable sources"]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in cross-reference: {str(e)}")
            return {'score': 0, 'error': str(e), 'reasons': ['Error in cross-reference verification']}
    
    def _pattern_analysis(self, text: str) -> Dict:
        """Analyze text for fake news patterns."""
        analysis = {
            'score': 70,  # Start neutral
            'detected_patterns': [],
            'warnings': [],
            'reasons': []
        }
        
        text_lower = text.lower()
        
        # Check each pattern category
        for category, patterns in self.fake_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern)
            
            if matches:
                analysis['detected_patterns'].append({
                    'category': category,
                    'matches': len(matches),
                    'patterns': matches
                })
                
                # Apply penalties based on category
                if category == 'death_hoax_patterns':
                    analysis['score'] -= 30
                    analysis['warnings'].append("Contains death hoax patterns")
                elif category == 'conspiracy_patterns':
                    analysis['score'] -= 25
                    analysis['warnings'].append("Contains conspiracy theory language")
                elif category == 'medical_misinformation':
                    analysis['score'] -= 35
                    analysis['warnings'].append("Contains medical misinformation patterns")
                elif category == 'clickbait_patterns':
                    analysis['score'] -= 15
                    analysis['warnings'].append("Contains clickbait patterns")
        
        # Check for legitimate journalism patterns
        journalism_patterns = [
            r'according to.*?(official|source|report)',
            r'(investigation|study|research).*?(shows|finds|reveals)',
            r'(spokesperson|representative).*?(said|confirmed|announced)',
            r'(data|statistics|numbers).*?(show|indicate|suggest)'
        ]
        
        journalism_matches = sum(1 for pattern in journalism_patterns if re.search(pattern, text_lower))
        if journalism_matches >= 2:
            analysis['score'] += 20
            analysis['reasons'].append("Contains legitimate journalism patterns")
        elif journalism_matches >= 1:
            analysis['score'] += 10
            analysis['reasons'].append("Some journalism patterns detected")
        
        # Ensure score bounds
        analysis['score'] = max(0, min(100, analysis['score']))
        
        return analysis
    
    def _semantic_consistency_check(self, text: str) -> Dict:
        """Check for semantic consistency and logical flow."""
        analysis = {
            'score': 60,
            'consistency_score': 0,
            'reasons': [],
            'warnings': []
        }
        
        try:
            sentences = text.split('.')
            if len(sentences) < 2:
                analysis['reasons'].append("Text too short for consistency analysis")
                return analysis
            
            # Check for contradictory statements
            contradiction_indicators = [
                ('not', 'is'), ('never', 'always'), ('impossible', 'confirmed'),
                ('denied', 'confirmed'), ('false', 'true')
            ]
            
            contradictions = 0
            for sent in sentences:
                sent_lower = sent.lower()
                for neg, pos in contradiction_indicators:
                    if neg in sent_lower and pos in sent_lower:
                        contradictions += 1
            
            if contradictions > 0:
                analysis['score'] -= contradictions * 10
                analysis['warnings'].append(f"Found {contradictions} potential contradictions")
            else:
                analysis['score'] += 10
                analysis['reasons'].append("No obvious contradictions found")
            
            # Check temporal consistency
            dates = re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', text)
            if dates:
                current_year = datetime.now().year
                future_dates = [d for d in dates if d.isdigit() and int(d) > current_year + 1]
                if future_dates:
                    analysis['score'] -= 20
                    analysis['warnings'].append("Contains suspicious future dates")
            
            analysis['consistency_score'] = analysis['score']
            return analysis
            
        except Exception as e:
            logger.error(f"Error in semantic consistency check: {str(e)}")
            return {'score': 50, 'error': str(e)}
    
    def _entity_verification(self, text: str) -> Dict:
        """Verify entities mentioned in the text."""
        analysis = {
            'score': 50,
            'entities_found': [],
            'verified_entities': [],
            'suspicious_entities': [],
            'reasons': []
        }
        
        if not self.nlp:
            analysis['reasons'].append("Entity verification unavailable (spaCy not loaded)")
            return analysis
        
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            analysis['entities_found'] = entities
            
            # Check for well-known entities
            known_entities = {
                'PERSON': ['narendra modi', 'joe biden', 'donald trump', 'elon musk'],
                'ORG': ['bbc', 'cnn', 'reuters', 'government', 'nasa', 'who'],
                'GPE': ['india', 'usa', 'china', 'united kingdom', 'delhi', 'mumbai']
            }
            
            verified_count = 0
            for ent_text, ent_label in entities:
                ent_lower = ent_text.lower()
                if ent_label in known_entities:
                    if any(known in ent_lower for known in known_entities[ent_label]):
                        verified_count += 1
                        analysis['verified_entities'].append(ent_text)
            
            if verified_count > 0:
                analysis['score'] += min(30, verified_count * 10)
                analysis['reasons'].append(f"Contains {verified_count} verifiable entities")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in entity verification: {str(e)}")
            return {'score': 50, 'error': str(e)}
    
    def _advanced_decision_engine(self, layers: Dict, original_text: str) -> Dict:
        """Advanced decision engine that combines all verification layers."""
        
        # Extract scores from each layer
        content_score = layers.get('content_analysis', {}).get('score', 50)
        source_score = layers.get('source_analysis', {}).get('score', 50)
        cross_ref_score = layers.get('cross_reference', {}).get('score', 0)
        pattern_score = layers.get('pattern_analysis', {}).get('score', 70)
        semantic_score = layers.get('semantic_analysis', {}).get('score', 60)
        entity_score = layers.get('entity_analysis', {}).get('score', 50)
        
        # Weighted scoring system
        weights = {
            'cross_reference': 0.30,  # Most important - external verification
            'pattern_analysis': 0.25, # Very important - fake news patterns
            'content_analysis': 0.20, # Important - content quality
            'source_analysis': 0.15,  # Moderately important - source credibility
            'semantic_analysis': 0.05, # Less important - consistency
            'entity_analysis': 0.05   # Less important - entity verification
        }
        
        # Calculate weighted score
        weighted_score = (
            cross_ref_score * weights['cross_reference'] +
            pattern_score * weights['pattern_analysis'] +
            content_score * weights['content_analysis'] +
            source_score * weights['source_analysis'] +
            semantic_score * weights['semantic_analysis'] +
            entity_score * weights['entity_analysis']
        )
        
        # Advanced decision logic
        reasoning = []
        warnings = []
        
        # Critical failure conditions (immediate FALSE verdict)
        critical_patterns = layers.get('pattern_analysis', {}).get('detected_patterns', [])
        death_hoax = any(p['category'] == 'death_hoax_patterns' for p in critical_patterns)
        medical_misinfo = any(p['category'] == 'medical_misinformation' for p in critical_patterns)
        
        if death_hoax and cross_ref_score < 30:
            verdict = 'FALSE'
            confidence = 95
            reasoning.append("Death hoax patterns detected with no credible source verification")
        elif medical_misinfo and cross_ref_score < 20:
            verdict = 'FALSE'
            confidence = 90
            reasoning.append("Medical misinformation patterns with no credible verification")
        
        # High confidence TRUE conditions
        elif cross_ref_score >= 80 and pattern_score >= 60:
            verdict = 'TRUE'
            confidence = min(95, weighted_score)
            reasoning.append("Multiple trusted sources verify the information")
        elif cross_ref_score >= 70 and content_score >= 70 and pattern_score >= 70:
            verdict = 'TRUE'
            confidence = min(90, weighted_score)
            reasoning.append("Good verification from trusted sources with quality content")
        
        # Moderate confidence TRUE conditions
        elif cross_ref_score >= 50 and pattern_score >= 65 and content_score >= 60:
            verdict = 'TRUE'
            confidence = min(75, weighted_score)
            reasoning.append("Moderate verification with good content quality")
        
        # Uncertain/Moderate conditions
        elif 40 <= weighted_score <= 60:
            verdict = 'UNCERTAIN'
            confidence = abs(weighted_score - 50) + 50
            reasoning.append("Mixed signals from verification layers - uncertain verdict")
        
        # FALSE conditions
        elif pattern_score < 40 or cross_ref_score < 20:
            verdict = 'FALSE'
            confidence = min(85, 100 - weighted_score)
            reasoning.append("Poor pattern analysis or lack of credible source verification")
        else:
            verdict = 'FALSE'
            confidence = min(80, 100 - weighted_score)
            reasoning.append("Overall analysis suggests unreliable information")
        
        # Special handling for sports content
        is_sports = any(word in original_text.lower() for word in [
            'ipl', 'cricket', 'match', 'team', 'won', 'victory', 'championship',
            'tournament', 'game', 'score', 'runs', 'wickets', 'rcb', 'csk', 'mi'
        ])
        
        if is_sports and cross_ref_score >= 30:
            # Be more lenient with sports content
            if verdict == 'FALSE' and confidence < 80:
                verdict = 'UNCERTAIN'
                confidence = 60
                reasoning.append("Sports content - requires more lenient verification")
        
        # Final adjustments
        is_trustworthy = verdict == 'TRUE'
        confidence_score = max(50, min(95, confidence))  # Ensure reasonable confidence bounds
        
        return {
            'is_trustworthy': is_trustworthy,
            'confidence_score': confidence_score,
            'final_verdict': verdict,
            'weighted_score': round(weighted_score, 2),
            'reasoning': reasoning,
            'warnings': warnings,
            'layer_scores': {
                'content': content_score,
                'source': source_score,
                'cross_reference': cross_ref_score,
                'patterns': pattern_score,
                'semantic': semantic_score,
                'entities': entity_score
            }
        }
    
    def _extract_advanced_keywords(self, text: str, n: int = 10) -> List[str]:
        """Extract advanced keywords using multiple techniques."""
        try:
            # Basic word extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Enhanced stopwords
            stopwords = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 'had',
                'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
                'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its',
                'let', 'put', 'say', 'she', 'too', 'use', 'with', 'have', 'from', 'they',
                'will', 'would', 'there', 'their', 'what', 'about', 'which', 'when', 'just',
                'also', 'said', 'says', 'like', 'were', 'been', 'more', 'could', 'should',
                'may', 'might', 'must', 'shall', 'this', 'that'
            }
            
            # Filter and count
            word_freq = Counter([w for w in words if w not in stopwords and len(w) > 2])
            
            # Get top keywords
            top_keywords = [word for word, freq in word_freq.most_common(n)]
            
            # Add entity-based keywords if available
            if self.nlp:
                doc = self.nlp(text)
                entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 2]
                # Combine with frequency-based keywords
                combined = list(set(top_keywords + entities))[:n]
                return combined
            
            return top_keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _search_news_api(self, query: str, max_results: int = 15) -> List[Dict]:
        """Search news API with enhanced parameters."""
        try:
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'pageSize': max_results,
                'sortBy': 'relevancy',
                'language': 'en',
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # Last week
            }
            
            response = requests.get(
                f"{self.news_api_base}/everything",
                params=params,
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                logger.warning(f"News API returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching news API: {str(e)}")
            return []
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on URL and title similarity."""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            title = article.get('title', '').lower()
            
            # Skip if URL already seen
            if url and url in seen_urls:
                continue
            
            # Skip if very similar title already seen
            if title and any(self._title_similarity(title, seen_title) > 0.8 for seen_title in seen_titles):
                continue
            
            seen_urls.add(url)
            seen_titles.add(title)
            unique_articles.append(article)
        
        return unique_articles
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _check_contradiction(self, original_text: str, comparison_text: str) -> bool:
        """Check if two texts contradict each other."""
        try:
            # Simple contradiction detection
            original_lower = original_text.lower()
            comparison_lower = comparison_text.lower()
            
            # Look for negation patterns
            contradiction_pairs = [
                ('confirmed', 'denied'), ('true', 'false'), ('yes', 'no'),
                ('won', 'lost'), ('alive', 'dead'), ('guilty', 'innocent')
            ]
            
            for pos, neg in contradiction_pairs:
                if ((pos in original_lower and neg in comparison_lower) or 
                    (neg in original_lower and pos in comparison_lower)):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking contradiction: {str(e)}")
            return False