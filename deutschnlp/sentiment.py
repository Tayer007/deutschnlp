import spacy
from typing import Dict, List, Union, Tuple
import os
import json


class SentimentAnalyzer:
    """
    German text sentiment analyzer.
    
    This class provides functionality for sentiment analysis of German text,
    using a combination of lexicon-based approaches and spaCy's language model.
    """
    
    def __init__(self, model_name="de_core_news_md"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the spaCy German model to use
        """
        self.nlp = spacy.load(model_name)
        
        # Load German sentiment lexicon
        # This is a simplified example - in a real implementation,
        # you would use a more comprehensive sentiment lexicon
        self.sentiment_lexicon = {
            # Positive words
            "gut": 1.0,
            "schön": 1.0,
            "großartig": 1.5,
            "ausgezeichnet": 1.5,
            "hervorragend": 1.5,
            "wunderbar": 1.5,
            "fantastisch": 1.5,
            "positiv": 1.0,
            "glücklich": 1.0,
            "zufrieden": 1.0,
            "freundlich": 1.0,
            "angenehm": 0.8,
            "perfekt": 1.5,
            "ideal": 1.2,
            "empfehlenswert": 1.2,
            "liebenswert": 1.0,
            "exzellent": 1.5,
            "super": 1.2,
            "toll": 1.0,
            "prima": 0.8,
            "genial": 1.3,
            
            # Negative words
            "schlecht": -1.0,
            "schrecklich": -1.5,
            "furchtbar": -1.5,
            "entsetzlich": -1.5,
            "grauenhaft": -1.5,
            "fürchterlich": -1.5,
            "miserabel": -1.5,
            "negativ": -1.0,
            "enttäuschend": -1.0,
            "traurig": -1.0,
            "ärgerlich": -1.0,
            "unangenehm": -0.8,
            "schrecklich": -1.5,
            "mangelhaft": -1.0,
            "unzureichend": -1.0,
            "unfreundlich": -1.0,
            "problematisch": -0.8,
            "katastrophal": -1.5,
            "unzumutbar": -1.2,
            "ungeeignet": -0.8,
            
            # Intensifiers
            "sehr": 1.5,
            "besonders": 1.3,
            "äußerst": 1.8,
            "unglaublich": 1.5,
            "extrem": 1.8,
            "ziemlich": 1.2,
            "total": 1.5,
            "absolut": 1.8,
            "komplett": 1.5,
            
            # Negations
            "nicht": -1.0,
            "keine": -1.0,
            "kein": -1.0,
            "nichts": -1.0,
            "nie": -1.0,
            "niemals": -1.0,
        }
        
        # Try to load a custom lexicon if available
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lexicon_path = os.path.join(current_dir, 'data', 'sentiment_lexicon_de.json')
            if os.path.exists(lexicon_path):
                with open(lexicon_path, 'r', encoding='utf-8') as f:
                    custom_lexicon = json.load(f)
                    self.sentiment_lexicon.update(custom_lexicon)
        except Exception as e:
            print(f"Could not load custom sentiment lexicon: {e}")
    
    def analyze(self, text: str) -> float:
        """
        Analyze the sentiment of German text.
        
        Args:
            text: German text to analyze
            
        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        doc = self.nlp(text)
        
        # Collect sentiment scores for all words
        scores = []
        intensifier = 1.0
        negation = 1.0
        
        for i, token in enumerate(doc):
            # Check for negations and intensifiers
            if token.lemma_.lower() in ["nicht", "keine", "kein", "nichts", "nie", "niemals"]:
                negation = -1.0
                continue
                
            if token.lemma_.lower() in ["sehr", "besonders", "äußerst", "unglaublich", 
                                        "extrem", "ziemlich", "total", "absolut", "komplett"]:
                intensifier = 1.5
                continue
            
            # Get sentiment score for the current word
            sentiment_score = self.sentiment_lexicon.get(token.lemma_.lower(), 0)
            
            # Apply negation and intensifier if needed
            if sentiment_score != 0:
                adjusted_score = sentiment_score * negation * intensifier
                scores.append(adjusted_score)
                
                # Reset modifiers
                negation = 1.0
                intensifier = 1.0
        
        # Compute final sentiment score
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def analyze_detailed(self, text: str) -> Dict:
        """
        Perform detailed sentiment analysis on German text.
        
        Args:
            text: German text to analyze
            
        Returns:
            Dictionary containing sentiment score and breakdown by sentence
        """
        doc = self.nlp(text)
        
        overall_score = self.analyze(text)
        
        # Analyze sentiment by sentence
        sentence_sentiments = []
        for sent in doc.sents:
            sent_score = self.analyze(sent.text)
            sentence_sentiments.append({
                "text": sent.text,
                "score": sent_score
            })
        
        return {
            "overall_score": overall_score,
            "sentences": sentence_sentiments,
            "interpretation": self._interpret_score(overall_score)
        }
    
    def _interpret_score(self, score: float) -> str:
        """
        Interpret a sentiment score as text.
        
        Args:
            score: Sentiment score
            
        Returns:
            String interpretation of the sentiment
        """
        if score >= 0.5:
            return "Sehr positiv"
        elif score >= 0.1:
            return "Positiv"
        elif score > -0.1:
            return "Neutral"
        elif score > -0.5:
            return "Negativ"
        else:
            return "Sehr negativ"


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Test with positive, negative, and neutral examples
    examples = [
        "Das Essen in diesem Restaurant war ausgezeichnet!",
        "Der Service war sehr gut und das Personal äußerst freundlich.",
        "Die Qualität des Produkts ist miserabel.",
        "Ich bin absolut unzufrieden mit dem Kundenservice.",
        "Das ist ein normaler Satz ohne emotionalen Inhalt."
    ]
    
    for example in examples:
        sentiment = analyzer.analyze(example)
        detailed = analyzer.analyze_detailed(example)
        print(f"Text: {example}")
        print(f"Sentiment score: {sentiment:.2f}")
        print(f"Interpretation: {detailed['interpretation']}")
        print("-" * 50)
