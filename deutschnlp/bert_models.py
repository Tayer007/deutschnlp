import torch
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification

class BERTModels:
    """
    Integration of BERT models for German text analysis.
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize BERT models for different NLP tasks.
        
        Args:
            use_gpu: Whether to use GPU for model inference
        """
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        
        # NER model
        print("Loading BERT NER model...")
        self.ner_model_name = "xlm-roberta-large-finetuned-conll03-german"
        self.ner_tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(self.ner_model_name)
        self.ner_pipeline = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer, device=self.device)
        
        # Sentiment analysis model
        print("Loading BERT sentiment model...")
        self.sentiment_model_name = "oliverguhr/german-sentiment-bert"
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.sentiment_model, tokenizer=self.sentiment_tokenizer, device=self.device)
    
    def analyze_entities(self, text):
        """
        Perform named entity recognition using BERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (entity, label)
        """
        entities = self.ner_pipeline(text)
        
        # Format results to match spaCy format
        formatted_entities = []
        for entity in entities:
            formatted_entities.append((entity['word'], entity['entity']))
        
        return formatted_entities
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using BERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Split text into sentences for sentence-level analysis
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Get overall sentiment
        overall_result = self.sentiment_pipeline(text)[0]
        
        # Convert to -1 to 1 scale to match our sentiment analyzer
        score_mapping = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }
        
        overall_score = score_mapping.get(overall_result['label'], 0) * overall_result['score']
        
        # Analyze each sentence
        sentence_results = []
        for sentence in sentences:
            result = self.sentiment_pipeline(sentence)[0]
            score = score_mapping.get(result['label'], 0) * result['score']
            sentence_results.append({
                'text': sentence,
                'score': score
            })
        
        # Interpret overall score
        interpretation = self._interpret_score(overall_score)
        
        return {
            'overall_score': overall_score,
            'interpretation': interpretation,
            'sentences': sentence_results
        }
    
    def _interpret_score(self, score):
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
