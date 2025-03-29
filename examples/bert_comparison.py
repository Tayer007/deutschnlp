"""
BERT vs spaCy comparison for German text analysis.

This script demonstrates how to use and compare BERT and spaCy models
for various NLP tasks on German text.
"""

import sys
import os
import time

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import DeutschNLP components
from deutschnlp.analyzer import GermanAnalyzer
from deutschnlp.sentiment import SentimentAnalyzer
from deutschnlp.bert_models import BERTModels

def compare_ner(text):
    """Compare Named Entity Recognition between spaCy and BERT."""
    print("\n" + "=" * 50)
    print("NAMED ENTITY RECOGNITION COMPARISON")
    print("=" * 50)
    print(f"Text: {text}")
    
    # spaCy NER
    spacy_analyzer = GermanAnalyzer()
    spacy_start = time.time()
    spacy_result = spacy_analyzer.analyze(text)
    spacy_time = time.time() - spacy_start
    spacy_entities = spacy_result.named_entities
    
    # BERT NER
    bert = BERTModels()
    bert_start = time.time()
    bert_entities = bert.analyze_entities(text)
    bert_time = time.time() - bert_start
    
    # Print results
    print("\nspaCy NER Results:")
    print(f"Processing time: {spacy_time:.4f} seconds")
    if spacy_entities:
        for entity, label in spacy_entities:
            print(f"  - {entity}: {label}")
    else:
        print("  No entities found")
    
    print("\nBERT NER Results:")
    print(f"Processing time: {bert_time:.4f} seconds")
    if bert_entities:
        for entity, label in bert_entities:
            print(f"  - {entity}: {label}")
    else:
        print("  No entities found")
    
    # Compare results
    spacy_set = {f"{e[0]}_{e[1]}" for e in spacy_entities}
    bert_set = {f"{e[0]}_{e[1]}" for e in bert_entities}
    
    common = spacy_set.intersection(bert_set)
    only_spacy = spacy_set - bert_set
    only_bert = bert_set - spacy_set
    
    print("\nComparison:")
    print(f"  Common entities: {len(common)}")
    print(f"  Only in spaCy: {len(only_spacy)}")
    print(f"  Only in BERT: {len(only_bert)}")
    print(f"  Speed comparison: BERT is {spacy_time/bert_time:.2f}x {'faster' if spacy_time > bert_time else 'slower'} than spaCy")

def compare_sentiment(text):
    """Compare Sentiment Analysis between spaCy and BERT."""
    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS COMPARISON")
    print("=" * 50)
    print(f"Text: {text}")
    
    # spaCy sentiment
    spacy_analyzer = SentimentAnalyzer()
    spacy_start = time.time()
    spacy_result = spacy_analyzer.analyze_detailed(text)
    spacy_time = time.time() - spacy_start
    
    # BERT sentiment
    bert = BERTModels()
    bert_start = time.time()
    bert_result = bert.analyze_sentiment(text)
    bert_time = time.time() - bert_start
    
    # Print results
    print("\nspaCy Sentiment Results:")
    print(f"Processing time: {spacy_time:.4f} seconds")
    print(f"  Overall score: {spacy_result['overall_score']:.2f}")
    print(f"  Interpretation: {spacy_result['interpretation']}")
    
    print("\nBERT Sentiment Results:")
    print(f"Processing time: {bert_time:.4f} seconds")
    print(f"  Overall score: {bert_result['overall_score']:.2f}")
    print(f"  Interpretation: {bert_result['interpretation']}")
    
    # Compare results
    sentiment_diff = abs(spacy_result['overall_score'] - bert_result['overall_score'])
    
    print("\nComparison:")
    print(f"  Sentiment score difference: {sentiment_diff:.2f}")
    print(f"  Agreement: {'High' if sentiment_diff < 0.3 else 'Medium' if sentiment_diff < 0.6 else 'Low'}")
    print(f"  Speed comparison: BERT is {spacy_time/bert_time:.2f}x {'faster' if spacy_time > bert_time else 'slower'} than spaCy")

if __name__ == "__main__":
    # Test examples
    ner_text = "Angela Merkel besuchte gestern Berlin und traf sich mit Vertretern von Volkswagen und BMW, um über die Zukunft der deutschen Automobilindustrie zu sprechen."
    sentiment_positive = "Das Essen in diesem Restaurant war ausgezeichnet! Der Service war sehr gut und das Personal äußerst freundlich."
    sentiment_negative = "Die Qualität des Produkts ist miserabel. Ich bin absolut unzufrieden mit dem Kundenservice."
    
    # Compare NER
    compare_ner(ner_text)
    
    # Compare sentiment analysis
    compare_sentiment(sentiment_positive)
    compare_sentiment(sentiment_negative)
