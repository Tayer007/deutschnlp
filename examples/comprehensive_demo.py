"""
Comprehensive demonstration of DeutschNLP toolkit capabilities.

This script demonstrates the main features of the DeutschNLP toolkit:
- Text analysis (tokenization, lemmatization, POS tagging, NER)
- Sentiment analysis
- Text comparison
- Dependency visualization
"""

import os
import sys
from pprint import pprint

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import DeutschNLP components
from deutschnlp.analyzer import GermanAnalyzer
from deutschnlp.sentiment import SentimentAnalyzer
from deutschnlp.similarity import TextComparator
from deutschnlp.visualizer import DependencyVisualizer

# Example texts
simple_text = "Er ist gestern sehr schnell mit dem Fahrrad nach Berlin gefahren."

complex_text = """
Die Bundesrepublik Deutschland ist ein demokratischer und sozialer Bundesstaat.
Berlin ist die Hauptstadt der Bundesrepublik Deutschland.
Die Deutsche Bahn hat gestern massive Verspätungen gemeldet.
Angela Merkel war von 2005 bis 2021 Bundeskanzlerin.
"""

positive_text = "Das Essen in diesem Restaurant war ausgezeichnet! Der Service war sehr gut."
negative_text = "Die Qualität des Produkts ist miserabel. Ich bin absolut unzufrieden."

similar_text1 = "Berlin ist die Hauptstadt von Deutschland."
similar_text2 = "Die Hauptstadt Deutschlands ist Berlin."


def demo_analysis():
    """Demonstrate text analysis capabilities."""
    print("\n" + "=" * 50)
    print("TEXT ANALYSIS DEMO")
    print("=" * 50)
    
    analyzer = GermanAnalyzer()
    
    print("\nSimple text analysis:")
    result = analyzer.analyze(simple_text)
    
    print(f"Text: {result.raw_text}")
    print("\nTokens:")
    print(", ".join(result.tokens))
    
    print("\nLemmas:")
    print(", ".join(result.lemmas))
    
    print("\nPOS Tags:")
    for token, pos in result.pos_tags:
        print(f"  {token}: {pos}")
    
    print("\nNamed Entities:")
    for entity, label in result.named_entities:
        print(f"  {entity}: {label}")
    
    print("\nComplex text analysis (named entities only):")
    result = analyzer.analyze(complex_text)
    
    print(f"Text:\n{result.raw_text}")
    print("\nNamed Entities:")
    for entity, label in result.named_entities:
        print(f"  {entity}: {label}")
    
    print("\nNoun Chunks:")
    for chunk in result.noun_chunks[:5]:  # Show first 5
        print(f"  {chunk}")


def demo_sentiment():
    """Demonstrate sentiment analysis capabilities."""
    print("\n" + "=" * 50)
    print("SENTIMENT ANALYSIS DEMO")
    print("=" * 50)
    
    sentiment_analyzer = SentimentAnalyzer()
    
    print("\nPositive text:")
    print(positive_text)
    score = sentiment_analyzer.analyze(positive_text)
    detailed = sentiment_analyzer.analyze_detailed(positive_text)
    print(f"Overall sentiment: {score:.2f} ({detailed['interpretation']})")
    
    print("\nNegative text:")
    print(negative_text)
    score = sentiment_analyzer.analyze(negative_text)
    detailed = sentiment_analyzer.analyze_detailed(negative_text)
    print(f"Overall sentiment: {score:.2f} ({detailed['interpretation']})")
    
    print("\nDetailed sentiment analysis by sentence:")
    for sentence in detailed['sentences']:
        print(f"  Sentence: {sentence['text']}")
        print(f"  Score: {sentence['score']:.2f}\n")


def demo_similarity():
    """Demonstrate text similarity capabilities."""
    print("\n" + "=" * 50)
    print("TEXT SIMILARITY DEMO")
    print("=" * 50)
    
    comparator = TextComparator()
    
    print("\nCompare similar texts:")
    print(f"Text 1: {similar_text1}")
    print(f"Text 2: {similar_text2}")
    
    result = comparator.compare(similar_text1, similar_text2)
    print(f"Similarity score: {result.score:.2f}")
    
    print("\nDetails:")
    for metric, score in result.details.items():
        print(f"  {metric}: {score:.2f}")
    
    print("\nCommon entities:", result.common_entities)
    print("Common nouns:", result.common_nouns)
    
    print("\nCompare different texts:")
    print(f"Text 1: {simple_text}")
    print(f"Text 2: {positive_text}")
    
    result = comparator.compare(simple_text, positive_text)
    print(f"Similarity score: {result.score:.2f}")
    
    print("\nFind similar sentences in longer texts:")
    similar_sentences = comparator.find_similar_sentences(
        complex_text, 
        "Berlin ist die deutsche Hauptstadt. Die Bahn meldet oft Verspätungen.",
        threshold=0.5
    )
    
    for sent1, sent2, score in similar_sentences:
        print(f"  Score: {score:.2f}")
        print(f"  Text 1: \"{sent1}\"")
        print(f"  Text 2: \"{sent2}\"\n")


def demo_visualization():
    """Demonstrate dependency visualization capabilities."""
    print("\n" + "=" * 50)
    print("DEPENDENCY VISUALIZATION DEMO")
    print("=" * 50)
    
    visualizer = DependencyVisualizer()
    
    print("\nGenerating visualization for:")
    print(simple_text)
    
    svg = visualizer.visualize(simple_text)
    
    # Save to file
    output_path = "dependency_tree.svg"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    
    print(f"Visualization saved to {os.path.abspath(output_path)}")
    print("Open this file in a web browser to view the dependency tree.")


if __name__ == "__main__":
    # Run all demos
    demo_analysis()
    demo_sentiment()
    demo_similarity()
    demo_visualization()
    
    print("\nAll demonstrations completed successfully!")
