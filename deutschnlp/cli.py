import argparse
import sys
import json
import os
from typing import List, Dict, Any

# Add parent directory to path for running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from deutschnlp.analyzer import GermanAnalyzer
from deutschnlp.sentiment import SentimentAnalyzer
from deutschnlp.similarity import TextComparator


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='DeutschNLP - German Language Analysis Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  deutschnlp analyze "Er ist gestern sehr schnell mit dem Fahrrad nach Berlin gefahren."
  
  # Sentiment analysis
  deutschnlp sentiment "Das Essen in diesem Restaurant war ausgezeichnet!"
  
  # Compare two texts
  deutschnlp compare "Berlin ist die Hauptstadt von Deutschland." "Die Hauptstadt Deutschlands ist Berlin."
  
  # Process a file
  deutschnlp analyze-file input.txt --output analysis.json
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Analyze text command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze German text')
    analyze_parser.add_argument('text', help='German text to analyze')
    analyze_parser.add_argument('--format', choices=['json', 'pretty'], default='pretty',
                              help='Output format (default: pretty)')
    
    # Analyze file command
    analyze_file_parser = subparsers.add_parser('analyze-file', help='Analyze German text from file')
    analyze_file_parser.add_argument('file', help='Path to text file')
    analyze_file_parser.add_argument('--output', help='Output file path (optional)')
    analyze_file_parser.add_argument('--format', choices=['json', 'pretty'], default='pretty',
                                   help='Output format (default: pretty)')
    
    # Sentiment analysis command
    sentiment_parser = subparsers.add_parser('sentiment', help='Analyze sentiment of German text')
    sentiment_parser.add_argument('text', help='German text to analyze')
    sentiment_parser.add_argument('--detailed', action='store_true', 
                                help='Show detailed sentiment analysis')
    
    # Compare texts command
    compare_parser = subparsers.add_parser('compare', help='Compare two German texts')
    compare_parser.add_argument('text1', help='First German text')
    compare_parser.add_argument('text2', help='Second German text')
    compare_parser.add_argument('--find-similar', action='store_true',
                              help='Find similar sentences')
    compare_parser.add_argument('--threshold', type=float, default=0.7,
                              help='Similarity threshold (default: 0.7)')
    
    return parser


def print_analysis_result(result, format_type='pretty'):
    """Pretty print analysis results."""
    if format_type == 'json':
        print(result.to_json())
        return
    
    print(f"Text: {result.raw_text}")
    print("\nTokens:")
    print(", ".join(result.tokens))
    
    print("\nLemmas:")
    print(", ".join(result.lemmas))
    
    print("\nPOS Tags:")
    for token, pos in result.pos_tags:
        print(f"  {token}: {pos}")
    
    print("\nNamed Entities:")
    if result.named_entities:
        for entity, label in result.named_entities:
            print(f"  {entity}: {label}")
    else:
        print("  None found")
    
    print("\nNoun Chunks:")
    if result.noun_chunks:
        for chunk in result.noun_chunks:
            print(f"  {chunk}")
    else:
        print("  None found")
    
    print("\nWord Frequencies:")
    for word, count in list(result.word_frequencies.items())[:10]:  # Show top 10
        print(f"  {word}: {count}")


def main():
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize appropriate analyzers based on command
    if args.command in ['analyze', 'analyze-file']:
        analyzer = GermanAnalyzer()
        
        if args.command == 'analyze':
            result = analyzer.analyze(args.text)
            print_analysis_result(result, args.format)
        else:  # analyze-file
            result = analyzer.analyze_file(args.file)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result.to_json(pretty=True))
                print(f"Analysis saved to {args.output}")
            else:
                print_analysis_result(result, args.format)
    
    elif args.command == 'sentiment':
        sentiment_analyzer = SentimentAnalyzer()
        
        if args.detailed:
            result = sentiment_analyzer.analyze_detailed(args.text)
            print(f"Text: {args.text}")
            print(f"Overall sentiment: {result['overall_score']:.2f} ({result['interpretation']})")
            print("\nSentiment by sentence:")
            for sentence in result['sentences']:
                print(f"  \"{sentence['text']}\"")
                print(f"  Score: {sentence['score']:.2f}")
                print()
        else:
            score = sentiment_analyzer.analyze(args.text)
            interpretation = sentiment_analyzer._interpret_score(score)
            print(f"Text: {args.text}")
            print(f"Sentiment: {score:.2f} ({interpretation})")
    
    elif args.command == 'compare':
        comparator = TextComparator()
        
        result = comparator.compare(args.text1, args.text2)
        print(f"Text 1: {args.text1}")
        print(f"Text 2: {args.text2}")
        print(f"Similarity score: {result.score:.2f}")
        
        print("\nDetails:")
        for metric, score in result.details.items():
            print(f"  {metric}: {score:.2f}")
        
        if result.common_entities:
            print("\nCommon entities:")
            for entity in result.common_entities:
                print(f"  {entity}")
        
        if result.common_nouns:
            print("\nCommon nouns:")
            for noun in result.common_nouns:
                print(f"  {noun}")
        
        if args.find_similar:
            similar_sentences = comparator.find_similar_sentences(
                args.text1, args.text2, threshold=args.threshold
            )
            
            if similar_sentences:
                print("\nSimilar sentences:")
                for sent1, sent2, score in similar_sentences:
                    print(f"  Score: {score:.2f}")
                    print(f"  Text 1: \"{sent1}\"")
                    print(f"  Text 2: \"{sent2}\"")
                    print()
            else:
                print("\nNo similar sentences found above threshold.")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
