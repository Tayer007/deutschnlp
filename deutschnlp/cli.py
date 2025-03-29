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
        for entity,
