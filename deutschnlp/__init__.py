"""
DeutschNLP - A comprehensive toolkit for analyzing German text
"""

__version__ = "0.1.0"

from deutschnlp.analyzer import GermanAnalyzer, GermanAnalysisResult
from deutschnlp.sentiment import SentimentAnalyzer
from deutschnlp.similarity import TextComparator, SimilarityResult
from deutschnlp.visualizer import DependencyVisualizer
self.settings = {
    'width': 1000,        # Increased SVG width
    'height': 600,        # SVG height (will adjust based on content)
    'padding': 30,        # Increased padding around the visualization
    'token_spacing': 100, # Increased horizontal space between tokens
    'level_height': 50,   # Increased vertical space between dependency levels
    'text_size': 14,      # Font size for token text
    'pos_size': 10,       # Font size for POS tags
    'arrow_size': 6,      # Size of dependency arrows
    'colors': {
        'text': '#2A2A2A',       # Token text color
        'pos': '#666666',        # POS tag color
        'arrow': '#888888',      # Arrow color
        'subject': '#C94C4C',    # Subject relations
        'object': '#4C6EC9',     # Object relations
        'root': '#2E8B57',       # Root node - changed to sea green
        'verb': '#2E8B57',       # Verb nodes - highlight in green
        'background': '#FFFFFF', # Background color
    }
}
