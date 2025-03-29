import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from deutschnlp.analyzer import GermanAnalyzer
from deutschnlp.sentiment import SentimentAnalyzer
from deutschnlp.similarity import TextComparator
from deutschnlp.visualizer import DependencyVisualizer

app = Flask(__name__)

# Initialize analyzers
analyzer = GermanAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
comparator = TextComparator()
visualizer = DependencyVisualizer()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text and return the results."""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Perform analysis
    result = analyzer.analyze(text)
    
    # Convert to dictionary for JSON response
    return jsonify(result.to_dict())

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment and return the results."""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Perform sentiment analysis
    result = sentiment_analyzer.analyze_detailed(text)
    
    return jsonify(result)

@app.route('/compare', methods=['POST'])
def compare_texts():
    """Compare two texts and return the results."""
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    if not text1 or not text2:
        return jsonify({'error': 'Two texts are required'}), 400
    
    # Perform comparison
    result = comparator.compare(text1, text2)
    
    # Convert to dictionary for JSON response
    return jsonify({
        'score': result.score,
        'common_entities': result.common_entities,
        'common_nouns': result.common_nouns,
        'details': result.details
    })

@app.route('/visualize', methods=['POST'])
def visualize_dependencies():
    """Generate dependency visualization for a sentence."""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Generate visualization
    svg_content = visualizer.visualize(text)
    
    return jsonify({'svg': svg_content})

@app.route('/similar_sentences', methods=['POST'])
def find_similar_sentences():
    """Find similar sentences between two texts."""
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    threshold = data.get('threshold', 0.7)
    
    if not text1 or not text2:
        return jsonify({'error': 'Two texts are required'}), 400
    
    # Find similar sentences
    result = comparator.find_similar_sentences(text1, text2, threshold=threshold)
    
    # Convert to list of dictionaries
    similar_sentences = []
    for sent1, sent2, score in result:
        similar_sentences.append({
            'sentence1': sent1,
            'sentence2': sent2,
            'score': score
        })
    
    return jsonify({'similar_sentences': similar_sentences})

if __name__ == '__main__':
    # Check if templates directory exists, if not, create a warning
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    if not os.path.exists(templates_dir):
        print(f"Warning: Templates directory not found at {templates_dir}")
        print("You may need to create the templates directory and add HTML templates.")
    
    if not os.path.exists(static_dir):
        print(f"Warning: Static directory not found at {static_dir}")
        print("You may need to create the static directory for CSS, JS, etc.")
    
    app.run(debug=True)
