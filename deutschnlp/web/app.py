import os
import sys
# Add parent directory to path so deutschnlp can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from deutschnlp.analyzer import GermanAnalyzer
from deutschnlp.sentiment import SentimentAnalyzer
from deutschnlp.similarity import TextComparator
from deutschnlp.visualizer import DependencyVisualizer
from deutschnlp.bert_models import BERTModels  # Import the new BERT models class

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Initialize analyzers
analyzer = GermanAnalyzer()
sentiment_analyzer = SentimentAnalyzer()
comparator = TextComparator()
visualizer = DependencyVisualizer()

# Initialize BERT models (lazy loading - will only initialize when needed)
bert_models = None

def get_bert_models():
    """Lazy initialization of BERT models"""
    global bert_models
    if bert_models is None:
        bert_models = BERTModels(use_gpu=False)
    return bert_models

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text and return the results."""
    data = request.json
    text = data.get('text', '')
    model = data.get('model', 'spacy')  # Default to spaCy if not specified
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Perform analysis based on selected model
    if model == 'spacy':
        # Use spaCy analyzer
        result = analyzer.analyze(text)
        # Convert to dictionary for JSON response
        return jsonify(result.to_dict())
    elif model == 'bert':
        # Use BERT models
        bert = get_bert_models()
        
        # Perform analysis with spaCy for basic features
        basic_result = analyzer.analyze(text)
        result_dict = basic_result.to_dict()
        
        # Replace named entities with BERT results
        bert_entities = bert.analyze_entities(text)
        result_dict['named_entities'] = bert_entities
        
        return jsonify(result_dict)
    else:
        return jsonify({'error': 'Invalid model specified'}), 400

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment and return the results."""
    data = request.json
    text = data.get('text', '')
    model = data.get('model', 'spacy')  # Default to spaCy if not specified
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Perform sentiment analysis based on selected model
    if model == 'spacy':
        # Use spaCy sentiment analyzer
        result = sentiment_analyzer.analyze_detailed(text)
        return jsonify(result)
    elif model == 'bert':
        # Use BERT sentiment analyzer
        bert = get_bert_models()
        result = bert.analyze_sentiment(text)
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid model specified'}), 400

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
