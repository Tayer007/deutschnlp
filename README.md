# German Language Analysis Toolkit (DeutschNLP)

A comprehensive toolkit for analyzing German text using spaCy and BERT models, providing advanced linguistic analysis, sentiment detection, text comparison, and dependency visualization.

## Features

- Complete linguistic analysis (tokens, lemmas, POS, dependencies, NER)
- German text sentiment analysis
- Frequency analysis and word importance
- Text similarity comparison
- Interactive visualization of parsed sentences
- Command-line interface for batch processing
- Web interface for interactive analysis
- Support for both spaCy and BERT models with comparison capability

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deutschnlp.git
cd deutschnlp

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
python -m spacy download de_core_news_md
pip install transformers torch
```

## Example Usage

### Basic Analysis
```python
from deutschnlp.analyzer import GermanAnalyzer

analyzer = GermanAnalyzer()
analysis = analyzer.analyze("Er ist gestern sehr schnell mit dem Fahrrad nach Berlin gefahren.")
print(analysis.tokens)
print(analysis.lemmas)
print(analysis.pos_tags)
print(analysis.named_entities)
print(analysis.dependencies)
```

### Sentiment Analysis
```python
from deutschnlp.sentiment import SentimentAnalyzer
from deutschnlp.bert_models import BERTModels

# Using spaCy-based analyzer
sentiment = SentimentAnalyzer()
score = sentiment.analyze("Das Essen in diesem Restaurant war ausgezeichnet!")
print(f"Sentiment score: {score}")  # Positive score

# Using BERT-based analyzer
bert = BERTModels()
result = bert.analyze_sentiment("Das Essen in diesem Restaurant war ausgezeichnet!")
print(f"BERT sentiment score: {result['overall_score']}")
```

### Text Comparison
```python
from deutschnlp.similarity import TextComparator

comparator = TextComparator()
similarity = comparator.compare(
    "Berlin ist die Hauptstadt von Deutschland.",
    "Die Hauptstadt Deutschlands ist Berlin."
)
print(f"Similarity score: {similarity.score}")
```

## Web Interface

The toolkit includes a web interface for interactive analysis.

```bash
# Start the web interface
cd deutschnlp
python -m deutschnlp.web.app
```

Then open your browser to http://localhost:5000

The web interface allows you to:
- Analyze German text with spaCy or BERT models
- Compare results from both models
- Perform sentiment analysis
- Compare text similarity
- Visualize dependency trees

## Command Line Interface

```bash
# Basic analysis
python -m deutschnlp.cli analyze "Er ist gestern sehr schnell mit dem Fahrrad nach Berlin gefahren."

# Sentiment analysis
python -m deutschnlp.cli sentiment "Das Essen in diesem Restaurant war ausgezeichnet!"

# Compare texts
python -m deutschnlp.cli compare "Berlin ist die Hauptstadt von Deutschland." "Die Hauptstadt Deutschlands ist Berlin."

# Process files
python -m deutschnlp.cli analyze-file input.txt --output analysis.json
```

## BERT Model Integration

DeutschNLP integrates state-of-the-art BERT models for German:

- **Named Entity Recognition**: Using XLM-RoBERTa model fine-tuned for German NER
- **Sentiment Analysis**: Using German sentiment BERT model

These provide higher accuracy compared to traditional models, particularly for complex texts. The web interface allows direct comparison between spaCy and BERT results.

## Project Structure

```
deutschnlp/
├── deutschnlp/
│   ├── analyzer.py         # Core spaCy analysis functionality
│   ├── sentiment.py        # German sentiment analysis
│   ├── visualizer.py       # Dependency visualization
│   ├── similarity.py       # Text comparison functionality
│   ├── utils.py            # Helper functions
│   ├── cli.py              # Command-line interface
│   ├── bert_models.py      # BERT model integration
│   └── web/                # Web interface with Flask
├── examples/               # Example usage scripts
└── tests/                  # Unit tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
