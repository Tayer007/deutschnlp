# German Language Analysis Toolkit (DeutschNLP)

A comprehensive toolkit for analyzing German text using spaCy, providing advanced linguistic analysis, sentiment detection, and text comparison features.

## Features

- Complete linguistic analysis (tokens, lemmas, POS, dependencies, NER)
- German text sentiment analysis
- Frequency analysis and word importance
- Text similarity comparison
- Interactive visualization of parsed sentences
- Command-line interface for batch processing
- Web interface for interactive analysis

## Project Structure

```
deutschnlp/
├── README.md
├── requirements.txt
├── setup.py
├── deutschnlp/
│   ├── __init__.py
│   ├── analyzer.py         # Core analysis functionality
│   ├── sentiment.py        # German sentiment analysis
│   ├── visualizer.py       # Dependency visualization
│   ├── similarity.py       # Text comparison functionality
│   ├── utils.py            # Helper functions
│   └── cli.py              # Command-line interface
├── web/
│   ├── app.py              # Flask web application
│   ├── templates/
│   └── static/
└── examples/
    ├── basic_analysis.py
    ├── sentiment_demo.py
    ├── similarity_demo.py
    └── visualization_demo.py
```

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

sentiment = SentimentAnalyzer()
score = sentiment.analyze("Das Essen in diesem Restaurant war ausgezeichnet!")
print(f"Sentiment score: {score}")  # Positive score
```

### Text Comparison
```python
from deutschnlp.similarity import TextComparator

comparator = TextComparator()
similarity = comparator.compare(
    "Berlin ist die Hauptstadt von Deutschland.",
    "Die Hauptstadt Deutschlands ist Berlin."
)
print(f"Similarity score: {similarity}")  # High similarity score
```

## Web Interface

The toolkit includes a web interface for interactive analysis.

```bash
# Start the web interface
cd web
python app.py
```

Then open your browser to http://localhost:5000

## Command Line Interface

```bash
# Basic analysis
deutschnlp analyze "Er ist gestern sehr schnell mit dem Fahrrad nach Berlin gefahren."

# Sentiment analysis
deutschnlp sentiment "Das Essen in diesem Restaurant war ausgezeichnet!"

# Process files
deutschnlp analyze-file input.txt --output analysis.json
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
