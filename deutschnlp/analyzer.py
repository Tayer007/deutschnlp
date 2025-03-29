import spacy
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os
import json


@dataclass
class GermanAnalysisResult:
    """Data container for German text analysis results."""
    raw_text: str
    tokens: List[str]
    lemmas: List[str]
    pos_tags: List[Tuple[str, str]]
    dependencies: List[Tuple[str, str, str]]
    named_entities: List[Tuple[str, str]]
    noun_chunks: List[str]
    sentences: List[str]
    word_frequencies: Dict[str, int]
    
    def to_dict(self):
        """Convert analysis results to dictionary for JSON serialization."""
        return {
            "raw_text": self.raw_text,
            "tokens": self.tokens,
            "lemmas": self.lemmas,
            "pos_tags": self.pos_tags,
            "dependencies": self.dependencies,
            "named_entities": self.named_entities,
            "noun_chunks": self.noun_chunks,
            "sentences": self.sentences,
            "word_frequencies": self.word_frequencies
        }
    
    def to_json(self, pretty=True):
        """Convert analysis results to JSON string."""
        if pretty:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        return json.dumps(self.to_dict(), ensure_ascii=False)


class GermanAnalyzer:
    """
    A comprehensive German text analyzer using spaCy.
    
    This class provides expanded functionality for analyzing German text,
    including tokenization, lemmatization, part-of-speech tagging,
    dependency parsing, named entity recognition, and more.
    """
    
    def __init__(self, model_name="de_core_news_md", load_proofread=True):
        """
        Initialize the German text analyzer.
        
        Args:
            model_name: Name of the spaCy German model to use
            load_proofread: Whether to load the German proofread component
        """
        self.nlp = spacy.load(model_name)
        
        # Extend pipeline with additional components if needed
        if load_proofread and not "proofread" in self.nlp.pipe_names:
            try:
                from spacy_german_proofread import GermanProofRead
                self.nlp.add_pipe("german_proofread")
                self.has_proofread = True
            except ImportError:
                print("German proofread component not available. Install with: pip install spacy-german-proofread")
                self.has_proofread = False
        else:
            self.has_proofread = False
    
    def analyze(self, text: str) -> GermanAnalysisResult:
        """
        Perform comprehensive linguistic analysis on German text.
        
        Args:
            text: German text to analyze
            
        Returns:
            GermanAnalysisResult object containing analysis results
        """
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract basic linguistic information
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        # Extract dependency relationships
        dependencies = [
            (token.text, token.dep_, token.head.text) 
            for token in doc
        ]
        
        # Named entity recognition
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract noun chunks (noun phrases)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        
        # Split text into sentences
        sentences = [sent.text for sent in doc.sents]
        
        # Calculate word frequencies (excluding punctuation and stop words)
        words = [
            token.lemma_.lower() for token in doc 
            if not token.is_punct and not token.is_stop
        ]
        word_frequencies = dict(Counter(words).most_common())
        
        # Create and return the analysis result
        return GermanAnalysisResult(
            raw_text=text,
            tokens=tokens,
            lemmas=lemmas,
            pos_tags=pos_tags,
            dependencies=dependencies,
            named_entities=named_entities,
            noun_chunks=noun_chunks,
            sentences=sentences,
            word_frequencies=word_frequencies
        )
    
    def proofread(self, text: str) -> Optional[List[Dict]]:
        """
        Perform proofreading on German text if the proofread component is available.
        
        Args:
            text: German text to proofread
            
        Returns:
            List of correction suggestions or None if proofreading is not available
        """
        if not self.has_proofread:
            return None
        
        doc = self.nlp(text)
        if hasattr(doc._, "suggestions"):
            return doc._.suggestions
        return None
    
    def analyze_file(self, filepath: str) -> GermanAnalysisResult:
        """
        Analyze German text from a file.
        
        Args:
            filepath: Path to the text file
            
        Returns:
            GermanAnalysisResult object containing analysis results
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return self.analyze(text)


if __name__ == "__main__":
    # Example usage
    analyzer = GermanAnalyzer()
    
    # Simple example
    text = "Er ist gestern sehr schnell mit dem Fahrrad nach Berlin gefahren."
    result = analyzer.analyze(text)
    
    print("Text:", result.raw_text)
    print("Tokens:", result.tokens)
    print("Lemmas:", result.lemmas)
    print("POS Tags:", result.pos_tags)
    print("Named Entities:", result.named_entities)
    print("Noun Chunks:", result.noun_chunks)
    print("Sentences:", result.sentences)
    print("Word Frequencies:", result.word_frequencies)
    
    # More complex example
    complex_text = """
    Die Bundesrepublik Deutschland ist ein demokratischer und sozialer Bundesstaat. 
    Berlin ist die Hauptstadt der Bundesrepublik Deutschland.
    Die Deutsche Bahn hat gestern massive Verspätungen gemeldet.
    Angela Merkel war von 2005 bis 2021 Bundeskanzlerin.
    """
    
    complex_result = analyzer.analyze(complex_text)
    print("\nNamed Entities in complex example:")
    for entity, label in complex_result.named_entities:
        print(f"  - {entity}: {label}")
