import spacy
import numpy as np
from typing import Dict, List, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    """Data container for text similarity results."""
    score: float
    common_entities: List[str]
    common_nouns: List[str]
    details: Dict[str, float]


class TextComparator:
    """
    German text comparison tool.
    
    This class provides functionality for comparing German texts,
    using different similarity metrics and linguistics features.
    """
    
    def __init__(self, model_name="de_core_news_md"):
        """
        Initialize the text comparator.
        
        Args:
            model_name: Name of the spaCy German model to use
        """
        self.nlp = spacy.load(model_name)
        self.tfidf_vectorizer = TfidfVectorizer()
    
    def compare(self, text1: str, text2: str) -> SimilarityResult:
        """
        Compare two German texts and return a detailed similarity analysis.
        
        Args:
            text1: First German text
            text2: Second German text
            
        Returns:
            SimilarityResult object with similarity metrics
        """
        # Parse texts
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        # Calculate semantic similarity using spaCy's word vectors
        semantic_similarity = doc1.similarity(doc2)
        
        # Find common entities
        entities1 = set(ent.text.lower() for ent in doc1.ents)
        entities2 = set(ent.text.lower() for ent in doc2.ents)
        common_entities = list(entities1.intersection(entities2))
        
        # Find common nouns
        nouns1 = set(token.lemma_.lower() for token in doc1 if token.pos_ == "NOUN")
        nouns2 = set(token.lemma_.lower() for token in doc2 if token.pos_ == "NOUN")
        common_nouns = list(nouns1.intersection(nouns2))
        
        # Calculate TF-IDF similarity
        tfidf_similarity = self._calculate_tfidf_similarity(text1, text2)
        
        # Calculate Jaccard similarity for tokens
        tokens1 = set(token.lemma_.lower() for token in doc1 
                      if not token.is_punct and not token.is_stop)
        tokens2 = set(token.lemma_.lower() for token in doc2 
                      if not token.is_punct and not token.is_stop)
        
        jaccard_similarity = len(tokens1.intersection(tokens2)) / max(1, len(tokens1.union(tokens2)))
        
        # Calculate final similarity score (weighted average)
        weights = {
            "semantic": 0.5,
            "tfidf": 0.3,
            "jaccard": 0.2
        }
        
        final_score = (
            weights["semantic"] * semantic_similarity +
            weights["tfidf"] * tfidf_similarity +
            weights["jaccard"] * jaccard_similarity
        )
        
        # Return detailed similarity result
        return SimilarityResult(
            score=final_score,
            common_entities=common_entities,
            common_nouns=common_nouns,
            details={
                "semantic_similarity": semantic_similarity,
                "tfidf_similarity": tfidf_similarity,
                "jaccard_similarity": jaccard_similarity
            }
        )
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate TF-IDF cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            TF-IDF cosine similarity score
        """
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        
        return similarity
    
    def find_similar_sentences(self, base_text: str, comparison_text: str, 
                              threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find similar sentence pairs between two texts.
        
        Args:
            base_text: Base text to compare from
            comparison_text: Text to compare against
            threshold: Minimum similarity score for a match
            
        Returns:
            List of tuples (sentence1, sentence2, similarity_score)
        """
        # Parse texts
        base_doc = self.nlp(base_text)
        comparison_doc = self.nlp(comparison_text)
        
        similar_pairs = []
        
        # Compare each sentence pair
        for sent1 in base_doc.sents:
            for sent2 in comparison_doc.sents:
                similarity = sent1.similarity(sent2)
                
                if similarity >= threshold:
                    similar_pairs.append((sent1.text, sent2.text, similarity))
        
        # Sort by similarity score (descending)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs


if __name__ == "__main__":
    # Example usage
    comparator = TextComparator()
    
    # Test with similar texts
    text1 = "Berlin ist die Hauptstadt von Deutschland."
    text2 = "Die Hauptstadt Deutschlands ist Berlin."
    
    result = comparator.compare(text1, text2)
    print(f"Similarity score: {result.score:.2f}")
    print(f"Common entities: {result.common_entities}")
    print(f"Common nouns: {result.common_nouns}")
    print(f"Details: {result.details}")
    
    # Test with different texts
    text3 = "Der Zug nach München fährt um 14 Uhr ab."
    text4 = "Das Museum öffnet morgen um 10 Uhr."
    
    result2 = comparator.compare(text3, text4)
    print(f"\nSimilarity score: {result2.score:.2f}")
    
    # Test similar sentence finding
    text5 = """
    Deutschland ist ein Land in Mitteleuropa.
    Berlin ist die Hauptstadt und das politische Zentrum.
    Die deutsche Sprache wird von etwa 100 Millionen Menschen gesprochen.
    Deutschland hat eine starke Wirtschaft und ist für seine Autos bekannt.
    """
    
    text6 = """
    Berlin ist die größte Stadt und zugleich Hauptstadt Deutschlands.
    Die deutsche Wirtschaft gehört zu den stärksten der Welt.
    In Deutschland leben etwa 83 Millionen Menschen.
    Deutsche Autos wie BMW und Mercedes sind weltbekannt.
    """
    
    similar_sentences = comparator.find_similar_sentences(text5, text6, threshold=0.5)
    print("\nSimilar sentences:")
    for sent1, sent2, score in similar_sentences:
        print(f"- Score: {score:.2f}")
        print(f"  1: {sent1}")
        print(f"  2: {sent2}")
