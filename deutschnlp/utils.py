import os
import re
import json
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('deutschnlp')


def clean_text(text: str) -> str:
    """
    Clean and normalize German text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes
    text = text.replace('„', '"').replace('"', '"')
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def load_json_resource(resource_name: str) -> Optional[Dict]:
    """
    Load a JSON resource file from the data directory.
    
    Args:
        resource_name: Name of the resource file
        
    Returns:
        Dictionary with the resource data or None if not found
    """
    try:
        # Get path to the data directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory at {data_dir}")
        
        # Load the resource
        resource_path = os.path.join(data_dir, resource_name)
        
        if not os.path.exists(resource_path):
            logger.warning(f"Resource not found: {resource_path}")
            return None
        
        with open(resource_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading resource {resource_name}: {e}")
        return None


def save_json_resource(resource_name: str, data: Dict) -> bool:
    """
    Save data to a JSON resource file in the data directory.
    
    Args:
        resource_name: Name of the resource file
        data: Data to save
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Get path to the data directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Save the resource
        resource_path = os.path.join(data_dir, resource_name)
        
        with open(resource_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved resource to {resource_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving resource {resource_name}: {e}")
        return False


def extract_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple rules for German.
    
    Note: This is a simple implementation. For better results,
    use the sentence segmentation from spaCy.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Clean the text first
    text = clean_text(text)
    
    # Split on sentence-ending punctuation followed by whitespace and capital letter
    # This is a simplified approach
    pattern = r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])'
    sentences = re.split(pattern, text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def get_stopwords() -> List[str]:
    """
    Get German stopwords from resources or fallback to a minimal list.
    
    Returns:
        List of German stopwords
    """
    stopwords = load_json_resource('stopwords_de.json')
    
    if stopwords:
        return stopwords
    
    # Fallback to minimal stopwords list
    return [
        'der', 'die', 'das', 'den', 'dem', 'des',
        'ein', 'eine', 'einer', 'eines', 'einem', 'einen',
        'und', 'oder', 'aber', 'wenn', 'weil', 'obwohl',
        'als', 'wie', 'so', 'zu', 'zur', 'zum',
        'in', 'im', 'an', 'am', 'auf', 'bei', 'mit', 'nach',
        'von', 'vor', 'für', 'über', 'unter', 'neben',
        'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr',
    ]


def is_german_text(text: str, threshold: float = 0.5) -> bool:
    """
    Check if the text is likely German based on character frequencies.
    
    This is a simple heuristic that works reasonably well for
    distinguishing German from English and many other languages.
    
    Args:
        text: Text to check
        threshold: Minimum ratio of German indicators to consider it German
        
    Returns:
        True if the text is likely German, False otherwise
    """
    # German-specific character sequences
    german_indicators = ['ä', 'ö', 'ü', 'ß', 'ch', 'sch', 'ei', 'ie', 'eu', 'äu']
    
    # Check for common German words
    common_german_words = [
        'der', 'die', 'das', 'und', 'ist', 'zu', 'von', 'mit',
        'für', 'nicht', 'auch', 'auf', 'eine', 'des', 'sich'
    ]
    
    # Normalize text for word checking
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Count German indicators
    indicator_count = 0
    for indicator in german_indicators:
        indicator_count += text_lower.count(indicator)
    
    # Count common German words
    word_count = 0
    for word in common_german_words:
        word_count += words.count(word)
    
    # Check if there are enough indicators relative to the text length
    text_length = len(text)
    if text_length == 0:
        return False
    
    indicator_ratio = indicator_count / text_length
    word_ratio = word_count / len(words) if words else 0
    
    # Combined score
    german_score = indicator_ratio * 0.7 + word_ratio * 0.3
    
    return german_score > threshold


if __name__ == "__main__":
    # Test the utilities
    example_text = """
    Der Prozess der digitalen
