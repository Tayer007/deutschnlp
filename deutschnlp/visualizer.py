import spacy
from typing import Optional, Dict, List, Tuple
import xml.etree.ElementTree as ET
import math
import re


class DependencyVisualizer:
    """
    Visualizer for German dependency trees.
    
    This class creates SVG visualizations of dependency trees for German sentences.
    """
    
    def __init__(self, model_name="de_core_news_md"):
        """
        Initialize the dependency visualizer.
        
        Args:
            model_name: Name of the spaCy German model to use
        """
        self.nlp = spacy.load(model_name)
        
        # Visualization settings
        self.settings = {
            'width': 800,         # SVG width
            'height': 600,        # SVG height (will adjust based on content)
            'padding': 20,        # Padding around the visualization
            'token_spacing': 80,  # Horizontal space between tokens
            'level_height': 40,   # Vertical space between dependency levels
            'text_size': 14,      # Font size for token text
            'pos_size': 10,       # Font size for POS tags
            'arrow_size': 6,      # Size of dependency arrows
            'colors': {
                'text': '#2A2A2A',       # Token text color
                'pos': '#666666',        # POS tag color
                'arrow': '#888888',      # Arrow color
                'subject': '#C94C4C',    # Subject relations
                'object': '#4C6EC9',     # Object relations
                'root': '#50A050',       # Root node
                'background': '#FFFFFF', # Background color
            }
        }
    
    def visualize(self, text):
    """
    Generate an SVG visualization of the dependency tree for a sentence.
    
    Args:
        text: German text to visualize (preferably a single sentence)
        
    Returns:
        SVG markup as a string
    """
    # Parse text with spaCy
    doc = self.nlp(text)
    
    # Get the first sentence if multiple
    if len(list(doc.sents)) > 1:
        sentence = next(doc.sents)
    else:
        sentence = doc
    
    # Adjust default width based on sentence length
    self.settings['width'] = max(800, len(sentence) * 120)
    
    # Create SVG element
    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': '100%',
        'height': str(self.settings['height']),
        'viewBox': f"0 0 {self.settings['width']} {self.settings['height']}",
        'style': 'overflow: visible;'
    })
    
    # Add background
    ET.SubElement(svg, 'rect', {
        'width': '100%',
        'height': '100%',
        'fill': self.settings['colors']['background'],
    })
    
    # Get tokens and their positions
    tokens, token_positions = self._layout_tokens(sentence)
    
    # Calculate actual height needed
    max_y = max([pos['y'] for pos in token_positions.values()]) + 150
    svg.set('height', str(max_y))
    svg.set('viewBox', f"0 0 {self.settings['width']} {max_y}")
    
    # Add dependencies (arrows)
    dependencies_group = ET.SubElement(svg, 'g', {'class': 'dependencies'})
    self._add_dependency_arrows(dependencies_group, sentence, token_positions)
    
    # Add tokens (text and POS tags)
    tokens_group = ET.SubElement(svg, 'g', {'class': 'tokens'})
    self._add_tokens(tokens_group, sentence, token_positions)
    
    # Convert to string
    return ET.tostring(svg, encoding='unicode')
    
   def _layout_tokens(self, sentence):
    """
    Calculate layout positions for tokens with improved layout for complex sentences.
    
    Args:
        sentence: spaCy sentence
        
    Returns:
        Tuple of (tokens list, token positions dictionary)
    """
    tokens = list(sentence)
    
    # Calculate token positions
    positions = {}
    token_width = self.settings['token_spacing']
    base_x = self.settings['padding']
    base_y = 200  # Start tokens a bit higher
    
    # First pass - assign horizontal positions
    for i, token in enumerate(tokens):
        positions[token.i] = {
            'x': base_x + i * token_width,
            'y': base_y,
            'width': token_width
        }
    
    # Adjust vertical positions based on dependency tree
    self._adjust_vertical_positions(tokens, positions)
    
    # Calculate the overall width needed
    max_x = max([pos['x'] + pos['width'] for pos in positions.values()]) + self.settings['padding']
    
    # Adjust SVG width if needed
    if max_x > self.settings['width']:
        self.settings['width'] = max_x
    
    return tokens, positions
    
    def _adjust_vertical_positions(self, tokens, positions):
        """
        Adjust the vertical positions of tokens based on dependency relations.
        
        Args:
            tokens: List of spaCy tokens
            positions: Dictionary of token positions
        """
        # Find the root
        root = None
        for token in tokens:
            if token.dep_ == "ROOT":
                root = token
                break
        
        if not root:
            return
        
        # Set levels based on distance from root
        visited = set()
        
        def set_levels(token, level=0):
            """Recursively set levels for tokens."""
            visited.add(token.i)
            
            # Adjust y position based on level
            positions[token.i]['y'] = (
                positions[token.i]['y'] - level * self.settings['level_height']
            )
            
            # Process children
            for child in token.children:
                if child.i in visited:
                    continue
                    
                if child.i < len(tokens):  # Only process children within this sentence
                    set_levels(child, level + 1)
        
        # Start from root with level 0
        set_levels(root)
    
    def _add_dependency_arrows(self, group, sentence, positions):
        """
        Add dependency arrows to the SVG.
        
        Args:
            group: SVG group element to add arrows to
            sentence: spaCy sentence
            positions: Dictionary of token positions
        """
        for token in sentence:
            if token.dep_ == "ROOT":
                continue
                
            # Get positions
            start_x = positions[token.head.i]['x'] + positions[token.head.i]['width'] / 2
            start_y = positions[token.head.i]['y']
            end_x = positions[token.i]['x'] + positions[token.i]['width'] / 2
            end_y = positions[token.i]['y']
            
            # Calculate control points for curved arrows
            # Vertical distance proportional to horizontal distance
            distance = abs(end_x - start_x)
            mid_y = min(start_y, end_y) - (distance / 10 + 10)
            
            # Choose color based on dependency type
            color = self.settings['colors']['arrow']
            if re.match(r'(subj|nsubj)', token.dep_):
                color = self.settings['colors']['subject']
            elif re.match(r'(obj|dobj|iobj)', token.dep_):
                color = self.settings['colors']['object']
            
            # Create curved path for the arrow
            path = ET.SubElement(group, 'path', {
                'd': f"M {start_x},{start_y} Q {(start_x + end_x) / 2},{mid_y} {end_x},{end_y}",
                'fill': 'none',
                'stroke': color,
                'stroke-width': '1.5',
                'marker-end': 'url(#arrowhead)'
            })
            
            # Add dependency label
            label_x = (start_x + end_x) / 2
            label_y = mid_y - 5
            
            label = ET.SubElement(group, 'text', {
                'x': str(label_x),
                'y': str(label_y),
                'text-anchor': 'middle',
                'font-size': str(self.settings['pos_size']),
                'fill': color
            })
            label.text = token.dep_
    
    def _add_tokens(self, group, sentence, positions):
    """
    Add token texts and POS tags to the SVG with improved styling.
    
    Args:
        group: SVG group element to add tokens to
        sentence: spaCy sentence
        positions: Dictionary of token positions
    """
    # First add markers definition
    defs = ET.SubElement(group, 'defs')
    marker = ET.SubElement(defs, 'marker', {
        'id': 'arrowhead',
        'markerWidth': str(self.settings['arrow_size']),
        'markerHeight': str(self.settings['arrow_size']),
        'refX': '0',
        'refY': str(self.settings['arrow_size'] / 2),
        'orient': 'auto'
    })
    
    arrow_path = ET.SubElement(marker, 'polygon', {
        'points': f"0 0, {self.settings['arrow_size']} {self.settings['arrow_size'] / 2}, 0 {self.settings['arrow_size']}",
        'fill': self.settings['colors']['arrow']
    })
    
    # Add tokens
    for token in sentence:
        x = positions[token.i]['x']
        y = positions[token.i]['y']
        
        # Token text
        text_color = self.settings['colors']['text']
        if token.dep_ == "ROOT":
            text_color = self.settings['colors']['root']
        elif token.pos_ == "VERB":
            text_color = self.settings['colors']['verb']
                
        text_elem = ET.SubElement(group, 'text', {
            'x': str(x),
            'y': str(y),
            'text-anchor': 'middle',
            'font-size': str(self.settings['text_size']),
            'font-weight': 'bold' if token.dep_ == "ROOT" or token.pos_ == "VERB" else 'normal',
            'fill': text_color
        })
        text_elem.text = token.text
        
        # POS tag
        pos_elem = ET.SubElement(group, 'text', {
            'x': str(x),
            'y': str(y + 20),
            'text-anchor': 'middle',
            'font-size': str(self.settings['pos_size']),
            'fill': self.settings['colors']['pos']
        })
        pos_elem.text = token.pos_


if __name__ == "__main__":
    # Example usage
    visualizer = DependencyVisualizer()
    
    # Generate visualization for a simple sentence
    sample_text = "Der schnelle braune Fuchs springt über den faulen Hund."
    svg = visualizer.visualize(sample_text)
    
    # Save to file
    with open("dependency_tree.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    
    print("Visualization saved to dependency_tree.svg")
    
    # More complex example
    complex_text = "Die Studenten, die fleißig studieren, werden morgen eine schwierige Prüfung schreiben."
    svg = visualizer.visualize(complex_text)
    
    # Save to file
    with open("complex_dependency_tree.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    
    print("Complex visualization saved to complex_dependency_tree.svg")
