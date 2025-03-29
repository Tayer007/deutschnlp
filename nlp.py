import re
import os
import nltk
import torch
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from termcolor import colored

# Create a specific directory for NLTK data in the current working directory
current_dir = os.getcwd()
nltk_data_dir = os.path.join(current_dir, "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK data path to use our specified directory
nltk.data.path.insert(0, nltk_data_dir)  # Add our directory as the first place to look

print(f"NLTK will download data to: {nltk_data_dir}")
print("Downloading required NLTK resources to explicit directory...")

# Download resources to our specific directory
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)

print("NLTK resources downloaded successfully!")

# Define a simple POS tagger in case NLTK's tagger still fails
def simple_pos_tagger(tokens):
    """A very basic POS tagger as fallback"""
    common_nouns = ['man', 'woman', 'dog', 'cat', 'house', 'car', 'book']
    common_verbs = ['is', 'am', 'are', 'was', 'were', 'be', 'run', 'go', 'drive', 'walk', 'eat']
    common_adjs = ['big', 'small', 'fast', 'slow', 'good', 'bad', 'new', 'old']
    common_advs = ['quickly', 'slowly', 'very', 'really', 'too']
    common_preps = ['in', 'on', 'at', 'to', 'from', 'with', 'by']
    
    tags = []
    for token in tokens:
        if token.lower() in common_nouns:
            tags.append((token, 'NN'))
        elif token.lower() in common_verbs:
            tags.append((token, 'VB'))
        elif token.lower() in common_adjs:
            tags.append((token, 'JJ'))
        elif token.lower() in common_advs:
            tags.append((token, 'RB'))
        elif token.lower() in common_preps:
            tags.append((token, 'IN'))
        elif token.lower() == 'the' or token.lower() == 'a' or token.lower() == 'an':
            tags.append((token, 'DT'))
        elif token.lower() in ['he', 'she', 'it', 'they', 'we', 'i', 'you', 'his', 'her', 'their', 'our', 'my']:
            tags.append((token, 'PRP'))
        elif token.lower().endswith('ing'):
            tags.append((token, 'VBG'))
        elif token.lower().endswith('ly'):
            tags.append((token, 'RB'))
        else:
            tags.append((token, 'NN'))  # Default to noun
    
    return tags

class NLPPipeline:
    def __init__(self, text):
        self.raw_text = text
        self.normalized_text = None
        self.tokens = None
        self.stems = None
        self.lemmas = None
        self.pos_tags = None
        
        # Initialize tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # BERT setup
        print("Loading BERT model (this may take a moment)...")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        print("BERT model loaded successfully!")
        
    def text_normalization(self):
        """Step 1: Text normalization - cleaning and standardizing the text"""
        print("\n" + "="*80)
        print(colored("STEP 1: TEXT NORMALIZATION", "green", attrs=["bold"]))
        print("="*80)
        print(colored("Raw text:", "yellow"), self.raw_text)
        
        # Convert to lowercase
        text = self.raw_text.lower()
        
        # Remove special characters and extra whitespaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        self.normalized_text = text
        print(colored("Normalized text:", "yellow"), self.normalized_text)
        return self.normalized_text
    
    def simple_tokenization(self):
        """Step 2: Tokenization - simple whitespace-based tokenization"""
        print("\n" + "="*80)
        print(colored("STEP 2: TOKENIZATION", "green", attrs=["bold"]))
        print("="*80)
        
        # Simple whitespace tokenization instead of using NLTK's word_tokenize
        self.tokens = self.normalized_text.split()
        
        print(colored("Simple tokenized text:", "yellow"), self.tokens)
        
        # BERT tokenization (subword tokenization)
        self.bert_tokens = self.bert_tokenizer.tokenize(self.normalized_text)
        print(colored("BERT subword tokenization:", "yellow"), self.bert_tokens)
        
        return self.tokens
    
    def morphological_normalization(self):
        """Step 3: Morphological normalization - stemming and lemmatization"""
        print("\n" + "="*80)
        print(colored("STEP 3: MORPHOLOGICAL NORMALIZATION", "green", attrs=["bold"]))
        print("="*80)
        
        # Stemming using Porter Stemmer
        self.stems = [self.stemmer.stem(token) for token in self.tokens]
        print(colored("Stemming:", "yellow"))
        for original, stemmed in zip(self.tokens, self.stems):
            print(f"  {original:15} -> {stemmed}")
        
        # Lemmatization using WordNet Lemmatizer
        self.lemmas = [self.lemmatizer.lemmatize(token) for token in self.tokens]
        print(colored("\nLemmatization:", "yellow"))
        for original, lemma in zip(self.tokens, self.lemmas):
            print(f"  {original:15} -> {lemma}")
        
        return {
            'stems': self.stems,
            'lemmas': self.lemmas
        }
    
    def pos_tagging(self):
        """Step 4: Part-of-speech tagging"""
        print("\n" + "="*80)
        print(colored("STEP 4: PART-OF-SPEECH TAGGING", "green", attrs=["bold"]))
        print("="*80)
        
        try:
            # First try NLTK's pos_tag with explicit data path
            self.pos_tags = nltk.pos_tag(self.tokens)
            print(colored("Using NLTK's POS tagger", "green"))
        except Exception as e:
            print(colored(f"NLTK POS tagger failed: {str(e)}", "red"))
            print(colored("Falling back to simple rule-based POS tagger", "yellow"))
            self.pos_tags = simple_pos_tagger(self.tokens)
        
        print(colored("POS Tags:", "yellow"))
        for token, tag in self.pos_tags:
            print(f"  {token:15} -> {tag}")
        
        # Display explanation of common POS tags
        print(colored("\nCommon POS tag meanings:", "yellow"))
        pos_explanations = {
            "NN": "Noun, singular",
            "NNS": "Noun, plural",
            "NNP": "Proper noun, singular",
            "NNPS": "Proper noun, plural",
            "VB": "Verb, base form",
            "VBP": "Verb, present tense, not 3rd person singular",
            "VBZ": "Verb, present tense, 3rd person singular",
            "VBG": "Verb, gerund or present participle",
            "VBD": "Verb, past tense",
            "JJ": "Adjective",
            "RB": "Adverb",
            "IN": "Preposition or subordinating conjunction",
            "DT": "Determiner",
            "PRP": "Personal pronoun"
        }
        
        for tag, explanation in pos_explanations.items():
            print(f"  {tag:5} : {explanation}")
        
        return self.pos_tags
    
    def simple_ner(self):
        """Step 5: Simple Named Entity Recognition using capitalization in original text"""
        print("\n" + "="*80)
        print(colored("STEP 5: SIMPLIFIED NAMED ENTITY RECOGNITION", "green", attrs=["bold"]))
        print("="*80)
        
        # A very simple NER approach using capitalization
        entities = []
        
        # Here, we'll use the original text to check for capitalized words
        original_words = self.raw_text.split()
        for word in original_words:
            # Check if the word starts with uppercase and is not at the start of a sentence
            if word[0].isupper() and word != original_words[0]:
                # Remove any punctuation
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word:  # Make sure we don't add empty strings
                    entities.append((clean_word, 'ENTITY'))
                
        print(colored("Simple NER results:", "yellow"))
        if entities:
            for entity, label in entities:
                print(f"  {entity:15} -> {label}")
        else:
            print("  No named entities detected with simple rules")
            
        return entities
    
    def bert_analysis(self):
        """Additional step: BERT contextual embeddings analysis"""
        print("\n" + "="*80)
        print(colored("ADDITIONAL STEP: BERT CONTEXTUAL EMBEDDINGS ANALYSIS", "green", attrs=["bold"]))
        print("="*80)
        
        # Convert input to BERT input format
        inputs = self.bert_tokenizer(self.normalized_text, return_tensors="pt")
        
        # Get the BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        # Get the embeddings from the last hidden state
        last_hidden_states = outputs.last_hidden_state
        
        print(colored("BERT Input Tokens:", "yellow"), self.bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
        print(colored("Embedding Shape:", "yellow"), last_hidden_states.shape)
        print(colored("This represents contextual embeddings for each token", "yellow"))
        
        # Just to demonstrate, we'll show the embedding vector for the first token
        first_token_embedding = last_hidden_states[0, 1, :].numpy()  # Skip [CLS] token
        print(colored("\nExample - First few dimensions of embedding for first token:", "yellow"))
        print(f"  {first_token_embedding[:5]}")
        print("  [...]")
        
        return {
            'tokens': self.bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            'embeddings': last_hidden_states.numpy()
        }
    
    def run_pipeline(self):
        """Run the complete NLP pipeline"""
        try:
            # Step 1: Text normalization
            self.text_normalization()
            
            # Step 2: Tokenization (simple)
            self.simple_tokenization()
            
            # Step 3: Morphological normalization
            self.morphological_normalization()
            
            # Step 4: POS tagging
            self.pos_tagging()
            
            # Step 5: Simple Named Entity Recognition
            entities = self.simple_ner()
            
            # Additional step: BERT analysis
            self.bert_analysis()
            
            # Summary of the pipeline
            print("\n" + "="*80)
            print(colored("PIPELINE COMPLETE", "green", attrs=["bold"]))
            print("="*80)
            print(colored("\nInputs:", "yellow"), self.raw_text)
            print(colored("Final processed tokens:", "yellow"), self.tokens)
            print(colored("Lemmatized tokens:", "yellow"), self.lemmas)
            
            if entities:
                print(colored("Recognized entities:", "yellow"), ", ".join([e[0] for e in entities]))
            
            print(colored("\nNote:", "yellow"), "BERT provides contextual embeddings that capture semantics based on surrounding words")
            print(colored("      ", "yellow"), "These embeddings can be used for many downstream NLP tasks")
            
        except Exception as e:
            print(colored(f"\nERROR in pipeline: {str(e)}", "red"))
            import traceback
            traceback.print_exc()


# Example usage
if __name__ == "__main__":
    # Example sentence that demonstrates various NLP concepts
    sample_text = "I am going to cook pasta"
    
    try:
        # Create and run the NLP pipeline
        pipeline = NLPPipeline(sample_text)
        pipeline.run_pipeline()

        # You can also run it on custom text
        print("\n\nWould you like to try with your own text? Enter it below (or press Enter to exit):")
        custom_text = input("> ")
        
        if custom_text:
            custom_pipeline = NLPPipeline(custom_text)
            custom_pipeline.run_pipeline()
            
    except Exception as e:
        print(f"\nFailed to run the NLP pipeline: {str(e)}")
        import traceback
        traceback.print_exc()