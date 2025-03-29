import spacy

# Load the German model
nlp = spacy.load("de_core_news_md")

# Example sentence
sentence = "Er ist gestern sehr schnell mit dem Fahrrad nach Berlin gefahren."

# Process the sentence
doc = nlp(sentence)

# Tokenization
print("Tokens:", [token.text for token in doc])

# Stemming/Lemmatization
print("Lemmas:", [token.lemma_ for token in doc])

# Part-of-Speech (POS) Tagging
print("POS Tags:", [(token.text, token.pos_) for token in doc])

# Named Entity Recognition (NER)
print("Named Entities:", [(ent.text, ent.label_) for ent in doc.ents])