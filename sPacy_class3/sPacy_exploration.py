from os import EX_NOINPUT
from numpy import positive, unique
import spacy
from pathlib import Path
from collections import Counter


from corpus_loader import corpus_loader

# Opening text corpus 
texts = corpus_loader("/Users/jakobgrohn/Desktop/Cognitive_Science/Cognitive Sceince 7th Semester/NLP/NLP-Assignments/sPacy_class3/data/train_corpus")

# Initialising language model
nlp = spacy.load("en_core_web_sm")

# Grabbing the first text as a test text
test_text = texts[0]

# Create a Doc object on the text'
doc = nlp(test_text)

##### EXERCISE 1 #####

# Function for filtering text and returning lemmas of adjectives, nouns and verbs
def filter_text(doc):
    lemmatized_words = []
    for token in doc:
        if token.pos_ in {'ADJ', 'NOUN', 'VERB'}:
            lemmatized_words.append(token.lemma_)
        
    return lemmatized_words

lemmatized_test_text = filter_text(doc)

##### EXERCISE 2 #####

# Defining function for calculating pos tag ratios 
def pos_ratios(doc):
    pos_counts = Counter([token.pos_ for token in doc])
    ratios = list(zip([j for j in pos_counts.keys()], [i/len(doc) for i in pos_counts.values()]))
    
    return ratios

pos_ratios_test_text = pos_ratios(doc)

##### EXERCISE 3 #####
def get_mdd(doc):
    dep_distances = 0
    for token in doc:
        dd = abs(token.i - token.head.i)
        dep_distances += dd

    mdd = dep_distances/len(doc)

    return mdd