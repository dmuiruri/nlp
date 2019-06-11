#! /usr/bin/env python
"""
In this script we determing named entities in the corpus and try to
identoity the topics coverage in the text.
"""
import spacy
import gensim
from os import listdir, path
nlp = spacy.load("en")
data_dir = path.dirname("./business/")

def get_tokens(article):
    """
    Get the tokens from a doc (spacy do
    """
    doc = nlp(article)
    for sent in doc.sents:
        for token in sent:
            print(token.text, token.lemma_, token.pos_, token.tag_)
    
def process_ner():
    """
    Get named entities in an article
    """
    for article in listdir("./business"):
        fp = path.join(data_dir, article)
        if path.isfile(fp):
            try:
                with open(fp, 'r') as f:
                    doc = nlp(f.read())
                print("\n Named Entities in  article {} ".format(article))
                print(doc.ents)
            except FileNotFoundError:
                print("File was not found or could not be openend")

if __name__ == '__main__':
    process_ner()
