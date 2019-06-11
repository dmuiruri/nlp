#! /usr/bin/env python
"""
In this script we determing named entities in the corpus and try to
identity the topics coverage in the text.

Search for a named entity in the corpus and see which articles are in
the corpus with the named entity.
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
    
def process_ner(entity='Nokia'):
    """
    Search for articles with a given named entity in an article

    Returns a list with files containing the named entity.
    """
    articles_list = []
    for article in listdir(data_dir):
        fp = path.join(data_dir, article)
        if path.isfile(fp):
            try:
                with open(fp, 'r') as f:
                    temp = f.read()
                    if temp.lower().find(entity.lower()) >= 0:  # lowercase to find match
                        articles_list.append(article)
                        doc = nlp(temp)
            except FileNotFoundError:
                print("File was not found or could not be openend")
    return articles_list

if __name__ == '__main__':
    print(process_ner(entity='WorldCom'))
