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
from nltk.corpus import stopwords
from gensim.models import LdaModel
from gensim import corpora

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

def topic_modelling(files=['114.txt', '100.txt', '465.txt', '059.txt']):
    """
    perform topic modelling for a given list of files
    """
    ntopics = 5
    articles = []
    fp = path.join(data_dir, '100.txt')
    stop_words = set(stopwords.words('english'))

    with open(fp) as f:
        text = f.read().split()
    articles.append([word for word in text if word not in stop_words])

    dictionary = corpora.Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]  # doc to BOW
    lda = LdaModel(corpus, id2word=dictionary, num_topics=ntopics, passes=100)
    for i in range(ntopics):
        topwords = lda.show_topic(i, topn=5)
        print("Top words in topic {}: {}\n".format(i+1, topwords))


if __name__ == '__main__':
    # print(process_ner(entity='WorldCom'))
    topic_modelling()
