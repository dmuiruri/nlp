#! /usr/bin/env python
"""
In this script we identify named entities in the corpus and try to
identity the topics coverage in the text for a given entity.

Search for a named entity in the corpus and see which articles are in
the corpus with the named entity.
"""
import spacy
import gensim
from os import listdir, path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import LdaModel
from gensim import corpora

nlp = spacy.load("en")
data_dir = path.dirname("./business/")

def get_tokens(article):
    """
    Get the tokens from a doc
    """
    doc = nlp(article)
    for sent in doc.sents:
        for token in sent:
            print(token.text, token.lemma_, token.pos_, token.tag_)
    
def process_ner(entity='Nokia'):
    """
    Search for articles with a given named entity in an article

    Returns a list with names of the files that contain the given
    named entity.
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
    found = len(articles_list)
    all_docs = len(listdir(data_dir))
    hit = found/all_docs
    print("Doc hit(%): {:.3f}, {} docs of {}".format(hit, found, all_docs))
    return articles_list

def topic_modelling(files=['114.txt', '100.txt', '465.txt', '059.txt']):
    """
    perform topic modelling for a given list of files
    """
    ntopics = 2
    articles = []
    stop_words = set(stopwords.words('english')) | {'Mr', 'The'}
    for f in files:
        fp = path.join(data_dir, f)
        with open(fp) as f:
            text = f.read().split()  # word_tokenize(
        articles.append([word for word in text if word not in stop_words])

    dictionary = corpora.Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]  # doc to BOW
    lda = LdaModel(corpus, id2word=dictionary, num_topics=ntopics, passes=500)
    for i in range(ntopics):
        topwords = lda.show_topic(i, topn=5)
        print("Top words in topic {}: {}\n".format(i+1, topwords))


if __name__ == '__main__':
    """
    Test identification of named entities in the corpus and try to
    determine the topic related to a given entity.
    """
    # print(process_ner(entity='WorldCom'))
    file_list = process_ner(entity='WorldCom')
    topic_modelling(files=file_list)
