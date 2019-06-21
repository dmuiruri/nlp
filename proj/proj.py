#! /usr/bin/env python
"""
In this script we identify named entities in the corpus and try to
identity the topics coverage in the text for a given entity.

Search for a named entity in the corpus and see which articles are in
the corpus with the named entity.
"""
import spacy
import gensim
import numpy as np
from os import listdir, path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import LdaModel
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    stop_words = set(stopwords.words('english')) | {'Mr', 'The', '-', 'said'}
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

def gen_docs(dpath=data_dir):
    """
    Get documents to create a corpus.
    """
    for article in listdir(dpath):
        fp = path.join(dpath, article)
        if path.isfile(fp):
            with open(fp, 'r') as f:
                yield f.read()

def get_doc_term_mat(docs_obj):
    """
    Generate a term document matrix using a basic word count model
    """
    corpus = docs_obj()
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)
    mat = X.toarray()
    return mat

def get_tf_idf_mat(docs_obj):
    """
    Generate a tf-idf matrix.

    This matrix is generated using log(1 + tf) to get the term
    frequency and N/log(N/df) for the inverse document frequency =>
    log(1 + tf) * log(N/df)
    """
    corpus = docs_obj()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(corpus)
    mat = X.toarray()
    return mat

def get_cosine_similarity(docs_obj, qt=[]):
    """
    Get the cosine similarity between documents or between a query
    term and the documents. If qt empty calculate cosine similarity
    between docs.

    docs_obj: A generator to read text files from local disk
    qt: A given query term
    """ 
    corpus = docs_obj()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(corpus)
    mat = X.toarray()
    if len(qt) > 0:
        print("The length of query term {}".format(len(qt)))
        cs = np.zeros((mat.shape[0], mat.shape[0]))
        qt = tfidf_vectorizer.transform(qt)
        return cosine_similarity(qt, mat) 
    else:
        # Depending on what value we use to determine closely similar
        # documents, we can create a similarity threshold.
        # (cosine_similarity > 0.5)*1
        return cosine_similarity(mat)
    
if __name__ == '__main__':
    """
    Test identification of named entities in the corpus and try to
    determine the topic related to a given entity.
    """
#     low_ner = 'Warner'
#     med_ner = 'WorldCom'
#     high_ner = 'Dollar'
#     file_list = process_ner(entity=high_ner)
#     topic_modelling(files=file_list)
#     print(get_doc_term_mat(gen_docs))
#     print(get_tf_idf_mat(gen_docs))
    print(get_cosine_similarity(gen_docs))
