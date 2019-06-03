#! /usr/bin/env python

"""
Vector Space Models and Lexical Semantics
"""
import nltk
import requests, xmltodict, pickle, os
import editdistance
import numpy as np
import pandas as pd
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords, cmudict
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
from gensim import corpora
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('cmudict')

lemmatizer = WordNetLemmatizer()

documents = ['Wage conflict in retail business grows',
             'Higher wages for cafeteria employees',
             'Retailing Wage Dispute Expands',
             'Train Crash Near Petershausen',
             'Five Deaths in Crash of Police Helicopter']

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

tfidf_vectorizer =  TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(documents)

def process_text(text):
    """
    Apply a pipeline of tools to process text.
    
    The function takes a paragraph as input, and splits the paragraph
    into sentences, applies word tokenization, POS tagging and
    lemmatization on all words.
    
    Returns a list containing two higher level lists containing pos
    and lemmatization results.
    """
    res = []
    # res.append([pos_tag(word_tokenize(line)) for line in sent_tokenize(text)])
    res.append([[lemmatizer.lemmatize(word) for word in item] for item in [word_tokenize(line) for line in sent_tokenize(text)]])
    return res

def filter_text(text):
    """
    Filter out stop words.
    """
    stopWords = set(stopwords.words('english'))
    # kept = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    pos = process_text(text)[0]
    sentences = [[word for word in sent if word[0] not in stopWords] for sent in pos] # remove stop words
    # return [[word for word in sent if word[1] in kept] for sent in sentences]
    return sentences

def document_voc():
    doc_voc = []
    for doc in documents:
        doc_voc.append(filter_text(doc.lower())[0])
    return doc_voc

def doc_vocab_matrix():
    doc_vocs = document_voc()
    n_docs = len(doc_vocs)
    vocs = {voc for doc in doc_vocs for voc in doc}
    df = pd.DataFrame(np.zeros((n_docs, len(vocs))), columns=vocs, index=['doc1', 'doc2', 'doc3', 'doc4', 'doc5'])
    for voc in df.columns:
        for doc in range(len(doc_vocs)):
            if voc in doc_vocs[doc]:
                df.loc["doc{}".format(doc + 1), voc] += 1 
    return df

def docs_contents():
    for doc in documents:
        print(doc)

def exercise12():
    """
    Use sklearn to create document matrix.

    Scikit-learn has a class called the CountVectorizer to build
    document-term matrices easily and includes a number of options
    such as removing stopwords, tokenizing, indicating encoding
    """
    return X.toarray()

def exercise21():
    """
    Using the dot product to rank documents.
    """
    queryterm = vectorizer.transform(['retail, wages'])
    rank = np.dot(X.toarray(), queryterm.toarray().transpose())
    return rank

def exercise21part2():
    """
    Normalize the count vectors of the doc-term matrix.
    """
    mat = X.toarray()
    rows = mat.shape[0]
    norm_mat = np.zeros(mat.shape)
    for i in range(len(documents)):
        norm_mat[i] = mat[i]/len(documents[i].split(' '))
    queryterm = vectorizer.transform(['retail, wages'])
    rank = np.dot(norm_mat, queryterm.toarray().transpose())
    return rank

def exercise22():
    """
    Calculate the Tf-IDF matrix
    """
    mat = X_tfidf.toarray()
    queryterm = tfidf_vectorizer.transform(['retail, wages'])
    rank = np.dot(mat, queryterm.toarray().transpose())
    return rank

def exercise31():
    """
    Find similar documents using cosine similarity
    """
    
    comb = list(itertools.combinations([i for i in range(len(documents))], 2)) 
    mat = X_tfidf.toarray()
    cs = np.zeros((5, 5)) # (mat.shape)
    for i in range(len(documents)):
        cs[i] = cosine_similarity(mat[i:i + 1], mat)[0]
    return cs

def exercise32():
    """
    Construct TF-IDF matrix for an unseen document
    """
    new_docs = [
        'Plane crash in Baden-Wuerttemberg',          # Doc 3a
        'The weather'                             # Doc 3b
        ]
    unseendocs = tfidf_vectorizer.transform(new_docs)
    return cosine_similarity(X_tfidf, unseendocs)

def exercise4(filename):
    """
    Topic Modelling
    """
    articles = []
    stopWords = set(stopwords.words('english'))

    stopWords = stopWords|{'</H1>', 'The', 'In', 'For', 'was', 'be', 'will', '<H1>'}  
    text = open(filename, 'r').read().split()
    index_start = list(np.where(np.array(text) == "<DOC")[0])
    for i in range(len(index_start)-1):
        start_art = index_start[i] + 2
        end_art = index_start[i+1]
        article = text[start_art:end_art]
        article = [word for word in article if word not in stopWords]
        articles.append(article)
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(a) for a in articles] # each doc to BOW
    n_topics = 2
    lda = LdaModel(common_corpus, id2word=common_dictionary, num_topics=n_topics, passes=200)
    for k in range(n_topics):
        top_words = lda.show_topic(k, topn=5)
        print("Top words in topic {}: {}\n".format(k+1, top_words))

if __name__ == '__main__':

    # Exercise 1.1
    docs_contents()
    print(">>>>> Exercise 1.1 >>> \n")
    print("{}\n".format(document_voc()))
    print("{}\n".format(doc_vocab_matrix()))
    print("Matrix shape is {}\n".format(doc_vocab_matrix().shape))


    # Excerice 1.2
    print("Document terms matrix shape {}\n".format(exercise12().shape))

    # Exercise 2.1
    print("Unnormalized rank:\n {}\n".format(exercise21()))
    print("Normalized rank:\n {}\n".format(exercise21part2()))

    # Exercise 2.2
    print("TF-IDF to weight words \n{}\n".format(exercise22()))

    # Exercise 3.1
    print("Cosine Similarity of the documents \n{}\n".format(exercise31()))

    # Exercise 3.2
    print("Cosine similarity of unseen docs \n{}\n".format(exercise32()))

    # Exercise 4
    print("Topic modelling \n{}\n".format(exercise4('./de-news.txt')))
