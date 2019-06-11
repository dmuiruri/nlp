#! /usr/bin/env python

import random
from nltk.corpus import treebank
from ass5utils import split_corpus, tagset
from nltk.metrics import ConfusionMatrix
from nltk.tag.hmm import HiddenMarkovModelTagger

total_docs = 100
total_relevant = 10

retrieved = ['R', 'N', 'N', 'R', 'R', 'N', 'N', 'N',
             'R', 'N', 'R', 'N', 'N', 'R', 'R']

num_R = retrieved.count('R')
num_N = retrieved.count('N')

s1_t = ['So', 'far', 'Mr.', 'Hahn', 'is', 'trying', 'to', 'entice', 'Nekoosa', 'into', 'negotiating', 'a', 'friendly',
      'surrender', 'while', 'talking', 'tough']
s2_t = ['Despite', 'the', 'economic', 'slowdown', 'there', 'are', 'few', 'clear', 'signs', 'that', 'growth', 'is',
      'coming', 'to', 'a', 'halt']
s3_t = ['The', 'real', 'battle', 'is', 'over', 'who', 'will', 'control', 'that', 'market', 'and', 'reap',
      'its', 'huge', 'rewards']

tags1 = ['IN', 'RB', 'NNP', 'NNP', 'VBZ', 'VBG', 'TO', 'VB', 'NNP', 'IN', 'VBG', 'DT', 'JJ', 'NN', 'IN', 'VBG', 'JJ']
tags2 = ['IN', 'DT', 'JJ', 'NN', 'EX', 'VBP', 'JJ', 'JJ', 'NNS', 'IN', 'NN', 'VBZ', 'VBG', 'TO', 'DT', 'NN']
tags3 = ['DT', 'JJ', 'NN', 'VBZ', 'IN', 'WP', 'MD', 'VB', 'DT', 'NN', 'CC', 'VB', 'PRP$', 'JJ', 'NNS']

training_sents, test_sents = split_corpus(treebank, 0.8)
test_tokens = [token[0] for sent in test_sents for token in sent]
test_tags = [tag[1] for sent in test_sents for tag in sent]

def predicted_tags(training_sentences):
    """
    Predict tags using HMM
    """
    hmm_tagger = HiddenMarkovModelTagger.train(training_sents)
    predicted_tags = [tag[1] for tag in hmm_tagger.tag(test_tokens)] # predict a tag given a token
    return predicted_tags

def evaluation(ref_tags, pred_tags, model='HMM'): #test_tags
    """
    Evaluate tagging model performance
    """
    correct_tags = 0
    total_tags = 0 # matrix

    # create confusion matrix to evaluate results
    cm = ConfusionMatrix(reference=ref_tags, test=pred_tags, sort_by_count=True)
    cm_tag_dict = {item.split()[0].split(':')[0]: item.split()[1] for item in cm.key().split('\n')[1:-1]}
    cm_tag_list = [item.split()[1] for item in cm.key().split('\n')[1:-1]]
    for tag_ref in cm_tag_list:
        for tag_test in cm_tag_list:
            if tag_ref == tag_test:
                correct_tags += cm[tag_ref, tag_test]
            total_tags += cm[tag_ref, tag_test]
    accuracy = correct_tags/total_tags
    print("Overall accuracy of the {} model: {:.3f}".format(model, accuracy))

    # Calculating Precision, Recall and F-Score for the 'NN' tag
    true_p = cm['NN', 'NN']
    false_p = sum([cm[item, 'NN'] for item in cm_tag_list if item != 'NN'])
    all_true_nn = sum([cm['NN', item] for item in cm_tag_list])  # reference
    precision = true_p/(true_p + false_p)
    recall = true_p/all_true_nn
    f_score = 2 * precision * recall /(precision + recall)
    print("Precision for 'NN' tag: {:.2f}".format(precision))
    print("Recall score for 'NN' tag {:.2f}".format(recall))
    print("F-score for 'NN' with Beta=1: {:.2f}".format(f_score))

def ex21():
    """
    Calculate evaluation metrics (Precision, Recall, Fscore) for the
    'NN' tag.
    """
    pred_tags = predicted_tags(training_sents)
    evaluation(test_tags, pred_tags, model="HMM")

def random_tagger(tagset, tokens):
    """
    Randomly generate a POS tag for given tokens.
    Returns a list of tuples (token, tag)
    """
    return [(token, random.choice(tagset)) for token in tokens]

def majority_tagger(train_sents, tokens):
    """
    Find most common tag in training sentences and tag each token with
    this tag.
    train_sents: sentences to be used to generate the most popular tag ("training")
    tokens: tokens to be tagged
    """
    common_tag_count = 0
    common_tag_name = ''
    [('Today', 'NN'), ('is', 'VBZ'), ('a', 'DT'), ('good', 'JJ'), ('day', 'NN')]
    [token[1] for sent in train_sents for token in sent]

    all_tags = [token[1] for sent in train_sents for token in sent]
    tags = set(all_tags)
    for tag in tags:
        tag_count = all_tags.count(tag)
        if tag_count > common_tag_count:
            common_tag_count = tag_count
            common_tag_name = tag

    return [(token, common_tag_name) for token in tokens]

def ex22():
    """
    Evaluate a random tagger and a majority tagger (baseline models to
    compare against an HMM model).
    """
    # Evaluate the random tagger
    print("Evaluating a random tagger: ")
    pred_tags = [item[1] for item in random_tagger(tagset, test_tokens)]
    evaluation(test_tags, pred_tags, model='Random Tagger')

    # Evaluate the majority tagger
    print("\nEvaluating a majority tagger: ")
    pred_tags = [item[1] for item in majority_tagger(training_sents, test_tokens)]
    evaluation(test_tags, pred_tags, model='Majority Tagger')

def ex23():
    """
    Evaluate a HMM language model.
    """
    hmm_tagger = HiddenMarkovModelTagger.train(training_sents)

    test_set = [(token, None) for token in test_tokens]
    log_prob = hmm_tagger.log_probability(test_set)
    no_of_tokens = len(test_set)

    # Calculate perplexity
    perplexity = 2 ** (-log_prob/no_of_tokens)
    print("Perplexity of HMM: {:.2f}".format(perplexity))

def ex31():
    """
    Annotate given sentences (manual)
    """
    s1 = ['DT', 'NN', 'NNP', 'NN', 'VB', 'VB', 'TO', 'VB', 'NN', 'DT', 'NN', 'DT', 'JJ',
          'VB', 'IN', 'VB', 'JJ']
    s2 = ['IN', 'DT', 'NN', 'NN', 'EX', 'DT', 'JJ', 'NN', 'JJ', 'NN', 'VB', 'DT',
          'VB', 'TO', 'DT', 'NN']
    s3 = ['DT', 'JJ', 'NN', 'VB', 'IN', 'WP', 'VB', 'NN', 'DT', 'NN', 'CC', 'NN',
          'PRP$', 'JJ', 'NN']

    print(s1, )
    print(s2)
    print(s3)

def raw_agreement(manual_tags, ref_tags):
    agree = 0
    for mtag in manual_tags:
        for rtag in ref_tags:
            if mtag == rtag:
                agree += 1
    return agree/len(ref_tags)
    
def ex32():
    """
    Calculate the raw agreement rate between manual annotations and
    the gold standard.
    """
    s1 = ['DT', 'NN', 'NNP', 'NN', 'VB', 'VB', 'TO', 'VB', 'NN', 'DT', 'NN', 'DT', 'JJ',
          'VB', 'IN', 'VB', 'JJ']
    s2 = ['IN', 'DT', 'NN', 'NN', 'EX', 'DT', 'JJ', 'NN', 'JJ', 'NN', 'VB', 'DT',
          'VB', 'TO', 'DT', 'NN']
    s3 = ['DT', 'JJ', 'NN', 'VB', 'IN', 'WP', 'VB', 'NN', 'DT', 'NN', 'CC', 'NN',
          'PRP$', 'JJ', 'NN']
    print("Raw agreement for sentence 1: {:.2f}".format(raw_agreement(s1, tags1)))
    print("Raw agreement for sentence 2: {:.2f}".format(raw_agreement(s2, tags2)))
    print("Raw agreement for sentence 3: {:.2f}".format(raw_agreement(s3, tags3)))


if __name__ == '__main__':
    true_pos = total_relevant
    false_pos = num_R - total_relevant
    true_neg = total_docs - total_relevant
    false_neg = true_neg - num_N
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    print("True positives: {}".format(true_pos))
    print("False positives: {}".format(false_pos))
    print("True negatives: {}".format(true_neg))
    print("Precision (P): {}".format(true_pos/(true_pos + false_pos)))
    print("Recall (R): {}".format(true_pos/(true_pos + false_neg)))
    print("F-Score with beta=1: {}".format((2*precision*recall)/(precision + recall)))
    print("Accuracy: {}".format(num_R/total_docs))

    ex21()
    ex22()
    ex23()
    ex32()
