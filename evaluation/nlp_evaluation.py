#! /usr/bin/env python

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

def ex21():
    """
    """
    correct_tags = 0
    total_tags = 0 # matrix
    training_sents, test_sents = split_corpus(treebank, 0.8)
    test_tokens = [token[0] for sent in test_sents for token in sent]
    test_tags = [tag[1] for sent in test_sents for tag in sent]
    hmm_tagger = HiddenMarkovModelTagger.train(training_sents)
    pred_tags = [tag[1] for tag in hmm_tagger.tag(test_tokens)] # predict a tag given a token
    
    # create confusion matrix to evaluate results
    cmat = ConfusionMatrix(reference=test_tags, test=pred_tags, sort_by_count=True)
    cmat_tag_dict = {item.split()[0].split(':')[0]: item.split()[1] for item in cmat.key().split('\n')[1:-1]}
    cmat_tag_list = [item.split()[1] for item in cmat.key().split('\n')[1:-1]]
    for tag_ref in cmat_tag_list:
        for tag_test in cmat_tag_list:
            if tag_ref == tag_test:
                correct_tags += cmat[tag_ref, tag_test]
            total_tags = cmat[tag_ref, tag_test]
    accuracy = correct_tags/total_tags
    print("Overall accuracy of the HMM model: {}".format(accuracy))

if __name__ == '__main__':
#     true_pos = total_relevant
#     false_pos = num_R - total_relevant
#     true_neg = total_docs - total_relevant
#     false_neg = true_neg - num_N
#     precision = true_pos/(true_pos + false_pos)
#     recall = true_pos/(true_pos + false_neg)
#     print("True positives: {}".format(true_pos))
#     print("False positives: {}".format(false_pos))
#     print("True negatives: {}".format(true_neg))
#     print("Precision (P): {}".format(true_pos/(true_pos + false_pos)))
#     print("Recall (R): {}".format(true_pos/(true_pos + false_neg)))
#     print("F-Score with beta=1: {}".format((2*precision*recall)/(precision + recall)))
#     print("Accuracy: {}".format(num_R/total_docs))

    print(ex21())
