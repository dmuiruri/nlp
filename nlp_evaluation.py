#! /usr/bin/env python

total_docs = 100
total_relevant = 10

retrieved = ['R', 'N', 'N', 'R', 'R', 'N', 'N', 'N',
             'R', 'N', 'R', 'N', 'N', 'R', 'R']

num_R = retrieved.count('R')
num_N = retrieved.count('N')


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
