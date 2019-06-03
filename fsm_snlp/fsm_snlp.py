#! /usr/bin/env python
"""
Finite State Methods and Statistical NLP
"""

import nltk
from nltk.corpus import masc_tagged
nltk.download('masc_tagged')

def tag_count(tag='VB'):
    """
    Count the number of a given tag in corpus.
    """
    return sum([True for sent in masc_tagged.tagged_sents() for pos_tag in sent if pos_tag[1] == tag]) # check tag for each sentence

def get_unique_tags():
    """
    Get the tags in a corpus.

    Using a set ensures we get distinctly unique tags across the entire corpus
    """
    return {pos_tag[1] for sent in masc_tagged.tagged_sents() for pos_tag in sent}

def next_tags(tag='VB'):
    """
    Calculate the transition distribution.
    
    The transition from one tag to other tags. Returns a list of tags
    that appear next to the given tag.
    """
    tags = get_unique_tags()
    dist = dict()
    next_tags = []
    # Check for the tag that is next 'VB'
    for sent in  masc_tagged.tagged_sents():
        for i in range (len(sent)):
            if sent[i][1] == tag and i+1 < len(sent):
                next_tags.append(sent[i+1][1])
    return next_tags

def count_next_tags(tag='VB'):
    """
    Get the number of tags next to the given tag

    Stats of the next tag.
    """
    ntags = next_tags(tag=tag)
    corpus_tags = get_unique_tags()
    return {tag:ntags.count(tag) for tag in get_unique_tags()}
    
def tag_transition_dist(tag='VB'):
    """
    Calculate the transition distribution for a given tag.
    """
    vb_next = count_next_tags()
    dt = vb_next['DT']
    total = 0
    for item in vb_next:
        total += vb_next[item]
    return dt/total

if __name__ == '__main__':
    # print("Tag count for {} tag: {}".format('VB', tag_count()))
    # print("Tags in the corpus {}".format(unique_tags()))

#     print("{}".format(next_tags()))
#     print("\n")
    print(count_next_tags())
    print("Transition distribution for 'DT' tag: {}".format(transition_dist()))
