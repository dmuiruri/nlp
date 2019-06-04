#! /usr/bin/env python
"""
Finite State Methods and Statistical NLP
"""

import nltk
import pickle
from os import path
from nltk.corpus import masc_tagged
from nltk.tag import hmm
from ass3utils import train_unsupervised

nltk.download('masc_tagged')

sentences = [
    "Once we have finished , we will go out .",
    "There is always room for more understanding between warring peoples .",
    "Evidently , this was one of Jud 's choicest tapestries , for the noble emitted a howl of grief and rage and leaped from his divan ."
    ]
sentences2 = [
    "Misjoggle in a gripty hifnipork .",
    "One fretigy kriptog is always better than several intersplicks ."
    ]
news = [
    "Strong demand triggers oil rally .",
    "Crude oil prices surged back above the $47 a barrel mark on Thursday after an energy market watchdog raised its forecasts for global demand ."
    ]
legal = ["If there is any conflict between the terms in the General Terms and the Additional Terms, then the Additional Terms govern .",
         "You may have additional rights under the law .",
         "We do not seek to limit those rights where it is prohibited to do so by law ."
          ]
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
    Get the number of unique tags next to a given tag

    Stats of the next tag.
    """
    ntags = next_tags(tag=tag)
    corpus_tags = get_unique_tags()
    return {tag:ntags.count(tag) for tag in get_unique_tags()}
    
def tag_transition_dist(tag='VB'):
    """
    Calculate the transition distribution for a given tag.
    
    This function calculates the probabiluty that the next tag is a DT
    (Determinant) given the current tag is a 'VB (Verb):
    p(t[i+1]=DT|t[i]=VB)
    """
    vb_next = count_next_tags()
    dt = vb_next['DT']
    total = 0
    for item in vb_next:
        total += vb_next[item]
    return dt/total

def tag_and_word(tag='VB', word='feel'):
    """
    Calculate the p(w[i]='feel'|t[i]=VB)

    Calculates the probability of a given word given a tag
    """
    tags = [word_tag for sent in masc_tagged.tagged_sents() for word_tag in sent if word_tag[1] == tag]
    return sum([True for tag in tags if tag[0] == word])/len(tags)

def nltk_hmm(sentences):
    """
    Using NLTK's library to perform tagging by training a Hidden
    Markov Model, the model uses Maximum Likelihood Estimates (MLE) to
    estimate the parameters.

    Note: The model appears to assign all unseen words as NN, one
    solution as indicated in the slides(Day 3) is "smoothing".
    """
    trainer = hmm.HiddenMarkovModelTrainer()
    model = trainer.train(masc_tagged.tagged_sents())
    for sentence in sentences:
        print("Tagging sentence: {}".format(sentence))
        print("{}\n".format(model.tag(sentence.split())))

def semi_supervised_hmm(sentences):
    """
    Train and test a tagging model with both labeled and unlabeled
    data.

    Trains a semi-supervised model where the model is trained with
    partly labeled data and partly unlabled data.

    sentences: Sentences to be tagged
    """
    with open('radio_planet_tokens.txt', 'r') as f:
        unlabeled_data = [[word for word in sent.split()] for sent in f.readlines()]  # Labeled data
    labeled_data = masc_tagged.tagged_sents()  # Unlabeled data
 
   # Check if the model exists, saved the first time it is
   # successfully created. Delete file to update model
    if path.exists("hmm_model.pkl"):
        # Load the model from disk
        print("Loading model from the disk")
        with open("hmm_model.pkl", 'rb') as model:
            hmm_model = pickle.load(model)
    else:
        # Train the model since not found
        print("Training the model...")
        hmm_model = train_unsupervised(labeled_data, unlabeled_data)
        with open("hmm_model.pkl", 'wb') as model:
            pickle.dump(hmm_model, model)

    # Tag sentences with trained model
    for sentence in sentences:
        print("Tagging sentence: {}".format(sentence))
        print("{}\n".format(hmm_model.tag(sentence.split())))

def log_prob_hmm():
    """
    """
    
if __name__ == '__main__':
#     print("Tag count for {} tag: {}".format('VB', tag_count()))
#     print("Tags in the corpus {}".format(unique_tags()))
    
#     print("{}".format(next_tags()))
#     print("\n")
#     print(count_next_tags())

#     # part 1a
#    print("Transition distribution for 'DT' tag: {}".format(tag_transition_dist()))
#     # Part 1b
#    print("Computing the probability of a word given a tag {}".format(tag_and_word()))

#     # part 2
#     print("Part 2: NLTK hmm Model \n")
#    nltk_hmm(sentences) # sentences seen during training
#    nltk_hmm(sentences2)  # sentences not seen during training
    # part 3
#    semi_supervised_hmm(sentences)

    # part 4
    nltk_hmm(news)
    semi_supervised_hmm(news)
    nltk_hmm(legal)
    semi_supervised_hmm(legal)
