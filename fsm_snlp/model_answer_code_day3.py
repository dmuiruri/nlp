#!../venv/bin/python3
# Model answer code for Intro to NLP intensive course 2019
# Day 3 assignments: POS tagging with HMMs
#
# Mark Granroth-Wilding <mark.granroth-wilding@helsinki.fi>, 2019
# Released under GPL v3

# 1. Load tagged data
#    Estimate transition probabilities for one tag.
#    Estimate emission probabilities for same tag.
print("###### Exercise 1 ######")
from nltk.corpus import masc_tagged
from collections import Counter
from itertools import islice
from operator import itemgetter


# Iterate over tag pairs
def tag_pairs(corp):
    for sent in corp:
        if len(sent) > 0:
            prev = sent[0][1]
            for word, tag in sent[1:]:
                yield (prev, tag)
                prev = tag

# Collect tag counts for VB
vb_tag_counts = Counter(tipo for (ti, tipo) in tag_pairs(masc_tagged.tagged_sents()) if ti == "VB")
# Sum up all counts for VB condition
vb_tag_total = float(sum(c for (p, c) in vb_tag_counts.items()))
# Estimate transition dist using MLE
trans_probs = dict(
    (tipo, float(count) / vb_tag_total) for (tipo, count) in vb_tag_counts.items()
)
print("Transitions from VB")
print("\n".join("{}: {:.2e}".format(tag, prob) for (tag, prob) in trans_probs.items()))
# Then answer the question
print("\nVB -> DT transition prob: {:.3f}".format(trans_probs["DT"]))

# Collect emission counts
vb_word_counts = Counter(word for sent in masc_tagged.tagged_sents() for (word, tag) in sent if tag == "VB")
# Sum up all counts (should be almost the same as above)
vb_tag_total2 = float(sum(c for (w, c) in vb_word_counts.items()))
# Estimate emission dist using MLE
em_probs = dict(
    (word, float(count) / vb_tag_total2) for (word, count) in vb_word_counts.items()
)
print("\nEmissions from VB (top 50")
print("\n".join("{}: {:.2e}".format(word, prob) for (word, prob) in
                islice(sorted(em_probs.items(), key=itemgetter(1), reverse=True), 50)
))
# Then answer the question
print("\nVB -> 'feel' emission prob: {:.3f}".format(em_probs["feel"]))


# 2. Use NLTK code to train full model on this corpus.
#    Try tagging a few sentences to see the output.
print("\n\n####### Exercise 2 ########")
from nltk.tag.hmm import HiddenMarkovModelTagger

example_sentences = [
    "Once we have finished , we will go out .".split(),
    "There is always room for more understanding between warring peoples .".split(),
    "Evidently , this was one of Jud 's choicest tapestries , for the noble "
    "emitted a howl of grief and rage and leaped from his divan".split(),
    "Misjoggle in a gripty hifnipork .".split(),
    "One fretigy kriptog is always better than several intersplicks .".split(),
]

print("Training HMM on MASC")
hmm_sup = HiddenMarkovModelTagger.train(masc_tagged.tagged_sents())

for sent in example_sentences:
    # POS tag each sentence
    tagged_sent = hmm_sup.tag(sent)
    print(tagged_sent)


# 3. Use NLTK code to run unsupervised training on unlabelled corpus.
#    Tag the same sentences again and see if the result is better.
import os
import pickle
from ass3utils import train_unsupervised

print("\n\n####### Exercise 3 #######")
# Don't train the unsupervised HMM every time
# If we've previously stored a pickled trained HMM, load that and skip the training
if os.path.exists("unsup_hmm.pkl"):
    with open("unsup_hmm.pkl", "rb") as f:
        hmm_unsup = pickle.load(f)
        print("Loaded pickled unsupervised HMM")
else:
    # Train a semi-supervised HMM
    with open("radio_planet_tokens.txt", "r") as f:
        unlab_sents = [[word for word in s.split()] for s in f.readlines()]

    print("Training HMM with labelled and unlabelled data")
    hmm_unsup = train_unsupervised(masc_tagged.tagged_sents(), unlab_sents)

    with open("unsup_hmm.pkl", "wb") as f:
        # Store so we can just load it next time
        pickle.dump(hmm_unsup, f)

# Add some new sentences to the examples
example_sentences.extend([
    "Yesterday these fiends operated upon Doggo .".split(),
    "For a time, his own soul and this brain - maggot struggled for supremacy .".split()
])

for sent in example_sentences:
    # Try tagging the sentences using both models
    tagged_sent = hmm_unsup.tag(sent)
    sup_tagged_sent = hmm_sup.tag(sent)
    print("\nUnsup:", tagged_sent)
    print("Sup:", sup_tagged_sent)


# 4. Tag some out-of-domain sentences using the MLE and Baum-Welch models.
#    Which looks better?
# Tag sentences as above...


# 5. Write some nonsense sentences using the words of the vocabulary.
#    Try using the HMM (both models) as a LM:
#     - put in each sentence and estimate its LM probability
#     - do the nonsense sentences get a lower probability?
print("\n\n###### Exercise 5 ######")

positive_sentences = [
    "This is a test .".split(),
    "There is always time .".split(),
] + example_sentences

print("\nReal sentences:")
for sent in positive_sentences:
    not_covered = [w for w in sent if w not in hmm_unsup._symbols]
    if not_covered:
        print("Words not in vocab: {}".format(", ".join(not_covered)))
    else:
        logprob = hmm_unsup.log_probability([(word, None) for word in sent])
        print("{}: {:.3e}".format(" ".join(sent), logprob))

print("\nNonsense sentences:")
nonsense_sentences = [
    "Friends plane note under in .".split(),
    "Outside eats whatever top on .".split(),
]
for sent in nonsense_sentences:
    not_covered = [w for w in sent if w not in hmm_unsup._symbols]
    if not_covered:
        print("Words not in vocab: {}".format(", ".join(not_covered)))
    else:
        logprob = hmm_unsup.log_probability([(word, None) for word in sent])
        print("{}: {:.3e}".format(" ".join(sent), logprob))


# 6. Use the HMM's random generator to generate some sentences.
#    Do they look like real sentences?
#    Why are they (usually) incoherent?
#    Why don't they look like the sentences in the training corpus?
#    Is the unsupervised model better?
print("\n\n###### Exercise 6 #####")
import random

print("\nRandomly sampled sentences (sup)")
for i in range(10):
    length = random.randint(5, 15)
    sent = hmm_sup.random_sample(random, length)
    print(" ".join(w for (w,t) in sent))

print("\nRandomly sampled sentences (unsup)")
for i in range(10):
    length = random.randint(5, 15)
    sent = hmm_unsup.random_sample(random, length)
    print(" ".join(w for (w,t) in sent))
