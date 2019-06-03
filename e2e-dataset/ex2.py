#! /usr/bin/env python

from ass6utils import read_file, score, MeaningRepresentation, bleu_single
import random

meaning_representations, references = read_file('devset.csv')

def generate(mr: MeaningRepresentation) -> str:
    # Aromi is a coffee shop providing Chinese food It is located in
    # the riverside. Its customer rating is 1 out of 5.

    family =  " considered as family friendly" if mr.family_friendly == 'yes' else "not considered as family friendly"
    rating = " with a {} customer rating.".format(mr.customer_rating) if mr.customer_rating else " " 
    loc = "at {} ".format(mr.area)
    sent_1 = "{} is a {} located {}near {} and provides {} Cuisines. ".format(
        mr.name, mr.eat_type, loc,  mr.near, mr.food,)
    sent_2 = " The restaurant has a {} price range ".format(mr.price_range) if mr.price_range else "." 
    sent_3 = " and is also {} ".format(family) +  rating

    return sent_1 + sent_2 + sent_3

score(generate, meaning_representations, references)

print('--')
for _ in range(10):
    print('\t', generate(random.choice(meaning_representations)))
print('\n')

def 
