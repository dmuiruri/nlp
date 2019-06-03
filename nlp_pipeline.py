#! /usr/bin/env python

import nltk
import requests, xmltodict, pickle, os
import editdistance
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords, cmudict
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('cmudict')

lemmatizer = WordNetLemmatizer()

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
    res.append([pos_tag(word_tokenize(line)) for line in sent_tokenize(text)])
    res.append([[lemmatizer.lemmatize(word) for word in item] for item in [word_tokenize(line) for line in sent_tokenize(text)]])
    return res

def filter_text(text):
    """
    Filter out stop words.
    """
    stopWords = set(stopwords.words('english'))
    kept = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    pos = process_text(text)[0]
    sentences = [[word for word in sent if word[0] not in stopWords] for sent in pos] # removed stop words
    return [[word for word in sent if word[1] in kept] for sent in sentences]

def food_words(file_path='./food_words.pkl'):
  if os.path.isfile(file_path): # load stored results
    with open(file_path, 'rb') as f:
      return pickle.load(f)

  url = 'http://ngrams.ucd.ie/therex3/common-nouns/head.action?head=food&ref=apple&xml=true'
  response = requests.get(url)
  result = xmltodict.parse(response.content)
  _root_content = result['HeadData']
  result_dict = dict(map(lambda r: tuple([r['#text'].replace('_', ' ').strip(), int(r['@weight'])]), _root_content['Members']['Member']))

  with open(file_path, 'wb') as f: # store the results locally (as a cache)
    pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)
  return result_dict

def pronounce(word):
    """
    Find the pronounciation of a word.
    """
    arpabet = cmudict.dict()
    return arpabet[word.lower()][0] if word.lower() in arpabet else None

def make_punny(text):
    """
    Select a token that is either a verb or noun and replace it with a
    similar sounding food related word.
    Assumes the food database is stored locally in current directory.
    """
    tokens_pos_tagged = process_text(text)[0]
    print(tokens_pos_tagged)
    noun_list = [[ word[0] for word in line if word[1] == 'NN' or 'NNP'] for line in tokens_pos_tagged]
    food_words_dict = food_words() 
#     food_words_dict = {'vegetable': 17227, 'chicken tamale': 3, 'duckling': 16, 'meat': 15967, 'currant buns': 8,
#                        'milk': 15580, 'prepared meat': 29, 'egg': 15448, 'green lolly': 10, 'strong game': 1, 'catfish stew': 18, 'fresh herb': 56,
#                        'drinking soda': 2, 'fish': 14626, 'local salmon': 8, 'fruit salad': 4, 'bird': 324, 'salad wrap': 8, 'aromatic cheese': 5,
#                        'shanghai noodle': 4, 'creamy soup': 96, 'braised meat': 14, 'sesame paste': 5
#                        }

    if noun_list[0][0]:
        print("First noun {}".format(noun_list[0][0]))
        temp_dist = max(food_words_dict.values())
        for word in food_words_dict:
            try:
                dist = editdistance.eval(pronounce(word), pronounce(noun_list[0][0]))
            except TypeError:
                continue
            print("testing word: {} dist: {} temp_dist: {}".format(word, dist, temp_dist))
            if  dist < temp_dist:
                temp_dist = dist
                word_temp = word
        return [item for item in food_words_dict if item == word_temp]
    else:
        print('No Noun found in the first sentence')
    

if __name__ == '__main__':

    text = "One morning I shot an elephant in my pajamas. How he got into my pajamas I'll never know." # by Groucho Marx
    sample_text = "Finger Lickin’ Good"
    print(process_text(text)[0])
    print(filter_text(text))    
    print(make_punny('Jurassic Park'))
