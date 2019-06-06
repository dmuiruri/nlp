#! /usr/bin/env python

import nltk
from nltk.parse.chart import BottomUpChartParser
from nltk.corpus import treebank
from nltk.grammar import Nonterminal

nltk.download('treebank')

cfg_rules1 = """
S -> NP-SBJ VP STOP
NP-SBJ -> DT NN NN
VP -> VBZ  NP
NP -> CD JJ NNS
DT -> 'the'
NN -> 'purchase' | 'price'
VBZ -> 'includes'
CD -> 'two'
JJ -> 'ancillary'
NNS -> 'companies'
STOP -> '.'
"""

cfg_rules2 = """
S -> NP-SBJ VP STOP
NP-SBJ -> DT NN
VP -> VBD NP PP-TMP
NP -> NP PP-DIR
NP -> DT NN
PPDIR -> IN NP
NP -> DT NN CC NN NN
PP-TMP -> IN NP
NP -> NNP CD

DT -> 'the'
NN -> 'guild'
VBD -> 'began'
DT -> 'a'
NN -> 'strike'
IN -> 'against'
DT -> 'the'
NN -> 'TV' | 'movie' | 'industry'
CC -> 'and'
IN -> 'in'
NNP -> 'March'
CD -> '1988'
STOP -> '.'
"""

cfg_rules3 = """
S -> NP-SBJ VP STOP
NP-SBJ -> DT NN NN
VP -> VBD NP PP-TMP
VP -> VBZ NP
NP -> NP PP-DIR
NP -> DT NN
PPDIR -> IN NP
NP -> DT NN CC NN NN
PP-TMP -> IN NP
NP -> NNP CD
NP -> CD JJ NNS

DT -> 'the'
VBD -> 'began' | 'bought'
VBZ -> 'includes'
DT -> 'a'
IN -> 'against'
DT -> 'the'
NN -> 'TV' | 'movie' | 'industry' | 'guild' | 'strike' | 'purchase' | 'price'
CC -> 'and'
IN -> 'in'
NNP -> 'March'
JJ -> 'ancillary'
NNS -> 'company' | 'companies'
CD -> '1988' | 'one' | 'two'
STOP -> '.'
"""
# Exercise 3
cfg_rules3_cnf = """
S ->  B1 STOP
B1 -> NP-SBJ VP
NP-SBJ -> B2 NN
B2 -> DT NN 
VP -> B3 PP-TMP
B3 -> VBD NP 
VP -> VBZ NP
NP -> NP PP-DIR
NP -> DT NN
PPDIR -> IN NP
NP -> B4 NN
B4 -> B5 NN
B5 -> B6 CC
B6 -> DT NN 
PP-TMP -> IN NP
NP -> NNP CD
NP -> B7 NNS
B7 -> CD JJ

DT -> 'the'
VBD -> 'began' | 'bought'
VBZ -> 'includes'
DT -> 'a'
IN -> 'against'
DT -> 'the'
NN -> 'TV' | 'movie' | 'industry' | 'guild' | 'strike' | 'purchase' | 'price'
CC -> 'and'
IN -> 'in'
NNP -> 'March'
JJ -> 'ancillary'
NNS -> 'company' | 'companies'
CD -> '1988' | 'one' | 'two'
STOP -> '.'
"""

sentences1 = [
    "the guild began a strike against the TV and movie industry in March 1988 .".split(),
    "the guild bought one ancillary company .".split(),
    "the purchase price includes two ancillary companies .".split()
    ]

sentences2 = [
    "Mr. Vinken is chairman .",
    "Stocks rose .",
    "Alan introduced a plan ."
    ]

cfg1 = nltk.CFG.fromstring(cfg_rules1) 
cfg2 = nltk.CFG.fromstring(cfg_rules2) 
cfg3 = nltk.CFG.fromstring(cfg_rules3)

# Exercise 4
parser = BottomUpChartParser(cfg_rules3_cnf)

# cfg1.check_coverage("the purchase price includes two ancillary companies .".split())
# cfg2.check_coverage("the guild began a strike against the TV and movie industry in March 1988 .".split())
# cfg3.check_coverage("the guild bought one ancillary company .".split())

def ex5():
    """
    Treebank Parser
    """
    # prd = list(set([p for tree in treebank.parsed_sents() for p in tree.productions()]))
    prd = list(set([p for tree in treebank.parsed_sents() for p in tree.productions()]))
    cfg_5 = nltk.CFG(Nonterminal("S"), prd)
    parser = BottomUpChartParser(cfg_5)
    for sent in sentences2:
        parses = list(parser.parse(sent.split()))
        print("Sentence: {}, Parse trees obtained for the sentence: {}".format(sent, len(parses)))

def ex6(symbol='S'):
    """
    PCFG: Probabilistic CFGs
    """
    productions = [p for tree in treebank.parsed_sents() for p in tree.productions()]
    return len([p for p in productions if p.lhs().symbol() == symbol])


if __name__ == '__main__':
    # Exercise 2
#     print("Exercise 2: Grammar Extension >>")
#     for s in sentences:
        
#         print("Checking coverage for sentence {}".format(s))
#         print("Parsing Errors: {}\n".format(cfg3.check_coverage(s)))

#     # Exercise 3
#     print("Exercise 3: CNF Form>>")
#     cfg3_cnf = nltk.CFG.fromstring(cfg_rules3_cnf)
#     for s in sentences:
        
#         print("Checking coverage for sentence {}".format(s))
#         print("Parsing Errors: {}\n".format(cfg3_cnf.check_coverage(s)))
#         print("Is the grammar is the CNF form: {}".format(cfg3_cnf.is_flexible_chomsky_normal_form()))

#     # Exercise 4
#     print("Exercise 4: Parsing with the Grammar ")
#     parser = BottomUpChartParser(cfg3_cnf)
#     for s in sentences:
#         print(list(parser.parse(s)))

    # Exercise 5
#    ex5()

    # Exercise 6
    print("Counting symbols in a production {}".format(ex6()))
