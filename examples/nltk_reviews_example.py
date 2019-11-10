'''

This module is used to analysis a graph to extract graphlet patterns

Supported graph models:
1 - Plain word graphs where nodes are simple words and edges are bigrams
2 - Graphs constructed from dependency parsing

'''

import sys
import nltk
from tqdm import tqdm
from nltk.corpus import movie_reviews, reuters, brown, inaugural
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk import bigrams, word_tokenize, pos_tag 
from nltk.corpus import stopwords
from gminer.algorithms import extract_graphlets

# settings and constants 
USE_POS_TAGGING = True # False
stopwordlist = list(set(stopwords.words('english')))

search_space_params = {
    'MAX_ORBIT_CAPACITY':10,
    'GRAPHLET_TYPE' : 'PRUNED',
    'PATTERN_FREQ_THRESHOLD_BY_STACK': {0:5,1:4,2:3,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2},
    'MAX_SEARCH_ITERATIONS': 10,
    'PRUNED_STACK_SIZE': 100000,
    'MIN_WORD_FREQ': 50,
    'WORD_SELECTION_RATIO': 0.5,
    #'CONTENT_WORD_REGEX_PATTERN': '^[A-z0-9]{3,}.*$', # raw word represenation
    'CONTENT_WORD_REGEX_PATTERN': '^[A-z0-9]{3,}_[NJV].*$', # POS tagged word representation
    'STOPWORD_LIST':stopwordlist
}

# loading corpus
#doc_collection = dict([ (id,movie_reviews.raw(id)) for id in movie_reviews.fileids() ] )
doc_collection = dict([ (id,reuters.raw(id)) for id in reuters.fileids() ] )

if USE_POS_TAGGING:
    print("Processing text docs for part of speech tagging...")
    docids = list(doc_collection.keys())
    for i in tqdm(range(len(docids))):
    #for (id,text) in doc_collection.items():
        id = docids[i]
        text = doc_collection[docids[i]]
        # sent tokenize is used to avoid having bigrams between words in two differnt lines
        line_seq = sent_tokenize(text.replace('_','-'))
        pos_tagged_lines_seq = []
        for line in line_seq:
            token_seq = word_tokenize(line)
            tag_seq = pos_tag(token_seq)
            # in the word graph, same word with two different pos tags will be represented by two distinct nodes
            # for example, a word with a past tense verb lexical representation can mean 1) verb and 2) adjective, thus two nodes will be required here
            word_tag = [a.lower() + "_" + b for (a,b) in tag_seq]
            pos_tagged_lines_seq.append(' '.join(word_tag))
        doc_collection[id] = '\n'.join(pos_tagged_lines_seq)
print("Done.")
# pattern extraction

word_patterns = extract_graphlets(doc_collection, search_space_params)

for g in word_patterns.keys():
    if len(g.split('|')) >= 2 and len(word_patterns[g]) >= 2:
        print(g + "==>" + str(list(set(word_patterns[g]))))

# persisting patterns to storage
with open('E:\\Projects\\Research\\Data\\analysis\\reuters_word_patterns.tsv', 'w',  encoding='utf-8') as filew:
    filew.write("OrbitPattern\tCenterWord\tWordDocIncidence\n")
    for graphlet_pattern in word_patterns.keys():
        if len(word_patterns[graphlet_pattern]) > 1:
            filew.write(graphlet_pattern + "\t" + str(list(set(list(word_patterns[graphlet_pattern])))) + '\n' )
