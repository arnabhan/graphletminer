'''

This module is used to analysis a graph to extract graphlet patterns

Supported graph models:
1 - Plain word graphs where nodes are simple words and edges are bigrams
2 - Graphs constructed from dependency parsing

'''
import sys
#sys.path.insert(0, "E:\Projects\Research\graphlet-miner")

from tqdm import tqdm
from nltk.corpus import movie_reviews, reuters, brown, inaugural
from gminer.algorithms import extract_graphlets
from gminer.text_processing import get_word_frequencies
import nltk

from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk import bigrams, word_tokenize, pos_tag 
from nltk.corpus import stopwords


root='ANC\\OANC-1.0.1-UTF8\\OANC\\data\\'
myreader= nltk.corpus.PlaintextCorpusReader(root + '\\written_2\\technical\\biomed', '.*\.txt') 
doc_collection = dict([ (id,myreader.raw(id)) for id in myreader.fileids() ] )

USE_POS_TAGGING = True
stopwordlist = list(set(stopwords.words('english')))

search_space_params = {
    'MAX_ORBIT_CAPACITY':10,
    'GRAPHLET_TYPE' : 'PRUNED',
    'PATTERN_FREQ_THRESHOLD_BY_STACK': {0:3,1:2,2:2,3:2,4:5},
    'MAX_SEARCH_ITERATIONS': 7,
    'PRUNED_STACK_SIZE': 10000,
    'MIN_WORD_FREQ': 40,
    'WORD_SELECTION_RATIO': 0.5,
    'CONTENT_WORD_REGEX_PATTERN': '^[A-z0-9]{3,}_[NJ].*$',
    'STOPWORD_LIST':[]
}

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

# persisting patterns to storage
with open('analysis\\anc_biomed_word_patterns.tsv', 'w',  encoding='utf-8') as filew:
    filew.write("OrbitPattern\tCenterWord\tWordDocIncidence\n")
    for graphlet_pattern in word_patterns.keys():
        if len(word_patterns[graphlet_pattern]) > 1:
            filew.write(graphlet_pattern + "\t" + str(list(set(list(word_patterns[graphlet_pattern])))) + '\n' )
