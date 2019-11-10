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

stopwordlist = list(set(stopwords.words('english')))

search_space_params = {
    'MAX_ORBIT_CAPACITY':10,
    'GRAPHLET_TYPE' : 'PRUNED',
    'PATTERN_FREQ_THRESHOLD_BY_STACK': {0:5,1:4,2:3,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2},
    'MAX_SEARCH_ITERATIONS': 10,
    'PRUNED_STACK_SIZE': 100000,
    'MIN_WORD_FREQ': 50,
    'WORD_SELECTION_RATIO': 0.5,
    'CONTENT_WORD_REGEX_PATTERN': '^[A-z0-9]{3,}.*$', # raw word represenation
    'STOPWORD_LIST':stopwordlist
}

# loading corpus
doc_collection = dict([ (id,reuters.raw(id)) for id in reuters.fileids() ] )

# pattern extraction

word_patterns = extract_graphlets(doc_collection, search_space_params)

# persisting patterns to storage
with open('E:\\Projects\\Research\\Data\\analysis\\reuters_gpatterns.tsv', 'w',  encoding='utf-8') as filew:
    filew.write("OrbitPattern\tCenterWord\tWordDocIncidence\n")
    for graphlet_pattern in word_patterns.keys():
        if len(word_patterns[graphlet_pattern]) > 1:
            filew.write(graphlet_pattern + "\t" + str(list(set(list(word_patterns[graphlet_pattern])))) + '\n' )