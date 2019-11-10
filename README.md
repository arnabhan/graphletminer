# Text Graphlet Miner

This project is a Python lib and utility app to extract graphlets from text corpora. Text graphlets are models like chemical atoms with a nucleus representing a hypothesized keyword and electrons (words with paths traced to the nucleus) organized in orbits around the nucleus. Orbits are numbered 1, 2 ... N. An orbit with rank i contains words that are at a distance i from the word at center. Graphlets are explored within word graphs. These graphs have nodes representing words and edges established between nodes with label words that constitute a bigram. 

Search for patterns starts with minimalistic graphlets with size=1 (with only center words) and then patterns in each search iteration are evaluated and expanded (by adding more nodes and orbits around the center word). The search is controlled by a set of parameters to reduce memory and running time complexity.

## Usage:
```bash
python examples/reuters.py
```

## Keywords appears in similar context
The lexical and syntactic contexts of keywords can be recurring across multiple segements in text corpora. Text graphlet patterns can help identify these set of keywords that share the same pattern.

## Data exaploration
Graphlet miner can be used for general purpose text genres analysis. It can identify recurring patterns in data that define the context in a which a set of words (at the center of graphlets) appear. Words flying in orbits around the center defines the context. The graphlet patterns can be used to compare different word usage across multile genres/corpora

## Text classification
Graphlet patterns can be used to represents documents in feature space for document categorization applications. Correlation with specific document categories can be an effective signal for text classifiers.

#### Example Pattern 1
This pattern was extracted within the American National Corpus (ANC) BioMed corpus collection:

1:different_JJ;<FUNC_OR_STOP_WORD>|2:cells_NNS;expression_NN;<FUNC_OR_STOP_WORD>;other_JJ	[('patterns_NNS', '1471-2091-3-15.txt'), ('classes_NNS', '1471-2091-3-15.txt')]

![Graphlet pattern](/docs/img/graphlet_pattern.jpg?raw=true)


# Installation

```bash
python setup.py install
```

# Example:

Graphlet miner needs a collection of documents to search for patterns within, in addition to a set of search parameters that can be effective in reducing running time. For instance, a minimum content word frequency threshold can be set. Only words above this minimum threshold will be added to graphlet patterns. A stack-based decoding is applied to explore pattens. A stack can contain graphlets of certain size. When set of pattens in stack_i are evaluated, new set of graphlets (with additonal nodes) are added stack_i+1. Number of patterns within each stack are pruned by a parameter that defines the minimum frequency of patterns in that stack. The PATTERN_FREQ_THRESHOLD_BY_STACK search parameter is a dictionary of the form: stack_id:min_pattern_freq. For example, this dictionary defines min pattern freq from stacks 0 through 10: {0:5,1:4,2:3,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2},


```python

# This example assumes NLTK corpus reuters is downloaded. Also, please make sure that stopwords are also downloaded from NLTK.

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

```

## Citation
Nabhan A.R., Shaalan K. (2016) Keyword Identification Using Text Graphlet Patterns. In: Métais E., Meziane F., Saraee M., Sugumaran V., Vadera S. (eds) Natural Language Processing and Information Systems. NLDB 2016. Lecture Notes in Computer Science, vol 9612. Springer, Cham

## License
[MIT](https://choosealicense.com/licenses/mit/)