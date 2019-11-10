from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk import bigrams, word_tokenize, pos_tag 

def get_bigrams(input_data): #, use_pos_tagging):
    bigrams_list = []
    line_seq = sent_tokenize(input_data) #sent_tokenize(input_data.replace('_','-'))
    for line in line_seq:
        token_seq = word_tokenize(line)
        bigrams_list.extend(bigrams([w for w in token_seq]))
    return bigrams_list
    
def get_freq_weighted_bigrams(bigram_list):
    weigthed_bigrams = {}
    for bigramseq in bigram_list:
        if bigramseq not in weigthed_bigrams.keys():
            weigthed_bigrams[bigramseq] = 1
        else:
            weigthed_bigrams[bigramseq] += 1
    return weigthed_bigrams

def get_word_frequencies(doc_collection, retention_ratio): # need to decouple pos tagging from here
    word_freq_map = {}
    
    for id, input_data in doc_collection.items():
        line_seq = sent_tokenize(input_data) #sent_tokenize(input_data.replace('_','-'))   
        for line in line_seq:
            token_seq = word_tokenize(line) 
            for token in token_seq:
                if token not in word_freq_map.keys():
                    word_freq_map[token] = 0
                word_freq_map[token] += 1

    sorted_word_freq_list = list(reversed(sorted(word_freq_map.items(), key=lambda kv: kv[1])))
    n = len(sorted_word_freq_list)

    return dict(sorted_word_freq_list[:int(n * retention_ratio)])
