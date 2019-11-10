'''

This module is used to analysis a graph to extract graphlet patterns

Supported graph models:
1 - Plain word graphs where nodes are simple words and edges are bigrams
2 - Graphs constructed from dependency parsing

'''

import os
import sys
import operator
import random

from itertools import combinations 
from networkx.algorithms.shortest_paths.generic import all_shortest_paths as path_finder
from networkx.algorithms.shortest_paths.unweighted import single_source_shortest_path as single_source_path_finder
from gminer.graphs import DocumentWordGraph, Graphlet
import gminer.text_processing as text_utils
from nltk.corpus import reuters
from nltk.corpus import stopwords
from tqdm import tqdm
from os.path import isfile, join
from os import listdir 

stopwordlist = list(set(stopwords.words('english')))
stopwordlist.extend(['.',',',';',':','-','"'])

def extract_graph_paths(graph, source_nodes, method='simple', max_depth=5):
    graph_paths = []
    for source_node in source_nodes:
        paths = single_source_path_finder(graph, source_node)
        for target in paths.keys():
            if target in source_nodes:
                distance = len(paths[target])
                if distance < max_depth:
                    graph_paths.append((source_node, target, distance))
    return graph_paths

def extract_max_graphlets_within_single_graph(doc_graph, max_orbits):
    '''
    TODO: extract paths witin connected components only. networkx throws exception if either ends of a path are not within the same component
    '''
    # identify all simple paths in a graph
    
    source_nodes = doc_graph.find_nonstopword_JNV_tagged_nodes() # DocumentWordGraph.find_nonstopword_JNV_tagged_nodes(doc_graph)
    doc_simple_paths = extract_graph_paths(doc_graph, source_nodes, method='simple', max_depth=max_orbits)
    
    # processing paths
    graphlet_db = {}
    for (center, peripheral, distance) in doc_simple_paths:
        if center not in graphlet_db.keys() and center not in stopwordlist:
            graphlet_db[center] = Graphlet(center)
        # two-way paths
        if peripheral not in graphlet_db.keys() and peripheral not in stopwordlist:
            graphlet_db[peripheral] = Graphlet(peripheral)
        if center not in stopwordlist:
            graphlet_db[center].put_node_on_orbit(peripheral,distance-1)
        if peripheral not in stopwordlist:
            graphlet_db[peripheral].put_node_on_orbit(center,distance-1)
            
    return graphlet_db.values()

def select_candidate_source_orbit(n_orbits):
    ''' picks a random orbit number.
    parameters
    ----------
    n_orbits : int
        number of orbits.
    returns
    -------
    int
        a randomly selected number that designate an orbit.
    '''
    return random.choice(list(range(0,n_orbits))) 

def get_candidate_neigbors_on_next_orbit(orbit_source_nodes, all_graphlet_nodes, content_word_set, doc_graph):
    ''' explores neighbors of source nodes of a givne orbit in a graphlet. The search process exploits the graph data structure of the document.

    parameters
    ----------
    orbit_source_nodes : list
        a list of word nodes that serves as a starting point of the search.
    all_graphlet_nodes : list
        all nodes in a graphlet object. This helps avoid generating candidates that are already in the graphlet pattern.
    content_word_set: list
        in a graph, this contains only words of meaning or content.
    doc_graph: DocumentWordGraph
        a graph data structure that allows for navigation through nodes 
    returns
    -------
    list
        an array of candidate words.
    '''
    content_word_neighbors = []
    functional_word_neighbors = []
    for x in orbit_source_nodes:
        for y in doc_graph.neighbors(x):
            if y not in all_graphlet_nodes:
                if y in content_word_set: # not visited before
                    content_word_neighbors.append(y)
                else:
                    functional_word_neighbors.append(y)
    return [content_word_neighbors,functional_word_neighbors]

def filter_candidate_neighbors_by_word_freq(concept_neighbor_words_set, word_freq):
    ''' takes a set of candidate words and filters them down to only the most frequent ones
    parameters
    ----------
    concept_neighbor_words_set : list
        an array of content words (non-stop/non-functional words)
    word_freq : dict
        a map from words to frequency values

    returns
    -------
    list
        filtered list of words based on word frequency map
    '''
    concept_neighbor_word_freq = {}
    for w in concept_neighbor_words_set:
        if w not in concept_neighbor_word_freq.keys():
            concept_neighbor_word_freq[w] = 0
        if w in word_freq.keys():
            concept_neighbor_word_freq[w] = word_freq[w]
    concept_neighbor_word_freq = dict(list(reversed(sorted(concept_neighbor_word_freq.items(), key=lambda kv: kv[1])))[:5])
    return list(concept_neighbor_word_freq.keys())

def expand_graphlet_candidates(graphlet, word_graph, word_freq):
    ''' expands text graphlet object by putting more nodes on orbits. 
    The choice of orbits is done randomly. 
    The choice of nodes is based on bigram models (words appear next to any node on the randomly selected orbit)
    Candidate nodes are furhter filtered by word frequencies estimated from unigrams

    parameters
    ----------
    graphlet : Graphlet
        an object that represents the graphlet pattern we would like to expand
    word_graph : DocumentWordGraph(networkx.Graph)
        a graph data object constructed from bigrams of text document 
    word_freq : dict
        a map from word to freq

    returns
    -------
    list
        an array of graphlets generated by adding extra nodes (and occasionally orbits) to input source graphlet
    '''
    graphlet_next_gen = []
    source_orbit_random = select_candidate_source_orbit(graphlet.get_number_of_orbits())
    source_nodes = graphlet.get_nodes_on_orbit(source_orbit_random)
    concept_neighbor_words_set, functional_neighbor_words_set = get_candidate_neigbors_on_next_orbit(source_nodes, graphlet.get_all_nodes(), word_graph.get_content_word_nodes(), word_graph)
    # filter neigbors by freq
    concept_neighbor_words_set = filter_candidate_neighbors_by_word_freq(concept_neighbor_words_set,word_freq)
    if len(concept_neighbor_words_set) == 0 and len(functional_neighbor_words_set) == 0:
        return graphlet_next_gen
    ### grow one content word at a time
    if len(concept_neighbor_words_set) > 0:
        concept_neighbor_words_set = concept_neighbor_words_set[:1]
    ###
    for candidate_node in concept_neighbor_words_set:
        new_graphlet = graphlet.clone()
        if source_orbit_random == new_graphlet.get_number_of_orbits() - 1:
            new_graphlet.add_orbit()

        new_graphlet.put_nodelist_on_orbit(functional_neighbor_words_set, source_orbit_random + 1)
        new_graphlet.put_node_on_orbit(candidate_node, source_orbit_random + 1)
        graphlet_next_gen.append(new_graphlet)
    return graphlet_next_gen

def get_top_scoring_graphlets(hypotheses, pattern_freq, search_iter_min_freq, max_pruning_threshold):
    return [(graphlet_pattern, graphlet, graph_id) \
            for (graphlet_pattern, graphlet, graph_id) in hypotheses \
                if graphlet_pattern == '' or graphlet_pattern in pattern_freq.keys() \
                and  pattern_freq[graphlet_pattern] > search_iter_min_freq \
        ][:max_pruning_threshold]

def extract_graphlets(doc_collection, params):
    '''
    Extracts text graphlet patterns within a collection of teext documents.
    The method first maps text document collection to a list of DocumentWordGraph objects. Then, graphlet patterns are extracted within Graph collections.

    parameters
    ----------
    doc_collection : list
        a list of string, each element represent string text of a document in the corpus/collection.
    word_freq : dict
        a map from word to freq
    min_freq: int
        minimum frequency of a word node (node degree) before it can be included as a center/neuclus node within th graphlet pattern.
    max_orbits: int
        maximum number of orbits a given graphlet pattern can have
    graphlet_type: str
        belongs to two types: "pruned" and "max". "pruned" type contains selected number of nodes on each orbit and also the number of orbits can vary.
        On the otherhand, "max" graphlets can include maximum reachable nodes and orbits.

    returns
    -------
    list
        an array of graphlets patterns.
    '''    

    ''' Method level constants '''
    
    max_orbits = params['MAX_ORBIT_CAPACITY']
    graphlet_type = params['GRAPHLET_TYPE']
    pattern_freq_threshold_by_stack = params['PATTERN_FREQ_THRESHOLD_BY_STACK']
    MAX_SEARCH_ITERATIONS = params['MAX_SEARCH_ITERATIONS']
    pruned_stack_size = params['PRUNED_STACK_SIZE'] 
    MIN_WORD_FREQ = params['MIN_WORD_FREQ'] 
    MAX_SEARCH_ITERATIONS = params['MAX_SEARCH_ITERATIONS']
    WORD_SELECTION_RATIO = params['WORD_SELECTION_RATIO']
    CONTENT_WORD_REGEX_PATTERN = params['CONTENT_WORD_REGEX_PATTERN']
    STOPWORD_LIST = params['STOPWORD_LIST']

    #MIN_WORD_FREQ = 30
    #MAX_SEARCH_ITERATIONS = 7
    #WORD_SELECTION_RATIO = 0.1
    search_iteration = 0

    ''' Data structures initializations '''
    graph_db = {}
    explored_hypothesis = []
    docids = list(doc_collection.keys())
    graphlet_search_stack = {}
    graphlet_search_stack[search_iteration] = []
    pattern_freq = {}
    word_patterns = {}
    
    ''' Initializations steps '''

    ''' Creating word frequency map '''
    print("Generating most freq tagged word map...")
    word_freq = text_utils.get_word_frequencies(doc_collection, WORD_SELECTION_RATIO)
    print('Number of keys in the filtered freq word map is ' + str(len(word_freq.keys())))
    print("Done.")

    ''' Initializing search stack with a list of seed graphlets with only one node at the center '''
    print('Initializing search space with seed graphlets with one word at the center...')
    graphlet_search_stack[0] = []
    for i in tqdm(range(len(docids))):
        word_graph = DocumentWordGraph(docids[i], doc_collection[docids[i]], source_type='text', stopwords=STOPWORD_LIST, content_word_pattern = CONTENT_WORD_REGEX_PATTERN)
        graph_db[word_graph.get_id()] = word_graph
        for word in word_graph.get_content_word_nodes():
            if word in word_freq.keys() and word_freq[word] > MIN_WORD_FREQ:
                graphlet_search_stack[0].append(('',Graphlet(word),word_graph.get_id()))# Adding null patterns '' as seeds
    print('Done.')
    ''' Performing graphlet search here ... '''
    ''' A search stack indexed by iterations. In each iteration, graphlets are expanded into next iteration stack    '''
    print("Starting Stack-based search for graphlet patterns...")
    for search_iteration in range(MAX_SEARCH_ITERATIONS):
        print("Starting search iter # {0}".format(search_iteration))
        if search_iteration == 2:
            x2=1
        graphlet_search_stack[search_iteration+1] = []
        hypotheses = graphlet_search_stack[search_iteration]
        batch = 0
        for h in tqdm(range(len(hypotheses))):
            (graphlet_pattern_key, graphlet, graph_id) = hypotheses[h]
            # retrieve graph record from db to start search             
            graph_obj = graph_db[graph_id]     
            #if batch % 10000 == 0:
            #    print(str(batch) + " states searched ..")  
            #batch += 1
            # expand graphlet by searching for neighboring nodes via graph_obj 
            expanded_graphlet = expand_graphlet_candidates(graphlet, graph_obj, word_freq)
            for graphlet_item in expanded_graphlet:
                graphlet_str_representation = graphlet_item.get_pattern_representation(graph_obj.get_content_word_nodes()) #str(graphlet_item)
                graphlet_graph_lookup_key = graph_id + "_" + graphlet_str_representation
                if graphlet_graph_lookup_key in explored_hypothesis:
                    continue
                explored_hypothesis.append(graphlet_graph_lookup_key)
                graphlet_pattern_key = '|'.join(graphlet_str_representation.split('|')[1:])
                pattern_freq[graphlet_pattern_key] = pattern_freq.get(graphlet_pattern_key,0) + 1
                if graphlet_pattern_key not in word_patterns.keys(): word_patterns[graphlet_pattern_key] = []
                word_patterns[graphlet_pattern_key].append( (graphlet_item.get_center_node(), graph_id) )
                graphlet_search_stack[search_iteration+1].append((graphlet_pattern_key, graphlet_item, graph_id))

        ''' Pruning search space by removing low scoring graphlets from next stack search iteration...'''
        #print("Number of candidate patterns for next search iteration = " + str(len(graphlet_search_stack[search_iteration+1])))
        freq_pruning_threshold = pattern_freq_threshold_by_stack.get(search_iteration,5)
        graphlet_search_stack[search_iteration+1] = get_top_scoring_graphlets(graphlet_search_stack[search_iteration+1], pattern_freq, freq_pruning_threshold, pruned_stack_size)
    print("Done.")
    return word_patterns
