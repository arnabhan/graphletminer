import copy
import re
import networkx

from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize
from nltk import bigrams, word_tokenize, pos_tag 
from nltk.corpus import stopwords
from os import listdir 
from os.path import isfile, join

from gminer.text_processing import get_bigrams, get_freq_weighted_bigrams

stopwordlist = list(set(stopwords.words('english')))

'''
This module provide functionality for constructing word graphs from a text document.
First supported type is PlainWordGraph, which is constructed from bigrams of input text documents
Second type is the DependencyWordGraph, that is constructed from dependency parsing of input sentences in a document
'''

class Graphlet(object):
    ''' An abstract data type that represents a graphlet object. 
    Graphlets are represented as a node at the center and a number of orbits 
    around that center. Links inter or intra orbits are not represented 
    explicitly here. All that is required is that a node in an orbit has a 
    source link from a node in the nearest neighbor inner orbit.     
    '''
    def __init__(self, center_node):     
        ''' initializes a graphlet object with one node at the center

        parameters
        ----------
        center_node : str
            a string label of the center node, typically a dictionary word or a word_tag string

        returns
        -------
        
        '''
        self.orbit_nodes = {}
        self.graphlet_nodes = [center_node] 
        self.orbit_nodes[0] = [center_node]
        
    def add_orbit(self):
        ''' creates a new orbit

        parameters
        ----------

        returns
        -------
        
        '''        
        next = max(list(self.orbit_nodes.keys())) + 1
        self.orbit_nodes[next] = []

    def put_node_on_orbit(self, node_name, orbit_id):
        ''' places a graph node with label node_name on the orbit specified by identifier orbit_id

        parameters
        ----------
        node_name : str
            a string label of the graph node, typically a dictionary word or a word_tag string

        orbit_id : int
            a number specifying an orbit in the graphlet

        returns
        -------
        
        '''
        if orbit_id not in self.orbit_nodes.keys():
            self.orbit_nodes[orbit_id] = []
        if node_name not in self.orbit_nodes[orbit_id]:
            self.orbit_nodes[orbit_id].append(node_name)
        self.graphlet_nodes.append(node_name)

    def put_nodelist_on_orbit(self, node_name_list, orbit_id):
        ''' places a list of graph nodes on the orbit specified by identifier orbit_id

        parameters
        ----------
        node_name_list : list of str
            a list of a string labels of the graph nodes, typically a list of word_tag string

        orbit_id : int
            a number specifying an orbit in the graphlet

        returns
        -------
        
        '''
        for node_name in node_name_list:
            if orbit_id not in self.orbit_nodes.keys():
                self.orbit_nodes[orbit_id] = []
            if node_name not in self.orbit_nodes[orbit_id]:
                self.orbit_nodes[orbit_id].append(node_name)
            self.graphlet_nodes.append(node_name)

    def put_node_on_outer_orbit(self, node_name):
        ''' places a node on the outermost orbit in the graphlet

        parameters
        ----------
        node_name : str
            a string label of the graph node that will be added to the graphlet, typically a list of word_tag string

        returns
        -------
        
        '''
        outer_orbit_id = max(list(self.orbit_nodes.keys()))
        self.put_node_on_orbit(node_name, outer_orbit_id)

    def put_nodelist_on_outer_orbit(self, node_name_list):
        ''' places a list of graph nodes on the outermost orbit in the graphlet

        parameters
        ----------
        node_name_list : list of str
            a list of a string labels of the graph nodes, typically a list of word_tag string

        returns
        -------
        
        '''
        outer_orbit_id = max(list(self.orbit_nodes.keys()))
        self.put_nodelist_on_orbit(node_name_list, outer_orbit_id)

    def get_nodes_on_orbit(self, orbit_id):
        ''' returns the list of nodes on the given graphlet orbit

        parameters
        ----------
        orbit_id : int
            a number specifying an orbit in the graphlet

        returns
        -------
        list, array of nodes on the given graphlet orbit
        '''
        assert(orbit_id in self.orbit_nodes.keys())
        return self.orbit_nodes[orbit_id]
    
    def get_nodes_on_outer_orbit(self):
        ''' returns the list of nodes on the outermost graphlet orbit

        parameters
        ----------

        returns
        -------
        list, array of nodes on the given graphlet's outermost orbit
        '''
        return self.orbit_nodes[max(list(self.orbit_nodes.keys()))]

    def get_number_of_orbits(self):
        ''' returns the number of orbits

        parameters
        ----------

        returns
        -------
        int, number of orbits
        '''
        return len(list(self.orbit_nodes.keys()))

    def get_all_nodes(self):
        ''' returns the list of every nodes on all orbits

        parameters
        ----------

        returns
        -------
        list, array of nodes
        '''
        return self.graphlet_nodes

    def get_center_node(self):
        ''' returns center node label

        parameters
        ----------

        returns
        -------
        str, node name/label
        '''
        return self.orbit_nodes[0][0]
        
    def get_size(self):
        ''' returns size of graphlet. Size is the number of nodes across on all orbits

        parameters
        ----------

        returns
        -------
        int, number of all nodes a graphlet has
        '''
        return len(self.get_all_nodes())

    def clone(self):
        ''' returns a deep copy of this graphlet. This will copy all orbits and
        all nodes.
        This method is used during expansion of state space search. For example,
        after cloning a graphlet, the new cloned object can be expanded by 
        adding a node on an orbit randomly chosen.

        parameters
        ----------

        returns
        -------
        Graphlet, a deep copy of this graphlet object
        '''
        return copy.deepcopy(self)

    def get_pattern_representation(self, content_word_list):
        ''' returns a string representation of graphlet.
        
        parameters
        ----------
        content_word_list : list of str
            a list of content words.

        returns
        -------
        str, text representation 
        
        Content words must be provided from the parent graph obj. 
        Graphlet pattern will then be all content words, plus a masked 
        representation of non-content words as <FUNC_OR_STOP_WORD> string 
        literal. 
        
        The pattern representation exclude the center node, it only focuses
        on the "context" representated as nodes on orbits floating around the
        center node. 

        Pattern Format:
        <orbit_1>:{<content_node>|<FUNC_OR_STOP_WORD>}+|<orbit_2>:{<content_node>|<FUNC_OR_STOP_WORD>}+|...

        the nodes on orbits are ordered lexicographically in that string represenation.

        '''        
        n_occuppied_orbits = len(self.orbit_nodes.keys())
        orbit_to_node_map = {}
        for orbit_index in range(0,n_occuppied_orbits):
            if orbit_index not in self.orbit_nodes.keys():
                orbit_to_node_map[orbit_index] = ['<EMPTY_ORBIT>']
            else:
                orbit_to_node_map[orbit_index] = []
                for node_id in self.orbit_nodes[orbit_index]:
                    if node_id in content_word_list:
                        orbit_to_node_map[orbit_index].append(node_id)
                    else:
                        orbit_to_node_map[orbit_index].append("<FUNC_OR_STOP_WORD>")   
                orbit_to_node_map[orbit_index].sort()
        return '|'.join([ str(node_id) + ":" + ';'.join(list(set(orbit_to_node_map[node_id]))) for node_id in range(0,n_occuppied_orbits)])

class DocumentWordGraph(networkx.Graph):
    ''' An abstract data type that represents an undirected graph object that 
    extends networkx.Graph. 
    DocumentWordGraph has nodes represented by words in a text document and 
    edges exist between adjacent words. Typically edges are drawn between 
    bigrams extracted witin a text document.
    DocumentWordGraph is the main object that is used for graphlet pattern 
    search. The grpahlet mining algorithm leverage graph structue and list of
    content words list to prune search space tree. Content words are specified
    as regular expressions.
    '''
    def __init__(self, graph_instance_id, input_source, source_type='file', stopwords=[], content_word_pattern='^[A-z0-9]{3,}$'):
        ''' initializes a DocumentWordGraph object
        The graph can be constructed by passing a text passage directly or via
        a path to a file that contains the text content.

        parameters
        ----------
        graph_instance_id : str
            a string identifier of the graph. This is ideally a document name
        input_source : str
            a string that either contains full text or path to file that contains text
        source_type : str
            values are 'file' or 'text'
        content_word_pattern : str
            a regex that specify what can be considered content word. 
            Content word pattern defines what is a content and what is otherwise stopword
            If part of speech tagging is used, this can be used to match nouns, adj, and verbs

        returns
        -------
        a graph object
        '''
        super().__init__()
        self.graph_id = graph_instance_id
        if source_type == 'file':
            with open(input_source,'r') as file_handler:
                text_blob = file_handler.read()
        else:
            text_blob = input_source
        bgram = get_bigrams(text_blob) #get_bigrams(text_blob,use_pos_tagging) # needs to be pushed up to the use app (users should have control over whether to supply a tagged or raw text)
        wbgram = get_freq_weighted_bigrams(bgram)        
        self.add_edges_from(list(wbgram.keys()))
        self.content_word_list = [word for word in self.nodes if re.match(content_word_pattern, word) and ( len(stopwords) == 0 or not word in stopwords ) ] # self.find_nonstopword_JNV_tagged_nodes()

    def get_id(self):
        ''' returns graph id

        parameters
        ----------

        returns
        -------
        str, graph id
        '''        
        return self.graph_id

    def get_content_word_nodes(self):
        ''' returns list of content words in a word graph

        parameters
        ----------

        returns
        -------
        list, content word list
        '''
        return self.content_word_list

'''
Place holder for Dependency Parsing based graphs
'''
