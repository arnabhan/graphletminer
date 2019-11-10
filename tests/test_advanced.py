# -*- coding: utf-8 -*-

#from .context import sample

import unittest
import nltk
from nltk.corpus import movie_reviews, reuters, brown, inaugural
from gminer.algorithms import extract_graphlets
from gminer.text_processing import get_word_frequencies

class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_pattern_extract(self):
        # settings and constants 
        GRAPHLET_TYPE = 'pruned' #'pruned' #'max'
        MIN_FREQ = 20
        MAX_ORBITS = 5
        MAX_NODES = 50
        MIN_NODES = 3
        MAX_SEARCH_ITERATIONS = 7
        PRUNED_STACK_SIZE = 500
        PATTERN_FREQ_THRESHOLD_BY_STACK = {0:10,1:5,2:5,3:5,4:3}

        # loading a portion of movive_reviews corpus for testing
        doc_collection = dict([ (id,movie_reviews.raw(id)) for id in movie_reviews.fileids() [:300] ] )

        # pattern extraction
        word_patterns = extract_graphlets(doc_collection, MIN_FREQ, MAX_ORBITS, GRAPHLET_TYPE, MAX_SEARCH_ITERATIONS, PATTERN_FREQ_THRESHOLD_BY_STACK, PRUNED_STACK_SIZE)
        n = len(word_patterns.keys())
        print("Number of patterns = " + str(n))
        self.assertIsNotNone(word_patterns.keys())
        self.assertNotEqual(n,0)

if __name__ == '__main__':
    unittest.main()
