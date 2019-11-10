'''

A module for input and output functionality, like saving patterns to files

'''

import os
from os.path import isfile, join
from os import listdir 

def save_graphlets_to_file(base_dir, filename, graph_list):
    filepath = join(base_dir, filename)
    with open(filepath,'w') as file_handler:
        file_handler.write('\n'.join(graph_list))
