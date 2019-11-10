import sys
sys.path.insert(0, "E:\Projects\Research\graphlet-miner")

import sys
from gminer.algorithms import extract_graphlets_within_text_corpus
import argparse
import fileio

if __name__ == '__main__':

    print('----- Graphlet miner - A graphlet based keyword extraction -----')
    '''
    Add arg_parse utils here ...
    At minimum, args should include:
    1) base directory to load files for processing
    2) min_freq of graphlet pattern
    3) output file path
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, help='root directory of input text documents')
    parser.add_argument('--min_freq', type=int, help='minimum support for graphet pattern frequency', default=5)
    parser.add_argument('--output_dir', type=str, help='root directory of output patterns')
    parser.add_argument('--output_file_prefix', type=str, help='prefix of output files')
    parser.add_argument('--min_orbits', type=int, help='minimum number of orbits', default=2)
    parser.add_argument('--max_orbits', type=int, help='max number of orbits', default=4)
    parser.add_argument('--use_edge_weights', type=bool, help='use edge weights while search for shortest paths', default=4)
    args = parser.parse_args() 

    norbits = 5 #args.max_orbits
    freq = 5 #args.min_freq
    root_dir = "E:\\Projects\\Research\\Data\\Corpora\\Reuters" #args.base_dir
    gtype = 'pruned' #'max' # 'pruned'
    out_dir = "E:\\Projects\\Research\\Data\\analysis"
    output_file_prefix = 'wordgraph_reuters.txt'

    graphlet_map = extract_graphlets_within_text_corpus(base_dir=root_dir, min_freq=freq, max_orbits =norbits,graphlet_type=gtype)
    print("Printing graphlets with freq > {0}".format(freq))
    graphletset = []
    for key in graphlet_map.keys():
           print(key + "==>" + str(graphlet_map[key] ))
           graphletset.append(key + '\t' + str(graphlet_map[key]))
    fileio.save_graphlets_to_file(out_dir, output_file_prefix,graphletset )