import argparse as ap
import pandas as pd
import numpy as np
import os
import pickle 
from tqdm import tqdm
import collections
from collections import Counter
import importlib
from example import read_corpus, read_wikipedia_corpus


import array
import io
import scipy.sparse as sp
import numbers

from glove import Corpus, Glove

def main(args):

    ############################
    # corpus_model = Corpus()
    # corpus_model.fit(read_corpus(args.corpus))
    # corpus_model.save('corpus_select.model')

    ############################
    # corpus_model = Corpus().load('corpus_select.model')
    # print('Dict size: %s' % len(corpus_model.dictionary))
    # print('Collocations: %s' % corpus_model.matrix.nnz)

    # with open('global_vocab.pkl', 'wb') as handle:
    #     pickle.dump(corpus_model.dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ############################

    # opening vocab to create the corpus object
    with open('global_vocab.pkl', 'rb') as f:
        vocab_dict = pickle.load(f)
    doc_model = Corpus(dictionary=vocab_dict)
    
    texts = list(read_corpus(args.corpus))
    
    #opening weight csv
    diff_bias = pd.read_csv(args.diff_bias, header=0)

    
    #col 2 is science/arts, col 3 is weapons/instruments

    total = {}
    # for i in range(10):
    for i in tqdm(range(len(texts))):
        doc = [texts[i]]
        doc_model.fit(doc)

        # we might not even need to save it, just put it into one matrix and save that

        coo = doc_model.matrix.todok()
        weight = diff_bias.iloc[i, 2]
        coo = {k:weight*v for k,v in coo.items()}
        total = Counter(coo) + Counter(total)


    def _dict_to_csr(term_dict):
        term_dict_v = term_dict.values()
        term_dict_k = term_dict.keys()
        term_dict_k_zip = zip(*term_dict_k)
        term_dict_k_zip_list = list(term_dict_k_zip)

        shape = (len(term_dict_k_zip_list[0]), len(term_dict_k_zip_list[1]))
        csr = sp.csr_matrix((list(term_dict_v), list(map(list, zip(*term_dict_k)))), shape = shape)
        coo = csr.tocoo()
        return coo

    total = dict(total)
    total = _dict_to_csr(total)
    print(total.get_shape())
    
    with open('doc_matrices_weighted.pkl', 'wb') as handle:
        pickle.dump(total, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # this will save the dict and the matrix, I kind of only need the matrix
        # with open(f'../coo_matrices/cooc_{i+1}.pkl', 'wb') as handle:
        #     pickle.dump(doc_model.matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
     ############################

    # with open('doc_matrices_weighted.pkl', 'rb') as f:
    #     bias_matrix = pickle.load(f)
    
    
    # print(bias_matrix.get_shape())
    # corpus_model = Corpus.load('corpus_select.model')
    # print(corpus_model.matrix.get_shape())

    # glove = Glove(no_components=100, learning_rate=0.05)
  
    # glove.fit(matrix=corpus_model.matrix, bias_matrix=bias_matrix, epochs=1,
    #             no_threads=2, verbose=True)
    # glove.add_dictionary(corpus_model.dictionary)
    # glove.save('glove_baseline.model')

    
    
       
if __name__=="__main__":

    # Set up command line parameters.
    parser = ap.ArgumentParser(description='Fit a GloVe model.')

    parser.add_argument('--vocab', '-v',
                        default="../../understanding-bias/embeddings/vocab-C0-V20.txt",
                        help=('The path to the vocab file for the whole corpus.'))
    parser.add_argument('--output_dir', '-o',
                        default="../coo_matrices",
                        help=('the path to the directory in which to store \
                            the resulting co-occurence matrix'))
    parser.add_argument('--corpus', '-c', action='store',
                        default="../../understanding-bias/corpora/simplewikiselect.txt",
                        help=('path to the raw text corpus'))
    parser.add_argument('--diff_bias', action='store',
                        default="../../understanding-bias/results/diff_bias/diff_bias-C0-V20-W8-D25-R0.05-E15-S1.csv",
                        help=('path to the raw text corpus'))
                    
    # parser.add_argument('--parallelism', '-p', action='store',
    #                     default=1,
    #                     help=('Number of parallel threads to use for training'))
    # parser.add_argument('--query', '-q', action='store',
    #                     default='',
    #                     help='Get closes words to this word.')
    args = parser.parse_args()

    main(args)

# ~/explicit-bias/understanding-bias/corpora/simplewikiselect.txt