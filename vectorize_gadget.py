#ref: https://github.com/johnb110/VDPython/blob/master/vectorize_gadget.py

import re

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
from gensim.models import KeyedVectors as word2vec
import numpy

# Sets for operators
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':' , ';',
    '{', '}'
    }

"""
Functionality to train Word2Vec model and vectorize gadgets
Buffers list of tokenized gadgets in memory
Trains Word2Vec model using list of tokenized gadgets
Uses trained model embeddings to create 2D gadget vectors
"""
class GadgetVectorizer:

    def __init__(self, vector_length):
        self.gadgets = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0



    """
    Tokenize entire gadget
    Tokenize each line and concatenate to one long list
    """
    @staticmethod
    def tokenize_gadget(gadget):
        tokenized = []
        function_regex = re.compile('FUN(\d)+')
        backwards_slice = False
        for line in gadget:
            tokens = GadgetVectorizer.tokenize(line)
            tokenized += tokens
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            else:
                backwards_slice = False
        return tokenized, backwards_slice

    """
    Add input gadget to model
    Tokenize gadget and buffer it to list
    """
    def add_gadget(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(gadget)
        self.gadgets.append(tokenized_gadget)
        if backwards_slice:
            self.backward_slices += 1
        else:
            self.forward_slices += 1

    """
    Uses Word2Vec to create a vector for each gadget
    Gets a vector for the gadget by combining token embeddings
    Number of tokens used is min of number_of_tokens and 50
    """
    def vectorize(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(gadget)
        vectors = numpy.zeros(shape=(50, self.vector_length))
        if backwards_slice:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[50 - 1 - i] = self.embeddings[tokenized_gadget[len(tokenized_gadget) - 1 - i]]
        else:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[i] = self.embeddings[tokenized_gadget[i]]
        return vectors

    """
    Done adding gadgets, now train Word2Vec model
    Only keep list of embeddings, delete model and list of gadgets
    """
    def train_model(self):
        # Set min_count to 1 to prevent out-of-vocabulary errors
        # word2vec 4 api
        # model = Word2Vec(self.gadgets, min_count=1, vector_size=self.vector_length, sg=1)
        # vectors_text_path = "gensim-model/java14_model/saved_model_iter8.release.data-00000-of-00001"
        vectors_text_path = "gensim-model/token_vecs.txt"
        print(vectors_text_path)
        # model = Word2Vec.load(vectors_text_path)
        model = word2vec.load_word2vec_format(vectors_text_path, binary=False)
        self.embeddings = model.wv
        del model
        del self.gadgets