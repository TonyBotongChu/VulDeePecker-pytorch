import numpy
import re

from transformers import AutoTokenizer, AutoModel

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

class BERTVectorizer():
    def __init__(self, vector_length):
        self.gadgets = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0
        self.tokenizer = AutoTokenizer.from_pretrained("codebert-base")
        self.tokenizer.add_tokens(['VAR'])

    """
    Takes a line of C++ code (string) as input
    Tokenizes C++ code (breaks down into identifier, variables, keywords, operators)
    Returns a list of tokens, preserving order in which they appear
    """
    @staticmethod
    def tokenize(line):
        tmp, w = [], []
        i = 0
        while i < len(line):
            # Ignore spaces and combine previously collected chars to form words
            if line[i] == ' ':
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # Check operators and append to final list
            elif line[i:i+3] in operators3:
                tmp.append(''.join(w))
                tmp.append(line[i:i+3])
                w = []
                i += 3
            elif line[i:i+2] in operators2:
                tmp.append(''.join(w))
                tmp.append(line[i:i+2])
                w = []
                i += 2
            elif line[i] in operators1:
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # Character appended to word list
            else:
                w.append(line[i])
                i += 1
        # Filter out irrelevant strings
        res = list(filter(lambda c: c != '', tmp))
        return list(filter(lambda c: c != ' ', res))


    def vectorize(self, gadget):
        tokenized = []
        function_regex = re.compile('FUN(\d)+')
        backwards_slice = False
        for line in gadget:
            tokens = self.tokenize(line)
            tokenized += tokens
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            else:
                backwards_slice = False
        # tokenized_gadget, backwards_slice = self.tokenize_gadget(gadget)
        vectors = numpy.zeros(shape=(50, self.vector_length))
        tokenized_gadget = []
        for g in gadget:
            tokenized_gadget+=self.tokenizer.tokenize(g)
        bert_vec = self.tokenizer.convert_tokens_to_ids(tokenized_gadget)
        if backwards_slice:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[50 - 1 - i] = bert_vec[len(bert_vec) - 1 - i]
                    # self.embeddings[tokenized_gadget[len(tokenized_gadget) - 1 - i]]
        else:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[i] = bert_vec[i]
                    # self.embeddings[tokenized_gadget[i]]
        return vectors
