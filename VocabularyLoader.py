import torch 
from torch.autograd import Variable
import json
import pdb
import numpy as np
from gensim.models import word2vec
from gensim.models import fasttext


class VocabularyLoader_char():
    def __init__(self, filename, device):
        self.character_table = {}
        self.index2char = {}
        self.n_chars = 0
        self.device = device
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                for w in line:
                    if w not in self.character_table:
                        self.character_table[w] = self.n_chars
                        self.index2char[self.n_chars] = w
                        self.n_chars += 1
        # print(self.n_chars)

    # Turn string into list of longs
    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor[c] = self.character_table[string[c]]
            except Exception as e:
                # pdb.set_trace()
                #print(string[c])
                print(e)
        return Variable(tensor).to(self.device)


class VocabularyLoader_token():
    def __init__(self, filename, device):
        self.token_table = {}
        self.index2token = {}
        self.n_tokens = 0
        self.device = device
        f = open(filename, 'r', encoding='UTF-8')
        for lines in f:
            ls = lines.replace('\n', ' ').replace('\t', ' ')
            token_lists = ls.split(' ')
            token_lists = [i for i in token_lists if (len(str(i))) != 0]
            for i in token_lists:
                if i is not '':
                    if i not in self.token_table:
                        self.token_table[i] = self.n_tokens
                        self.index2token[self.n_tokens] = i
                        self.n_tokens += 1
        self.token_table['UNKNOWN'] = self.n_tokens
        self.index2token[self.n_tokens] = 'UNKNOWN'
        self.n_tokens+=1
        # print(self.n_tokens)

    # Turn tokens into list of longs
    def token_tensor(self, tokens):
        tensor = torch.zeros(len(tokens)).long()
        for c in range(len(tokens)):
            try:
                tensor[c] = self.token_table[tokens[c]]
            except Exception as e:
                pdb.set_trace()

        return Variable(tensor).to(self.device)

class VocabularyLoader_ast():
    def __init__(self, filename, ast_file, device):
        self.node_table = {}
        self.index2char = {}
        self.n_chars_ast = 0
        self.device = device

        self.token_table = {}
        self.index2token = {}
        self.n_tokens = 0

        with open(ast_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                for w in line:
                    if w not in self.node_table:
                        self.node_table[w] = self.n_chars_ast
                        self.index2char[self.n_chars_ast] = w
                        self.n_chars_ast += 1
                        # print(self.n_chars)

        f = open(filename, 'r', encoding='UTF-8')
        for lines in f:
            ls = lines.replace('\n', ' ').replace('\t', ' ')
            token_lists = ls.split(' ')
            token_lists = [i for i in token_lists if (len(str(i))) != 0]
            for i in token_lists:
                if i is not '':
                    if i not in self.token_table:
                        self.token_table[i] = self.n_tokens
                        self.index2token[self.n_tokens] = i
                        self.n_tokens += 1
        self.token_table['UNKNOWN'] = self.n_tokens
        self.index2token[self.n_tokens] = 'UNKNOWN'
        self.n_tokens += 1

    # Turn string into list of longs
    def char_tensor(self, path):

        path = path.split()

        #tensor = torch.zeros(self.n_chars_ast).long()
        #tensor = torch.zeros(len(path))
        pathlist = [0] * self.n_chars_ast
        for c in range(len(path)):
            try:
                #tensor[self.node_table[path[c]]] += 1
                #tensor[c] = self.node_table[path[c]]
                #pathlist.append(self.node_table[path[c]])
                pathlist[self.node_table[path[c]]] += 1

            except Exception as e:
                pdb.set_trace()
                #print(path[c])
        return pathlist
        #Variable(tensor).to(self.device)

    def token_tensor(self, tokens):
        tensor = torch.zeros(len(tokens)).long()
        for c in range(len(tokens)):
            try:
                tensor[c] = self.token_table[tokens[c]]
            except Exception as e:
                pdb.set_trace()
        return Variable(tensor).to(self.device)

class VocabularyLoader_newast():
    def __init__(self, filename, ast_newfile, astdim, device):
        self.newastnode_table = {}
        self.index2newast = {}
        self.n_token_newast = 0
        self.device = device
        self.ast_file = ast_newfile
        self.astdim = astdim

        self.token_table = {}
        self.index2token = {}
        self.n_tokens = 0

        with open(ast_newfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                for w in line:
                    if w not in self.newastnode_table:
                        self.newastnode_table[w] = self.n_token_newast
                        self.index2newast[self.n_token_newast] = w
                        self.n_token_newast += 1
                        # print(self.n_chars)

        f = open(filename, 'r', encoding='UTF-8')
        for lines in f:
            ls = lines.replace('\n', ' ').replace('\t', ' ')
            token_lists = ls.split(' ')
            token_lists = [i for i in token_lists if (len(str(i))) != 0]
            for i in token_lists:
                if i is not '':
                    if i not in self.token_table:
                        self.token_table[i] = self.n_tokens
                        self.index2token[self.n_tokens] = i
                        self.n_tokens += 1
        self.token_table['UNKNOWN'] = self.n_tokens
        self.index2token[self.n_tokens] = 'UNKNOWN'
        self.n_tokens += 1

    def token_tensor(self, tokens):
        tensor = torch.zeros(len(tokens)).long()
        for c in range(len(tokens)):
            try:
                tensor[c] = self.token_table[tokens[c]]
            except Exception as e:
                pdb.set_trace()
        return Variable(tensor).to(self.device)

    def word2vec(self):
        sentences = word2vec.LineSentence(self.ast_file)
        model = fasttext.FastText(sentences, size=self.astdim, window=3, min_count=1, iter=10, min_n=3, max_n=6, word_ngrams=0, max_vocab_size=933)
        #model = word2vec.Word2Vec(sentences, size=self.astdim)
        model.save(u"ast.model")
        return model

    def ast_path_tensor(self, path, model):
        pathtensor = [0] * self.astdim
        for p in path.split():
            pathtensor = torch.from_numpy(model[p]) + torch.tensor(pathtensor, dtype=torch.float)

        return Variable(torch.tensor(pathtensor)).to(self.device)


