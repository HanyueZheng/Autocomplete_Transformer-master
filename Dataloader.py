from VocabularyLoader import VocabularyLoader_char,VocabularyLoader_token, VocabularyLoader_ast
import random
import torchsnooper
import numpy as np 
import torch
from torch.autograd import Variable
import os
import torch.nn as nn


class DataLoader_char():
    def __init__(self, filename, chunk_len, device):
        with open(filename,'r',encoding='UTF-8') as f:
            lines=f.readlines()
        self.content = "".join(lines)
        self.file_len = len(self.content)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_char(filename, self.device)

    def next_chunk(self):
        chunk = self.__random_chunk()
        input = chunk[:-1]
        target = chunk[1:]
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.char_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.char_tensor(self.content[start_index:end_index])


class DataLoader_token():
    def __init__(self, filename, chunk_len, device):
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        self.content = "".join(lines)
        self.token_list = self.content.replace('\n', ' ').replace('\t', ' ').split(' ')
        self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
        self.file_len = len(self.token_list)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_token(filename, self.device)


    # 1/1,2/1,3/1,4/1...n/1 输入形式
    # def next_chunk(self):
    #     chunk = self.__random_chunk()
    #     input_target_pair = []
    #     for i in range(1,self.chunk_len):
    #         input = torch.zeros(self.chunk_len-1).long()
    #         for j in range(self.chunk_len-i-1):
    #             input[j]=self.vocabularyLoader.n_tokens-1
    #         input[-i:] = chunk[:i]
    #         target = chunk[i-1:i+1]
    #         input=input.to(self.device)
    #         target=target.to(self.device)
    #         input_target_pair.append((input,target))
    #     return input_target_pair

    def next_chunk(self):
        chunk = self.__random_chunk()
        input = chunk[:-1]
        target = chunk[1:]
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.token_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index])


class DataLoader_token_kg():
    def __init__(self, filename, kg, chunk_len, device):
        self.kg = kg
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        self.content = "".join(lines)
        self.token_list = self.content.replace('\n', ' ').replace('\t', ' ').split(' ')
        self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
        self.file_len = len(self.token_list)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_token(filename, self.device)

    def next_chunk(self):
        # TODO : add [UNK]
        chunk, content = self.__random_chunk()
        ents_list = []
        ents = [24] * self.chunk_len  # UNK = 24 to be modified
        contents = ""
        for i in range(len(content)):
            contents = contents + content[i] + " "
        # TODO modify
        for i in range(len(self.kg)):
            if contents.find(" " + self.kg[i] + " ") != -1:
                ents_list.append(self.kg[i])
        for i in range(len(ents_list)):
            key = ents_list[i].strip().split()
            if content.index(key[0]) >= 0:
                ents[content.index(key[0])] = self.kg.index(" ".join(key))
        ents = torch.Tensor(ents).long()
        input_target_pair = []
        for i in range(1, self.chunk_len):
            input = torch.zeros(self.chunk_len - 1).long()
            # length of ent?
            ent = torch.zeros(self.chunk_len - 1).long()
            for j in range(self.chunk_len - i - 1):
                input[j] = self.vocabularyLoader.n_tokens - 1
            input[-i:] = chunk[:i]
            target = chunk[i - 1:i + 1]
            ent[-i:] = ents[:i]
            input = input.to(self.device)
            target = target.to(self.device)
            ent = ent.to(self.device)
            input_target_pair.append((input, ent, target))
        return input_target_pair

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.token_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index]), \
                   self.token_list[start_index:end_index]

class DataLoader_token_ast():
    def __init__(self, filename, kg, chunk_len, device, ast_file, ast_token_num):
        self.kg = kg
        self.astfile = ast_file
        self.ast_token_num = ast_token_num
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        self.content = "".join(lines)
        self.token_list = self.content.replace('\n', ' ').replace('\t', ' ').split(' ')
        self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
        self.file_len = len(self.token_list)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_ast(filename, ast_file, self.device)

    def find_path(self, filename):
        pathdic = {}
        key = ""
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line != ""):
                    if line[0] == "[":
                        pathdic[line] = []
                        key = line
                    else:
                        pathdic[key].append(line)
                #line = f.readline()
                #print(line)
            #print(pathdic)
        #print(pathdic)
        return pathdic

    def path_encode(self, pathdic):
        path_encodedic = {}
        for k in pathdic.keys():

            path_encodedic[k] = []
            path = pathdic[k]

            for p in path:
                pencode = self.vocabularyLoader.char_tensor(p)
                #print("chartensor success")
                path_encodedic[k].append(pencode)
        return path_encodedic

    def next_chunk(self):
        # TODO : add [UNK]
        chunk, content = self.__random_chunk()
        #print("content: ")
        #print(content)
        path_encodedic = self.path_encode(self.find_path(self.astfile))
        #pathdic = self.find_path(self.astfile)
        ents_list = []
        ents = [24] * self.chunk_len  # UNK = 24 to be modified
        contents = ""
        for i in range(len(content)):
            contents = contents + content[i] + " "

        #path_tensor = Variable(torch.zeros(self.ast_token_num - 1).long()).to(self.device)
        path_addlist = [0] * self.ast_token_num

        keylist = []
        start = 0
        while contents.find("[", start) > 0:
            #print(path_encodedic)
            start = contents.find("[ iTC", start)
            #print("start: " + str(start))
            if contents.find("]", start) > 0:
                end = contents.find("]", start)
                #print("end:" + str(end))
                key = contents[start: end + 1]
                key = key.replace(" ", "") + "\n"
                if key != "" and key in path_encodedic.keys():
                    keylist.append(key)
                start = end
            else:
                start = len(contents) - 1
        for k in keylist:
            ast_list = path_encodedic[k]

            for t in ast_list :
                #t.resize(t.size(), path_tensor.size())
                # print(t.size())
#                trans = nn.Embedding(t.size(), self.ast_token_num - 1)

                path_addlist = np.sum([t, path_addlist], axis = 0)

        ast = torch.Tensor(path_addlist).long()
        #print(ast.size())

        # TODO modify
        for i in range(len(self.kg)):
            if contents.find(" " + self.kg[i] + " ") != -1:
                ents_list.append(self.kg[i])
        for i in range(len(ents_list)):
            key = ents_list[i].strip().split()
            if content.index(key[0]) >= 0:
                ents[content.index(key[0])] = self.kg.index(" ".join(key))
        ents = torch.Tensor(ents).long()
        #print(ents.size())
        input_target_pair = []
        for i in range(1, self.chunk_len):
            input = torch.zeros(self.chunk_len - 1).long()
            # length of ent?
            ent = torch.zeros(self.chunk_len - 1).long()

            for j in range(self.chunk_len - i - 1):
                input[j] = self.vocabularyLoader.n_tokens - 1
            #print("chunck:")
            #print(chunk)
            input[-i:] = chunk[:i]
            target = chunk[i - 1:i + 1]
            ent[-i:] = ents[:i]
            input = input.to(self.device)
            # print(input)
            # print(input.size())
            target = target.to(self.device)
            # print(target)
            # print(target.size())
            ent = ent.to(self.device)
            # print(ent)
            # print(ent.size())
            ast = ast.to(self.device)
            # print(ast)
            # print(ast.size())
            input_target_pair.append((input, ent, target, ast))
        return input_target_pair

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.token_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index]), \
                   self.token_list[start_index:end_index]