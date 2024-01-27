import rdkit 
import multiprocessing
import torch
import torch.nn as nn 
import copy 
import math 
import numpy as np
import pickle


def get_smi_list(path) :
    with open(path, 'r') as file :
        return [smi[:-1] for smi in file.readlines()]
    

def replace_atom(smi) :
    return smi.replace('Cl', 'L').replace('Br', 'R') 

def get_mol(smi) :
    return rdkit.Chem.MolFromSmiles(smi)

def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)



def get_dic(smi_list) :
    dic = {'<START>': 0, '<END>': 1, '<PAD>': 2}
    for smi in smi_list :
        for char in smi :
            if char not in dic :
                dic[char] = len(dic) 
    return dic 

def tokenize(smi) :
    return [0] + [smi_dic[char] for char in smi] + [1]

def pad(smi) :
    return smi + [2] * (max_len - len(smi))

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def get_mask(target) :
    mask = (target != smi_dic['<PAD>']).unsqueeze(-2)
    return mask & subsequent_mask(target.size(-1)).type_as(mask.data)



def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [0, pi] for plots: 

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L    



class MyDataset(torch.utils.data.Dataset) :
    def __init__(self, token_list) :
        self.token_list = token_list

    def __len__(self) :
        return len(self.token_list)

    def __getitem__(self, idx) :   
        return torch.tensor(self.token_list[idx], dtype=torch.long)







