import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import rdkit 
import multiprocessing
import copy
import math 
import random
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


class MyDataset(torch.utils.data.Dataset) :
    def __init__(self, token_list) :
        self.token_list = token_list

    def __len__(self) :
        return len(self.token_list)

    def __getitem__(self, idx) :   
        return torch.tensor(self.token_list[idx], dtype=torch.long)
    
def get_mask(target, smi_dic) :
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


def gen_beta(start, end, T1, T2, T3):
    for i in range(T1):
        yield start
    log_s = np.log(start)
    log_e = np.log(end)
    T = T2 - T1
    AT = T3 - T1
    for i in range(T):
        cur_beta = np.exp(log_s + (log_e - log_s) / AT * i)
        yield cur_beta

    T = T3 - T2
    delta_beta = (end - cur_beta) / T
    for i in range(T):
        cur_beta += delta_beta
        yield cur_beta

    while True:
        yield end




with open('data/chembl24_canon_train.pickle', 'rb') as file :
    smi_list = pickle.load(file) 


smi_list = [smi for smi in smi_list if len(smi) < 40] # Choose only smiles with length < 40

print(f'Number of data: {len(smi_list)}')


smi_dic = get_dic(smi_list)

inv_dic = {v:k for k, v in smi_dic.items()}

token_list = parallel_f(tokenize, smi_list)
max_len = len(max(token_list, key=len))
token_list = parallel_f(pad, token_list)



BATCH_SIZE = 256

dataset = MyDataset(token_list)
train_set, val_set = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    

    
class Attention(nn.Module) :
    def __init__(self, dim_model, num_head) :
        super(Attention, self).__init__()
        assert dim_model % num_head == 0, 'dim_model % num_head != 0'
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_head = dim_model // num_head

        self.Q = nn.Linear(dim_model, dim_model)
        self.K = nn.Linear(dim_model, dim_model)
        self.V = nn.Linear(dim_model, dim_model)

        self.out = nn.Linear(dim_model, dim_model)

    def forward(self, Q, K, V, mask = None) :
        B = Q.size(0) 

        Q, K, V = self.Q(Q), self.K(K), self.V(V)

        len_Q, len_K, len_V = Q.size(1), K.size(1), V.size(1)

        Q = Q.reshape(B, self.num_head, len_Q, self.dim_head)
        K = K.reshape(B, self.num_head, len_K, self.dim_head)
        V = V.reshape(B, self.num_head, len_V, self.dim_head)
        
        K_T = K.transpose(2,3).contiguous()

        attn_score = Q @ K_T

        attn_score = attn_score / (self.dim_head ** 1/2)
        if mask is not None :
            attn_score = attn_score.masked_fill(mask == 0, -1e9)

        attn_distribution = torch.softmax(attn_score, dim = -1)

        attn = attn_distribution @ V

        attn = attn.reshape(B, len_Q, self.num_head * self.dim_head)
        
        attn = self.out(attn)

        return attn, attn_distribution
    



class EncoderOne(nn.Module) :
    def __init__(self, dim_model, dim_latent, num_head, num_layer, dropout) :
        super(EncoderOne, self).__init__()

        self.layers = clones(EncoderOneLayer(dim_model, dim_latent, num_head, dropout), num_layer)

        self.mu = nn.Linear(dim_model, dim_latent)
        self.sigma = nn.Linear(dim_model, dim_latent)

        
    def forward(self, x) :
        for layer in self.layers : 
            x = layer(x) 

        return self.mu(x), self.sigma(x) 


class EncoderOneLayer(nn.Module) :
    def __init__(self, dim_model, dim_latent, num_head, dropout) :
        super(EncoderOneLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim_model)
        self.drop1 = nn.Dropout(dropout)
        self.self_attn = Attention(dim_model, num_head) 

        self.norm2 = nn.LayerNorm(dim_model)
        self.feed_foward = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_model, dim_model)
        )
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(dim_model) 

    def forward(self, x) :
        x = self.norm1(x)
        attn, self_attn = self.self_attn(x, x, x)
        x = x + self.drop1(attn)

        x = self.norm2(x)
        x = self.feed_foward(x)
        x = x + self.drop2(x)   

        x = self.norm3(x)

        return x 


class Decoder(nn.Module) :
    def __init__(self, dim_model, dim_expansion, dim_latent, num_head, num_layer, dropout, smi_dic) :
        super(Decoder, self).__init__()

        self.layers = clones(DecoderLayer(dim_model, dim_expansion, dim_latent, num_head, dropout, smi_dic), num_layer) 
        self.expand = nn.Sequential(
            nn.Linear(dim_latent, dim_model),
            nn.LeakyReLU(),
            nn.Dropout()
        )

    def forward(self, memory, target, mask) :
        memory = self.expand(memory)
        for layer in self.layers : 
            target = layer(memory, target, mask) 
        return target

class DecoderLayer(nn.Module) :
    def __init__(self, dim_model, dim_expansion, dim_latent, num_head, dropout, smi_dic) :
        super(DecoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(dim_model)   
        self.self_attn = Attention(dim_model, num_head)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim_model)
        self.cross_attn = Attention(dim_model, num_head)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(dim_model)
        self.feed_foward = nn.Sequential(
            nn.Linear(dim_model, dim_expansion),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(dim_expansion, dim_model),
            nn.LeakyReLU()
        )
        self.drop3 = nn.Dropout(dropout)
        
    def forward(self, memory, target, mask) :

        target = self.norm1(target)
        attn, self_attn = self.self_attn(target, target, target, mask)
        target = target + self.drop1(attn)

        target = self.norm2(target)
        attn, cross_attn = self.cross_attn(target, memory, memory)
        target = target + self.drop2(attn)

        target = self.norm3(target)
        target = self.feed_foward(target)
        target = target + self.drop3(target)

        return target


class TransformerVAE(nn.Module) :
    def __init__(self, dim_model, dim_expansion, dim_latent, num_head, num_layer, dropout, smi_dic) :
        super(TransformerVAE, self).__init__()
        self.dim_model = dim_model
        self.smi_dic = smi_dic
        
        self.embed = nn.Embedding(len(smi_dic), dim_model)
        self.pos = PositionalEncoding(dim_model, dropout)

        self.encoder_1 = EncoderOne(dim_model, dim_latent, num_head, num_layer, dropout)

        # self.pos_z = PositionalEncoding(dim_latent, dropout)


        self.embed_tgt = nn.Embedding(len(smi_dic), dim_model) 
        self.pos_tgt = PositionalEncoding(dim_model, dropout)

        self.decoder = Decoder(dim_model, dim_expansion, dim_latent, num_head, num_layer, dropout, smi_dic) 

        self.norm = nn.LayerNorm(dim_model)
        self.proj = nn.Linear(dim_model, len(smi_dic))

    def reparameterization(self, mu, sigma) :
        eps = torch.rand_like(sigma).to(device)
        z = mu + torch.exp(sigma) * eps
        return z 

    def get_mask(self, target, smi_dic) :
        mask = (target != smi_dic['<PAD>']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)
    
    def inference(self, z, target) :

        mask = self.get_mask(target, self.smi_dic)
        mask = mask.unsqueeze(1).to(device) 

        target = self.embed_tgt(target) * (self.dim_model ** 0.5)
        target = self.pos_tgt(target)

        target = self.decoder(z, target, mask) 

        target = self.norm(target) 
        target = self.proj(target)
        target = F.log_softmax(target, dim = -1)

        return target
    
    def forward(self, x, target) :
        x = self.embed(x) * (self.dim_model ** 0.5)   
        x = self.pos(x) 

        mu, sigma = self.encoder_1(x) 
        z = self.reparameterization(mu, sigma) 

        # z = self.pos_z(z) 

        # memory = self.encoder_2(z) 

        mask = self.get_mask(target, self.smi_dic)
        mask = mask.unsqueeze(1).to(device) 

        target = self.embed_tgt(target) * (self.dim_model ** 0.5)
        target = self.pos_tgt(target)

        target = self.decoder(z, target, mask) 

        target = self.norm(target) 
        target = self.proj(target)
        target = F.log_softmax(target, dim = -1)

        return target, mu, sigma



model = TransformerVAE(dim_model=256,
                       dim_expansion=256,
                       dim_latent = 64,
                       num_head=8,
                       num_layer=4,
                       dropout=0.5,
                       smi_dic=smi_dic).to(device)


# loss_fn = nn.NLLLoss()
optim = torch.optim.Adam(model.parameters(), lr = 0.0003) 


def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(smi_dic)), tgt[:, 1:].reshape(-1))
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) * beta 
    return reconstruction_loss + kl_loss

NUM_EPOCH = 20



