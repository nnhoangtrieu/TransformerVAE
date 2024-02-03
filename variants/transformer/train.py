import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
import numpy as np
import copy 
import pickle
import math 
import re
import rdkit 
from rdkit.Chem import MolFromSmiles as get_mol
import argparse
import multiprocessing
import os
rdkit.rdBase.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Description of your script.')
parser.add_argument('--d_model', type=int, default=256, help='model dimension')
parser.add_argument('--d_latent', type=int, default=128, help='latent dimension')
parser.add_argument('--d_ff', type=int, default=256, help='feed forward dimension')
parser.add_argument('--num_head', type=int, default=8, help='number of attention heads')
parser.add_argument('--num_layer', type=int, default=8, help='number of layers')
parser.add_argument('--dropout', type=float, default=1, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_len', type=int, default=40, help='longest length of input for dataset')

parser.add_argument('--kl_start', type=int, default=40, help='longest length of input for dataset')
parser.add_argument('--kl_w_start', type=float, default=40, help='longest length of input for dataset')
parser.add_argument('--kl_w_end', type=float, default=40, help='longest length of input for dataset')


#python train.py --d_model 512 --d_latent 128 --d_ff 512 --num_head 8 --num_layer 3 --dropout 0.3 --lr 0.0003 --epochs 30 --batch_size 256 --max_len 35 --kl_start 5 --kl_w_start 0.0001 --kl_w_end 0.001
#python train.py --d_model 512 --d_latent 128 --d_ff 1024 --num_head 8 --num_layer 3 --dropout 0.3 --lr 0.0003 --epochs 30 --batch_size 256 --max_len 35 --kl_start 5 --kl_w_start 0.0001 --kl_w_end 0.001

#python train.py --d_model 512 --d_latent 128 --d_ff 512 --num_head 8 --num_layer 4 --dropout 0.3 --lr 0.0003 --epochs 30 --batch_size 256 --max_len 35 --kl_start 5 --kl_w_start 0.0001 --kl_w_end 0.001
#python train.py --d_model 512 --d_latent 128 --d_ff 1024 --num_head 8 --num_layer 4 --dropout 0.3 --lr 0.0003 --epochs 30 --batch_size 256 --max_len 35 --kl_start 5 --kl_w_start 0.0001 --kl_w_end 0.001

#python train.py --d_model 512 --d_latent 128 --d_ff 512 --num_head 8 --num_layer 5 --dropout 0.3 --lr 0.0003 --epochs 30 --batch_size 256 --max_len 35 --kl_start 5 --kl_w_start 0.0001 --kl_w_end 0.001
#python train.py --d_model 512 --d_latent 128 --d_ff 1024 --num_head 8 --num_layer 5 --dropout 0.3 --lr 0.0003 --epochs 30 --batch_size 256 --max_len 35 --kl_start 5 --kl_w_start 0.0001 --kl_w_end 0.001

arg = parser.parse_args()
######## Utils Functions ########
def seed_torch(seed=910):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
def tokenizer(smile):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens
def get_vocab(smi_list) :
    dic = {'<START>': 0, '<END>': 1, '<PAD>': 2}
    for smi in smi_list :
        for char in smi :
            if char not in dic :
                dic[char] = len(dic) 
    return dic 
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
def get_mask( target, smi_dic) :
        mask = (target != smi_dic['<PAD>']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)
def pad(smi, max_len) :
    return smi + [2] * (max_len - len(smi))
def encode(smi, vocab) :
    return [0] + [vocab[char] for char in smi] + [1]
def parallel_f(f, input_list) :
    pool = multiprocessing.Pool()
    return pool.map(f, input_list)
def read_gen_smi(t) : 
    smiles = ''.join([inv_vocab[i] for i in t])
    smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
    return smiles 
def get_valid(smi) : 
    return smi if get_mol(smi) else None 
def get_novel(smi) : 
    return smi if smi not in smi_list else None 
class MyDataset(torch.utils.data.Dataset) :
    def __init__(self, token_list) :
        self.token_list = token_list

    def __len__(self) :
        return len(self.token_list)

    def __getitem__(self, idx) :   
        return torch.tensor(self.token_list[idx], dtype=torch.long)
    
######## Utils Layer ########
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
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
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
class KLAnnealer:
    def __init__(self, n_epoch):
        self.i_start = arg.kl_start
        self.w_start = arg.kl_w_start
        self.w_max = arg.kl_w_end
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc
######## Encoder ########
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
class Encoder(nn.Module):
    def __init__(self, layer, N, d_latent):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.mu = nn.Linear(layer.size, d_latent)
        self.sigma = nn.Linear(layer.size, d_latent)
    def get_z(self, mu, sigma) : 
        eps = torch.rand_like(sigma).to(device)
        z = mu + torch.exp(0.5 * sigma) * eps
        return z 
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return self.get_z(mu, sigma), mu, sigma

######## Decoder ########
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)    
class Decoder(nn.Module):
    def __init__(self, layer, N, d_latent):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.upsize = nn.Linear(d_latent, layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        memory = F.relu(self.upsize(memory))
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

######## Model ########
class Transformer(nn.Module):
    def __init__(self, d_model, d_latent, d_ff, num_head, num_layer, dropout, vocab) : 
        super(Transformer, self).__init__()
        c = copy.deepcopy

        attn = MultiHeadedAttention(num_head, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff)
        position = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layer, d_latent)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), num_layer, d_latent)
        self.src_embedding = nn.Sequential(Embeddings(d_model, len(vocab)), c(position))
        self.tgt_embedding = nn.Sequential(Embeddings(d_model, len(vocab)), c(position))
        self.generator = nn.Linear(d_model, len(vocab))

    def inference(self, src_z, tgt, src_mask, tgt_mask) :
        tgt = self.tgt_embedding(tgt)
        out = self.decoder(tgt, src_z, src_mask, tgt_mask)
        out = F.log_softmax(self.generator(out), dim = -1)
        return out
        
    def forward(self, src, tgt, src_mask, tgt_mask):   
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        z, mu, sigma = self.encoder(src, src_mask)
        out = self.decoder(tgt, z, src_mask, tgt_mask)
        out = F.log_softmax(self.generator(out), dim = -1)

        return out, mu, sigma


######### Data #########
with open('../data/chembl24_canon_train.pickle','rb') as file :
    smi_list = pickle.load(file) 
    smi_list = [smi for smi in smi_list if len(smi) <= arg.max_len]
    print(f'Number of data: {len(smi_list)}')
    token_list = [tokenizer(s) for s in smi_list]
    vocab = get_vocab(token_list)
    inv_vocab = {v: k for k, v in vocab.items()}
    token_list = [encode(t, vocab) for t in token_list]
    max_len = len(max(token_list, key=len))
    token_list = [pad(t, max_len) for t in token_list]

    

dataset = MyDataset(token_list)
# train_set, val_set = random_split(dataset, [0.9, 0.1])

train_loader = DataLoader(dataset, batch_size=arg.batch_size)
# val_loader = DataLoader(val_set, batch_size=arg.batch_size, shuffle=True)
seed_torch()
model = Transformer(d_model=arg.d_model,
                    d_latent=arg.d_latent,
                    d_ff=arg.d_ff,
                    num_head=arg.num_head,
                    num_layer=arg.num_layer,
                    dropout=arg.dropout,
                    vocab=vocab).to(device)

for i in model.parameters() : 
    print(i)
optim = torch.optim.Adam(model.parameters(), lr = arg.lr, weight_decay=1e-6)
def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(vocab)), tgt.reshape(-1))
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()).mean() / arg.batch_size
    return  reconstruction_loss + kl_loss * beta 




#########################################################################
############################## TRAINING #################################
########################################################################

print(f'############################## TRAINING #################################')

annealer = KLAnnealer(arg.epochs)

for epoch in range(arg.epochs) : 
    train_loss = 0
    val_loss = 0
    beta = annealer(epoch)
    model.train()
    for i, src in enumerate(train_loader) : 
        src = src.to(device)    
        src_mask = (src != 2).unsqueeze(-2)
        tgt = src
        tgt_mask = get_mask(tgt[:, :-1], vocab)

        
        out, mu, sigma = model(src, tgt[:, :-1], src_mask, tgt_mask)
        loss = loss_fn(out, tgt[:, 1:], mu, sigma, beta)
        train_loss += loss.item()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        optim.step()
        optim.zero_grad()
    
    model.eval()
    # for src in val_loader : 
    #     beta = 0 if i < len(val_loader) * 0.99 else 0.00001
    #     src = src.to(device)    
    #     src_mask = (src != 2).unsqueeze(-2)
    #     tgt = src
    #     tgt_mask = get_mask(tgt[:, :-1], vocab)
    #     out, mu, sigma = model(src, tgt[:, :-1], src_mask, tgt_mask)
    #     loss = loss_fn(out, tgt[:, 1:], mu, sigma, beta)
    #     val_loss += loss.item()
        

    NUM_GEN = 500
    gen_mol = torch.empty(0).to(device)
    
    for _ in range(60) :

        src_z = torch.randn(NUM_GEN, max_len, arg.d_latent).to(device)
        src_mask = torch.ones(NUM_GEN, 1, max_len).to(device)
        tgt = torch.zeros(NUM_GEN, 1, dtype = torch.long).to(device)
        tgt_mask = get_mask(tgt, vocab).to(device)
        
        for _ in range(max_len - 1) :
            out = model.inference(src_z, tgt, src_mask, tgt_mask)
            _, idx = torch.topk(out, 1, dim = -1)
            idx = idx[:, -1, :]
            tgt = torch.cat([tgt, idx], dim = 1)
        
        gen_mol = torch.cat((gen_mol, tgt), dim =0)
    
    gen_mol = gen_mol.tolist()
    gen_mol = parallel_f(read_gen_smi, gen_mol)
    valid_mol = parallel_f(get_valid, gen_mol)
    valid_mol = [m for m in valid_mol if m is not None]
    validity = (len(valid_mol) / 30000) * 100 

    unique_mol = set(valid_mol)
    try : 
        unique = (len(unique_mol) / len(valid_mol)) * 100 
    except : 
        unique = 0

    novel_mol = parallel_f(get_novel, valid_mol)
    novel_mol = [m for m in novel_mol if m is not None]
    novelty = (len(novel_mol) / len(unique_mol)) * 100
     
    unique_novel_mol = set(novel_mol)



    with open(f'gen_mol/{d_model}_{d_latent}_{d_ff}_{num_head}_{num_layer}_{dropout}_{lr}_{epochs}_{batch_size}_{max_len}_{kl_start}_{kl_w_start}_{kl_w_end}.txt', 'a') as file :
        for i, m in enumerate(unique_novel_mol) :
            file.write(f'{i+1}. {m}\n')
        file.write(f"epoch {epoch + 1} --- train loss: {train_loss:3f}\n")
        file.write(f'validity: {validity:.2f}%, novelty: {novelty:.2f}%, unique: {unique:.2f}%')

    for i, m in enumerate(unique_novel_mol) :
        print(f'{i+1}. {m}')

    print(f"epoch {epoch + 1} --- train loss: {train_loss:3f}")
    print(f'validity: {validity:.2f}%, novelty: {novelty:.2f}%, unique: {unique:.2f}%')
    