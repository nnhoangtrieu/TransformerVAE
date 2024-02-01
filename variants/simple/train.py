import pickle
import rdkit 
from rdkit.Chem import MolFromSmiles as get_mol
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
import model 
import utils
import numpy as np 
from model import Model 
from utils import parallel_f, get_dic, MyDataset
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
parser = argparse.ArgumentParser(description='Description of your script.')
parser.add_argument('--dim_model', type=int, default=256, help='number of epochs')
parser.add_argument('--num_head', type=int, default=8, help='number of epochs')
parser.add_argument('--dropout', type=float, default=1, help='number of epochs')
parser.add_argument('--norm', type=bool, default=True, help='norm')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_len', type=int, default=40, help='batch size')
arg_list = parser.parse_args()
#python train.py --dim_model 256 --num_head 8 --dropout 0.5 --norm True --lr 0.003 --epochs 10 --batch_size 256 --max_len 40

def tokenize(smi) :
    return [0] + [smi_dic[char] for char in smi] + [1]
def pad(smi) :
    return smi + [2] * (max_len - len(smi))

def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L    

def loss_fn(pred, tgt, mu, sigma, beta) :
    print(f'pred: {pred.shape}')
    print(f'tgt: {tgt.shape}')

    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(smi_dic)), tgt.view(-1))
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) 
    return  reconstruction_loss + kl_loss * beta 

with open('../data/chembl24_canon_train.pickle','rb') as file :
    smi_list = pickle.load(file) 
    smi_list = [smi for smi in smi_list if len(smi) <= arg_list.max_len]
    print(f'Number of data: {len(smi_list)}')

smi_dic = get_dic(smi_list)
inv_dic = {v:k for k, v in smi_dic.items()}
token_list = parallel_f(tokenize, smi_list)
max_len = len(max(token_list, key=len))
token_list = parallel_f(pad, token_list)

dataset = MyDataset(token_list)
train_set, val_set = random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_set, batch_size = arg_list.batch_size, shuffle = True)
val_loader = DataLoader(val_set, batch_size = arg_list.batch_size, shuffle = True)

model = Model(dim_model=arg_list.dim_model,
              num_head=arg_list.num_head,
              dropout=arg_list.dropout,
              smi_dic=smi_dic,
              norm = arg_list.norm).to(device)

optim = torch.optim.Adam(model.parameters(), lr = arg_list.lr) 


beta_np_cyc = frange_cycle_cosine(0.0, 1.0, len(train_loader), 2) 

rdkit.rdBase.DisableLog('rdApp.*')
print('\n\n')
print('///////////////////////////////////////////////////')
print('///////////////// START TRAINING //////////////////')
print('///////////////////////////////////////////////////')
print('\n\n')
for epoch in range(1, arg_list.epochs + 1) :
    train_loss, val_loss = 0, 0 
    # beta = beta_np_cyc[epoch - 1]
    for i, input in enumerate(train_loader) :
        model.train() 

        input = input.to(device)
        pred, mu, sigma = model(input, input[:, :-1])
        loss = loss_fn(pred, input[:, 1:], mu, sigma, 0)
        train_loss += loss.item() 
        loss.backward()
        optim.step()
        optim.zero_grad() 


        model.eval()

        z = torch.randn(1, max_len, arg_list.dim_model // 4).to(device)
        target = torch.zeros(1, 1, dtype = torch.long).to(device)
        for _ in range(max_len - 1) :
            out = model.inference(z, target)
            _, idx = torch.topk(out, 1, dim = -1)
            idx = idx[:, -1, :]
            target = torch.cat([target, idx], dim = 1)
        target = target.squeeze(0).tolist()
        smiles = ''.join([inv_dic[i] for i in target])
        smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")
        valid = "Valid" if get_mol(smiles) else "Not"

        print(f'{smiles} - {valid}')

    print(f'epoch : {epoch}, train loss : {train_loss / len(train_loader)} val loss : {val_loss / len(val_loader)}')
