import utils
from utils import * 
import pickle 
import argparse
import torch 
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import model 
from model import TransformerVAE
import rdkit 
from rdkit.Chem import MolFromSmiles as get_mol

parser = argparse.ArgumentParser(description='Description of your script.')

parser.add_argument('--dim_model', type=int, default=256, help='number of epochs')
parser.add_argument('--dim_expansion', type=int, default=256, help='number of epochs')
parser.add_argument('--dim_latent', type=int, default=128, help='number of epochs')
parser.add_argument('--num_head', type=int, default=8, help='number of epochs')
parser.add_argument('--num_layer', type=int, default=1, help='number of epochs')
parser.add_argument('--dropout', type=float, default=1, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--max_len', type=int, default=40, help='batch size')

arg_list = parser.parse_args()

#python train.py --dim_model 256 --dim_expansion 256 --dim_latent 128 --num_head 8 --num_layer 1 --dropout 0.5 --lr 0.0003 --epochs 10 --batch_size 256 --max_len 40

def tokenize(smi) :
    return [0] + [smi_dic[char] for char in smi] + [1]

def pad(smi) :
    return smi + [2] * (max_len - len(smi))

with open('../data/chembl24_canon_train.pickle', 'rb') as file :
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerVAE(dim_model = arg_list.dim_model,
                       dim_expansion = arg_list.dim_expansion,
                       dim_latent = arg_list.dim_latent,
                       num_head = arg_list.num_head,
                       num_layer = arg_list.num_layer,
                       dropout = arg_list.dropout,
                       smi_dic = smi_dic).to(device)

beta_np_cyc = frange_cycle_cosine(0.0, 1.0, arg_list.epochs, 1)

optim = torch.optim.Adam(model.parameters(), lr = 0.0003) 


def loss_fn(pred, tgt, mu, sigma, beta) :
    reconstruction_loss = F.nll_loss(pred.reshape(-1, len(smi_dic)), tgt[:, 1:].reshape(-1), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) 
    return  reconstruction_loss + kl_loss * beta 

print('\n\n')
print('///////////////////////////////////////////////////')
print('///////////////// START TRAINING //////////////////')
print('///////////////////////////////////////////////////')
print('\n\n')

def metric(pred, smi_list, inv_dic) : 
    pred = pred.cpu().tolist() 
    gen_smi = []
    
    valid_count = 0 
    novel_count = 0

    for i in pred : 
        smi = ''.join([inv_dic[c] for c in i])
        smi = smi.replace("<START>", "").replace("<PAD>","").replace("<END>","")
        if get_mol(smi) != None : 
            gen_smi.append(smi) 
            valid_count += 1 

    gen_set = set(gen_smi) 
    try : 
        unique_count = (len(gen_set) / len(gen_smi)) * 100
    except : 
        unique_count = 0 

    for smi in gen_smi : 
        if smi not in smi_list : 
            novel_count += 1 

    return (valid_count / len(pred)) * 100, unique_count, (novel_count / len(pred)) * 100, gen_set


NUM_GENERATE = 500

rdkit.rdBase.DisableLog('rdApp.*')

for epoch in range(1, arg_list.epochs + 1) :
    train_loss = 0
    beta = beta_np_cyc[epoch-1]
    for i, input in enumerate(train_loader) :
        model.train()

        input = input.to(device)
        pred, mu, sigma = model(input, input[:, :-1])

        loss = loss_fn(pred, input, mu, sigma, beta)
        train_loss += loss.item()
        loss.backward()
        optim.step()
        optim.zero_grad()


        model.eval()
        if i % (len(train_loader) // 20) == 0 :
            z = torch.randn(NUM_GENERATE, max_len, 128).to(device)
            target = torch.zeros(NUM_GENERATE, 1, dtype=torch.long).to(device)

            for i in range(max_len - 1) :
                out = model.inference(z, target)
                _, idx = torch.topk(out, 1, dim = -1)
                idx = idx[:, -1, :]
                target = torch.cat([target, idx], dim = 1)


            validity, uniqueness, novelty, gen_set = metric(target, smi_list, inv_dic) 
            print(f'\nValid: {validity:.4f}% --- Uniqueness: {uniqueness:.4f}% --- Novelty: {novelty:.4f}%\n')
            
            for i, smi in enumerate(gen_set) : 
                print(f'{i}: {smi}')

            # target = target.cpu().tolist()
            # smiles = ''.join([inv_dic[i] for i in target])
            # smiles = smiles.replace("<START>", "").replace("<PAD>", "").replace("<END>","")

            # print(f'{smiles}')

    print(f'\n\nepoch : {epoch}, train loss : {train_loss / len(train_loader)}\n\n')