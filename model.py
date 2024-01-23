import torch
import torch.nn as nn
import torch.nn.functional as F 
import math 
import utils
from utils import get_mask

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
    

class FirstEncoder(nn.Module) :
    def __init__(self, dim_model, dim_latent, num_head, dropout, vocab_size) :
        super(FirstEncoder, self).__init__()
        self.dim_model = dim_model

        self.embed = nn.Embedding(vocab_size, dim_model)
        self.pos = PositionalEncoding(dim_model, dropout) 

        self.norm1 = nn.LayerNorm(dim_model)
        self.drop1 = nn.Dropout(dropout)
        self.self_attn = Attention(dim_model, num_head) 

        self.norm2 = nn.LayerNorm(dim_model)
        self.feed_foward = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(dim_model, dim_model)
        )
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(dim_model)

        self.mu, self.sigma = nn.Linear(dim_model, dim_latent), nn.Linear(dim_model, dim_latent)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() 
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x) :
        x = self.embed(x) * (self.dim_model ** 0.5) 
        x = self.pos(x)

        x = self.norm1(x)
        attn, self_attn = self.self_attn(x, x, x)
        x = x + self.drop1(attn)

        x = self.norm2(x)
        x = self.feed_foward(x)
        x = x + self.drop2(x)   

        x = self.norm3(x)

        mu, sigma = self.mu(x), torch.exp(self.sigma(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1/2).sum()

        return z



class SecondEncoder(nn.Module) :
    def __init__(self, dim_model,  dim_latent, num_head, dropout) :
        super(SecondEncoder, self).__init__()

        self.norm1 = nn.LayerNorm(dim_latent)
        self.drop1 = nn.Dropout(dropout)
        self.self_attn = Attention(dim_latent, num_head) 

        self.norm2 = nn.LayerNorm(dim_latent)
        self.feed_foward = nn.Sequential(
            nn.Linear(dim_latent, dim_model),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(dim_model, dim_model)
        )
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(dim_model)
    
    def forward(self, z) :
        z = self.norm1(z)
        attn, self_attn = self.self_attn(z, z, z)
        z = z + self.drop1(attn)

        z = self.norm2(z)
        z = self.feed_foward(z)
        z = z + self.drop2(z)   

        z = self.norm3(z)

        return z
class Decoder(nn.Module) :
    def __init__(self, dim_model, num_head, dropout, vocab_size) :
        super(Decoder, self).__init__()

        self.dim_model = dim_model
        
        self.embed = nn.Embedding(vocab_size, dim_model)
        self.pos = PositionalEncoding(dim_model, dropout)

        self.norm1 = nn.LayerNorm(dim_model)   
        self.self_attn = Attention(dim_model, num_head)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim_model)
        self.cross_attn = Attention(dim_model, num_head)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(dim_model)
        self.feed_foward = nn.Sequential(
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU()
        )
        self.drop3 = nn.Dropout(dropout)
        
        self.norm4 = nn.LayerNorm(dim_model)

        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, memory, target) :
        mask = get_mask(target)
        mask = mask.unsqueeze(1).to(device)

        target = self.embed(target) * (self.dim_model ** 0.5)
        target = self.pos(target)

        target = self.norm1(target)
        attn, self_attn = self.self_attn(target, target, target, mask)
        target = target + self.drop1(attn)

        target = self.norm2(target)
        attn, cross_attn = self.cross_attn(target, memory, memory)
        target = target + self.drop2(attn)

        target = self.norm3(target)
        target = self.feed_foward(target)
        target = target + self.drop3(target)

        target = self.norm4(target)

        target = self.proj(target)
        target = F.log_softmax(target, dim = -1)

        return target 