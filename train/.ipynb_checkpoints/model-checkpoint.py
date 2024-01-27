import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 
import utils 
from utils import clones, subsequent_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.mu = nn.Sequential(
            nn.Linear(dim_model, dim_latent),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_latent),
        )
        self.sigma = nn.Sequential(
            nn.Linear(dim_model, dim_latent),
            nn.LeakyReLU(),
            nn.LayerNorm(dim_latent),
        )

        
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



class EncoderTwo(nn.Module) :
    def __init__(self, dim_model, dim_latent, num_head, num_layer, dropout) :
        super(EncoderTwo, self).__init__()

        self.layers = clones(EncoderTwoLayer(dim_latent, dim_latent, num_head, dropout), num_layer)
        self.expand = EncoderTwoLayer(dim_model, dim_latent, num_head, dropout)

    def forward(self, z) :
        for layer in self.layers : 
            z = layer(z) 
        x = self.expand(z)
        return x 

class EncoderTwoLayer(nn.Module) :
    def __init__(self, dim_model, dim_latent, num_head, dropout) :
        super(EncoderTwoLayer, self).__init__()

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
    def __init__(self, dim_model, dim_expansion, num_head, num_layer, dropout, smi_dic) :
        super(Decoder, self).__init__()

        self.layers = clones(DecoderLayer(dim_model, dim_expansion, num_head, dropout, smi_dic), num_layer) 

    def forward(self, memory, target, mask) :
        for layer in self.layers : 
            target = layer(memory, target, mask) 
        return target

class DecoderLayer(nn.Module) :
    def __init__(self, dim_model, dim_expansion, num_head, dropout, smi_dic) :
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

        self.pos_z = PositionalEncoding(dim_latent, dropout)

        self.encoder_2 = EncoderTwo(dim_model, dim_latent, num_head, num_layer, dropout)

        self.embed_tgt = nn.Embedding(len(smi_dic), dim_model) 
        self.pos_tgt = PositionalEncoding(dim_model, dropout)

        self.decoder = Decoder(dim_model, dim_expansion, num_head, num_layer, dropout, smi_dic) 

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
        z = self.pos_z(z) 

        memory = self.encoder_2(z) 

        mask = self.get_mask(target, self.smi_dic)
        mask = mask.unsqueeze(1).to(device) 

        target = self.embed_tgt(target) * (self.dim_model ** 0.5)
        target = self.pos_tgt(target)

        target = self.decoder(memory, target, mask) 

        target = self.norm(target) 
        target = self.proj(target)
        target = F.log_softmax(target, dim = -1)

        return target
    
    def forward(self, x, target) :
        x = self.embed(x) * (self.dim_model ** 0.5)   
        x = self.pos(x) 

        mu, sigma = self.encoder_1(x) 
        z = self.reparameterization(mu, sigma) 

        z = self.pos_z(z) 

        memory = self.encoder_2(z) 

        mask = self.get_mask(target, self.smi_dic)
        mask = mask.unsqueeze(1).to(device) 

        target = self.embed_tgt(target) * (self.dim_model ** 0.5)
        target = self.pos_tgt(target)

        target = self.decoder(memory, target, mask) 

        target = self.norm(target) 
        target = self.proj(target)
        target = F.log_softmax(target, dim = -1)

        return target, mu, sigma