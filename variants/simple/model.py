import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

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


class Encoder(nn.Module) : 
    def __init__(self, dim_model, num_head, dropout, smi_dic, norm = True) :
        super(Encoder, self).__init__() 
        
        self.dim_model = dim_model

        self.embed = nn.Embedding(len(smi_dic), dim_model)
        self.pos = PositionalEncoding(dim_model, dropout) 

        self.norm1 = nn.LayerNorm(dim_model) if norm else nn.Dropout(0) 
        self.drop1 = nn.Dropout(dropout) 
        self.self_attn = Attention(dim_model, num_head) 

        # Add ResNet
        self.norm2 = nn.LayerNorm(dim_model) if norm else nn.Dropout(0) 
        self.feed_forward = nn.Sequential(

            # nn.Linear(dim_model, dim_model),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),

            nn.Linear(dim_model, dim_model // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            # nn.Linear(dim_model // 2, dim_model // 2),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
        )

        self.norm3 = nn.LayerNorm(dim_model // 2) if norm else nn.Dropout(0) 

        self.mu = nn.Linear(dim_model // 2, dim_model // 4) 
        self.sigma = nn.Linear(dim_model // 2, dim_model // 4) 

    def forward(self, x) :
        mask = (x != 2).unsqueeze(-2).unsqueeze(1)
        x = self.embed(x) * (self.dim_model ** 0.5)
        x = self.pos(x) 

        x = self.norm1(x) 
        attn, self_attn = self.self_attn(x, x, x) 
        x = x + self.drop1(attn) 

        x = self.norm2(x) 
        x = self.feed_forward(x) 

        x = self.norm3(x) 

        mu, sigma = self.mu(x), self.sigma(x) 

        return mu, sigma 
    



class Decoder(nn.Module) : 
    def __init__(self, dim_model, num_head, dropout, smi_dic, norm = True) :
        super(Decoder, self).__init__() 
        self.dim_model = dim_model

        self.embed = nn.Embedding(len(smi_dic), dim_model)
        self.pos = PositionalEncoding(dim_model, dropout)

        self.norm1 = nn.LayerNorm(dim_model) if norm else nn.Dropout(0) 
        self.self_attn = Attention(dim_model, num_head)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim_model) if norm else nn.Dropout(0)
        self.cross_attn = Attention(dim_model, num_head)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(dim_model) if norm else nn.Dropout(0)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_model * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            nn.Linear(dim_model * 2, dim_model * 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),

            # nn.Linear(dim_model * 4, dim_model * 8),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout)
        )
        self.drop3 = nn.Dropout(dropout)

        self.norm4 = nn.LayerNorm(dim_model * 4)
        self.proj = nn.Linear(dim_model * 4, len(smi_dic))



    def forward(self, memory, target, mask) : 
        target = self.embed(target) * (self.dim_model ** 0.5)   
        target = self.pos(target)

        target = self.norm1(target)
        attn, self_attn = self.self_attn(target, target, target, mask)
        target = target + self.drop1(attn)

        target = self.norm2(target)
        attn, cross_attn = self.cross_attn(target, memory, memory)
        target = target + self.drop2(attn)

        target = self.norm3(target)
        target = self.feed_forward(target)
        target = target + self.drop3(target)

        target = self.norm4(target)
        target = self.proj(target)

        return target
    

class Model(nn.Module) :
    def __init__(self, dim_model, num_head, dropout, smi_dic, norm = True) : 
        super(Model, self).__init__()
        self.smi_dic = smi_dic

        self.encoder = Encoder(dim_model, num_head, dropout, smi_dic, norm)
        self.decoder = Decoder(dim_model // 4, num_head, dropout, smi_dic, norm)

    def get_mask(self, target) :
        mask = (target != self.smi_dic['<PAD>']).unsqueeze(-2)
        return mask & subsequent_mask(target.size(-1)).type_as(mask.data)
    
    def reparameterization(self, mu, sigma) :
        eps = torch.rand_like(sigma).to(device)
        z = mu + torch.exp(sigma) * eps
        return z 
    
    def inference(self, z, target) :
        mask = self.get_mask(target)
        mask = mask.unsqueeze(1).to(device)

        target = self.decoder(z, target, mask) 
        target = F.log_softmax(target, dim=-1)

        return target

    def forward(self, x, target) :
        mu, sigma = self.encoder(x)

        z = self.reparameterization(mu, sigma) 

        mask = self.get_mask(target)
        mask = mask.unsqueeze(1).to(device) 

        target = self.decoder(z, target, mask) 
        
        target = F.log_softmax(target, dim = -1) 

        return target, mu, sigma
    




