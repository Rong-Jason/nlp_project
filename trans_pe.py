"""
方案6: Transformer with Position Encoding Comparison
特点: 对比正弦位置编码 vs 学习式位置编码
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cfg


class SinusoidalPE(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LearnedPE(nn.Module):
    """学习式位置编码"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.pe(pos)


class MHA(nn.Module):
    """多头注意力"""
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.h, self.dk = h, d // h
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
    
    def forward(self, q, k, v, mask=None, causal=False):
        B, L, _ = q.shape
        Q = self.wq(q).view(B, L, self.h, self.dk).transpose(1, 2)
        K = self.wk(k).view(B, -1, self.h, self.dk).transpose(1, 2)
        V = self.wv(v).view(B, -1, self.h, self.dk).transpose(1, 2)
        
        sc = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)
        if causal:
            cm = torch.triu(torch.ones(L, L, device=q.device), 1).bool()
            sc = sc.masked_fill(cm, -1e9)
        if mask is not None:
            sc = sc.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        at = self.drop(F.softmax(sc, -1))
        out = (at @ V).transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)


class EncLayer(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.attn = MHA(d, h, drop)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), self.n1(x), self.n1(x), mask)
        return x + self.ff(self.n2(x))


class DecLayer(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.attn1 = MHA(d, h, drop)
        self.attn2 = MHA(d, h, drop)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
    
    def forward(self, x, mem, src_mask=None):
        x = x + self.attn1(self.n1(x), self.n1(x), self.n1(x), causal=True)
        x = x + self.attn2(self.n2(x), mem, mem, src_mask)
        return x + self.ff(self.n3(x))


class TransformerPE(nn.Module):
    """Transformer with configurable PE"""
    def __init__(self, src_v, tgt_v, d, h, n_enc, n_dec, ff, drop, pe_type='sinusoidal'):
        super().__init__()
        self.d = d
        self.src_emb = nn.Embedding(src_v, d, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_v, d, padding_idx=0)
        
        # 位置编码选择
        PE = SinusoidalPE if pe_type == 'sinusoidal' else LearnedPE
        self.src_pe = PE(d)
        self.tgt_pe = PE(d)
        self.pe_type = pe_type
        
        self.enc = nn.ModuleList([EncLayer(d, h, ff, drop) for _ in range(n_enc)])
        self.dec = nn.ModuleList([DecLayer(d, h, ff, drop) for _ in range(n_dec)])
        self.out = nn.Linear(d, tgt_v)
        self._init()
    
    def _init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src):
        mask = (src == 0)
        x = self.src_pe(self.src_emb(src) * math.sqrt(self.d))
        for layer in self.enc:
            x = layer(x, mask)
        return x, mask
    
    def forward(self, src, tgt):
        mem, mask = self.encode(src)
        y = self.tgt_pe(self.tgt_emb(tgt) * math.sqrt(self.d))
        for layer in self.dec:
            y = layer(y, mem, mask)
        return self.out(y)
    
    @torch.no_grad()
    def generate(self, src, max_len=100):
        self.eval()
        mem, mask = self.encode(src)
        ys = torch.full((src.size(0), 1), cfg.tokens.BOS, device=src.device)
        for _ in range(max_len):
            y = self.tgt_pe(self.tgt_emb(ys) * math.sqrt(self.d))
            for layer in self.dec:
                y = layer(y, mem, mask)
            next_t = self.out(y[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_t], 1)
            if (next_t == cfg.tokens.EOS).all():
                break
        return ys


def build_transformer(src_v, tgt_v, device, pe_type='sinusoidal'):
    c = cfg.transformer
    return TransformerPE(src_v, tgt_v, c.d_model, c.heads, c.enc_layers, 
                        c.dec_layers, c.ff_dim, c.dropout, pe_type).to(device)

