"""
方案6数据处理 (增强版)
集成 data_utils.py 的细致清洗逻辑与分词优化
"""
import json, re, nltk
from collections import Counter
import jieba
import torch
from torch.utils.data import Dataset, DataLoader
import cfg

# 确保NLTK数据可用
for res in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        nltk.download(res)


def zh_tok(s):
    """jieba分词，正则清洗：保留汉字、字母、数字及基本标点"""
    s = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:，。！？；：]', '', s.strip())
    return [t for t in jieba.cut(s) if t.strip()]

def en_tok(s):
    """ NLTK 分词及小写化处理"""
    s = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', s.strip().lower())
    return nltk.word_tokenize(s)

class Vocab:
    def __init__(self):
        # 统一使用 PAD=0, UNK=1, BOS=2, EOS=3
        self.w2i = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.i2w = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}

    def fit(self, sents, min_f=2):
        """集成词频过滤逻辑"""
        c = Counter(w for s in sents for w in s)
        for w, f in c.most_common():
            if f >= min_f and w not in self.w2i:
                i = len(self.w2i)
                self.w2i[w], self.i2w[i] = i, w
        return self

    def enc(self, toks, add_se=True):
        ids = [self.w2i.get(t, 1) for t in toks] # 1 为 <unk>
        return [2] + ids + [3] if add_se else ids # 2 为 <bos>, 3 为 <eos>

    def dec(self, ids):
        return [self.i2w.get(i, '<unk>') for i in ids if i not in {0, 2, 3}]

    def __len__(self): return len(self.w2i)

def load_pairs(path, max_len=100):
    pairs = []
    with open(path, encoding='utf-8') as f:
        for ln in f:
            if not ln.strip(): continue
            d = json.loads(ln)
            zh, en = zh_tok(d['zh']), en_tok(d['en'])
            # 长度过滤逻辑，确保不为空且不超过 max_len
            if 0 < len(zh) <= max_len and 0 < len(en) <= (max_len - 2):
                pairs.append((zh, en))
    return pairs

class MTDataset(Dataset):
    def __init__(self, pairs, sv, tv, ml):
        # 预先编码以加快训练速度
        self.items = [(sv.enc(z), tv.enc(e)) for z, e in pairs]

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    """处理 Batch 内的 Padding 填充"""
    ss, ts = zip(*batch)
    sl = torch.tensor([len(s) for s in ss])
    ms, mt = max(len(s) for s in ss), max(len(t) for t in ts)
    sp = torch.zeros(len(batch), ms, dtype=torch.long) # 默认为 0 (pad_idx)
    tp = torch.zeros(len(batch), mt, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        sp[i, :len(s)] = torch.tensor(s)
        tp[i, :len(t)] = torch.tensor(t)
    return sp, sl, tp

def get_loaders():
    tr = load_pairs(cfg.data.train, cfg.data.max_len)
    va = load_pairs(cfg.data.valid, cfg.data.max_len)

    sv, tv = Vocab(), Vocab()
    sv.fit([p[0] for p in tr], cfg.data.min_freq)
    tv.fit([p[1] for p in tr], cfg.data.min_freq)

    print(f'Vocab: src={len(sv)} tgt={len(tv)} | Train={len(tr)} Val={len(va)}')

    tr_dl = DataLoader(MTDataset(tr, sv, tv, cfg.data.max_len), cfg.train.batch, True, collate_fn=collate)
    va_dl = DataLoader(MTDataset(va, sv, tv, cfg.data.max_len), cfg.train.batch, False, collate_fn=collate)

    return tr_dl, va_dl, sv, tv