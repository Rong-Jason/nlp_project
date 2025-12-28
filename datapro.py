"""
方案6数据处理
支持多种分词方案对比:
- 中文: Jieba (Baseline), HanLP (Advanced)
- 英文: NLTK (Baseline), SentencePiece (BPE/Subword)
"""
import json
import re
import os
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import cfg

# NLTK Check
import nltk

for res in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        print(f"Downloading NLTK resource: {res}...")
        nltk.download(res)

# Jieba (Default)
import jieba

# HanLP (Optional)
hanlp_tok = None


def get_hanlp():
    global hanlp_tok
    if hanlp_tok is None:
        try:
            import hanlp
            print("Loading HanLP model (this may take a while)...")
            # 加载轻量级模型，如果显存足够可换用 hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH
            hanlp_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        except ImportError:
            raise ImportError("Please install hanlp: pip install hanlp")
    return hanlp_tok


# SentencePiece (Optional for BPE)
sp_model = None


def get_sp_model(model_path='eng_bpe.model', data_path=None, vocab_size=8000):
    global sp_model
    try:
        import sentencepiece as spm
    except ImportError:
        raise ImportError("Please install sentencepiece: pip install sentencepiece")

    if sp_model is None:
        # 如果模型不存在且提供了数据路径，则进行训练
        if not os.path.exists(model_path):
            if data_path is None:
                raise ValueError("BPE model not found and no data_path provided for training.")

            print(f"Training SentencePiece BPE model on {data_path}...")
            # 提取英文语料用于训练 BPE
            temp_corpus = 'temp_en_corpus.txt'
            with open(data_path, 'r', encoding='utf-8') as f, open(temp_corpus, 'w', encoding='utf-8') as out:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        # 简单预清洗，移除换行符
                        clean_en = re.sub(r'\s+', ' ', d['en']).strip()
                        out.write(clean_en + '\n')

            # 训练 BPE 模型
            spm.SentencePieceTrainer.train(
                input=temp_corpus,
                model_prefix=model_path.replace('.model', ''),
                vocab_size=vocab_size,
                model_type='bpe',
                character_coverage=1.0,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3
            )
            print("BPE Training done.")
            if os.path.exists(temp_corpus): os.remove(temp_corpus)

        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        sp_model = sp
    return sp_model


# ==========================================
# 2. 分词逻辑封装
# ==========================================

def get_tokenizers(zh_method='jieba', en_method='bpe', train_path=None):
    """
    工厂函数：根据配置返回对应的分词函数
    :param zh_method: 'jieba' or 'hanlp'
    :param en_method: 'nltk' or 'bpe'
    :param train_path: 用于训练BPE的数据路径
    """

    # --- 中文分词器 ---
    if zh_method == 'hanlp':
        model = get_hanlp()

        def zh_func(s):
            # HanLP 也会保留标点，这里做基础清洗
            s = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:，。！？；：]', '', s.strip())
            return model(s)
    else:
        # Default: Jieba
        def zh_func(s):
            s = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:，。！？；：]', '', s.strip())
            return [t for t in jieba.cut(s) if t.strip()]

    # --- 英文分词器 ---
    if en_method == 'bpe':
        # SentencePiece
        sp = get_sp_model(data_path=train_path)

        def en_func(s):
            s = s.strip().lower()  # 小写化
            # Encode as pieces (strings) rather than IDs to fit existing Vocab pipeline
            return sp.encode_as_pieces(s)
    else:
        # Default: NLTK
        def en_func(s):
            s = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', s.strip().lower())
            return nltk.word_tokenize(s)

    return zh_func, en_func


# ==========================================
# 3. 数据处理与加载
# ==========================================

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
        ids = [self.w2i.get(t, 1) for t in toks]  # 1 为 <unk>
        return [2] + ids + [3] if add_se else ids  # 2 为 <bos>, 3 为 <eos>

    def dec(self, ids):
        return [self.i2w.get(i, '<unk>') for i in ids if i not in {0, 2, 3}]

    def __len__(self):
        return len(self.w2i)


def load_pairs(path, zh_tok_func, en_tok_func, max_len=100):
    pairs = []
    print(f"Loading data from {path}...")
    with open(path, encoding='utf-8') as f:
        for i, ln in enumerate(f):
            if not ln.strip(): continue
            try:
                d = json.loads(ln)
                zh = zh_tok_func(d['zh'])
                en = en_tok_func(d['en'])
                # 长度过滤逻辑，确保不为空且不超过 max_len
                if 0 < len(zh) <= max_len and 0 < len(en) <= (max_len - 2):
                    pairs.append((zh, en))
            except Exception as e:
                print(f"Error parsing line {i}: {e}")
                continue
    return pairs


class MTDataset(Dataset):
    def __init__(self, pairs, sv, tv):
        # 预先编码以加快训练速度
        self.items = [(sv.enc(z), tv.enc(e)) for z, e in pairs]

    def __len__(self): return len(self.items)

    def __getitem__(self, i): return self.items[i]


def collate(batch):
    """处理 Batch 内的 Padding 填充"""
    ss, ts = zip(*batch)
    sl = torch.tensor([len(s) for s in ss])
    ms, mt = max(len(s) for s in ss), max(len(t) for t in ts)
    sp = torch.zeros(len(batch), ms, dtype=torch.long)  # 默认为 0 (pad_idx)
    tp = torch.zeros(len(batch), mt, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        sp[i, :len(s)] = torch.tensor(s)
        tp[i, :len(t)] = torch.tensor(t)
    return sp, sl, tp


def get_loaders(zh_method='jieba', en_method='bpe'):
    """
    获取 DataLoader
    :param zh_method: 'jieba' (默认) 或 'hanlp'
    :param en_method: 'nltk' (默认) 或 'bpe'
    """
    # 1. 获取分词器
    zh_tok, en_tok = get_tokenizers(zh_method, en_method, train_path=cfg.data.train)

    # 2. 加载数据并分词
    print(f"Start processing data with ZH={zh_method}, EN={en_method}...")
    tr = load_pairs(cfg.data.train, zh_tok, en_tok, cfg.data.max_len)
    va = load_pairs(cfg.data.valid, zh_tok, en_tok, cfg.data.max_len)

    # 3. 构建词表
    # 注意：如果是 BPE，词表大小实际上由 SentencePiece 训练时的 vocab_size 决定，
    # 但为了兼容现有架构，我们这里仍然统计一次 token。
    sv, tv = Vocab(), Vocab()
    sv.fit([p[0] for p in tr], cfg.data.min_freq)
    tv.fit([p[1] for p in tr], cfg.data.min_freq)

    print(f'Vocab: src={len(sv)} tgt={len(tv)} | Train={len(tr)} Val={len(va)}')

    # 4. 构建 DataLoader
    tr_dl = DataLoader(MTDataset(tr, sv, tv), cfg.train.batch, True, collate_fn=collate)
    va_dl = DataLoader(MTDataset(va, sv, tv), cfg.train.batch, False, collate_fn=collate)

    return tr_dl, va_dl, sv, tv