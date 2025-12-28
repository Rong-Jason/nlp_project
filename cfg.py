"""
方案6配置文件
风格: 使用SimpleNamespace，紧凑配置
特点: Deep GRU + 乘性注意力 + TF Decay + 位置编码对比
"""
from types import SimpleNamespace

# 数据配置
data = SimpleNamespace(
    train='./datasets/train_mixed_v2.jsonl',
    valid='./datasets/valid_hy.jsonl',
    test='./datasets/test_hy.jsonl',
    max_len=100,
    min_freq=2
)

# 特殊token
tokens = SimpleNamespace(PAD=0, UNK=1, BOS=2, EOS=3)

# Deep GRU配置 (4层!)
gru = SimpleNamespace(
    emb_size=256,
    hidden_size=384,
    num_layers=4,  # 深层GRU
    dropout=0.35,
    attention='additive',  # 乘性注意力 multiplicative 加性 additive 点乘 dot

    # Teacher Forcing Decay
    tf_init=1.0,
    tf_final=0.2,
    tf_decay_rate=0.05  # 每epoch衰减
)

# Transformer配置
transformer = SimpleNamespace(
    d_model=512,
    heads=8,
    enc_layers=6,
    dec_layers=6,
    ff_dim=2048,
    dropout=0.1,
    # 位置编码对比实验
    pos_types=['sinusoidal', 'learned']
)

# 训练配置
train = SimpleNamespace(
    batch=32,
    lr=2e-4,
    epochs=50,
    warmup=1000,
    clip=1.0,
    label_smooth=0.1,
    device='cuda:1',
    mode='gru'
)

# 输出路径
output = SimpleNamespace(
    model_dir='./train/ckpts_gru_add_bpe_jieba_hy_adm_2e-4_50',
    log_dir='./logs'
)

