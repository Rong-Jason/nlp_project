"""
方案6: Deep GRU模型
特点: 4层GRU + 多种注意力机制(Dot/Additive/Multiplicative) + Teacher Forcing Decay
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cfg


class DotProductAttn(nn.Module):
    """点积(Dot-Product)注意力: score = h^T s"""
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        # 点积注意力通常要求 Encoder 和 Decoder 的隐层维度一致
        # 如果不一致，通常需要先投影，或者改用 Multiplicative
        assert enc_dim == dec_dim, "Dot-product attention requires equal hidden sizes."

    def forward(self, h, enc_out, mask=None):
        # h: (B, D), enc_out: (B, L, D)
        # score = h^T * s -> (B, L)
        score = torch.bmm(enc_out, h.unsqueeze(2)).squeeze(2)

        if mask is not None:
            score = score.masked_fill(mask, -1e9)

        attn = F.softmax(score, dim=-1)
        # ctx: (B, D)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx, attn


class MultiplicativeAttn(nn.Module):
    """乘性(General)注意力: score = h^T W s"""
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(self, h, enc_out, mask=None):
        # h: (B, D), enc_out: (B, L, E)
        # W(h): (B, E)
        # enc_out * W(h) -> (B, L)
        score = torch.bmm(enc_out, self.W(h).unsqueeze(2)).squeeze(2)

        if mask is not None:
            score = score.masked_fill(mask, -1e9)

        attn = F.softmax(score, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx, attn


class AdditiveAttn(nn.Module):
    """加性(Additive/Bahdanau)注意力: score = v^T tanh(W1 h + W2 s)"""
    def __init__(self, enc_dim, dec_dim, attn_dim=None):
        super().__init__()
        # 如果未指定中间维度，默认使用解码器隐层维度
        if attn_dim is None:
            attn_dim = dec_dim

        self.W_h = nn.Linear(dec_dim, attn_dim, bias=False) # 变换解码器状态
        self.W_s = nn.Linear(enc_dim, attn_dim, bias=False) # 变换编码器输出
        self.v = nn.Linear(attn_dim, 1, bias=False)         # 计算分数

    def forward(self, h, enc_out, mask=None):
        # h: (B, D), enc_out: (B, L, E)

        # query: (B, 1, attn_dim)
        query = self.W_h(h).unsqueeze(1)
        # key: (B, L, attn_dim)
        key = self.W_s(enc_out)

        # energy: (B, L, attn_dim) -> tanh(W1 h + W2 s)
        energy = torch.tanh(query + key)

        # score: (B, L, 1) -> (B, L)
        score = self.v(energy).squeeze(2)

        if mask is not None:
            score = score.masked_fill(mask, -1e9)

        attn = F.softmax(score, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)
        return ctx, attn


class DeepGRUEncoder(nn.Module):
    """深层GRU编码器"""
    def __init__(self, vocab_sz, emb_sz, hid_sz, n_layers, drop):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=0)
        self.gru = nn.GRU(emb_sz, hid_sz, n_layers, batch_first=True,
                         dropout=drop if n_layers > 1 else 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        pk = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.gru(pk)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, h


class DeepGRUDecoder(nn.Module):
    """深层GRU解码器+可选注意力机制"""
    def __init__(self, vocab_sz, emb_sz, hid_sz, n_layers, drop, attn_type='multiplicative'):
        super().__init__()
        self.vocab_sz = vocab_sz
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=0)

        # 根据参数选择注意力机制
        if attn_type == 'dot':
            self.attn = DotProductAttn(hid_sz, hid_sz)
        elif attn_type == 'additive':
            self.attn = AdditiveAttn(hid_sz, hid_sz)
        else:
            # 默认为乘性注意力 (multiplicative/general)
            self.attn = MultiplicativeAttn(hid_sz, hid_sz)

        self.gru = nn.GRU(emb_sz + hid_sz, hid_sz, n_layers, batch_first=True,
                         dropout=drop if n_layers > 1 else 0)
        self.fc = nn.Linear(hid_sz * 2 + emb_sz, vocab_sz)
        self.drop = nn.Dropout(drop)

    def step(self, tok, h, enc_out, mask):
        e = self.drop(self.emb(tok.unsqueeze(1)))
        # 使用选定的注意力机制计算 context vector
        ctx, _ = self.attn(h[-1], enc_out, mask)

        inp = torch.cat([e, ctx.unsqueeze(1)], -1)
        out, h = self.gru(inp, h)
        pred = self.fc(torch.cat([out.squeeze(1), ctx, e.squeeze(1)], -1))
        return pred, h


class BeamHypothesis:
    """用于Beam Search中存储单条假设的辅助类"""

    def __init__(self, token_id, log_prob, hidden):
        self.token_id = token_id  # 当前这一步的token
        self.log_prob = log_prob  # 累积对数概率
        self.hidden = hidden  # 当前时刻的隐层状态 (Layers, 1, Hid)
        self.sequence = [token_id]  # 以此结尾的完整序列

    def extend(self, token_id, log_prob, hidden):
        """扩展当前假设，生成新的假设"""
        new_hyp = BeamHypothesis(token_id, self.log_prob + log_prob, hidden)
        new_hyp.sequence = self.sequence + [token_id]
        return new_hyp

    @property
    def score(self):
        # 可以添加长度惩罚 (Length Penalty): score / (len(seq)^alpha)
        return self.log_prob / (len(self.sequence) ** 0.7)



class DeepGRUSeq2Seq(nn.Module):
    """完整模型"""
    def __init__(self, enc, dec, device):
        super().__init__()
        self.enc, self.dec, self.device = enc, dec, device

    def forward(self, src, src_len, tgt, tf_ratio=1.0):
        B, T = tgt.shape
        outs = torch.zeros(B, T, self.dec.vocab_sz, device=self.device)
        enc_out, h = self.enc(src, src_len)
        mask = (src == 0)

        tok = tgt[:, 0]
        for t in range(1, T):
            pred, h = self.dec.step(tok, h, enc_out, mask)
            outs[:, t] = pred
            # Teacher Forcing Decay
            use_tf = torch.rand(1).item() < tf_ratio
            tok = tgt[:, t] if use_tf else pred.argmax(-1)
        return outs

    @torch.no_grad()
    def translate(self, src, src_len, max_len=100):
        self.eval()
        enc_out, h = self.enc(src, src_len)
        mask = (src == 0)

        tok = torch.full((src.size(0),), cfg.tokens.BOS, device=self.device)
        result = [tok]

        for _ in range(max_len):
            pred, h = self.dec.step(tok, h, enc_out, mask)
            tok = pred.argmax(-1)
            result.append(tok)
            if (tok == cfg.tokens.EOS).all():
                break

        return torch.stack(result, dim=1)

    @torch.no_grad()
    def translate_greedy(self, src, src_len, max_len=100):
        """贪婪搜索"""
        self.eval()
        enc_out, h = self.enc(src, src_len)
        mask = (src == 0)

        # 初始 token: BOS
        tok = torch.tensor([cfg.tokens.BOS], device=self.device)
        result = []

        for _ in range(max_len):
            # pred: (1, vocab_sz)
            pred, h = self.dec.step(tok, h, enc_out, mask)
            tok = pred.argmax(-1)  # 贪婪选择最大概率

            token_id = tok.item()
            if token_id == cfg.tokens.EOS:
                break
            result.append(token_id)

        return result

    @torch.no_grad()
    def translate_beam(self, src, src_len, beam_width=3, max_len=100):
        """集束搜索"""
        self.eval()
        enc_out, h = self.enc(src, src_len)
        mask = (src == 0)

        # 初始化 Beam: [Hypothesis]
        start_hyp = BeamHypothesis(cfg.tokens.BOS, 0.0, h)
        beams = [start_hyp]
        completed = []

        for _ in range(max_len):
            candidates = []

            # 如果所有 beam 都停止了，退出
            if not beams:
                break

            # 扩展当前的每一个 beam
            for hyp in beams:
                tok_tensor = torch.tensor([hyp.token_id], device=self.device)

                # 解码一步
                pred, next_h = self.dec.step(tok_tensor, hyp.hidden, enc_out, mask)

                # 取 Log Softmax
                log_probs = F.log_softmax(pred, dim=-1).squeeze(0)  # (vocab_sz)

                # 选取当前这一步的 Top K (减少计算量)
                # 注意：这里我们选 beam_width 个，保证足够多的候选
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)

                for k in range(beam_width):
                    token_id = topk_ids[k].item()
                    log_p = topk_log_probs[k].item()
                    candidates.append(hyp.extend(token_id, log_p, next_h))

            # 排序并修剪 (Pruning)
            # 按分数从高到低排序
            candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

            new_beams = []
            for cand in candidates:
                if cand.token_id == cfg.tokens.EOS:
                    completed.append(cand)
                else:
                    if len(new_beams) < beam_width:
                        new_beams.append(cand)

            # 如果收集到了足够的完成句，也可以提前退出
            if len(completed) >= beam_width:
                break

            beams = new_beams

        # 选择最终结果
        if not completed:
            completed = beams

        # 全局选最好的
        best_hyp = sorted(completed, key=lambda x: x.score, reverse=True)[0]

        # 结果中去掉开头的 BOS (BOS id 是我们在 sequence[0] 初始化的)
        return best_hyp.sequence[1:]


def build_gru_model(src_vocab_sz, tgt_vocab_sz, device, attn_type='multiplicative'):
    """构建模型，支持指定 attention 类型"""
    c = cfg.gru
    enc = DeepGRUEncoder(src_vocab_sz, c.emb_size, c.hidden_size, c.num_layers, c.dropout)
    dec = DeepGRUDecoder(tgt_vocab_sz, c.emb_size, c.hidden_size, c.num_layers, c.dropout, attn_type=attn_type)
    return DeepGRUSeq2Seq(enc, dec, device).to(device)


def get_tf_ratio(epoch):
    """计算Teacher Forcing比例 (指数衰减)"""
    c = cfg.gru
    ratio = c.tf_init * (1 - c.tf_decay_rate) ** epoch
    return max(ratio, c.tf_final)