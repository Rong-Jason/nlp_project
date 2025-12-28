"""
方案6: 推理脚本
Usage: python inference.py --model gru --ckpt ./ckpts/gru.pt --text "中文"
"""
import argparse
import json
import torch
import cfg
import sacrebleu
from data import zh_tok
from deep_gru import build_gru_model
from trans_pe import build_transformer


def load_gru(path, device):
    ck = torch.load(path, map_location=device)
    sv, tv = ck['sv'], ck['tv']
    model = build_gru_model(len(sv), len(tv), device, 'dot')
    model.load_state_dict(ck['model'])
    model.eval()
    return model, sv, tv


def load_trans(path, device):
    ck = torch.load(path, map_location=device)
    sv, tv = ck['sv'], ck['tv']
    pe = ck.get('pe', 'sinusoidal')
    #pe = ck.get('pe', 'learned')
    model = build_transformer(len(sv), len(tv), device, pe)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, sv, tv


def eval_trans(model, dl, tv, device):
    model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for src, _, tgt in dl:
            src = src.to(device)
            pred = model.generate(src)
            for i in range(src.size(0)):
                hyps.append(' '.join(tv.dec(pred[i].cpu().tolist())))
                refs.append([' '.join(tv.dec(tgt[i].tolist()))])
    return sacrebleu.corpus_bleu(hyps, refs).score


def translate(model, text, sv, tv, device, mtype):
    toks = zh_tok(text)
    ids = sv.enc(toks)
    src = torch.tensor([ids], device=device)
    sl = torch.tensor([len(ids)], device=device)

    with torch.no_grad():
        if mtype == 'gru':
            pred = model.translate(src, sl)
        else:
            pred = model.generate(src)

    return ' '.join(tv.dec(pred[0].cpu().tolist()))


def batch_translate(model, in_f, out_f, sv, tv, device, mtype):
    results = []
    hyps = []  # 预测结果
    refs = []  # 参考答案

    with open(in_f, 'r', encoding='utf-8') as f:
        for ln in f:
            d = json.loads(ln)
            pred = translate(model, d['zh_hy'], sv, tv, device, mtype)

            # 获取参考答案，如果没有则为空字符串
            ref_text = d.get('en', '')

            results.append({'zh': d['zh'], 'en_ref': ref_text, 'en_pred': pred})

            # 只有当参考答案存在时，才加入BLEU计算列表
            if ref_text:
                hyps.append(pred)
                refs.append(ref_text)

    # 保存预测结果
    with open(out_f, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Saved: {out_f}')

    # 计算BLEU
    if refs:
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        print(f'BLEU: {bleu.score:.2f}')
    else:
        print('No references found for BLEU calculation.')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='trans', choices=['gru', 'trans'])
    p.add_argument('--ckpt', default='./train/ckpts_trans_su_njtk_adm_2e-4_50/trans_sinusoidal_28.68.pt', type=str)
    p.add_argument('--text', type=str)
    p.add_argument('--file', default='/home/test/data/hrh/hw_nlp/datasets/test_hy.jsonl', type=str)
    p.add_argument('--out', default='./train/ckpts_trans_su_njtk_adm_2e-4_50/hy-pred.jsonl')
    return p.parse_args()


def main():
    args = parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model == 'gru':
        model, sv, tv = load_gru(args.ckpt, device)
    else:
        model, sv, tv = load_trans(args.ckpt, device)
    
    if args.text:
        result = translate(model, args.text, sv, tv, device, args.model)
        print(f'In: {args.text}')
        print(f'Out: {result}')
    elif args.file:
        batch_translate(model, args.file, args.out, sv, tv, device, args.model)


if __name__ == '__main__':
    main()

