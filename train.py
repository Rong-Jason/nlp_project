"""
方案6: 训练脚本
"""
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import sacrebleu
import csv

import cfg
from datapro import get_loaders
from deep_gru import build_gru_model, get_tf_ratio
from trans_pe import build_transformer


def train_gru():
    """训练Deep GRU (2 layers)"""
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    tr_dl, va_dl, sv, tv = get_loaders()

    model = build_gru_model(len(sv), len(tv), device, cfg.gru.attention)

    # 1. 创建并初始化 CSV 文件
    os.makedirs(cfg.output.model_dir, exist_ok=True)
    log_path = os.path.join(cfg.output.model_dir, 'train_log.csv')

    # 打开文件准备写入，使用 'w' 模式覆盖或创建
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(log_file)
    # 写入表头
    csv_writer.writerow(['epoch', 'loss', 'bleu', 'tf_ratio'])

    opt = optim.AdamW(model.parameters(), lr=cfg.train.lr)
    # opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    # 2. 定义调度器: ReduceLROnPlateau
    # mode='max': 我们希望 BLEU 越大越好 (如果监控 loss 则用 'min')
    # factor=0.5: 触发时，学习率变为原来的 0.5 倍
    # patience=5: 如果 5 个 epoch 指标都没提升，才降低 LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5
    )

    best = 0
    try:
        for ep in range(cfg.train.epochs):
            model.train()
            tf = get_tf_ratio(ep)
            loss_sum = 0

            for src, sl, tgt in tqdm(tr_dl, desc=f'Ep{ep + 1}'):
                src, sl, tgt = src.to(device), sl.to(device), tgt.to(device)
                opt.zero_grad()
                out = model(src, sl, tgt, tf)
                loss = crit(out[:, 1:].reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                opt.step()
                loss_sum += loss.item()

            avg_loss = loss_sum / len(tr_dl)
            bleu = eval_gru(model, va_dl, tv, device)

            print(f'Ep{ep + 1}: loss={avg_loss:.4f} bleu={bleu:.2f} tf={tf:.2f}')
            # current_lr = opt.param_groups[0]['lr'] # 获取当前的学习率 (用于记录)
            # scheduler.step(avg_loss)

            # 3. 写入log数据
            csv_writer.writerow([ep + 1, avg_loss, bleu, tf])
            log_file.flush()  # 立即写入硬盘，防止程序崩溃数据丢失

            if ep % 2 == 0:
                best = bleu
                torch.save({'model': model.state_dict(), 'sv': sv, 'tv': tv},
                           f'{cfg.output.model_dir}/{bleu:.2f}.pt')
    finally:
        # --- 3. 关闭文件 ---
        log_file.close()

    print(f'Best: {best:.2f}')


def eval_gru(model, dl, tv, device):
    model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for src, sl, tgt in dl:
            src, sl = src.to(device), sl.to(device)
            pred = model.translate(src, sl)
            for i in range(src.size(0)):
                hyps.append(' '.join(tv.dec(pred[i].cpu().tolist())))
                refs.append([' '.join(tv.dec(tgt[i].tolist()))])
    return sacrebleu.corpus_bleu(hyps, refs).score


def train_transformer(pe_type='learned'):# sinusoidal learned
    """训练Transformer (指定位置编码类型)"""
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    tr_dl, va_dl, sv, tv = get_loaders()

    model = build_transformer(len(sv), len(tv), device, pe_type)

    # 1. 创建 CSV 文件 (文件名包含 pe_type)
    os.makedirs(cfg.output.model_dir, exist_ok=True)
    log_path = os.path.join(cfg.output.model_dir, f'trans_{pe_type}.csv')

    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['epoch', 'loss', 'bleu'])  # 表头

    opt = optim.AdamW(model.parameters(), lr=cfg.train.lr)
    # opt = optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=cfg.train.label_smooth)

    # 2. 定义调度器: ReduceLROnPlateau
    # mode='max': 我们希望 BLEU 越大越好 (如果监控 loss 则用 'min')
    # factor=0.5: 触发时，学习率变为原来的 0.5 倍
    # patience=5: 如果 5 个 epoch 指标都没提升，才降低 LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.1, patience=5
    )

    best = 0
    try:
        for ep in range(cfg.train.epochs):
            model.train()
            loss_sum = 0

            for src, _, tgt in tqdm(tr_dl, desc=f'Ep{ep + 1}'):
                src, tgt = src.to(device), tgt.to(device)
                opt.zero_grad()
                out = model(src, tgt[:, :-1])
                loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
                opt.step()
                loss_sum += loss.item()

            avg_loss = loss_sum / len(tr_dl)
            bleu = eval_trans(model, va_dl, tv, device)

            print(f'Ep{ep + 1}: loss={avg_loss:.4f} bleu={bleu:.2f}')
            # current_lr = opt.param_groups[0]['lr'] # 获取当前的学习率 (用于记录)
            # scheduler.step(avg_loss)

            # 3. 写入log数据
            csv_writer.writerow([ep + 1, avg_loss, bleu])
            log_file.flush()  # 确保实时写入

            if ep % 2 == 0:
                best = bleu
                torch.save({'model': model.state_dict(), 'sv': sv, 'tv': tv, 'pe': pe_type},
                           f'{cfg.output.model_dir}/trans_{pe_type}_{bleu:.2f}.pt')
    finally:
        log_file.close()

    print(f'Best ({pe_type}): {best:.2f}')


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


def compare_pe():
    """对比两种位置编码"""
    print("=== Sinusoidal PE ===")
    train_transformer('sinusoidal')
    print("\n=== Learned PE ===")
    train_transformer('learned')


if __name__ == '__main__':
    import sys

    # mode = sys.argv[1] if len(sys.argv) > 1 else 'gru'
    mode = cfg.train.mode

    if mode == 'gru':
        train_gru()
    elif mode == 'trans':
        train_transformer()
    elif mode == 'compare':
        compare_pe()
