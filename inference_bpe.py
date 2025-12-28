import argparse
import json
import torch
import sacrebleu
from tqdm import tqdm
import cfg
from mydata import SPTokenizer
from deep_gru import build_gru_model
from trans_pe import build_transformer

def load_model_checkpoint(path, model_type, device):
    """
    加载模型和Tokenizer
    """
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device)

    # 从checkpoint中恢复Tokenizer对象 (pickle)
    sv = checkpoint['sv']
    tv = checkpoint['tv']

    if model_type == 'gru':
        model = build_gru_model(len(sv), len(tv), device)
    else:
        # Transformer
        # pe_type = checkpoint.get('pe', 'sinusoidal')
        pe_type = checkpoint.get('pe', 'learned')
        model = build_transformer(len(sv), len(tv), device, pe_type)

    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model, sv, tv

def translate_single_sentence(model, text, sv, tv, device, model_type):
    """
    单句推理函数
    """
    # 修正点：mydata.py中的SPTokenizer可以直接encode字符串，
    # 不需要像inference.py那样先调用zh_tok
    ids = sv.enc(text)

    # 转换为Tensor
    src = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        if model_type == 'gru':
            # GRU 需要传入长度张量
            sl = torch.tensor([len(ids)], device=device)
            pred_ids = model.translate(src, sl)
        else:
            # Transformer (假设实现了 generate 方法)
            pred_ids = model.generate(src)

    if isinstance(pred_ids, torch.Tensor):
        pred_ids = pred_ids.squeeze(0).cpu().tolist()

    return tv.dec(pred_ids)

def evaluate_file(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    model, sv, tv = load_model_checkpoint(args.ckpt, args.model, device)

    results = []
    hyps = [] # 预测列表
    refs = [] # 参考答案列表

    print(f"Starting inference on {args.input}...")

    # 2. 读取测试文件并推理
    with open(args.input, 'r', encoding='utf-8') as f:
        # 读取所有行以便显示进度条
        lines = f.readlines()

        for line in tqdm(lines, desc="Translating"):
            data = json.loads(line)
            src_text = data['zh_hy']
            ref_text = data.get('en', '') # 测试集必须包含 'en' 字段才能计算BLEU

            # 执行翻译
            pred_text = translate_single_sentence(model, src_text, sv, tv, device, args.model)

            # 收集结果
            results.append({
                'zh': src_text,
                'en_ref': ref_text,
                'en_pred': pred_text
            })

            if ref_text:
                hyps.append(pred_text)
                refs.append(ref_text)

    # 3. 保存预测结果
    with open(args.output, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    print(f"Predictions saved to {args.output}")

    # 4. 计算 BLEU
    if refs:
        # SacreBLEU expects list of references where each reference is a list of sentences
        # refs needs to be wrapped: [ [ref1_sent1, ref1_sent2...], [ref2_sent1 (optional)...] ]
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        print("-" * 30)
        print(f"BLEU Score: {bleu.score:.2f}")
        print(f"Details: {bleu}")
        print("-" * 30)
    else:
        print("Warning: No reference translations found in input file. BLEU not calculated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Translation Model")

    # 默认路径根据 cfg.py 和 inference.py 中的注释设定
    parser.add_argument('--model', default='trans', choices=['gru', 'trans'], help='Model architecture')
    # 请根据实际训练好的 .pt 文件路径修改 default
    parser.add_argument('--ckpt', default='./train/ckpts_trans_le_bpe_adm_2e-4_50/trans_learned_89.77.pt', type=str, help='Path to model checkpoint (.pt)')
    parser.add_argument('--input', default='./datasets/test_hy.jsonl', type=str, help='Input test file (jsonl)')
    parser.add_argument('--output', default='./train/ckpts_trans_le_bpe_adm_2e-4_50/hy_pred.jsonl', type=str, help='Output predictions file')

    args = parser.parse_args()

    # 检查输入文件是否存在
    import os
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        # 尝试使用 cfg 中的默认值
        if os.path.exists(cfg.data.test):
            print(f"Falling back to cfg.data.test: {cfg.data.test}")
            args.input = cfg.data.test
        else:
            exit(1)

    evaluate_file(args)