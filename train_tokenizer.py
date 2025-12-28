import sentencepiece as spm
import json

def train_spm(input_file, model_prefix, vocab_size=8000):
    # 1. 提取所有文本保存临时文件
    txt_file = 'train_temp_corpus_gru.txt'
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(txt_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            # 将中文和英文都写入，让模型同时学习两种语言的子词
            f_out.write(data['zh'] + '\n')
            f_out.write(data['en'] + '\n')

    # 2. 训练 SentencePiece 模型
    # --input: 输入文件
    # --model_prefix: 模型保存名前缀
    # --vocab_size: 词表大小 (小数据集建议 4000-8000，大数据集 32000)
    # --model_type: bpe (机器翻译常用) 或 unigram
    # --character_coverage: 1.0 (覆盖所有字符，中文必须设高，默认0.9995可能会丢弃罕见汉字)
    spm.SentencePieceTrainer.train(
        input=txt_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=[]
    )
    print(f"Model trained and saved to {model_prefix}.model")

if __name__ == '__main__':
    # 假设你的数据文件叫 data.jsonl
    train_spm('./datasets/train_100k.jsonl', 'spm_bpe_gru', vocab_size=8000)