# 机器翻译项目 (Machine Translation Project)

本项目旨在探索和对比基于 RNN (GRU) 与 Transformer 架构的中英机器翻译模型性能。项目包含深度 GRU 网络的实现以及 Transformer 位置编码的对比实验。

## ✨ 核心特点 (Core Features)

本项目涵盖了以下核心实验方案：

1.  **深层 GRU 架构 (Deep GRU Architecture)**
    * 实现了 **2层** GRU 结构。
    * 旨在深入探索循环神经网络在序列翻译任务中的效果与瓶颈。

2.  **指数 Teacher Forcing 衰减 (Exponential Teacher Forcing Decay)**
    * 引入动态调整的训练策略。
    * 随着训练进行，Teacher Forcing 的比例呈指数级衰减，实现从“强制教学”到“自回归生成”的平滑过渡，缓解 Exposure Bias 问题。

3.  **位置编码对比实验 (Position Encoding Comparison)**
    * 针对 Transformer 模型，专门设计了对比实验。
    * 对比 **正弦位置编码 (Sinusoidal)** 与 **可学习位置编码 (Learned)** 对模型性能的影响。

---

## 🛠️ 环境依赖 (Dependencies)

在运行代码之前，请确保安装以下 Python 库：

```bash
pip install torch jieba sacrebleu tqdm nltk
