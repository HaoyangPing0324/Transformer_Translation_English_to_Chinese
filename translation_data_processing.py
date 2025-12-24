"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/16
@Desc    : 使用 Transformer模型 进行英文到中文翻译（数据处理）
@License : MIT License (MIT)
@Version : 1.0
"""

#### 导入库
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
import re

# -------------------------------
# 1. 数据加载与预处理（终极优化）
# -------------------------------

# 加载预训练的 BERT tokenizer
# 增加do_lower_case确保英文小写化，提升词表复用率
tokenizer_en = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    cache_dir="../data",
    local_files_only=True,
    do_lower_case=True  # 新增：英文强制小写
)
tokenizer_zh = BertTokenizer.from_pretrained(
    'bert-base-chinese',
    cache_dir="../data",
    local_files_only=True
)

# 文本清理函数（增强版）
def clean_text(text: str) -> str:
    """清理文本：去除多余空格、特殊符号，修复常见格式问题"""
    if not isinstance(text, str):
        text = str(text)
    # 去除多余空格（多个空格→单个）
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格
    text = text.strip()
    # 去除不可见字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 保留常见标点（避免语义丢失）
    text = re.sub(r'[^\w\s.,!?;:()""''-]', '', text)
    return text

# 分词函数（增强鲁棒性）
def tokenize_en(sentence):
    """英文分词：新增文本清理+空值处理+长度适配"""
    # 空值/非字符串处理
    if sentence is None or sentence == "":
        return []
    # 文本清理
    sentence = clean_text(sentence)
    # 分词（BERT tokenizer已处理子词）
    tokens = tokenizer_en.tokenize(sentence)
    # 放宽长度限制（适配80长度训练）
    return tokens[:80]  # 限制为80，匹配Config.MAX_LEN

def tokenize_zh(sentence):
    """中文分词：新增文本清理+空值处理+长度适配"""
    if sentence is None or sentence == "":
        return []
    sentence = clean_text(sentence)
    tokens = tokenizer_zh.tokenize(sentence)
    # 中文长度放宽到100（适配英文长句）
    return tokens[:100]

# 特殊token（保持不变）
PAD_TOKEN = '<pad>'  # 用于填充（padding）序列，使得同一批次中的所有句子长度一致
SOS_TOKEN = '<sos>'  # 表示句子的开始（Start Of Sentence）
EOS_TOKEN = '<eos>'  # 表示句子的结束（End Of Sentence）
UNK_TOKEN = '<unk>'  # 表示未知词（Unknown Token）

# 构建词表（核心优化：解决[UNK]问题）
class Vocab:
    def __init__(self, tokens, max_size=20000, min_freq=1):  # 关键修改：
        # 1. max_size从10000→20000（英文词表扩容）
        # 2. min_freq从2→1（保留所有出现过的token，消除[UNK]）
        self.freq = {}
        # 优化：遍历所有token，统计词频（去重前）
        for token in tokens:
            if token.strip() == "":  # 过滤空token
                continue
            self.freq[token] = self.freq.get(token, 0) + 1

        # 根据词频降序排序
        sorted_tokens = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)

        # 保留特殊token
        self.itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        for token, count in sorted_tokens:
            if count < min_freq:    # 过滤低频token（min_freq=1，无过滤）
                continue
            if token in self.itos:  # 已有特殊token
                continue
            self.itos.append(token)
            if len(self.itos) >= max_size:
                break
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def numericalize(self, tokens):
        """数值化：增加空token处理"""
        if not tokens:  # 空token列表
            return []
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens]

    def __len__(self):
        return len(self.itos)

# 自定义Dataset类（适配长序列+增强鲁棒性）
class TranslationDataset(Dataset):
    def __init__(self, data, en_vocab=None, zh_vocab=None, build_vocab=False, max_samples=None):
        """
        data: 包含 'translation' 字段的列表，每个元素为字典，如 {'en':..., 'zh':...}
        build_vocab: 如果为True，则构建词表
        max_samples: 可选，限制样本数量（比如不少于 2万对）
        """
        # 空数据处理
        if not data:
            self.data = []
            self.en_sentences = []
            self.zh_sentences = []
            self.en_vocab = en_vocab
            self.zh_vocab = zh_vocab
            return

        # 限制样本数量
        if max_samples and max_samples > 0:
            data = data[:max_samples]
        self.data = data

        # 分词后的句子存储（优化：增加异常处理）
        self.en_sentences = []
        self.zh_sentences = []
        for item in self.data:
            try:
                # 确保item是字典且包含translation字段
                if not isinstance(item, dict) or 'translation' not in item:
                    continue
                # 分词
                en_tokens = tokenize_en(item['translation'].get('en', ""))
                zh_tokens = tokenize_zh(item['translation'].get('zh', ""))
                # 过滤空句子（放宽长度限制）
                if len(en_tokens) < 1 or len(zh_tokens) < 1:
                    continue
                self.en_sentences.append(en_tokens)
                self.zh_sentences.append(zh_tokens)
            except Exception as e:
                # 跳过异常样本，不中断程序
                continue

        # 构建词表（核心优化：扩容中文词表）
        if build_vocab:
            # 合并所有token（过滤空列表）
            all_en_tokens = [token for sent in self.en_sentences if sent for token in sent]
            all_zh_tokens = [token for sent in self.zh_sentences if sent for token in sent]
            # 构建词表（增大词表容量）
            self.en_vocab = Vocab(all_en_tokens, max_size=20000)  # 英文词表扩容到20000
            self.zh_vocab = Vocab(all_zh_tokens, max_size=8000)   # 中文词表从5000→8000
        else:
            self.en_vocab = en_vocab
            self.zh_vocab = zh_vocab

    def __len__(self):
        return len(self.en_sentences)  # 修正：返回有效样本数

    def __getitem__(self, idx):
        """获取样本：增加越界处理+空token处理+长序列支持"""
        try:
            # 越界处理
            if idx >= len(self.en_sentences) or idx >= len(self.zh_sentences):
                # 返回空序列（由collate_fn处理padding）
                en_indices = [self.en_vocab.stoi[SOS_TOKEN], self.en_vocab.stoi[EOS_TOKEN]]
                zh_indices = [self.zh_vocab.stoi[SOS_TOKEN], self.zh_vocab.stoi[EOS_TOKEN]]
                return torch.tensor(en_indices), torch.tensor(zh_indices)

            # 数值化，并加上 SOS 和 EOS
            en_tokens = self.en_sentences[idx]
            zh_tokens = self.zh_sentences[idx]

            en_indices = [self.en_vocab.stoi[SOS_TOKEN]] + \
                        self.en_vocab.numericalize(en_tokens) + \
                        [self.en_vocab.stoi[EOS_TOKEN]]
            zh_indices = [self.zh_vocab.stoi[SOS_TOKEN]] + \
                        self.zh_vocab.numericalize(zh_tokens) + \
                        [self.zh_vocab.stoi[EOS_TOKEN]]

            # 适配长序列训练（放宽到100）
            max_seq_len = 100
            en_indices = en_indices[:max_seq_len]
            zh_indices = zh_indices[:max_seq_len]

            return torch.tensor(en_indices), torch.tensor(zh_indices)
        except Exception as e:
            # 异常样本返回默认值
            en_indices = [self.en_vocab.stoi[SOS_TOKEN], self.en_vocab.stoi[EOS_TOKEN]]
            zh_indices = [self.zh_vocab.stoi[SOS_TOKEN], self.zh_vocab.stoi[EOS_TOKEN]]
            return torch.tensor(en_indices), torch.tensor(zh_indices)

def create_dataloader(
        data_list,  # 原始数据列表（如官方train/validation/test的list）
        max_length,  # 最大句子长度（如80）
        max_samples,  # 最大样本量（如50000）
        batch_size,  # batch size（如16）
        build_vocab,  # 是否构建词表（训练集设为True，其他设为False）
        shuffle=False,  # 是否打乱数据（训练集设为True，其他设为False）
        en_vocab=None,  # 可选，外部传入的英文词表（非训练集时需要）
        zh_vocab=None  # 可选，外部传入的中文词表（非训练集时需要）
):
    """
    生成指定配置的DataLoader，用于处理官方的train/validation/test数据集
    优化：适配长序列+增强鲁棒性
    """
    # 空数据处理
    if not data_list:
        raise ValueError("数据列表不能为空！")

    # 步骤1：定义长度过滤函数（适配80长度训练）
    def filter_length(item):
        """过滤函数：支持更长的序列（max_length=80）"""
        try:
            if not isinstance(item, dict) or 'translation' not in item:
                return False
            en_tokens = tokenize_en(item['translation'].get('en', ""))
            zh_tokens = tokenize_zh(item['translation'].get('zh', ""))
            # 过滤空token和超长token（放宽到80）
            return (0 < len(en_tokens) <= max_length) and (0 < len(zh_tokens) <= max_length + 20)  # 中文多20个长度
        except Exception as e:
            return False

    # 步骤2：执行数据过滤（优化：保留更多有效样本）
    filtered_data = list(filter(filter_length, data_list))
    if not filtered_data:
        raise ValueError(f"过滤后无有效数据！max_length={max_length}可能过小")

    # 步骤3：限制最大样本量
    if max_samples and max_samples > 0:
        limited_data = filtered_data[:max_samples]
    else:
        limited_data = filtered_data
    print(f"过滤并限制后的数据量：{len(limited_data)}")

    # 步骤4：构建TranslationDataset
    if build_vocab:
        # 训练集：构建词表
        dataset = TranslationDataset(limited_data, build_vocab=True, max_samples=max_samples)
        # 保存词表
        en_vocab = dataset.en_vocab
        zh_vocab = dataset.zh_vocab
        print(f"英文词表大小：{len(en_vocab)}, 中文词表大小：{len(zh_vocab)}")
    else:
        # 验证/测试集：使用外部传入的词表
        if en_vocab is None or zh_vocab is None:
            raise ValueError("非训练集（build_vocab=False）时，必须传入en_vocab和zh_vocab！")
        dataset = TranslationDataset(
            limited_data,
            build_vocab=False,
            en_vocab=en_vocab,
            zh_vocab=zh_vocab,
            max_samples=max_samples
        )

    # 步骤5：定义collate_fn（优化：处理空序列）
    def collate_fn(batch):
        # 拆分批次中的英文和中文张量
        en_batch, zh_batch = zip(*batch)
        # 过滤空张量
        en_batch = [t for t in en_batch if len(t) > 0]
        zh_batch = [t for t in zh_batch if len(t) > 0]
        # 空批次处理
        if not en_batch or not zh_batch:
            return torch.empty(0), torch.empty(0)
        # 对张量进行padding，统一长度
        pad_idx_en = en_vocab.stoi[PAD_TOKEN]
        pad_idx_zh = zh_vocab.stoi[PAD_TOKEN]
        en_batch_padded = pad_sequence(en_batch, padding_value=pad_idx_en, batch_first=True)
        zh_batch_padded = pad_sequence(zh_batch, padding_value=pad_idx_zh, batch_first=True)
        return en_batch_padded, zh_batch_padded

    # 步骤6：生成并返回DataLoader（优化：设置num_workers=0，避免多进程问题）
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,  # 新增：禁用多进程，避免Windows下的兼容性问题
        pin_memory=True  # 新增：启用内存锁定，提升GPU传输速度
    )

    # 返回结果
    if build_vocab:
        return dataloader, en_vocab, zh_vocab
    else:
        return dataloader