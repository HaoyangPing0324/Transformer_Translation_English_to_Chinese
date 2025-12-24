"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/16
@Desc    : 使用 Transformer模型 进行英文到中文翻译（模型定义）
@License : MIT License (MIT)
@Version : 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# 导入数据处理文件中的特殊token（需保证translation_data_processing.py在同级目录）
from translation_data_processing import PAD_TOKEN
# -------------------------------

# 位置编码（Positional Encoding）
# -------------------------------
class PositionalEncoding(nn.Module):
    """适配batch_first=True的位置编码（翻译任务专用）"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 生成位置编码核心矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # 偶数维sin，奇数维cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展batch维度 (1, max_len, d_model)，方便广播到任意batch_size
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不参与参数更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]（翻译任务标准输入格式）
        返回：加位置编码+Dropout后的特征
        """
        # 截取前seq_len个位置编码，广播相加
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        return x

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        生成后续token掩码（Decoder自注意力用），防止看到未来token
        sz: 序列长度
        返回: (sz, sz) 的掩码矩阵，True表示该位置被mask
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def generate_padding_mask(src: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """生成padding掩码（Encoder/Decoder的注意力用），忽略PAD_TOKEN"""
    mask = (src == pad_idx).unsqueeze(1)  # (batch_size, 1, seq_len)
    return mask

def create_masks(
            src: torch.Tensor, tgt: torch.Tensor,
            pad_idx_en: int, pad_idx_zh: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成训练/推理所需的所有掩码"""
        batch_size, src_len = src.shape
        batch_size, tgt_len = tgt.shape

        # 1. Decoder后续token掩码
        tgt_mask = generate_square_subsequent_mask(tgt_len, device)
        # 2. 源语言padding掩码
        src_key_padding_mask = generate_padding_mask(src, pad_idx_en).squeeze(1)
        # 3. 目标语言padding掩码
        tgt_key_padding_mask = generate_padding_mask(tgt, pad_idx_zh).squeeze(1)
        # 4. Memory padding掩码
        memory_key_padding_mask = src_key_padding_mask

        return tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask

# ======================  单个Encoder层 ======================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 多头自注意力层（自注意力：关注英文句子内部的依赖）
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 自注意力层
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化
        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)  # 残差连接
        src = self.norm2(src)  # 层归一化
        return src

# ======================  单个Decoder层 ======================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 新增：保存交叉注意力权重
        self.attention_weights = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 1. 掩码自注意力
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. 交叉注意力（保存权重）
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        self.attention_weights = attn_weights  # 保存权重：[batch_size, tgt_len, src_len]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3. 前馈网络
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

# ====================== 4. 完整的Transformer翻译模型 ======================
class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        # 英文词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)  # 0是PAD_TOKEN索引
        # 中文词嵌入层
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        # 位置编码层
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)

        # 自定义Encoder（由多个EncoderLayer组成）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        # 自定义Decoder（由多个DecoderLayer组成）
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_key_padding_mask=None):
        # 词嵌入 + 位置编码
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # 经过所有EncoderLayer
        for layer in self.encoder_layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)
        return src  # [batch_size, src_len, d_model]

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 词嵌入 + 位置编码
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)

        # 新增：自动生成后续token掩码
        if tgt_mask is None and tgt.size(1) > 0:
            tgt_mask = generate_square_subsequent_mask(tgt.size(1), tgt.device)

        # 保存最后一层的Encoder-Decoder Attention权重
        attn_weights = None
        # 经过所有DecoderLayer
        for i, layer in enumerate(self.decoder_layers):
            tgt = layer(tgt, memory, tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
            # 提取最后一层的交叉注意力权重（需要修改DecoderLayer，让它返回权重）
            if i == len(self.decoder_layers) - 1:
                attn_weights = layer.attention_weights  # 正确：取DecoderLayer的attention_weights属性  # 新增：保存权重
        return tgt, attn_weights

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Encoder
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        # Decoder（返回输出和Attention权重）
        output, attn_weights = self.decode(tgt, memory, tgt_mask=tgt_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask,
                                           memory_key_padding_mask=memory_key_padding_mask)
        # 输出层
        output = self.fc_out(output)
        return output, attn_weights  # 同时返回输出和权重