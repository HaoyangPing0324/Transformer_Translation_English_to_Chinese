"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/16
@Desc    : Transformer翻译模型训练/验证/推理函数库（仅函数定义，供调用）
@License : MIT License (MIT)
@Version : 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# 导入自定义模型
from transformer_model import (
  TransformerTranslator,
  generate_padding_mask,
  generate_square_subsequent_mask,
  create_masks
)
from translation_data_processing import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN  # 补充导入UNK_TOKEN

# ====================== 训练/验证核心函数（优化：仅Epoch级打印Loss） ======================
def train_epoch(
    model: TransformerTranslator, loader, criterion: nn.Module,
    optimizer: optim.Optimizer, pad_idx_en: int, pad_idx_zh: int, device: torch.device
) -> float:
    """单轮训练，返回平均损失（仅Epoch级打印，移除批次级打印）"""
    model.train()
    total_loss = 0.0
    for batch_idx, (src, tgt) in enumerate(loader):
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher Forcing：用tgt[:-1]预测tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # 生成掩码
        tgt_mask, src_pad_mask, tgt_pad_mask, memory_pad_mask = create_masks(
            src, tgt_input, pad_idx_en, pad_idx_zh, device
        )

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        output, _ = model(
            src=src,
            tgt=tgt_input,
            src_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask
        )

        # 计算损失（展平维度）
        loss = criterion(
            output.reshape(-1, output.shape[-1]),
            tgt_output.reshape(-1)
        )

        # 反向传播+梯度裁剪（防止梯度爆炸）
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # 移除：每10个批次打印Loss的代码 ↓
        # if (batch_idx + 1) % 10 == 0:
        #     print(f"Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}")

    # 可选：在Epoch结束后打印一次平均Loss（也可由main.py统一打印）
    avg_loss = total_loss / len(loader)
    return avg_loss

def val_epoch(
    model: TransformerTranslator, loader, criterion: nn.Module,
    pad_idx_en: int, pad_idx_zh: int, device: torch.device
) -> float:
    """单轮验证（无梯度计算），返回平均损失"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 生成掩码
            tgt_mask, src_pad_mask, tgt_pad_mask, memory_pad_mask = create_masks(
                src, tgt_input, pad_idx_en, pad_idx_zh, device
            )

            # 前向传播
            output, _ = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_pad_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=memory_pad_mask
            )

            # 计算损失
            loss = criterion(
                output.reshape(-1, output.shape[-1]),
                tgt_output.reshape(-1)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss

# ====================== 推理函数（优化束搜索+解决翻译不完整） ======================
def translate_with_attention(
    model: TransformerTranslator, en_sentence: str, tokenize_en,
    en_vocab, zh_vocab, pad_idx_en: int, pad_idx_zh: int,
    device: torch.device, max_len: int = 100, beam_size: int = 5
) -> Tuple[str, Optional[np.ndarray], List[str], List[str]]:
    """推理并返回翻译结果+注意力权重（优化束搜索，解决长句翻译不完整）
    返回：zh_sentence, attn_weights, en_tokens, zh_tokens
    """
    model.eval()
    with torch.no_grad():
        # 1. 预处理英文句子（增强鲁棒性）
        if not isinstance(en_sentence, str):
            en_sentence = str(en_sentence)
        en_sentence_clean = en_sentence.strip()
        en_tokens = tokenize_en(en_sentence_clean)

        # 处理空token
        if not en_tokens:
            return "", None, [], []

        # 数值化+添加特殊符号（保留原始长度）
        en_indices = [en_vocab.stoi[SOS_TOKEN]] + en_vocab.numericalize(en_tokens) + [en_vocab.stoi[EOS_TOKEN]]
        src = torch.tensor(en_indices).unsqueeze(0).to(device)  # [1, src_len]

        # 2. 初始化束搜索：(序列, 累计对数概率, 注意力权重)
        # 降低EOS优先级：给EOS序列减惩罚分，避免过早终止
        beams: List[Tuple[torch.Tensor, float, Optional[torch.Tensor]]] = [
            (torch.tensor([[zh_vocab.stoi[SOS_TOKEN]]], device=device), 0.0, None)
        ]
        completed_beams: List[Tuple[torch.Tensor, float, Optional[torch.Tensor]]] = []

        # 3. 束搜索主循环（增大max_len，降低EOS优先级）
        for _ in range(max_len):
            if not beams:
                break

            new_beams: List[Tuple[torch.Tensor, float, Optional[torch.Tensor]]] = []
            for seq, score, attn in beams:
                # 如果已到EOS，加入完成束但减惩罚分（避免过早终止）
                if seq[0, -1] == zh_vocab.stoi[EOS_TOKEN]:
                    completed_beams.append((seq, score - 0.5, attn))  # 减0.5惩罚分
                    continue

                # 生成掩码（适配长序列）
                tgt_mask = generate_square_subsequent_mask(seq.shape[1], device)
                src_pad_mask = generate_padding_mask(src, pad_idx_en).squeeze(1)
                tgt_pad_mask = generate_padding_mask(seq, pad_idx_zh).squeeze(1)

                # 前向传播
                output, current_attn = model(
                    src=src,
                    tgt=seq,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    memory_key_padding_mask=src_pad_mask
                )

                # 取最后一个token的概率（对数概率，避免下溢）
                log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                # 取top-k候选（束宽自适应）
                top_probs, top_idxs = log_probs.topk(beam_size, dim=-1)

                # 扩展束（保留更多候选）
                for i in range(beam_size):
                    new_score = score + top_probs[0, i].item()
                    new_idx = top_idxs[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, new_idx], dim=1)
                    new_beams.append((new_seq, new_score, current_attn))

            # 按得分排序，保留top beam_size个束（增加容错）
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        # 4. 合并完成束和未完成束，选最优（优先未完成束，避免截断）
        all_beams = completed_beams + beams
        if not all_beams:
            return "", None, en_tokens, []

        # 优化：优先选择长度更长的序列（解决翻译不完整）
        def beam_score(beam):
            seq, score, _ = beam
            # 长度奖励：越长的序列加分，鼓励生成完整翻译
            length_reward = len(seq[0]) * 0.01
            return score + length_reward

        best_seq, _, best_attn = max(all_beams, key=beam_score)

        # 5. 还原中文句子（优化token过滤）
        zh_indices = best_seq.squeeze(0).cpu().numpy()
        zh_tokens = []
        for idx in zh_indices:
            if idx < len(zh_vocab.itos):  # 防止索引越界
                tok = zh_vocab.itos[idx]
                if tok not in [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]:  # 过滤UNK
                    zh_tokens.append(tok)
        # 处理空翻译
        if not zh_tokens:
            zh_sentence = ""
        else:
            zh_sentence = ''.join(zh_tokens)

        # 6. 处理注意力权重（适配长序列）
        attn_weights = None
        if best_attn is not None:
            attn_weights = best_attn.squeeze(0).cpu().numpy()
            # 智能截断到有效token长度
            attn_weights = attn_weights[:len(zh_tokens)+1, :len(en_tokens)+2]

        return zh_sentence, attn_weights, en_tokens, zh_tokens

# ====================== 注意力热图绘制（兼容低版本seaborn/matplotlib） ======================
def plot_attention_heatmap(
    attn_weights: np.ndarray, en_tokens: List[str], zh_tokens: List[str],
    save_path: str = 'attention_heatmap.png'
):
    """绘制注意力热图（兼容低版本库，移除不兼容参数）"""
    # 预处理token，避免空值/过长
    en_tokens_full = [SOS_TOKEN] + en_tokens + [EOS_TOKEN]
    zh_tokens_full = [SOS_TOKEN] + zh_tokens + [EOS_TOKEN]

    # 截断注意力权重到匹配token长度（防止维度不匹配）
    attn_weights = attn_weights[:len(zh_tokens_full), :len(en_tokens_full)]

    # 增大画布+优化显示（兼容低版本）
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        attn_weights,
        cmap='viridis',
        xticklabels=en_tokens_full,
        yticklabels=zh_tokens_full,
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'}
    )
    # 单独设置旋转角度（替代不兼容的xticklabels_rotation参数）
    plt.xticks(rotation=45)  # 英文token旋转45度
    plt.yticks(rotation=0)   # 中文token不旋转
    plt.xlabel('Source (English) Tokens', fontsize=12)
    plt.ylabel('Target (Chinese) Tokens', fontsize=12)
    plt.title('Encoder-Decoder Attention Weight Heatmap', fontsize=14)
    plt.tight_layout()  # 适配布局
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 高清保存
    plt.close()  # 关闭画布，释放内存

# ====================== Loss曲线绘制（优化） ======================
def plot_loss_curve(
    train_losses: List[float], val_losses: List[float],
    save_path: str = 'loss_curve.png'
):
    """绘制训练/验证Loss曲线（优化样式）"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses,
             label='Train Loss', marker='o', linewidth=2, markersize=6)
    plt.plot(range(1, len(val_losses)+1), val_losses,
             label='Val Loss', marker='s', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ====================== 模型保存/加载（无修改） ======================
def save_model(
    model: TransformerTranslator, optimizer: optim.Optimizer,
    train_losses: List[float], val_losses: List[float],
    save_path: str = 'transformer_translator.pth'
):
    """保存模型权重和训练状态"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, save_path)

def load_model(
    model: TransformerTranslator, optimizer: optim.Optimizer,
    load_path: str = 'transformer_translator.pth', device: torch.device = torch.device('cpu')
) -> Tuple[List[float], List[float]]:
    """加载模型权重和训练状态，返回train_losses/val_losses"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    print(f"模型已从：{load_path} 加载")
    return train_losses, val_losses