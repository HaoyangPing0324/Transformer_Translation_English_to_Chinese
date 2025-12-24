"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/22
@Desc    : 使用 Transformer模型 进行英文到中文翻译（主函数）
@License : MIT License (MIT)
@Version : 1.0
"""

#### 导入库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import BertTokenizer
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
import translation_data_processing as tdp
from transformer_model import TransformerTranslator
from trainer import (
    train_epoch, val_epoch, translate_with_attention,
    plot_loss_curve, plot_attention_heatmap, save_model, load_model
)

# ====================== 全局配置（终极优化：解决翻译不完整+[UNK]） ======================
class Config:
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型超参数（增大模型规模提升效果）
    D_MODEL = 512  # 从256→512，增强特征表达
    NHEAD = 8      # 从4→8，提升注意力捕捉能力
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024  # 从512→1024，增强前馈网络能力
    DROPOUT = 0.1
    MAX_LEN = 80   # 从50→80，大幅放宽训练序列长度限制（适配中英文长度差异）
    # 训练超参数（优化训练策略）
    EPOCHS = 20    # 从10→20，增加训练轮数
    BATCH_SIZE = 16  # 从32→16，适配80长度+512维度的显存需求（避免OOM）
    LR = 5e-4      # 从1e-3→5e-4，减小学习率避免震荡
    # 数据配置（增大训练数据量）
    MAX_TRAIN_SAMPLES = 50000  # 从28000→50000，用更多数据训练
    MAX_VAL_SAMPLES = 5000
    MAX_TEST_SAMPLES = 10000
    # 推理配置（单独定义推理最大长度，避免和训练长度耦合）
    INFER_MAX_LEN = 100  # 推理时中文翻译的最大长度（远大于训练长度，确保完整）
    BEAM_SIZE = 5        # 束搜索宽度（从3→5，提升长句生成质量）
    # 保存路径
    MODEL_SAVE_PATH = 'transformer_translator_best.pth'
    LOSS_CURVE_PATH = 'loss_curve_optimized.png'

# ====================== 主函数 ======================
def main():
    # 1. 初始化配置
    cfg = Config()
    print(f"使用设备：{cfg.DEVICE}")
    print(f"训练序列最大长度：{cfg.MAX_LEN} | 推理翻译最大长度：{cfg.INFER_MAX_LEN} | 束搜索宽度：{cfg.BEAM_SIZE}")
    print("="*50 + " 加载数据集 " + "="*50)

    # 2. 加载数据集（Parquet格式）
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": "../data/iwslt2017/iwslt2017-en-zh/iwslt2017-train.parquet",
            "validation": "../data/iwslt2017/iwslt2017-en-zh/iwslt2017-validation.parquet",
            "test": "../data/iwslt2017/iwslt2017-en-zh/iwslt2017-test.parquet"
        }
    )

    # 3. 构建DataLoader（训练集构建词表，验证/测试集复用）
    # 注：词表扩容需配合translation_data_processing.py的Vocab类修改（max_size=20000, min_freq=1）
    train_loader, en_vocab, zh_vocab = tdp.create_dataloader(
        data_list=list(dataset['train']),
        max_length=cfg.MAX_LEN,
        max_samples=cfg.MAX_TRAIN_SAMPLES,
        batch_size=cfg.BATCH_SIZE,
        build_vocab=True,
        shuffle=True
    )

    # 4. 处理验证集（build_vocab=False，传入训练集的词表，shuffle=False）
    val_loader = tdp.create_dataloader(
        data_list=list(dataset['validation']),
        max_length=cfg.MAX_LEN,
        max_samples=cfg.MAX_VAL_SAMPLES,
        batch_size=cfg.BATCH_SIZE,
        build_vocab=False,
        shuffle=False,
        en_vocab=en_vocab,
        zh_vocab=zh_vocab
    )

    # 5. 处理测试集（和验证集一样）
    test_loader = tdp.create_dataloader(
        data_list=list(dataset['test']),
        max_length=cfg.MAX_LEN,
        max_samples=cfg.MAX_TEST_SAMPLES,
        batch_size=cfg.BATCH_SIZE,
        build_vocab=False,
        shuffle=False,
        en_vocab=en_vocab,
        zh_vocab=zh_vocab
    )

    # 6. 获取PAD索引
    pad_idx_en = en_vocab.stoi[tdp.PAD_TOKEN] if tdp.PAD_TOKEN in en_vocab.stoi else 0
    pad_idx_zh = zh_vocab.stoi[tdp.PAD_TOKEN] if tdp.PAD_TOKEN in zh_vocab.stoi else 0
    print(f"英文PAD索引：{pad_idx_en} | 中文PAD索引：{pad_idx_zh}")
    print(f"英文词表大小：{len(en_vocab)} | 中文词表大小：{len(zh_vocab)}")

    print("="*50 + " 初始化模型 " + "="*50)
    # 7. 初始化模型、损失函数、优化器
    model = TransformerTranslator(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(zh_vocab),
        d_model=cfg.D_MODEL,
        nhead=cfg.NHEAD,
        num_encoder_layers=cfg.NUM_ENCODER_LAYERS,
        num_decoder_layers=cfg.NUM_DECODER_LAYERS,
        dim_feedforward=cfg.DIM_FEEDFORWARD,
        dropout=cfg.DROPOUT,
        max_len=1000  # 位置编码最大长度（远大于推理长度，避免位置编码不足）
    ).to(cfg.DEVICE)

    # 损失函数（忽略PAD_TOKEN的损失）
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx_zh)
    # 优化器（增加权重衰减，防止过拟合）
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    # 学习率调度器（仅删除verbose=True，修复低版本PyTorch兼容问题）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2  # 移除verbose=True
    )

    print("="*50 + " 开始训练 " + "="*50)
    # 8. 多轮训练+验证
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, cfg.EPOCHS + 1):
        # 训练
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            pad_idx_en=pad_idx_en,
            pad_idx_zh=pad_idx_zh,
            device=cfg.DEVICE
        )
        # 验证
        val_loss = val_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            pad_idx_en=pad_idx_en,
            pad_idx_zh=pad_idx_zh,
            device=cfg.DEVICE
        )

        # 学习率调度
        scheduler.step(val_loss)

        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 打印本轮结果
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model=model,
                optimizer=optimizer,
                train_losses=train_losses,
                val_losses=val_losses,
                save_path=cfg.MODEL_SAVE_PATH
            )

    # 9. 绘制Loss曲线
    plot_loss_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        save_path=cfg.LOSS_CURVE_PATH
    )

    print("="*50 + " 推理测试 " + "="*50)
    # 10. 加载最优模型（可选，也可直接用训练后的模型）
    train_losses, val_losses = load_model(
        model=model,
        optimizer=optimizer,
        load_path=cfg.MODEL_SAVE_PATH,
        device=cfg.DEVICE
    )

    # 11. 从测试集随机抽取10个样本推理（保留10个，不修改）
    test_samples = random.sample(list(dataset['test']), 10)
    # 保存翻译结果到文件（解决终端中文显示框框问题）
    with open("translation_results_optimized.txt", "w", encoding="utf-8") as f:
        for idx, sample in enumerate(test_samples):
            # 获取测试样本的英文句子
            test_en_sentence = sample['translation']['en']
            # 翻译并获取注意力权重（使用100的推理长度+5的束宽，确保翻译完整）
            zh_sentence, attn_weights, en_tokens, zh_tokens = translate_with_attention(
                model=model,
                en_sentence=test_en_sentence,
                tokenize_en=tdp.tokenize_en,
                en_vocab=en_vocab,
                zh_vocab=zh_vocab,
                pad_idx_en=pad_idx_en,
                pad_idx_zh=pad_idx_zh,
                device=cfg.DEVICE,
                max_len=cfg.INFER_MAX_LEN,  # 推理长度设为100，远大于训练长度
                beam_size=cfg.BEAM_SIZE     # 束宽设为5，提升长句翻译质量
            )

            # 打印翻译结果
            print(f"\n【随机测试样本 {idx+1}】")
            print(f"英文输入：{test_en_sentence}")
            print(f"中文翻译：{zh_sentence}")
            # 写入文件（中文正常显示）
            f.write(f"【随机测试样本 {idx+1}】\n")
            f.write(f"英文输入：{test_en_sentence}\n")
            f.write(f"中文翻译：{zh_sentence}\n\n")

            # 绘制注意力热图（按样本索引命名，避免覆盖）
            if attn_weights is not None:
                heatmap_path = f'attention_heatmap_sample_{idx+1}_optimized.png'
                plot_attention_heatmap(
                    attn_weights=attn_weights,
                    en_tokens=en_tokens,
                    zh_tokens=zh_tokens,
                    save_path=heatmap_path
                )
                print(f"注意力热图已保存至：{heatmap_path}")

    print("\n翻译结果已保存至：translation_results_optimized.txt")
    print("\n" + "="*50 + " 全部流程完成 " + "="*50)

if __name__ == "__main__":
    main()