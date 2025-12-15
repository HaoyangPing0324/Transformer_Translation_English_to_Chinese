# 使用 Transformer模型 进行英文到中文翻译

## 项目来源
苏州大学未来科学与工程学院《人工智能》课程
任课老师：吴洪状

## 任务描述
本任务要求你构建一个基于 PyTorch 的Transformer模型，将英文句子翻译成中文。你将使用 HuggingFace Datasets 加载并预处理IWSLT 2016数据集，实现模型训练和推理过程。

## 任务目标
构建一个英文到中文的翻译系统，主要任务包括：
加载并预处理 IWSLT 2017数据集中的英文-中文句对，注意：需要安装datasets和transformers 库；
构建简易的Transformer模型（包括Positional Encoding、encoder、decoder）；
实现训练与验证流程；
实现推理功能：输入英文句子，输出翻译的中文句子；
对模型的训练过程和性能进行基本分析。

## 技术要求
使用 PyTorch 内置函数实现 Transformer 模型；
使用 HuggingFace Datasets 加载IWSLT 2017数据集：
模型训练可使用 teacher forcing；

## 任务要求
使用不少于 1万对训练句对；
英文、中文分别构建词表，词表大小设置阈值（如 10k）
训练时注意合理划分训练/验证集（如 90%/10%）
模型训练轮数至少为 10 epoch，并记录 loss 曲线

## 可视化
随机推理测试数据中的几个样本，显示输入英文句子和机器翻译的中文句子，并绘制attention 权重热图（encoder-decoder attention）。
