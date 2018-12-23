#!/usr/bin/python
# -*- coding: utf-8 -*-


class TRNNConfig(object):
    """RNN配置参数"""
    # 模型参数
    embedding_dim = 128      # 词向量维度
    seq_length = 30        # 序列长度
    num_classes = 5        # 类别数"B","E","M","S"
    vocab_size = 5000       # 词汇表达小
    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru
    dropout_keep_prob = 0.7 # dropout保留比例
    learning_rate = 1e-3    # 学习率
    batch_size = 64       # 每批训练大小
    num_epochs = 20          # 总迭代轮次
    print_per_batch = 10  # 每多少轮输出一次结果
    crf=0                   #使用crf or not
