#!/usr/bin/python
# -*- coding:utf8 -*-
################################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
# Power by work 2020-07-27 16:35:39
"""

import os
import jieba
import logging
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import gensim
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from tqdm import trange
from torch.utils.data import Dataset


def build_words_file(df_name):
    """
    对原始数据集分词，生成新的有分词结果的数据集
    Args：
        df_name：DataFrame，没有分词列的原始数据集路径
    Returns：
        data：DataFrame，有分词列的新数据集
    """
    new_df_name = df_name + "_words"
    if os.path.exists(new_df_name):
        print("加载数据集: {0}".format(new_df_name))
        data = pd.read_csv(new_df_name, sep="\t")
    else:
        logger = logging.getLogger()
        jieba.default_logger = logger
        sen_cut_list = []
        print("数据集分词...: {0}".format(df_name))
        data = pd.read_csv(df_name, sep="\t")
        for i in trange(0, len(data)):
            words = list(jieba.cut(data.sen[i]))
            sen_cut_list.append(words)
        data["sen_cut"] = sen_cut_list
        data.to_csv(new_df_name, sep="\t", index=False)
    return data


def build_tokenizer(max_seq_len, tokenizer_dat, df_file):
    """
    基于训练集建立字典，赋予每个词语一个索引
    Args：
        max_seq_len：int，模型处理文本的最大长度，大于这个长度的文本会被截断。
            注意，该参数借助Tokenizer类保存在tokenizer.dat文件中，若需修改最大文本长度，请先删除该文件。
        tokenizer_dat：str，tokenizer.dat文件的保存路径。
        df_file：DataFrame，列表中保存了建立字典时需要基于的语料。
    Returns：
        tokenizer：tokenizer类。
    """
    if os.path.exists(tokenizer_dat):
        print("加载字典: {0}".format(tokenizer_dat))
        tokenizer = pickle.load(open(tokenizer_dat, 'rb'))
    else:
        text = []
        print("建立字典...: {0}".format(tokenizer_dat))
        for i in range(0, len(df_file)):
            text.extend(eval(str(df_file.sen_cut[i])))
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)  # 此处text包含重复单词
        pickle.dump(tokenizer, open(tokenizer_dat, 'wb'))
    return tokenizer


def _load_glove(pre_trained_file, word2idx=None):
    """
    加载Glove格式的预训练词向量
    Args：
        pre_trained_file：str，预训练词向量的路径。
        word2idx：dic，基于语料建立的字典，保存了单词到索引的映射。
    Returns：
        word_vec：dic，基于字典建立的单词到词向量的映射，映射中的单词都在语料中出现过。
    """

    def get_coefs(word, *arr):
        """
        获取单词及对应词向量
        Args：
            word：str，单词。
            *arr：str，单词对应的词向量。
        Returns：
            预训练词向量文件在当前行存储的单词和对应的向量。
        """
        return word, np.asarray(arr, dtype='float32')

    raw_dic = dict(get_coefs(*o.split(" ")) for o in open(pre_trained_file, encoding='utf-8', errors='ignore'))

    word_vec = {}
    for word in raw_dic.keys():
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = raw_dic[word]
    print('the ratio coverage of pre-trained embedding is : %.4f' % (len(word_vec) / float(len(word2idx))))
    return word_vec


def _load_word_vec(pre_trained_file, binary=False, word2idx=None):
    """
    加载word2vec格式的预训练词向量
    Args：
        pre_trained_file：str，预训练词向量的路径。
        binary：bool，读取的文件是否是二进制的形式。
        word2idx：dic，基于语料建立的字典，保存了单词到索引的映射。
    Returns：
        word_vec：dic，基于字典建立的单词到词向量的映射，映射中的单词都在语料中出现过。
    """
    if binary == True:
        raw_dic = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_file, binary=True)
    else:
        raw_dic = gensim.models.KeyedVectors.load_word2vec_format(pre_trained_file)

    word_vec = {}
    for word in raw_dic.wv.vocab.keys():
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = raw_dic[word]
    print('the ratio coverage of pre-trained embedding is : %.4f' % (len(word_vec) / float(len(word2idx))))
    return word_vec


def _load_fasttext(pre_trained_file, word2idx=None):
    """
    加载fasttext格式的词向量
    Args：
        pre_trained_file：str，预训练词向量的路径。
        word2idx：dic，基于语料建立的字典，保存了单词到索引的映射。
    Returns：
        word_vec：dic，基于字典建立的单词到词向量的映射，映射中的单词都在语料中出现过。
    """
    raw_dic = {}
    with open(pre_trained_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        for line in fin:
            values = line.rstrip().split(' ')
            raw_dic[values[0]] = np.asarray(values[1:], dtype='float32')

    word_vec = {}
    for word in raw_dic.keys():
        if word2idx is None or word in word2idx.keys():
            word_vec[word] = raw_dic[word]
    print('the ratio coverage of pre-trained embedding is : %.4f' % (len(word_vec) / float(len(word2idx))))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, format, binary, pre_trained_file, initial_embedding_matrix):
    """
    建立词向量文件，文件中的单词均在语料中出现过。
    该函数从原始的预训练词向量大文件中提取当前语料用到的部分，保存在一个小文件中，以加快加载速度。
    Args：
        word2idx：dic，基于语料建立的字典，保存了单词到索引的映射。
        embed_dim：int，词向量的维度。
        format：str，词向量的格式，有glove、word2vec、fasttext三种。
        binary：bool，在读取word2vec格式的词向量时，读取的文件是否是二进制形式的。
        pre_trained_file：str，预训练词向量的路径。
        initial_embedding_matrix：str，已经保存好的定制化词向量文件的路径。
    Returns：
        embedding_matrix：词向量文件，文件中的单词均在语料中出现过。
    """
    if os.path.exists(initial_embedding_matrix):
        print('loading initial_matrix : ', initial_embedding_matrix)
        embedding_matrix = pickle.load(open(initial_embedding_matrix, 'rb'))
    else:
        print('loading pre-trained embedding file : ', pre_trained_file)
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))
        if format == 'glove':
            word_vec = _load_glove(pre_trained_file, word2idx=word2idx)
        elif format == 'word2vec':
            word_vec = _load_word_vec(pre_trained_file, binary=binary, word2idx=word2idx)
        elif format == 'fasttext':
            word_vec = _load_fasttext(pre_trained_file, word2idx=word2idx)
        print('build initial_embedding_matrix : ', initial_embedding_matrix)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # 0行和len(word2idx)+1行是全0向量，没有对应预训练向量的词也用全0向量表示
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(initial_embedding_matrix, 'wb'))
    return embedding_matrix


def pad_and_trunc(sequence, max_seq_len, dtype='int64', padding='post', truncating='post', value=0.):
    """
    文本截断和补齐
    Args：
        sequence：list，文本转化得到的id序列。
        max_seq_len：int，文本的最大长度，超过这个长度的文本需要截断。
        dtype：str，返回的id序列的类型。
        padding：str，补齐方式，默认为在句子尾部补齐。
        truncating：str，截断方式，默认为截断句子尾部。
        value：float，补齐标记对应的id，用该值来初始化索引序列，默认为0.0。
    Returns：
        x：list，截断或者补齐后的文本的id序列。
    """
    x = (np.ones(max_seq_len) * value).astype(dtype)
    if truncating == 'post':
        trunc = sequence[: max_seq_len]
    else:
        trunc = sequence[-max_seq_len:]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    """
    基于语料建立字典的类。
    """

    def __init__(self, max_seq_len):
        """
        初始化
        Args：
            max_seq_len：int，文本的最大长度，超过这个长度的文本需要截断。
        Returns：
            None
        """
        self.max_seq_len = max_seq_len
        self.word_freq_dic = {}
        self.sorted_words = []
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, all_word_list):
        """
        根据输入的所有单词（含重复单词）的列表建立字典。
        Args：
            all_word_list：从语料获取的含有重复单词的列表。
        Returns：
            None
        """
        for word in all_word_list:
            if word not in self.word_freq_dic:
                self.word_freq_dic[word] = 1
            else:
                self.word_freq_dic[word] += 1
        # 返回单词列表，由词频从高到低排列
        self.sorted_words = sorted(self.word_freq_dic, key=self.word_freq_dic.__getitem__, reverse=True)
        for word in self.sorted_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, sen_cut, reverse=False, padding='post', truncating='post'):
        """
        将单个文本转换为id序列。
        Args：
            sen_cut：list，句子分词后的列表。
            reverse：bool，是否反转句子顺序。
            padding：str，句子补齐的方式，默认补齐句子的尾部。
            truncating：str，句子截断的方式，默认截断句子的尾部。
        Returns：
            paded_sequence：numpy，截断或者补齐后的文本id序列。
        """
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in sen_cut]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        paded_sequence = pad_and_trunc(sequence, self.max_seq_len, padding=padding, truncating=truncating)
        return paded_sequence


class MyDataset(Dataset):
    """
    数据类，继承pytorch的Dataset类
    """

    def __init__(self, df_name, tokenizer):
        """
        初始化
        Args：
            df_name：str，训练集或验证集数据的路径，需要是经过预处理的文件。
            tokenizer：根据语料建立的字典类。
        Returns：
            None
        """
        all_data = []
        for i in range(len(df_name)):
            x_data = tokenizer.text_to_sequence(eval(str(df_name.sen_cut[i])))
            y_label = int(df_name.label[i])
            data = {
                'x_data': x_data,
                'y_label': y_label
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        """
        得到某个样本数据。
        Args：
            index：int，样本的索引。
        Returns：
            index索引对应样本的文本和标签。
        """
        return self.data[index]

    def __len__(self):
        """
        得到数据集规模
        Args：
            None
        Returns：
            数据集的样本总数。
        """
        return len(self.data)