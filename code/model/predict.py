#!/usr/bin/python
#-*- coding:utf8 -*-
################################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
# Power by work 2020-07-27 16:35:39
"""


import random
import jieba
import pickle
import copy
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor():
    """
    模型的预测类
    """
    def __init__(self, json_path):
        """
        初始化
        Args：
            json_path：模型训练时保存的json文件
        Returns：
            None
        """
        with open(json_path, 'r', encoding='utf-8') as fr:
            json_data = json.load(fr)
        model_path = json_data["model_path"]
        tokenizer_path = json_data["tokenizer_path"]
        device = json_data["device"]
        cuda_idx = json_data["cuda_idx"]

        print("加载模型...")
        self.model = torch.load(model_path)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == "cuda":
            self.device = torch.device(device + ": " + cuda_idx)
        else:
            self.device = torch.device("cpu")
        self.model.to(device)

        print("加载字典...")
        self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))


    def predict(self, texts, label=None):
        """
        文本预测
        Args：
            texts：list[str]，一维文本列表，保存了多段需要预测的文本
            label：list[int]，标签列表，与文本列表中的文本一一对应
        Returns：
            如果数据集没有标签，返回softmax打分（list[list[numpy.float64]]）、预测标签（list[int]）
            如果数据集存在标签，返回softmax打分、预测标签、真实标签（list[int]）
        """
        self.model.eval()
        pred_list, cls_list = [], []
        for text in tqdm(texts):
            text_ids = [self.tokenizer.text_to_sequence(list(jieba.cut(text)))]
            text_ids = torch.tensor(text_ids, dtype=torch.int64).to(self.device)
            t_inputs = [text_ids]
            t_outputs = self.model(t_inputs)
            # softmax打分
            y_pred = F.softmax(t_outputs, dim=-1).cpu().detach().numpy().tolist()
            pred_list.append(y_pred[0])
            # 默认分类：使用打分最高的类别作为预测类别
            y_cls = np.array(y_pred).argmax(axis=-1)
            cls_list.append(int(y_cls))
        if label is None:
            return pred_list, cls_list
        else:
            return pred_list, cls_list, label


# if __name__ == "__main__":

    # Predictor
    # cnn_model = Predictor("./cnn_files/json_files/metric_0.513.json")
    # y_pred, y_cls = cnn_model.predict(["推荐一个微信公众号——猪猪晚安每天晚上会整合推荐一篇睡前故事。"])
    # print(y_pred)
    # print(y_cls)