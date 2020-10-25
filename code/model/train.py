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
import math
import random
import argparse
import copy
import json
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cnn_model import CnnModel
from cnn_model import CnnModel
from pytorch_data_utils import build_words_file
from pytorch_data_utils import build_tokenizer
from pytorch_data_utils import MyDataset

from evaluate_metrics import threshold_recall_by_precision


class Trainer():
    """
    模型的训练类：
        输入：训练集和验证集，均带标签
        输出：json文件，保存了模型训练的 1.超参数 2.最好模型的路径 3.最好模型全局的各个指标(p、r、f、a、50~90精度下对应的召回)
    """

    def __init__(
            self,
            train_path=None,
            valid_path=None,
            save_path=None,
            tokenizer_path=None):
        """
        初始化
        Args：
            None
        Returns：
            None
        """
        # parameters
        parser = argparse.ArgumentParser()
        # path parameter
        parser.add_argument(
            '--train_path',
            default="./data/train_10w_shuf",
            type=str)
        parser.add_argument(
            '--valid_path',
            default="./data/valid",
            type=str)
        parser.add_argument(
            '--save_path',
            default="./cnn_files/",
            type=str)
        parser.add_argument(
            '--tokenizer_path',
            default="./cnn_files/tokenizer.dat",
            type=str)
        # general parameter
        parser.add_argument('--num_epoch', default=3, type=int)
        parser.add_argument('--learning_rate', default=0.0005, type=float)
        parser.add_argument('--polarities_dim', default=2, type=int, help="类别数")
        parser.add_argument('--l2reg', default=0.00001, type=float)
        parser.add_argument('--initializer', default='xavier_uniform_', type=str)
        parser.add_argument('--optimizer', default='adam', type=str)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--device', default=None, type=str)
        parser.add_argument('--cuda_idx', default=0, type=int)
        parser.add_argument('--n_gpu', default=1, type=int)
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--optimize_param', default=None, type=str)
        parser.add_argument(
            '--log_step',
            default=256,
            type=int,
            help="参数每更新log_step次，进行模型评估")
        parser.add_argument(
            '--given_precision',
            default="0.5 0.6 0.7 0.8 0.9",
            type=str,
            help="给定精度，求对应召回")
        parser.add_argument(
            '--max_seq_len',
            default=48,
            type=int,
            help="文本序列的最大长度")
        # model parameter
        parser.add_argument('--emb_dim', default=50, type=int)
        parser.add_argument('--kernel_size', default=3, type=int)
        parser.add_argument('--kernel_sizes', default='2 3 5', type=str)
        parser.add_argument('--kernel_num', default=64, type=int)
        parser.add_argument('--mlp_dim', default=128, type=int)
        parser.add_argument('--dropout', default=0.5, type=float)
        parser.add_argument('--input_drop', default=0.5, type=float)
        self.opt = parser.parse_args()

        if train_path is not None:
            self.opt.train_path = train_path
        if valid_path is not None:
            self.opt.valid_path = valid_path
        if train_path is not None:
            self.opt.save_path = save_path
        if train_path is not None:
            self.opt.tokenizer_path = tokenizer_path

        initializers = {
            'xavier_uniform_': torch.nn.init.xavier_uniform_,
            'xavier_normal_': torch.nn.init.xavier_normal,
            'orthogonal_': torch.nn.init.orthogonal_
        }
        optimizers = {
            'adadelta': torch.optim.Adadelta,  # default lr=1.0
            'adagrad': torch.optim.Adagrad,  # default lr=0.01
            'adam': torch.optim.Adam,  # default lr=0.001
            'adamax': torch.optim.Adamax,  # default lr=0.002
            'asgd': torch.optim.ASGD,  # default lr=0.01
            'rmsprop': torch.optim.RMSprop,  # default lr=0.01
            'sgd': torch.optim.SGD
        }
        self.opt.initializer = initializers[self.opt.initializer]
        self.opt.optimizer = optimizers[self.opt.optimizer]
        self.opt.inputs_cols = ['x_data']

        if self.opt.device is None:
            self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.opt.device == "cuda":
            self.opt.device = torch.device(self.opt.device + ": " + str(self.opt.cuda_idx))
        else:
            self.opt.device = torch.device("cpu")

        def set_seed(num):
            """
            设置种子
            Args：
                num：int，种子
            Returns：
                None
            """
            os.environ['PYTHONHASHSEED'] = str(num)
            np.random.seed(num)
            random.seed(num)
            torch.manual_seed(num)
            torch.cuda.manual_seed(num)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        set_seed(self.opt.seed)
        self.max_metric = 0.0  # 记录模型的最好评估指标
        self.json_path = ""

    def _reset_params(self):
        """
        模型可训练参数初始化
        Args：
            None
        Returns：
            None
        """
        for child in self.model.children():
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def fit(self):
        """
        在训练集上训练模型，并在验证集上评估模型，保存在验证集上评估指标最好的模型
        Args：
            None
        Returns：
            best_model_path：训练得到的最佳模型的保存路径
        """
        if not os.path.exists(self.opt.save_path):
            os.mkdir(self.opt.save_path)
        if not os.path.exists(self.opt.save_path + "model_files/"):
            os.mkdir(self.opt.save_path + "model_files/")
        if not os.path.exists(self.opt.save_path + "json_files/"):
            os.mkdir(self.opt.save_path + "json_files/")
        # 分词
        self.train_df = build_words_file(self.opt.train_path)
        self.valid_df = build_words_file(self.opt.valid_path)
        # 字典
        self.tokenizer = build_tokenizer(
            max_seq_len=self.opt.max_seq_len,
            tokenizer_dat=self.opt.tokenizer_path,
            df_file=self.train_df
        )
        self.opt.vocab_num = len(self.tokenizer.word2idx) + 2
        # 模型
        self.model = CnnModel(self.opt).to(self.opt.device)
        # 数据
        self.trainset = MyDataset(self.train_df, self.tokenizer)
        self.devset = MyDataset(self.valid_df, self.tokenizer)
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dataset=self.devset, batch_size=self.opt.batch_size, shuffle=False)
        # 训练
        self._reset_params()
        self._train(train_data_loader, dev_data_loader)
        return self.json_path

    def _train(self, train_data_loader, val_data_loader):
        """
        训练函数
        Args：
            train_data_loader：训练集数据迭代器
            val_data_loader：验证集数据迭代器
        Returns：
            None
        """
        # 是否使用多GPU
        if self.opt.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # 损失函数 & 优化器
        criterion = nn.CrossEntropyLoss()
        optimizer_grouped_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(
            optimizer_grouped_parameters,
            lr=self.opt.learning_rate,
            weight_decay=self.opt.l2reg
        )

        self.model.zero_grad()
        for epoch in range(self.opt.num_epoch):
            print('>' * 66)
            print('epoch: {}'.format(epoch))
            global_step = 0
            n_correct, n_total, loss_total = 0, 0, 0

            for _, sample_batched in enumerate(train_data_loader):
                self.model.train()
                global_step += 1
                inputs = [
                    torch.tensor(sample_batched[col], dtype=torch.long).to(self.opt.device)
                    for col in self.opt.inputs_cols
                ]
                targets = torch.tensor(sample_batched['y_label'], dtype=torch.long).to(self.opt.device)
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                if self.opt.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                    # validate per log_steps
                    self._evaluate(val_data_loader, self.opt.given_precision)
        # validate at the end of the train
        self._evaluate(val_data_loader, self.opt.given_precision)

    def _evaluate(self, data_loader, given_precision):
        """
        验证函数，返回当前模型在给定指标上对应的其他指标
        Args：
            data_loader：需验证数据集的迭代器
            given_precision：给定的指标，如给定0.9精度
        Returns：
            None
        """
        n_correct, n_total = 0, 0
        targets_all, outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for _, sample_batched in enumerate(data_loader):
                inputs = [
                    torch.tensor(sample_batched[col], dtype=torch.long).to(self.opt.device)
                    for col in self.opt.inputs_cols
                ]
                targets = torch.tensor(sample_batched['y_label'], dtype=torch.long).to(self.opt.device)
                outputs = self.model(inputs)
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                if targets_all is None:
                    targets_all = targets
                    outputs_all = outputs
                else:
                    targets_all = torch.cat((targets_all, targets), dim=0)
                    outputs_all = torch.cat((outputs_all, outputs), dim=0)
        # 模型的全局准确率
        overall_a = n_correct / n_total
        # 模型在要求精度下的阈值和召回
        y_true = targets_all.cpu().numpy().tolist()
        y_pred = F.softmax(outputs_all, dim=-1)[:, 1].cpu().numpy().tolist()
        threshold_list, recall_list = [], []
        for given_p in given_precision.split(" "):
            given_p = float(given_p)
            threshold, recall = threshold_recall_by_precision(y_pred, y_true, given_p)
            threshold_list.append(threshold)
            recall_list.append(recall)
        print("> val_acc: {:.4f}, required_threshold: {:.4f}, required_recall: {:.4f}". \
              format(overall_a, threshold_list[-1], recall_list[-1]))
        # 输出
        js = copy.deepcopy(self.opt.__dict__)
        for k, v in js.items():
            js[k] = str(v)
        if recall > self.max_metric:
            # 保存模型
            self.max_metric = recall
            model_path = self.opt.save_path + "model_files/metric_{0}".format(round(self.max_metric, 4))
            self.saveModel(save_path=model_path)
            # 保存json
            precision_threshold, precision_recall = {}, {}
            for p, t in zip(given_precision.split(" "), threshold_list):
                precision_threshold[float(p)] = float(t)
            for p, r in zip(given_precision.split(" "), recall_list):
                precision_recall[float(p)] = float(r)
            js["precision_threshold"] = precision_threshold
            js["precision_recall"] = precision_recall
            js["model_path"] = model_path
            self.json_path = self.opt.save_path + "/json_files/metric_{0}.json".format(round(self.max_metric, 4))
            with open(self.json_path, "w") as fw:
                fw.write(json.dumps(js))
            print(" >>> best result updated <<< ")
        print('*' * 33)

    def saveModel(self, save_path):
        """
        模型保存
        Args：
            save_path：模型保存的路径
        Returns：
            None
        """
        torch.save(self.model, save_path)
        print('>> saved: {}'.format(save_path))

# if __name__ == "__main__":

# Trainer
# obj = Trainer()
# obj.fit()