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

import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnModel(nn.Module):
    """
    CNN模型类，实现了模型的初始化及前馈过程
    """
    def __init__(self, opt):
        """
        CNN模型初始化
        Args：
            opt：参数设置，由CNN类传入
        Returns：
            None
        """
        super(CnnModel, self).__init__()
        self.embed = nn.Embedding(opt.vocab_num, opt.emb_dim)
        self.input_drop = nn.Dropout(opt.input_drop)
        self.convs = nn.ModuleList([nn.Conv1d(opt.emb_dim, opt.kernel_num, int(K)) for K in opt.kernel_sizes.split()])
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.kernel_size * opt.kernel_num, opt.polarities_dim)

    def forward(self, inputs):
        """
        tensor在CNN模型中的前馈过程
        Args:
            inputs：tensor，模型每个batch的输入
        Returns:
            logits：模型在各个类别上的未归一化概率分布
        """
        # ids to embedding
        sen_indicies = inputs[0]
        sen_emb = self.embed(sen_indicies)
        sen_feature = self.input_drop(sen_emb)

        # produce feature maps
        conv_list = []
        for conv in self.convs:
            conv_L = conv(sen_feature.transpose(1, 2))
            conv_L = self.dropout(conv_L)
            conv_L = F.max_pool1d(conv_L, conv_L.size(2)).squeeze(2)
            conv_list.append(conv_L)

        sen_out = [i.view(i.size(0), -1) for i in conv_list]
        sen_out = torch.cat(sen_out, dim=1)

        # classification
        logits = self.dense(sen_out)
        return logits