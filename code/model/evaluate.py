#!/usr/bin/python
# -*- coding:utf8 -*-
################################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""
# Power by work 2020-06-30 20:50:39
"""

import numpy as np


# from matplotlib import pyplot as plt


def precision_recall_by_threshold(y_pred, y_true, given_threshold, given_label=1):
    """
    Function:
        给定某种标签阈值，计算精度、召回
    Args:
        y_pred: list，模型打分
        y_true: list，真实标签
        given_threshold: float，给定的阈值
        label: int，正例对应的标签
    Returns:
        精度，float
        召回，float
    """
    all_metrics = cal_all_metrics(y_pred, y_true, given_label=given_label)

    for i in range(len(all_metrics["threshold"])):
        if all_metrics["threshold"][i] < given_threshold:
            return all_metrics["precision"][i], all_metrics["recall"][i]


def threshold_recall_by_precision(y_pred, y_true, given_precision, given_label=1):
    """
    Function:
        给定某种标签精度，计算阈值、召回
    Args:
        y_pred: list，模型打分
        y_true: list，真实标签
        given_precision: float，给定的精度
        label: int，正例对应的标签
    Returns:
        阈值，float
        召回，float
    """
    all_metrics = cal_all_metrics(y_pred, y_true, given_label=given_label)
    # plt.plot(all_metrics["recall"], all_metrics["precision"]) # P-R折线图
    # plt.show()

    if max(all_metrics["precision"]) < given_precision:
        return 0.0, 0.0

    for i in range(len(all_metrics["precision"]) - 1, -1, -1):
        if float(all_metrics["precision"][i]) > given_precision:
            return float(all_metrics["threshold"][i]), float(all_metrics["recall"][i])


def threshold_precision_by_recall(y_pred, y_true, given_recall, given_label=1):
    """
    Function:
        给定某种标签召回，计算阈值、精度
    Args:
        y_pred: list，模型打分
        y_true: list，真实标签
        given_recall: float，给定的召回
        label: int，正例对应的标签
    Returns:
        阈值，float
        精度，float
    """
    all_metrics = cal_all_metrics(y_pred, y_true, given_label=given_label)

    for i in range(len(all_metrics["recall"])):
        if all_metrics["recall"][i] > given_recall:
            return all_metrics["threshold"][i], all_metrics["precision"][i]


def cal_all_metrics(y_pred, y_true, given_label=1):
    """
    Function:
        计算某种标签的的各种评估指标
    Args:
        y_pred: list，模型打分，按数据集原始顺序排序
        y_true: list，真实标签的列表，按数据集原始顺序排序
    Returns:
        all_metrics: dict，包括键值：threshold、true_label、precision、recall、f1、fpr
            分别表示阈值、真实标签、精度、召回、f1、ROC曲线的横轴
    Steps:
        1.从高到低对y_pred排序，分别记录排序前和排序后样本的索引
        2.按照排序后样本的索引，对y_true做排序
        3.计算精度：TP / (TP + FP)
            TP：遍历到当前样本时，累计 y_true==given_label 的次数
            TP + FP：当前样本在内遍历过样本的总数
        4.计算召回：TP / (TP + FN)
            TP：遍历到当前样本时，累计 y_true==given_label 的次数
            TP + FN：所有样本中累计 y_true==given_label 的次数
        5.计算f1：2 * P * R / (P + R)
        6.计算ROC曲线：纵轴TPR，横轴FPR
            TPR = recall = TP / (TP + FN)
            FPR = FP / (FP + TN)
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    new_idx = np.argsort(-y_pred)
    # print(new_idx)
    # print(type(new_idx)) # numpy类型

    new_y_pred = y_pred[new_idx]
    new_y_true = y_true[new_idx]
    # print(new_y_pred)
    # print(new_y_true)

    # 遍历，计算精度、召回、f1
    recall_base = 0
    fpr_base = 0
    precision_base = []
    cnt_tp, tps = 0, []
    cnt_fp, fps = 0, []
    for idx, label in enumerate(new_y_true):
        if label == given_label:
            recall_base += 1
            cnt_tp += 1
        else:
            fpr_base += 1
            cnt_fp += 1
        fps.append(cnt_fp)
        tps.append(cnt_tp)
        precision_base.append(idx + 1)

    threshold_arr = np.array(new_y_pred)
    true_arr = np.array(new_y_true)
    precision_arr = np.divide(np.array(tps), np.array(precision_base))
    recall_arr = (np.array(tps)) / recall_base
    f1_arr = 2 * np.multiply(precision_arr, recall_arr) / (precision_arr + recall_arr)
    fpr_arr = np.array(fps) / fpr_base

    # print(threshold_arr, end='\n\n')
    # print(true_arr, end="\n\n")
    # print(precision_arr, end='\n\n')
    # print(recall_arr, end='\n\n')
    # print(fpr_arr, end='\n\n')
    # print(f1_arr, end='\n\n')

    # 阈值、真实标签、precision、recall、f1、fpr
    all_metrics = {
        "threshold": threshold_arr,
        "true_label": true_arr,
        "precision": precision_arr,
        "recall": recall_arr,
        "f1": f1_arr,
        "fpr": fpr_arr
    }
    return all_metrics


def threshold_align(old_threshold, new_threshold, new_scores):
    """
    进行旧版模型 v1 和新版模型 v2 的阈值对齐
    Args：
        old_threshold：list，旧模型各个精度下的阈值
        new_threshold：list，新模型各个精度下的阈值
        new_scores：list，新模型对实例的打分
    Returns：
        old_scores：list，新模型打分在旧模型上的映射
    """
    assert len(old_threshold) == len(new_threshold), "新、旧模型的阈值需要一一对应"
    old_scores = []
    old_threshold = [0.0] + old_threshold + [1.0]
    new_threshold = [0.0] + new_threshold + [1.0]
    for new_s in new_scores:
        for i in range(len(new_threshold) - 1):
            if new_threshold[i] <= new_s < new_threshold[i + 1]:
                tmp = (old_threshold[i + 1] - old_threshold[i]) * \
                      (new_s - new_threshold[i]) / (new_threshold[i + 1] - new_threshold[i]) + old_threshold[i]
                old_scores.append(tmp)
    return old_scores

# def test_eg1():
#     y_pred = [1 for _ in range(30)]
#     y_true = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,0,0,1]
#     all_metrcis = cal_all_metrics(y_pred, y_true)
#     pre = all_metrcis["precision"]
#     rec = all_metrcis["recall"]
#     plt.plot(rec, pre)
#     plt.show()
#     # precision_recall_by_threshold(model, valid_file, 0.999, label=1)


# def test_eg2():
#     old_threshold = [0.46, 0.95]
#     new_threshold = [0.52, 0.92]
#     new_scores = [0.08, 0.15, 0.35, 0.44, 0.56, 0.69, 0.77, 0.99]
#     old_scores = threshold_align(old_threshold, new_threshold, new_scores)
#     print(old_scores)


# if __name__ == "__main__":
# test_eg1()
# test_eg2()