import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
import numpy as np
import torch
import math



import yaml
import os

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_config(config_path=None):
    """获取配置，支持命令行参数覆盖"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'run/human-config.yml')
    
    # 加载基础配置
    config = load_config(config_path)
    return config


def caculate_metric(pred_y, labels, pred_prob):
    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # print('tp\tfp\ttn\tfn')
    # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

    ACC = float(tp + tn) / test_num

    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    labels = labels.cpu()
    pred_prob = pred_prob.cpu()
    labels = labels.numpy().tolist()
    pred_prob = pred_prob.numpy().tolist()

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  
    AUC = auc(fpr, tpr)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)
    AUCPR = auc(recall, precision)
    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, AUCPR, MCC, tp, fp, tn, fn])

    # ROC(fpr, tpr, AUC)
    # PRC(recall, precision, AP)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data

# draw ROC
def ROC(fpr, tpr, roc_auc, draw_path):
    """绘制 ROC (受试者工作特征) 曲线
    
    参数:
        fpr: 假阳性率（False Positive Rate）
        tpr: 真阳性率（True Positive Rate）
        roc_auc: ROC曲线下面积
        draw_path: 图像保存路径
    """
    plt.figure()
    lw = 2  # 线宽
    plt.figure(figsize=(10, 10))  # 设置图像大小为10x10
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  
    # 绘制对角线（随机猜测的基准线）
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # 设置坐标轴范围
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # 设置坐标轴标签和标题，使用较大字号
    plt.xlabel('False Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.ylabel('True Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.title('Receiver operating characteristic example', fontdict={'weight': 'normal', 'size': 30})
    plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 30})
    # 保存图像
    plt.savefig(draw_path + os.sep + 'roc.png')


# draw PRC
def PRC(recall, precision, AP, draw_path):
    """绘制 PRC (精确率-召回率) 曲线
    
    参数:
        recall: 召回率
        precision: 精确率
        AP: 平均精确率（Average Precision）
        draw_path: 图像保存路径
    """
    plt.figure()
    # 绘制阶梯形的PRC曲线
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')  # post表示阶梯的右侧对齐
    # 填充曲线下方区域
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    # 设置坐标轴标签和标题
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])  # 设置y轴范围
    plt.xlim([0.0, 1.05])  # 设置x轴范围
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(AP))
    # 保存图像
    plt.savefig(draw_path + os.sep + 'prc.png')

def calculateConfusionMatrixOn1DList(pred_prob_1d, lab_1d, thres=0.5):
    if isinstance(pred_prob_1d, torch.Tensor):
        prob = pred_prob_1d.tolist()
        lab = lab_1d.tolist()
    else:
        prob = pred_prob_1d
        lab = lab_1d

    length = len(lab_1d)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(length):
        if abs(lab[i] - 1.0) < 0.001:
            if prob[i] >= thres:
                tp += 1
            else:
                fn += 1
        else:
            if prob[i] >= thres:
                fp += 1
            else:
                tn += 1
    return tp, fn, fp, tn

def calculateMaxMCCThresOnValidset(pred_prob_1d, lab_1d, start_threshold=0.001, threshold_step=0.001, end_threshold=1.0, is_valid=False, test_thre=None, draw=False, draw_path=None):
    if isinstance(lab_1d, torch.Tensor):
        valid_num = lab_1d.size(0)
        labels = lab_1d.cpu()
        pred_prob = pred_prob_1d.cpu()
        labels = labels.numpy().tolist()
        pred_prob = pred_prob.numpy().tolist()
    else:
        valid_num = len(lab_1d)
        labels = lab_1d
        pred_prob = pred_prob_1d


    maxMCC = -1
    corrThres = -1
    corrTP = 0
    corrFN = 0
    corrFP = 0
    corrTN = 0

    if (is_valid and test_thre==None):
        for thres in np.arange(start_threshold, end_threshold, threshold_step):
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            _tp, _fn, _fp, _tn = calculateConfusionMatrixOn1DList(pred_prob_1d, lab_1d, thres)
            tp += _tp
            fn += _fn
            fp += _fp
            tn += _tn

            fenmu = math.sqrt(1.*(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            if fenmu < 0.001:
                fenmu = 0.001
            mcc = 1.*(tp*tn - fp*fn)/fenmu

            if mcc > maxMCC:
                maxMCC = mcc
                corrThres = thres
                corrTP = tp
                corrFN = fn
                corrFP = fp
                corrTN = tn

    else:
        corrTP, corrFN, corrFP, corrTN = calculateConfusionMatrixOn1DList(pred_prob_1d, lab_1d, test_thre)
        fenmu = math.sqrt(1. * (corrTP + corrFP) * (corrTP + corrFN) * (corrTN + corrFP) * (corrTN + corrFN))
        if fenmu < 0.001:
            fenmu = 0.001
        maxMCC = 1. * (corrTP * corrTN - corrFP * corrFN) / fenmu
        corrThres = test_thre

    ACC = float(corrTP + corrTN) / valid_num

    # precision
    if corrTP + corrFP == 0:
        Precision = 0
    else:
        Precision = float(corrTP) / (corrTP + corrFP)

    # SE
    if corrTP + corrFN == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(corrTP) / (corrTP + corrFN)

    # SP
    if corrTN + corrFP == 0:
        Specificity = 0
    else:
        Specificity = float(corrTN) / (corrTN + corrFP)

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)



    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1) 
    AUC = auc(fpr, tpr)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)
    AUCPR = auc(recall, precision)

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, AUCPR, maxMCC, corrTP, corrFP, corrTN, corrFN])
    if draw:
        ROC(fpr, tpr, AUC, draw_path)
        PRC(recall, precision, AP, draw_path)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]

    return metric, roc_data, prc_data, corrThres

def single_MCC(tp, fn, fp, tn):
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return MCC

def Output_prelab(positive, test_thre=None, pre_path=None):
    if isinstance(positive, np.ndarray):
        prob = positive.tolist()
    else:
        prob = positive
    list = []
    if pre_path != None:
        pre = open(pre_path, 'w')
        if test_thre != None:
            for i in prob:
                if float(i) >= test_thre:
                    list.append(1.0)
                    pre.write(str('{:.3f}'.format(i)) + '\n')
                else:
                    list.append(0.0)
                    pre.write(str('{:.3f}'.format(i)) + '\n')
        pre.close()
    else:
        print("请输入预测值文件路径")
        exit("DEBUG")
    return list





