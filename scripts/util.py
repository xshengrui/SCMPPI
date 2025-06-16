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
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_config(config_path=None):
    """Get configuration, supporting command-line argument override"""
    if config_path is None:
        # Default config path if not provided
        config_path = os.path.join(os.path.dirname(__file__), 'run/human-config.yml')
    
    # Load base configuration
    config = load_config(config_path)
    return config

def caculate_metric(pred_y, labels, pred_prob):
    """
    Calculate various classification metrics (ACC, Precision, Recall, Specificity, F1, MCC, AUC, AUCPR).
    
    Args:
        pred_y (numpy.ndarray or list): Predicted binary labels (0 or 1).
        labels (torch.Tensor or numpy.ndarray or list): True binary labels.
        pred_prob (torch.Tensor or numpy.ndarray or list): Predicted probabilities.
    
    Returns:
        tuple: A tuple containing:
            - metric (torch.Tensor): Tensor of calculated metrics.
            - roc_data (list): [fpr, tpr, AUC] for ROC curve.
            - prc_data (list): [recall, precision, AP] for PRC curve.
    """
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

    ACC = float(tp + tn) / test_num

    # Precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # Recall (Sensitivity)
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC (Matthews Correlation Coefficient)
    denominator_mcc = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator_mcc == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt(denominator_mcc))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # Convert tensors to numpy arrays for sklearn functions if needed
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().tolist()
    if isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.cpu().numpy().tolist()

    # ROC and AUC (Receiver Operating Characteristic and Area Under the Curve)
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1) 
    AUC = auc(fpr, tpr)

    # PRC and AP (Precision-Recall Curve and Average Precision)
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)
    AUCPR = auc(recall, precision)

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, AUCPR, MCC, tp, fp, tn, fn])

    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data

def ROC(fpr, tpr, roc_auc, draw_path):
    """
    Draws the Receiver Operating Characteristic (ROC) curve.
    
    Args:
        fpr (numpy.ndarray): False Positive Rate.
        tpr (numpy.ndarray): True Positive Rate.
        roc_auc (float): Area Under the ROC Curve.
        draw_path (str): Path to save the image.
    """
    plt.figure()
    lw = 2  # Line width
    plt.figure(figsize=(10, 10))  # Set image size to 10x10
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  
    # Plot the diagonal line (baseline for random guessing)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Set axis labels and title with larger font size
    plt.xlabel('False Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.ylabel('True Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.title('Receiver operating characteristic example', fontdict={'weight': 'normal', 'size': 30})
    plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 30})
    # Save image
    plt.savefig(draw_path + os.sep + 'roc.png')

def PRC(recall, precision, AP, draw_path):
    """
    Draws the Precision-Recall Curve (PRC).
    
    Args:
        recall (numpy.ndarray): Recall values.
        precision (numpy.ndarray): Precision values.
        AP (float): Average Precision.
        draw_path (str): Path to save the image.
    """
    plt.figure()
    # Plot the step-like PRC curve
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')  # 'post' aligns steps to the right
    # Fill the area under the curve
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    # Set axis labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])  # Set y-axis limit
    plt.xlim([0.0, 1.05])  # Set x-axis limit
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(AP))
    # Save image
    plt.savefig(draw_path + os.sep + 'prc.png')

def calculateConfusionMatrixOn1DList(pred_prob_1d, lab_1d, thres=0.5):
    """
    Calculates the confusion matrix components (TP, FN, FP, TN) for 1D lists
    given a threshold.
    """
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
        if abs(lab[i] - 1.0) < 0.001:  # Positive label
            if prob[i] >= thres:
                tp += 1
            else:
                fn += 1
        else:  # Negative label
            if prob[i] >= thres:
                fp += 1
            else:
                tn += 1
    return tp, fn, fp, tn

def calculateMaxMCCThresOnValidset(pred_prob_1d, lab_1d, start_threshold=0.001, threshold_step=0.001, end_threshold=1.0, is_valid=False, test_thre=None, draw=False, draw_path=None):
    """
    Calculates evaluation metrics, optionally finding the best MCC threshold
    on a validation set, or using a fixed threshold on a test set.
    """
    if isinstance(lab_1d, torch.Tensor):
        valid_num = lab_1d.size(0)
        labels = lab_1d.cpu().numpy().tolist()
        pred_prob = pred_prob_1d.cpu().numpy().tolist()
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

    if (is_valid and test_thre == None):
        for thres in np.arange(start_threshold, end_threshold, threshold_step):
            _tp, _fn, _fp, _tn = calculateConfusionMatrixOn1DList(pred_prob_1d, lab_1d, thres)
            tp = _tp
            fn = _fn
            fp = _fp
            tn = _tn

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

    else: # Use fixed test_thre or if not validation set
        corrTP, corrFN, corrFP, corrTN = calculateConfusionMatrixOn1DList(pred_prob_1d, lab_1d, test_thre)
        fenmu = math.sqrt(1. * (corrTP + corrFP) * (corrTP + corrFN) * (corrTN + corrFP) * (corrTN + corrFN))
        if fenmu < 0.001:
            fenmu = 0.001
        maxMCC = 1. * (corrTP * corrTN - corrFP * corrFN) / fenmu
        corrThres = test_thre

    ACC = float(corrTP + corrTN) / valid_num

    # Precision
    if corrTP + corrFP == 0:
        Precision = 0
    else:
        Precision = float(corrTP) / (corrTP + corrFP)

    # Recall (Sensitivity)
    if corrTP + corrFN == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(corrTP) / (corrTP + corrFN)

    # Specificity
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
    """Calculates MCC for given TP, FN, FP, TN values."""
    denominator_mcc = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator_mcc == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt(denominator_mcc))
    return MCC

def Output_prelab(positive, test_thre=None, pre_path=None):
    """
    Outputs predicted labels based on a threshold and saves them to a file.
    
    Args:
        positive (numpy.ndarray or list): Predicted probabilities.
        test_thre (float, optional): Threshold for classification. Defaults to None.
        pre_path (str, optional): Path to save the prediction file. Defaults to None.
        
    Returns:
        list: List of predicted binary labels (0.0 or 1.0).
    """
    if isinstance(positive, np.ndarray):
        prob = positive.tolist()
    else:
        prob = positive
    
    pred_labels = []
    if pre_path is not None:
        with open(pre_path, 'w') as pre:
            if test_thre is not None:
                for i in prob:
                    if float(i) >= test_thre:
                        pred_labels.append(1.0)
                        pre.write(str('{:.3f}'.format(i)) + '\n')
                    else:
                        pred_labels.append(0.0)
                        pre.write(str('{:.3f}'.format(i)) + '\n')
    else:
        print("Please provide the prediction file path.")
        exit("DEBUG") # Exit for debugging, ideally raise an error in production
    return pred_labels