"""
多标签分类评估指标和预测函数
"""

import numpy as np

# 标准学习策略
def get_dynamic_threshold_prediction(probabilities, T):
    predictions = np.zeros_like(probabilities, dtype=int)
    
    for i in range(len(probabilities)):
        max_prob = np.max(probabilities[i])  # 找到最大概率
        for j in range(len(probabilities[i])):
            # 如果该标签的概率与最大概率的差距小于阈值T，则预测为1
            if (max_prob - probabilities[i][j]) < T:
                predictions[i][j] = 1
    
    return predictions


def sample_based_metrics(all_preds, all_labels):
    n = len(all_preds)
    m = all_preds.shape[1]

    # ATR (Absolute True Rate)
    ATR = np.mean([int(np.array_equal(all_preds[i], all_labels[i])) for i in range(n)])

    # Acc (Accuracy)
    Acc = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.union1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0]))
                   if len(np.union1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) > 0 else 1
                   for i in range(n)])

    # Pre (Precision)
    Pre = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.where(all_preds[i] == 1)[0]) if len(np.where(all_preds[i] == 1)[0]) > 0 else 1
                   for i in range(n)])

    # Rec (Recall)
    Rec = np.mean([len(np.intersect1d(np.where(all_preds[i] == 1)[0], np.where(all_labels[i] == 1)[0])) /
                   len(np.where(all_labels[i] == 1)[0]) if len(np.where(all_labels[i] == 1)[0]) > 0 else 1
                   for i in range(n)])

    # F1 (F1 Score)
    F1 = 2 * Pre * Rec / (Pre + Rec) if (Pre + Rec) > 0 else 0

    return ATR, Acc, Pre, Rec, F1


def location_based_metrics_micro(all_preds, all_labels):
    # 汇总所有标签的 TP, TN, FP, FN
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    # 微观 (micro) 度量计算
    micro_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    micro_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    micro_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return micro_accuracy, micro_precision, micro_recall, micro_f1


def print_sample_based_metrics(ATR, Acc, Pre, Rec, F1):
    """
    打印基于样本的评估指标
    """
    print(f"Sample-based Metrics:\nATR: {ATR:.4f}\nAcc: {Acc:.4f}\nPre: {Pre:.4f}\nRec: {Rec:.4f}\nF1: {F1:.4f}\n")


def print_location_based_metrics(Acc_prime, Pre_prime, Rec_prime, F1_prime):
    """
    打印基于位置的评估指标
    """
    print(f"Location-based Metrics:\nAcc': {Acc_prime:.4f}\nPre': {Pre_prime:.4f}\nRec': {Rec_prime:.4f}\nF1': {F1_prime:.4f}\n")


def compute_co_occurrence_matrix(labels):
    # 确保为numpy数组
    labels = np.asarray(labels)
    n_labels = labels.shape[1]
    co_matrix = np.zeros((n_labels, n_labels), dtype=float)
    
    for i in range(n_labels):
        li = labels[:, i] == 1
        for j in range(n_labels):
            lj = labels[:, j] == 1
            intersection = np.sum(li & lj)
            union = np.sum(li | lj)
            co_matrix[i, j] = (intersection / union) if union > 0 else 0.0
    
    return co_matrix


def compute_co_occurrence_matrix_cos(labels, binarize=True, threshold=0.5):
    # 转为 numpy 数组
    L = np.asarray(labels)
    if not np.issubdtype(L.dtype, np.number):
        L = L.astype(float)
    
    # 按需二值化
    if binarize:
        L = (L >= threshold).astype(float)
    else:
        L = L.astype(float)

    # 点积矩阵
    dot_mat = (L.T @ L).astype(float)

    # 各标签范数（对二值数据等于 sqrt(该标签出现次数)）
    norms = np.linalg.norm(L, axis=0).astype(float)
    denom = norms[:, None] * norms[None, :]

    n_labels = L.shape[1]
    co_matrix = np.divide(
        dot_mat,
        denom,
        out=np.zeros((n_labels, n_labels), dtype=float),
        where=denom > 0
    )
    return co_matrix