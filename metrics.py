import numpy as np
from sklearn.metrics import *


def compute_acc(labels, preds):
    acc = accuracy_score(labels, preds)
    return acc


def compute_auc_roc(labels, preds):
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    auc_roc = auc(fpr, tpr)
    return auc_roc


def compute_auc_pr(labels, preds):
    p, r, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    auc_pr = auc(r, p)
    return auc_pr


def compute_mcc(labels, preds):
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc


def computer_ss(labels, preds):
    labels = labels.flatten()
    preds = preds.flatten()
    confusion = confusion_matrix(labels, preds)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    print('Specificity:', TN / float(TN + FP))

def compute_performance(labels, preds):
    labels = labels.flatten()
    preds = preds.flatten()

    predictions_max = None
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    for t in range(1, 100):
        local_threshold = t / 100.0
        predictions = (preds > local_threshold).astype(np.int32)
        p = 0.0
        r = 0.0
        total = 0
        p_total = 0

        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp

        if tp == 0 and fp == 0 and fn == 0:
            continue
        total += 1
        if tp != 0:
            p_total += 1
            precision = tp / (1.0 * (tp + fp))
            recall = tp / (1.0 * (tp + fn))
            p += precision
            r += recall

        if total > 0 and p_total > 0:
            r /= total
            p /= p_total
            if p + r > 0:
                f = 2 * p * r / (p + r)
                if f_max < f:
                    f_max = f
                    p_max = p
                    r_max = r
                    t_max = local_threshold
                    predictions_max = predictions

    return f_max, p_max, r_max, t_max, predictions_max
