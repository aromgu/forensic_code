import torch
import numpy as np
import keras
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
SMOOTH = 1e-6
EPS = 1e-10

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])

def dice_coefficient(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice

def get_f1score(precision, recall) :
    return 2 * precision * recall / (precision + recall + 1e-8)

def dice_coef(y_true, y_pred, smooth=1):
    # y_pred[y_pred >= 0.5] = 1
    # y_pred[y_pred < 0.5] = 0
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def pixelAcc(gt, pred, threshold = 0.5):
    # pred[pred >= threshold] = 1; pred[pred < threshold] = 0
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(gt, pred)

    return acc

def get_metrices(pred, label, pred_):
    # from sklearn import metrics
    from sklearn.metrics import jaccard_score

    pred = np.asarray(pred.flatten(), dtype=np.int64)
    pred_ = np.asarray(pred_.flatten(), dtype=np.int64)
    label = np.asarray(label.flatten(), dtype=np.int64)

    # acc = accuracy_score(label, pred)
    pre = precision_score(label, pred, average='macro')
    rec = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')
    iou_out = jaccard_score(label, pred, average='macro')
    auc = 0.0
    # y_true = np.expand_dims(label, axis=0)
    # y_pred = np.expand_dims(pred, axis=0)
    #if np.count_nonzero(label)==0 and np.count_nonzero(pred)==0:
    #    auc = 1
    #else:
    # fpr, tpr, thresholds = metrics.roc_auc_score(label, pred_, average='macro')
    # auc = metrics.auc(fpr, tpr)
    #
    # if np.isnan(auc):
    #     print(np.count_nonzero(label), np.count_nonzero(pred))
    # m = tf.keras.metrics.AUC(num_thresholds=3)
    # m.update_state(label, pred)
    # auc = m.result().numpy()
    # print(auc)

    return auc, pre, rec, f1, iou_out
