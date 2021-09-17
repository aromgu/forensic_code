import torch
import numpy as np
from keras import backend as K
import tensorflow as tf
import torch.nn.functional as F
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
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def pixelAcc(gt, pred, threshold = 0.5):
    pred[pred >= threshold] = 1; pred[pred < threshold] = 0
    pred = np.asarray(pred, dtype=np.int)
    gt = np.asarray(gt, dtype=np.int)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(gt, pred)

    return acc

def get_metrices(pred, label, threshold=0.5):
    pred = F.sigmoid(pred)
    pred = pred.cpu().detach().numpy()
    pred[pred >= threshold] = 1; pred[pred < threshold] = 0
    label = label.cpu().detach().numpy()

    acc = pixelAcc(label,pred)
    iou = tf.keras.metrics.MeanIoU(num_classes=2)
    iou.update_state(label, pred)
    iou_out = iou.result().numpy()

    pred = np.asarray(pred, dtype=np.int)
    label = np.asarray(label, dtype=np.int)
    pred = pred.reshape(-1)
    label = label.reshape(-1)

    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    # cm = confusion_matrix(label, pred, labels=[0, 1])
    # cm = [[TN FP
    #        FN TP]]
    # precision = cm[1, 1] / (cm[1, 1] + cm[0, 1] + 1e-10)
    # recall = cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-10)
    # plot_confusion_matrix(cm, target_names=['normal', 'forgery'])
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = get_f1score(precision, recall)
    precision = precision_score(label, pred, average='weighted')
    recall = recall_score(label, pred, average='weighted')
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    return acc, precision, recall, f1_score, iou_out

