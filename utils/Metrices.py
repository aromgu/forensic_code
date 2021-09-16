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

def iou_numpy(pred, labels):
    pred = torch.sigmoid(pred)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.cpu().detach().numpy().squeeze(1).astype(int)
    labels = labels.cpu().detach().numpy().astype(int)

    intersection = np.bitwise_and(pred, labels).sum((1, 2))
    union = (pred | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded.mean()  # Or thresholded.mean()

def dice_coefficient(hist):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.
    Args:
        hist: confusion matrix.
    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice

def get_precision(pred,gt,threshold=0.5):
    # pred = PR.cpu().detach().numpy()
    pred[pred >= threshold] = 1; pred[pred < threshold] = 0
    # gt = GT.int().cpu().detach().numpy()
    pred = np.asarray(pred, dtype=np.int)
    gt = np.asarray(gt, dtype=np.int)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    from sklearn.metrics import precision_score
    precision = precision_score(gt, pred, average='weighted')

    return precision
    # PR = PR > threshold
    # GT = GT == torch.max(GT)
    # TP = ((PR==1)&(GT==1))
    # FP = ((PR==1)&(GT==0))
    # # print(TP, FP)
    # Precision = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FP)) + 1e-6)
    # return Precision

def get_recall(pred,gt,threshold=0.5):
    # pred = PR.cpu().detach().numpy()
    pred[pred >= threshold] = 1; pred[pred < threshold] = 0
    # gt = GT.int().cpu().detach().numpy()
    pred = np.asarray(pred, dtype=np.int)
    gt = np.asarray(gt, dtype=np.int)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    from sklearn.metrics import recall_score
    recall = recall_score(gt, pred, average='weighted')

    return recall
    # PR = PR > threshold # 1
    # GT = GT == torch.max(GT)
    # print(torch.max(GT), 'MAX')
    # print(GT==1 ,'?')
    # TP = ((PR==1)&(GT==1))
    # FN = ((PR==0)&(GT==0))
    # print(float(torch.sum(TP)), 'tp')
    # print(float(torch.sum(FN)),'fn')
    # Recall = float(torch.sum(TP))/(float(torch.sum(TP)+torch.sum(FN)) + 1e-6)
    # return Recall

def get_f1score(precision, recall) :
    # precision = precision.cpu().detach().numpy()
    # recall = recall.cpu().detach().numpy()
    return 2 * precision * recall / (precision + recall + 1e-8)

def dice_coef(y_true, y_pred, smooth=1):
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    # y_pred = y_pred.cpu().detach().numpy()
    # y_true = y_true.cpu().detach().numpy()
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def pixelAcc(gt, pred, threshold = 0.5):
    # if target.shape != predicted.shape:
    #     print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
    #     return
    #
    # if target.dim() != 4:
    #     print("target has dim", target.dim(), ", Must be 4.")
    #     return
    #
    # accsum = 0
    # for i in range(target.shape[0]):
    #     target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
    #     # predicted_arr =
    #     predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy()#.argmax(0)
    #
    #     predicted_arr[predicted_arr >= 0.5] = 1
    #     predicted_arr[predicted_arr < 0.5] = 0
    #     same = (target_arr == predicted_arr).sum()
    #     a, b = target_arr.shape
    #     total = a * b
    #     accsum += same / total
    #
    # pixelAccuracy = accsum / target.shape[0]
    pred[pred >= threshold] = 1; pred[pred < threshold] = 0
    # gt = GT.int().cpu().detach().numpy()
    pred = np.asarray(pred, dtype=np.int)
    gt = np.asarray(gt, dtype=np.int)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(gt, pred)

    return acc

def get_metrices(pred, label):
    pred = F.sigmoid(pred)
    pred = pred.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    acc = pixelAcc(label,pred)
    precision = get_precision(pred, label)
    recall = get_recall(pred, label)
    f1 = get_f1score(precision, recall)
    iou = tf.keras.metrics.MeanIoU(num_classes=2)
    iou.update_state(label, pred)
    iou_out = iou.result().numpy()
    return acc, precision, recall, f1, iou_out