import torch
import numpy as np
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

def get_metrices(pred, label):
    import tensorflow.keras as keras
    from sklearn.metrics import jaccard_score

    pred = np.asarray(pred.flatten(), dtype=np.int64)
    label = np.asarray(label.flatten(), dtype=np.int64)
    acc = accuracy_score(label, pred)
    pre = precision_score(label, pred, average='macro')
    rec = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')
    iou_out = jaccard_score(label, pred, average='macro')
    # iou = keras.metrics.MeanIoU(num_classes=2)
    # iou.update_state(label, pred)
    # iou_out = iou.result().numpy()
    # print(iou_out)
    # iou_out = np.array(iou_out

    return acc, pre, rec, f1, iou_out

    # acc = pixelAcc(label, pred)
    # print("accuracy", acc)
    # print("precision", pre)
    # print("recall", rec)
    # print("f1_score", f1_score)
    # pred = F.sigmoid(pre
    # pred = pred.cpu().detach().numpy()
    # label = label.cpu().detach().numpy()
    # pred[pred > threshold] = 1; pred[pred <= threshold] = 0
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(pred.squeeze(), cmap='gray')
    # ax[1].imshow(label.squeeze(), cmap='gray')
    # plt.show()
    #
    # acc = pixelAcc(label,pred)

    #
    # pred = np.asarray(pred, dtype=np.int)
    # label = np.asarray(label, dtype=np.int)
    # pred = pred.reshape(-1) 이게 reshape(-1, 1)로 되있었음 아 저거 원래 -1 맞았는데 내가 막 threshold랑 셰입 바꿔보다가 집간거였엉 아아 ㅇㅋㅇㅋ 그리고 최종 결과 보여줄께

    # label = label.reshape(-1)
    #
    # from sklearn.metrics import confusion_matrix, precision_score, recall_score
    # # pred[pred >= threshold] = 1; pred[pred < threshold] = 0
    #
    # cm = confusion_matrix(label, pred, labels=[0, 1])
    # print(cm)
    #
    # precision = cm[1, 1] / (cm[1, 1] + cm[0, 1] + 1e-10)
    # recall = cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-10)
    #
    # # f1 = get_f1score(precision, recall)
    # # precision = precision_score(label, pred, average='weighted')
    # # recall = recall_score(label, pred, average='weighted')
    # f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    # # print('precision',precision); print('recall',recall) #; print('iou',iou_out);print('acc',acc)
    #
    #
    # return acc, precision, recall, f1_score, iou_out

