import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
SMOOTH = 1e-6
EPS = 1e-10

def get_metrices(label, pred):
    label = label.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred = (pred >= 0.5).astype(np.int_)

    pred = np.asarray(pred.flatten(), dtype=np.int64)
    label = np.asarray(label.flatten(), dtype=np.int64)

    pre = precision_score(label, pred, average='macro')
    rec = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')
    iou_out = jaccard_score(label, pred, average='macro')
    auc = 0.0

    return auc, pre, rec, f1, iou_out


# def get_metrices(pred, label, pred_):
#     # from sklearn import metrics
#     from sklearn.metrics import jaccard_score
#
#     pred = np.asarray(pred.flatten(), dtype=np.int64)
#     pred_ = np.asarray(pred_.flatten(), dtype=np.int64)
#     label = np.asarray(label.flatten(), dtype=np.int64)
#
#     # acc = accuracy_score(label, pred)
#     pre = precision_score(label, pred, average='macro')
#     rec = recall_score(label, pred, average='macro')
#     f1 = f1_score(label, pred, average='macro')
#     iou_out = jaccard_score(label, pred, average='macro')
#     auc = 0.0
#     # y_true = np.expand_dims(label, axis=0)
#     # y_pred = np.expand_dims(pred, axis=0)
#     #if np.count_nonzero(label)==0 and np.count_nonzero(pred)==0:
#     #    auc = 1
#     #else:
#     # fpr, tpr, thresholds = metrics.roc_auc_score(label, pred_, average='macro')
#     # auc = metrics.auc(fpr, tpr)
#     #
#     # if np.isnan(auc):
#     #     print(np.count_nonzero(label), np.count_nonzero(pred))
#     # m = tf.keras.metrics.AUC(num_thresholds=3)
#     # m.update_state(label, pred)
#     # auc = m.result().numpy()
#     # print(auc)
#
#     return auc, pre, rec, f1, iou_out