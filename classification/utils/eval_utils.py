import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, auc, roc_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
import h5py


def compute_metrics(run_name, num_split=20):
    h5f = h5py.File(run_name + '.h5', 'r')
    y_ohe = h5f['y'][:]
    y_pred_ohe = h5f['y_pred'][:]
    y_var = h5f['y_var'][:]
    h5f.close()

    y = np.argmax(y_ohe, axis=1)
    y_pred = np.argmax(y_pred_ohe, axis=1)

    umin = np.min(y_var)
    umax = np.max(y_var)
    N_tot = np.prod(y.shape)
    wrong_pred = (y != y_pred).astype(int)
    right_pred = (y == y_pred).astype(int)

    precision_, recall_, threshold = precision_recall_curve(wrong_pred.reshape([-1]), y_var.reshape([-1]))

    fpr, tpr, threshold_ = roc_curve(wrong_pred.reshape([-1]), y_var.reshape([-1]))

    # TODO: what is the best way to pick thresholds?!
    # uT = np.linspace(umin, umax, num_split)
    uT = threshold

    npv, recall, acc, precision, T = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    counter = 0
    for ut in uT:
        t = (ut - umin) / (umax - umin)
        counter += 1
        uncertain = (y_var >= ut).astype(int)
        certain = (y_var < ut).astype(int)
        TP = np.sum(uncertain * wrong_pred)
        TN = np.sum(certain * right_pred)
        N_w = np.sum(wrong_pred)
        N_c = np.sum(certain)
        N_unc = np.sum(uncertain)
        recall = np.append(recall, TP / N_w)
        npv = np.append(npv, TN / N_c)
        precision = np.append(precision, TP/N_unc)
        acc = np.append(acc, (TN + TP) / N_tot)
        T = np.append(T, t)
    auc_ = auc(recall_, precision_)
    roc_auc = auc(fpr, tpr)
    return recall, npv, acc, precision_, recall_, auc_, T, fpr, tpr, roc_auc