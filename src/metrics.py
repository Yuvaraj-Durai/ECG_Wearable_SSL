"""Label-based and example-based metrics (Lai et al. 2023, "Metrics")."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def valid_classes(y):
    return np.where((y.sum(0) > 0) & (y.sum(0) < len(y)))[0]


def auroc_auprc(y, p):
    c = valid_classes(y)
    au = np.array([roc_auc_score(y[:, i], p[:, i]) for i in c])
    ap = np.array([average_precision_score(y[:, i], p[:, i]) for i in c])
    return c, au, ap


def f1_thresholds(y, p):
    """Per-class threshold maximising F1 (chosen on validation data)."""
    th = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        if y[:, i].sum() == 0:
            continue
        pr, rc, t = precision_recall_curve(y[:, i], p[:, i])
        f = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
        th[i] = t[np.argmax(f[:-1])]
    return th


def label_metrics(y, p, th):
    """Per-class TP/TN/FP/FN, sensitivity, specificity, F1 at thresholds th."""
    b = p >= th
    tp = (b & (y == 1)).sum(0)
    tn = (~b & (y == 0)).sum(0)
    fp = (b & (y == 0)).sum(0)
    fn = (~b & (y == 1)).sum(0)
    sen = tp / np.maximum(tp + fn, 1)
    spe = tn / np.maximum(tn + fp, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, sen=sen, spe=spe, f1=f1)


def example_metrics(y, p, th):
    """Per-ECG counts averaged over ECGs, and sample-wise Sen/Spe/F1/Acc."""
    b = p >= th
    tp = (b & (y == 1)).sum(1)
    tn = (~b & (y == 0)).sum(1)
    fp = (b & (y == 0)).sum(1)
    fn = (~b & (y == 1)).sum(1)
    return dict(tp=tp.mean(), tn=tn.mean(), fp=fp.mean(), fn=fn.mean(),
                sen=np.mean(tp / np.maximum(tp + fn, 1)), spe=np.mean(tn / np.maximum(tn + fp, 1)),
                f1=np.mean(2 * tp / np.maximum(2 * tp + fp + fn, 1)), acc=np.mean((tp + tn) / y.shape[1]))


def summary(y, p, th=None):
    c, au, ap = auroc_auprc(y, p)
    out = dict(auroc=float(au.mean()), auprc=float(ap.mean()), n_classes=int(len(c)))
    if th is not None:
        lm = label_metrics(y, p, th)
        out.update(f1=float(lm["f1"][c].mean()), sen=float(lm["sen"][c].mean()), spe=float(lm["spe"][c].mean()))
    return out
