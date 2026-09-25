"""CPSC2018 open-source evaluation (Lai et al. 2023, "Open-source dataset evaluation").

The 10 cross-validation models are combined by simple voting (a class is positive when
more than half of the models, each at its own validation-tuned threshold, say so) and by
probability averaging, and scored on the held-out test set with the challenge metric:
the mean over the 9 classes of per-class F1.

  python -m src.cpsc
"""
import glob
import json
import os

import numpy as np

from .datasets import cpsc2018
from .metrics import auroc_auprc, f1_thresholds, label_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "results", "runs", "cpsc2018")


def evaluate(prefix):
    files = sorted(glob.glob(os.path.join(RUNS, f"{prefix}_f*.npz")))
    if len(files) < 10:
        return None
    d = cpsc2018()
    z = [np.load(f) for f in files]
    te = z[0]["idx_test"]
    y = d["y"][te]
    votes = np.mean([zz["p_test"] >= zz["th"] for zz in z], 0)
    vote_pred = votes > 0.5
    # averaged probabilities with thresholds tuned on the out-of-fold validation predictions
    oof_p = np.concatenate([zz["p_val"] for zz in z])
    oof_y = np.concatenate([d["y"][zz["idx_val"]] for zz in z])
    th = f1_thresholds(oof_y, oof_p)
    p_mean = np.mean([zz["p_test"] for zz in z], 0)
    out = dict(n_models=len(z), n_test=int(len(te)), classes=d["names"])
    for name, pred in [("vote", vote_pred), ("mean", p_mean >= th)]:
        lm = label_metrics(y, pred.astype(float), 0.5)
        out[name] = dict(f1=float(lm["f1"].mean()), f1_per_class=lm["f1"].round(4).tolist())
    single = [float(label_metrics(y, zz["p_test"], zz["th"])["f1"].mean()) for zz in z]
    out["single_model_f1_mean"], out["single_model_f1_std"] = float(np.mean(single)), float(np.std(single))
    _, au, ap = auroc_auprc(y, p_mean)
    out["mean_auroc"], out["mean_auprc"] = float(au.mean()), float(ap.mean())
    return out


def main():
    res = {p: evaluate(p) for p in ["cpsc_pw_aug", "cpsc_aug"]}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "results", "cpsc2018.json"), "w"), indent=1)
    for k, v in res.items():
        if v:
            print(f"{k}: vote F1 {v['vote']['f1']:.3f}  mean-prob F1 {v['mean']['f1']:.3f}  "
                  f"single {v['single_model_f1_mean']:.3f}±{v['single_model_f1_std']:.3f}  AUROC {v['mean_auroc']:.3f}")


if __name__ == "__main__":
    main()
