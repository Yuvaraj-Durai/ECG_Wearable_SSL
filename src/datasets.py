"""Loading cached datasets, label vocabularies and fixed splits."""
import os

import numpy as np
import pandas as pd

from .data import CACHE, CPSC_CLASSES, CPSC_NAMES, RAW, ROOT

MIN_POS = 50  # Chapman terms with >= 50 positives become classes (62 terms; the paper uses 60)
SPLIT_SEED = 0


def snomed_names():
    """SNOMED code -> (abbreviation, full name) from the PhysioNet 2021 mapping and Chapman's list."""
    out = {}
    ch = pd.read_csv(os.path.join(RAW, "chapman", "ConditionNames_SNOMED-CT.csv"))
    ch.columns = ["abbr", "name", "code"]
    for r in ch.itertuples():
        out[str(r.code)] = (r.abbr, r.name)
    for f in ["dx_mapping_unscored.csv", "dx_mapping_scored.csv"]:
        d = pd.read_csv(os.path.join(ROOT, "data", f))
        for r in d.itertuples():
            out[str(r.SNOMEDCTCode)] = (r.Abbreviation, r.Dx)
    return out


def multihot(dx, classes):
    idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(dx), len(classes)), np.float32)
    for r, d in enumerate(dx):
        for c in str(d).split():
            if c in idx:
                y[r, idx[c]] = 1
    return y


def load(name, mmap=True):
    X = np.load(os.path.join(CACHE, f"{name}_X.npy"), mmap_mode="r" if mmap else None)
    R = np.load(os.path.join(CACHE, f"{name}_R.npy"))
    meta = pd.read_csv(os.path.join(CACHE, f"{name}_meta.csv"), dtype={"dx": str})
    return X, R, meta


def chapman():
    """Chapman-Shaoxing/Ningbo multi-label set with a fixed 80/10/10 split.
    Each record is from a different patient, so a record-wise split is also patient-wise."""
    X, R, meta = load("chapman")
    codes = pd.Series([c for d in meta.dx.fillna("") for c in d.split()]).value_counts()
    classes = sorted(codes.index[codes >= MIN_POS], key=lambda c: -codes[c])
    y = multihot(meta.dx.fillna(""), classes)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(y))
    n = len(y)
    split = np.empty(n, "U5")
    split[perm[:int(0.8 * n)]] = "train"
    split[perm[int(0.8 * n):int(0.9 * n)]] = "val"
    split[perm[int(0.9 * n):]] = "test"
    nm = snomed_names()
    names = [nm.get(c, (c, c))[0] for c in classes]
    full = [nm.get(c, (c, c))[1] for c in classes]
    return dict(X=X, R=R, y=y, classes=classes, names=names, full_names=full, split=split, ids=meta.id.values)


def cpsc2018():
    """CPSC2018 (9 classes): a stratified 15% held-out test set (fold -1) and 10 CV folds (0-9)
    on the rest, by iterative stratification on the label matrix. The paper's hidden challenge
    test set (2,954 ECGs) is not public, so the held-out set stands in for it."""
    X, R, meta = load("cpsc2018")
    y = multihot(meta.dx.fillna(""), CPSC_CLASSES)
    keep = y.sum(1) > 0
    props = np.array([0.15] + [0.085] * 10)
    desired = props[:, None] * y[keep].sum(0)[None]
    counts = np.zeros_like(desired)
    group = np.full(len(y), -1)
    rng = np.random.default_rng(SPLIT_SEED)
    for c in np.argsort(y[keep].sum(0)):  # rarest class first
        for i in rng.permutation(np.where((y[:, c] == 1) & (group == -1))[0]):
            g = int(np.argmax(desired[:, c] - counts[:, c]))
            group[i] = g
            counts[g] += y[i]
    fold = np.where(group == 0, -1, group - 1)
    fold[~keep] = -3
    return dict(X=X, R=R, y=y, classes=CPSC_CLASSES, names=CPSC_NAMES, fold=fold, test=fold == -1,
                length=meta["len"].values, ids=meta.id.values)
