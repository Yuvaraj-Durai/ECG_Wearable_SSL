"""Datasets at 500 Hz, 10 s (5000 samples x 12 leads), float16, in mV.

Preprocessing follows Lai et al. (Nat. Commun. 2023): 5th-order Butterworth
high-pass at 0.5 Hz. R-peak positions are pre-computed once for the cycle-mask
augmentation.

  chapman  - labelled multi-label set (PhysioNet ecg-arrhythmia 1.0.0, 45,152 ECGs,
             SNOMED-CT terms); stands in for the paper's 164,538 annotated ECGs.
  code15   - unlabelled pre-training pool (CODE-15%, 400 Hz resampled to 500 Hz,
             amplitude-harmonised to Chapman); stands in for the 493,948 unannotated ECGs.
  cpsc2018 - open-source evaluation (6,877 ECGs, 9 classes, 10-fold CV).

  python -m src.data chapman|code15|cpsc2018
"""
import glob
import os
import sys
import zipfile
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OLD = os.path.join(ROOT, "..", "ECG_12lead", "data")  # raw downloads shared with ECG_12lead
CACHE = os.path.join(ROOT, "data", "cache")
FS, T = 500, 5000
MAX_R = 128  # enough beats for 60 s CPSC records
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SOS = butter(5, 0.5, "highpass", fs=FS, output="sos")
MIN_POS = 100  # a Chapman term becomes a class with >= this many positives
CPSC_CLASSES = ["426783006", "164889003", "270492004", "164909002", "59118001",
                "284470004", "164884008", "429622005", "164931005"]  # SNOMED as in the 2020 challenge
CPSC_NAMES = ["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]


def preprocess(x, fs, T=T):
    """[t, 12] raw -> ([T, 12] float32 high-passed, zero-padded; valid length)."""
    x = np.nan_to_num(x.astype(np.float32))
    if fs != FS:
        g = np.gcd(int(fs), FS)
        x = resample_poly(x, FS // g, int(fs) // g, axis=0).astype(np.float32)
    x = sosfiltfilt(SOS, x, axis=0).astype(np.float32)
    n = min(len(x), T)
    out = np.zeros((T, 12), np.float32)
    out[:n] = x[:n]
    return out, n


def r_peaks(x, n):
    """R-peak sample indices from the summed absolute derivative of all leads."""
    e = np.abs(np.diff(x[:n], axis=0)).sum(1)
    e = np.convolve(e, np.ones(25) / 25, mode="same")
    p, _ = find_peaks(e, distance=int(0.3 * FS), height=0.35 * np.percentile(e, 99))
    # refine to the largest absolute amplitude of lead II near the energy peak
    p = [max(0, q - 25) + int(np.argmax(np.abs(x[max(0, q - 25):q + 25, 1]))) for q in p]
    out = np.full(MAX_R, -1, np.int32)
    out[:min(len(p), MAX_R)] = p[:MAX_R]
    return out


def _wfdb(args):
    """(header, max length, min length) -> (id, X, R-peaks, length, SNOMED codes)."""
    hea, tmax, tmin = args
    try:
        r = wfdb.rdrecord(hea[:-4])
        if r.p_signal.shape[1] != 12:
            return None
        x, n = preprocess(r.p_signal, r.fs, tmax)
        if n < tmin or not np.isfinite(x).all() or np.abs(x).sum() == 0:
            return None
        dx = next((c.split(":", 1)[1].strip() for c in r.comments if c.startswith("Dx")), "")
        return os.path.basename(hea[:-4]), x.astype(np.float16), r_peaks(x, n), n, dx.split(",")
    except Exception:
        return None


def _save(name, X, R, extra):
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, f"{name}_X.npy"), X)
    np.save(os.path.join(CACHE, f"{name}_R.npy"), R)
    pd.DataFrame(extra).to_csv(os.path.join(CACHE, f"{name}_meta.csv"), index=False)


def build_wfdb(name, pattern, tmax=T, tmin=int(0.9 * T)):
    heas = sorted(glob.glob(pattern))
    with Pool(96) as p:
        out = [o for o in p.imap(_wfdb, [(h, tmax, tmin) for h in heas], chunksize=32) if o is not None]
    ids, X, R, n, dx = zip(*out)
    _save(name, np.stack(X), np.stack(R), {"id": ids, "len": n, "dx": [" ".join(d) for d in dx]})
    print(name, len(ids), "records of", len(heas))


def _code15_part(z):
    """One CODE-15% part -> temporary files (X float16 [n, 5000, 12], R, exam ids); returns n."""
    h5 = os.path.join(CACHE, os.path.basename(z).replace(".zip", ".hdf5"))
    with zipfile.ZipFile(z) as f:
        f.extract(os.path.basename(h5), CACHE)
    X, R, ids = [], [], []
    with h5py.File(h5, "r") as f:
        tr, ex = f["tracings"], f["exam_id"][:]
        for s in range(0, len(tr), 1024):
            b = tr[s:s + 1024]
            for i in range(len(b)):
                nz = np.where(np.abs(b[i]).sum(1) > 0)[0]
                if len(nz) < 0.95 * 7 * 400 or not np.isfinite(b[i]).all():
                    continue
                x, n = preprocess(b[i, nz[0]:nz[-1] + 1], 400)
                X.append(x.astype(np.float16))
                R.append(r_peaks(x, n))
                ids.append(ex[s + i])
    os.remove(h5)
    tag = os.path.join(CACHE, "tmp_" + os.path.basename(z)[:-4])
    np.save(tag + "_X.npy", np.stack(X))
    np.save(tag + "_R.npy", np.stack(R))
    np.save(tag + "_id.npy", np.array(ids))
    print(os.path.basename(z), len(ids), flush=True)
    return tag


def mad(a):
    return np.median(np.abs(a - np.median(a, 0)), 0)


def build_code15():
    parts = sorted(glob.glob(os.path.join(OLD, "code15", "exams_part*.zip")))
    os.makedirs(CACHE, exist_ok=True)
    with Pool(9) as p:  # ~4 GB hdf5 each, extracted to the cache temporarily
        tags = p.map(_code15_part, parts)
    sizes = [len(np.load(t + "_id.npy")) for t in tags]
    X = np.lib.format.open_memmap(os.path.join(CACHE, "code15_X.npy"), "w+", np.float16, (sum(sizes), T, 12))
    o = 0
    for t, m in zip(tags, sizes):
        X[o:o + m] = np.load(t + "_X.npy", mmap_mode="r")
        o += m
    R = np.concatenate([np.load(t + "_R.npy") for t in tags])
    ids = np.concatenate([np.load(t + "_id.npy") for t in tags])
    # match each lead's median absolute deviation to Chapman
    ch = np.load(os.path.join(CACHE, "chapman_X.npy"), mmap_mode="r")
    rng = np.random.default_rng(0)
    a = ch[rng.choice(len(ch), 3000, replace=False)].astype(np.float32).reshape(-1, 12)
    b = X[rng.choice(len(X), 3000, replace=False)].astype(np.float32).reshape(-1, 12)
    b = b[np.abs(b).sum(1) > 0]
    scale = (mad(a) / mad(b)).astype(np.float32)
    print("CODE-15 per-lead scale:", scale.round(3))
    for s in range(0, len(X), 8192):
        X[s:s + 8192] = (X[s:s + 8192].astype(np.float32) * scale).astype(np.float16)
    X.flush()
    np.save(os.path.join(CACHE, "code15_R.npy"), R)
    pd.DataFrame({"id": ids}).to_csv(os.path.join(CACHE, "code15_meta.csv"), index=False)
    np.save(os.path.join(CACHE, "code15_scale.npy"), scale)
    for t in tags:
        for k in ("_X", "_R", "_id"):
            os.remove(t + k + ".npy")
    print("code15", X.shape)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "chapman":
        build_wfdb("chapman", os.path.join(OLD, "chapman", "WFDBRecords", "*", "*", "*.hea"))
    elif what == "cpsc2018":
        # 6-60 s records: keep up to 60 s; training crops 10 s windows, testing slides over the record
        build_wfdb("cpsc2018", os.path.join(ROOT, "data", "cpsc2018", "g*", "*.hea"), tmax=12 * T, tmin=int(0.55 * T))
    elif what == "code15":
        build_code15()
