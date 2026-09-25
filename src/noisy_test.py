"""An "inferior-quality" copy of the Chapman test split (stands in for the paper's 1,791
artifact-laden wearable ECGs, Fig. 2d).

Real recorded noise from the MIT-BIH Noise Stress Test Database (electrode motion 'em',
muscle artifact 'ma', baseline wander 'bw'; 360 Hz, resampled to 500 Hz) is added to every
test ECG at a per-record SNR drawn from U(0, 12) dB, with an independent noise segment per
lead; 10% of records also get one lead detached (flat line) for 2-6 s. The result is passed
through the same 0.5 Hz high-pass as all other data. None of this noise is used in training.

  python -m src.noisy_test
"""
import os

import numpy as np
import wfdb
from scipy.signal import resample_poly, sosfiltfilt

from .data import CACHE, FS, ROOT, SOS, T
from .datasets import chapman


def main():
    rng = np.random.default_rng(2023)
    noise = {}
    for r in ["em", "ma", "bw"]:
        x = wfdb.rdrecord(os.path.join(ROOT, "data", "nstdb", r)).p_signal.astype(np.float32)
        noise[r] = resample_poly(x, 25, 18, axis=0).astype(np.float32)  # 360 -> 500 Hz
    d = chapman()
    te = np.sort(np.where(d["split"] == "test")[0])
    X = d["X"][te].astype(np.float32)
    out = np.empty_like(X)
    snr = rng.uniform(0, 12, len(te))
    for i in range(len(te)):
        x = X[i]
        mix = rng.dirichlet([1, 1, 1])  # proportions of em / ma / bw
        n = np.zeros_like(x)
        for w, r in zip(mix, ["em", "ma", "bw"]):
            src = noise[r]
            for l in range(12):
                s = rng.integers(0, len(src) - T)
                seg = src[s:s + T, rng.integers(0, src.shape[1])]
                n[:, l] += w * (seg - seg.mean()) / (seg.std() + 1e-6)
        ps = (x ** 2).mean(0) + 1e-8  # per-lead signal power
        pn = (n ** 2).mean(0) + 1e-8
        y = x + n * np.sqrt(ps / pn / 10 ** (snr[i] / 10))
        if rng.random() < 0.1:
            l, L = rng.integers(0, 12), int(rng.uniform(2, 6) * FS)
            s = rng.integers(0, T - L)
            y[s:s + L, l] = 0
        out[i] = sosfiltfilt(SOS, y, axis=0)
    np.save(os.path.join(CACHE, "chapman_test_noisy_X.npy"), out.astype(np.float16))
    np.save(os.path.join(CACHE, "chapman_test_noisy_snr.npy"), snr)
    print("noisy test set", out.shape, "median SNR", np.median(snr).round(2))


if __name__ == "__main__":
    main()
