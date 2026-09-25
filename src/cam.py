"""Class activation maps (Lai et al. 2023, Fig. 3c).

CAM_c(t) = sum_k w_ck F_k(t) from the last feature map of MSDNN, upsampled to the input
length. For each chosen class the most confident true-positive test ECG is drawn (lead II)
with the CAM as a red background.

  python -m src.cam --tag msdnn_pw_aug_s0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .datasets import chapman
from .models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHOW = [("164931005", "ST elevation"), ("164889003", "Atrial fibrillation"), ("17338001", "Premature ventricular contractions")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="msdnn_pw_aug_s0")
    a = ap.parse_args()
    d = chapman()
    z = np.load(os.path.join(ROOT, "results", "runs", "chapman", f"{a.tag}.npz"))
    net = build_model("msdnn", len(d["classes"]))
    net.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "chapman", f"{a.tag}.pt"), map_location="cpu", weights_only=True))
    net.eval()
    te, p = z["idx_test"], z["p_test"]
    show = [(c, n) for c, n in SHOW if c in d["classes"]]
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 2.6 * len(show)))
    for ax, (code, name) in zip(np.atleast_1d(axes), show):
        c = d["classes"].index(code)
        pos = np.where(d["y"][te, c] == 1)[0]
        i = pos[np.argmax(p[pos, c])]
        x = torch.from_numpy(d["X"][te[i]].astype(np.float32).T[None])
        with torch.no_grad():
            f = net.features(x)
            cam = torch.einsum("k,bkt->bt", net.fc.weight[c], f)
            cam = F.interpolate(cam[:, None], size=x.shape[-1], mode="linear", align_corners=False)[0, 0].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        t = np.arange(x.shape[-1]) / 500
        sig = x[0, 1].numpy()
        lo, hi = sig.min() - 0.1, sig.max() + 0.1
        ax.imshow(cam[None], aspect="auto", cmap="Reds", extent=[0, t[-1], lo, hi], alpha=0.6, vmin=0, vmax=1)
        ax.plot(t, sig, color="black", lw=0.8)
        ax.set_ylim(lo, hi)
        ax.set_title(f"CAM of {name} (test ECG {d['ids'][te[i]]}, p = {p[i, c]:.2f}; lead II)", fontsize=10, loc="left")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mV")
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "fig3c_cam.png"), dpi=150)
    fig.savefig(os.path.join(ROOT, "figures", "fig3c_cam.pdf"))
    print("saved figures/fig3c_cam.png")


if __name__ == "__main__":
    main()
