"""Class activation maps (Lai et al. 2023, Fig. 3c).

CAM_c(t) = sum_k w_ck F_k(t) from the last feature map of MSDNN, upsampled to the input
length. For each chosen class the most confident true-positive test ECG is drawn in the lead
where the finding is clearest; the red background is the CAM (transparent = low attention).

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

from matplotlib.colors import LinearSegmentedColormap

from .datasets import chapman
from .models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# (SNOMED code, name, lead index, line colour) - the paper's three examples
SHOW = [("164931005", "ST elevation", 7, "black"), ("164889003", "Atrial fibrillation", 1, "navy"),
        ("427172004", "Premature ventricular contractions", 1, "darkred")]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CMAP = LinearSegmentedColormap.from_list("cam", [(1, 1, 1, 0), (1, 0.6, 0.6, 0.35), (0.85, 0, 0, 0.8)])


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
    show = [s for s in SHOW if s[0] in d["classes"]]
    fig, axes = plt.subplots(len(show), 1, figsize=(14, 2.4 * len(show)), squeeze=False)
    for ax, (code, name, lead, col) in zip(axes[:, 0], show):
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
        sig = x[0, lead].numpy()
        lo, hi = sig.min() - 0.1, sig.max() + 0.1
        im = ax.imshow(cam[None] ** 2, aspect="auto", cmap=CMAP, extent=[0, t[-1], lo, hi], vmin=0, vmax=1)
        ax.plot(t, sig, color=col, lw=0.9)
        ax.set_ylim(lo, hi)
        ax.set_title(f"CAM of {name}: lead {LEADS[lead]}, test ECG {d['ids'][te[i]]}, model probability {p[i, c]:.2f}",
                     fontsize=10, loc="left")
        ax.set_ylabel("mV")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.subplots_adjust(hspace=0.5)
    fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.015, pad=0.01, label="model attention (CAM, normalised)")
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "figures", "fig3c_cam.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(ROOT, "figures", "fig3c_cam.pdf"), bbox_inches="tight")
    print("saved figures/fig3c_cam.png")


if __name__ == "__main__":
    main()
