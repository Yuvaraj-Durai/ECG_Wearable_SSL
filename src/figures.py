"""Figures in the layout of Lai et al. (Nat. Commun. 2023), drawn from the saved runs.

  fig1_ecg_augmentations  Fig. 1d-f: a clean ECG, the four augmentations, real artifacts
  fig2a ... fig2g         the Fig. 2 panels, one file each
  fig2_all                all Fig. 2 panels on one page, as in the paper
  fig_pretraining_loss    MoCo pre-training curves

Violins: one dot per diagnostic term (or per seed in 2c), black lines at the 25th/50th/75th
percentiles, mean printed above. Colours follow the paper's Fig. 2a.

  called by python -m src.report (make_all receives that module)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve

from .augment import channel_mask, crop_resize, cycle_mask, freq_dropout

R = None  # the running src.report module (its data and run helpers), set by make_all

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})

COL = {"dnn": "#e41a1c", "ssdnn": "#8c8c8c", "msdnn": "#ff7f00", "msdnn_pw": "#4daf4a",
       "msdnn_aug": "#377eb8", "msdnn_pw_aug": "#984ea3"}
SHORT = {"dnn": "DNN\n(34-layer)", "ssdnn": "Single-scale\nDNN-18", "msdnn": "MSDNN", "msdnn_pw": "MSDNN\nwith PW",
         "msdnn_aug": "MSDNN\nwith Aug", "msdnn_pw_aug": "MSDNN\nwith PW&Aug"}
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PR_CLASSES = [("427172004", "PVC"), ("429622005", "STD"), ("713426002", "IRBBB")]  # paper: PVC, STD, PRBBB


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=16)
    ax.text(-0.08, 1.08, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


def violin(ax, data, labels, colors, ylim=(0, 1)):
    pos = np.arange(1, len(data) + 1)
    parts = ax.violinplot(data, pos, widths=0.8, showextrema=False)
    for b, c in zip(parts["bodies"], colors):
        b.set_facecolor(c)
        b.set_edgecolor("black")
        b.set_linewidth(0.6)
        b.set_alpha(0.3)
    rng = np.random.default_rng(0)
    for x, d, c in zip(pos, data, colors):
        d = np.asarray(d)
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(d)), d, s=10, color=c, edgecolor="none", zorder=3)
        for q, lw in zip(np.percentile(d, [25, 50, 75]), (0.8, 1.8, 0.8)):
            ax.hlines(q, x - 0.28, x + 0.28, color="black", lw=lw, zorder=4)
        ax.text(x, ylim[1], f"mean {d.mean():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(pos, labels)
    ax.set_ylim(*ylim)


def save(fig, name):
    fig.savefig(os.path.join(R.FIG, name + ".png"))
    fig.savefig(os.path.join(R.FIG, name + ".pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------------ Fig. 2
def fig2a(ax):
    keys = [k for k, _ in R.ABL if R.seeds(k)]
    data = [R.seed_avg_ap(R.seeds(k)) for k in keys]
    violin(ax, data, [SHORT[k] for k in keys], [COL[k] for k in keys])
    ax.set_ylabel("AUPRC per diagnostic term")
    panel(ax, "a", f"Ablation: each dot is one of {len(data[0])} terms (mean of 3 seeds)")


def fig2b(ax):
    n_tr = int((R.D["split"] == "train").sum())
    curves = {}
    for pw, lab, c, mk in [("", "MSDNN (random initialisation)", COL["msdnn"], "o"),
                           ("_pw", "MSDNN with PW (pre-trained)", COL["msdnn_pw"], "s")]:
        pts = [(f * n_tr, r["test"]["auprc"]) for f in R.FRACS
               for r in [R.run(f"msdnn{pw}_frac{f}_s0") if f < 1 else R.run(f"msdnn{pw}_s0")] if r]
        if pts:
            x, y = map(np.array, zip(*pts))
            ax.plot(x, y, marker=mk, color=c, label=lab, ms=5)
            curves[pw] = (x, y)
    if len(curves) == 2 and len(curves[""][0]) == len(curves["_pw"][0]):
        (x, a), (_, b) = curves[""], curves["_pw"]
        ax.fill_between(x, a, b, color=COL["msdnn_pw"], alpha=0.1, label="gain from pre-training")
        for i in (0, len(x) - 1):
            ax.annotate(f"+{b[i] - a[i]:.3f}", (x[i], (a[i] + b[i]) / 2), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=COL["msdnn_pw"])
    ax.set(xlabel="Number of labelled training ECGs", ylabel="Macro AUPRC (test)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    panel(ax, "b", "Pre-trained vs random initialisation by training-set size")


def fig2c(ax):
    base, comb = R.seeds("msdnn"), R.seeds("msdnn_aug")
    ks = [(k, n) for k, n in R.AUG if R.seeds(f"msdnn_aug-{k}")]
    data = [[r["test"]["auprc"] for r in R.seeds(f"msdnn_aug-{k}")] for k, _ in ks]
    b = np.mean([r["test"]["auprc"] for r in base])
    c = np.mean([r["test"]["auprc"] for r in comb]) if comb else b
    lo, hi = min(min(map(min, data)), b) - 0.01, max(max(map(max, data)), c) + 0.012
    violin(ax, data, [n.replace(" ", "\n") for _, n in ks], ["#bdbdbd"] * len(ks), ylim=(lo, hi))
    for v, col, lab in [(b, COL["msdnn"], f"MSDNN without Aug ({b:.3f})"), (c, "#17becf", f"MSDNN with all four Aug ({c:.3f})")]:
        ax.axhline(v, color=col, lw=1.5)
        ax.text(len(ks) + 0.45, v, lab, color=col, fontsize=8, va="bottom", ha="right")
    ax.set(ylabel="Macro AUPRC (test)", xlabel="ECG augmentation used alone")
    panel(ax, "c", "Each augmentation alone (dots = 3 seeds)")


def fig2d(ax):
    a = [r for r in R.seeds("msdnn") if "p_test_noisy" in r]
    b = [r for r in R.seeds("msdnn_aug") if "p_test_noisy" in r]
    data = [R.seed_avg_ap(a), R.seed_avg_ap(a, "p_test_noisy"), R.seed_avg_ap(b), R.seed_avg_ap(b, "p_test_noisy")]
    violin(ax, data, ["Clean\ntest set", "Noisy\ntest set"] * 2, [COL["msdnn"]] * 2 + [COL["msdnn_aug"]] * 2)
    ax.axvline(2.5, color="grey", lw=0.6, ls=":")
    ax.legend(handles=[Patch(color=COL["msdnn"], alpha=0.5, label="MSDNN"),
                       Patch(color=COL["msdnn_aug"], alpha=0.5, label="MSDNN with Aug")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.set_ylabel("AUPRC per diagnostic term")
    panel(ax, "d", "Robustness to real artifacts (NSTDB noise)")


def fig2e(ax):
    names = {"lead-I": "1 lead\n(I)", "lead-holter": "3 leads\n(II, V1, V5)", "lead-frank": "3 leads\n(I, aVF, V2)", "": "12 leads"}
    rs = [(k, R.seeds(f"msdnn_pw_aug_{k}" if k else "msdnn_pw_aug")) for k, _ in R.LEADS]
    rs = [(k, v) for k, v in rs if v]
    violin(ax, [R.seed_avg_ap(v) for _, v in rs], [names[k] for k, _ in rs],
           [COL["msdnn_pw_aug"] if k == "" else "#8c8c8c" for k, _ in rs])
    ax.set_ylabel("AUPRC per diagnostic term")
    panel(ax, "e", "Fewer leads (simulated 1-3 lead devices)")


def fig2f(axes):
    rs = {k: R.run(f"{k}_s0") for k, _ in R.ABL if k != "ssdnn"}
    rs = {k: v for k, v in rs.items() if v}
    for i, (ax, (code, name)) in enumerate(zip(axes, PR_CLASSES)):
        k = R.D["classes"].index(code)
        for key, r in rs.items():
            y, p = R.D["y"][r["idx_test"], k], r["p_test"][:, k]
            pr, rc, _ = precision_recall_curve(y, p)
            c, _, ap = R.per_class(r)
            ax.step(rc, pr, where="post", color=COL[key], lw=1.2,
                    label=f"{SHORT[key].replace(chr(10), ' ')}: {ap[list(c).index(k)]:.3f}")
        ax.set(xlabel="Sensitivity (recall)", ylabel="PPV (precision)", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(title="AUPRC", fontsize=7, title_fontsize=7, loc="lower left" if i == 0 else "upper right")
        ax.grid(alpha=0.3)
        full = R.D["full_names"][k]
        if i == 0:
            panel(ax, "f", f"PR curve of {name} ({full})")
        else:
            ax.set_title(f"PR curve of {name} ({full})", loc="left", pad=16)


def fig2g(axes):
    r = R.run("msdnn_pw_aug_s0")
    y, p, ft, be, of = R.af_points(r)
    pr, rc, t = precision_recall_curve(y, p)
    ax = axes[0]
    ax.plot(rc, pr, color=COL["msdnn_pw_aug"], lw=1.2)
    ax.plot([0, 1], [0, 1], color="black", lw=0.6)
    for th, mk, c, lab in [(be, "o", "blue", "Break-even point"), (of, "s", "green", "Optimal-F1 point"),
                           (ft, "*", "red", "High-sensitivity point")]:
        b = p >= th
        tp = (b & (y == 1)).sum()
        ax.scatter(tp / y.sum(), tp / max(b.sum(), 1), marker=mk, s=70, color=c, label=f"{lab} (t = {th:.2f})", zorder=3)
    ax.set(xlabel="Sensitivity (recall)", ylabel="PPV (precision)", xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend(loc="upper left", fontsize=8)
    panel(ax, "g", "Three operating points on the PR curve of AF")
    ax = axes[1]
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    ax.plot(t, pr[:-1], color=COL["msdnn_pw_aug"], label="Precision")
    ax.plot(t, rc[:-1], color=COL["msdnn"], label="Recall")
    ax.plot(t, f1[:-1], color="#17becf", label="F1")
    for th, c, lab in [(ft, "red", "High-sensitivity"), (be, "blue", "Break-even"), (of, "green", "Optimal F1")]:
        ax.axvline(th, color=c, lw=1)
        ax.text(th, 0.5, " " + lab, rotation=90, color=c, fontsize=8, va="center", ha="right")
    ax.set(xlabel="Decision threshold", ylabel="Value", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    ax.set_title("Precision, recall and F1 versus threshold (AF)", loc="left", pad=16)


def fig2_all():
    fig = plt.figure(figsize=(15, 19))
    g = fig.add_gridspec(4, 6, hspace=0.6, wspace=0.6)
    fig2a(fig.add_subplot(g[0, :3]))
    fig2b(fig.add_subplot(g[0, 3:]))
    fig2c(fig.add_subplot(g[1, :2]))
    fig2d(fig.add_subplot(g[1, 2:4]))
    fig2e(fig.add_subplot(g[1, 4:]))
    fig2f([fig.add_subplot(g[2, 2 * i:2 * i + 2]) for i in range(3)])
    fig2g([fig.add_subplot(g[3, :3]), fig.add_subplot(g[3, 3:])])
    save(fig, "fig2_all")


def fig2_panels():
    for f, name, size in [(fig2a, "fig2a_ablation", (10, 4.5)), (fig2b, "fig2b_training_size", (7, 4.5)),
                          (fig2c, "fig2c_augmentations", (7, 4.5)), (fig2d, "fig2d_robustness", (7, 4.5)),
                          (fig2e, "fig2e_leads", (7, 4.5))]:
        fig, ax = plt.subplots(figsize=size)
        f(ax)
        save(fig, name)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig2f(axes)
    save(fig, "fig2f_pr_curves")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig2g(axes)
    save(fig, "fig2g_operating_points_af")


# ------------------------------------------------------------------------------------ Fig. 1
def fig1():
    """d: a clean ECG; e: the four augmentations at the top of their training ranges (grey = original,
    red band = masked); f: the same ECG from the noisy test set (real NSTDB artifacts)."""
    D = R.D
    te = np.sort(np.where(D["split"] == "test")[0])
    nsr = D["y"][te, D["classes"].index("426783006")] == 1
    cand = te[nsr & (D["y"][te].sum(1) == 1)][:300]
    hf = [np.abs(np.diff(D["X"][i, :, 0].astype(np.float32), 2)).mean() for i in cand]  # pick a smooth one
    i = int(cand[int(np.argmin(hf))])
    x = torch.from_numpy(D["X"][i].astype(np.float32).T[None])
    Rp = torch.from_numpy(D["R"][i][None]).long()
    lead, L = 0, 2500  # lead I, first 5 s
    t = np.arange(L) / 500
    torch.manual_seed(0)
    cyc = cycle_mask(x, Rp, offset=(0.04, 0.04), width=(0.12, 0.12))
    chan = channel_mask(x, p=0.5)
    while not (chan[0].abs().sum(1) == 0).any():
        chan = channel_mask(x, p=0.5)
    masked = int(torch.where(chan[0].abs().sum(1) == 0)[0][0])  # show one masked lead
    noisy = np.load(os.path.join(R.ROOT, "data", "cache", "chapman_test_noisy_X.npy"), mmap_mode="r")
    snr = np.load(os.path.join(R.ROOT, "data", "cache", "chapman_test_noisy_snr.npy"))
    j = int(np.searchsorted(te, i))
    rows = [("d", "Good-quality ECG (lead I)", x[0, lead], "green", None),
            ("e", "Frequency dropout (10% of DCT coefficients removed)", freq_dropout(x, rate=(0.1, 0.1))[0, lead], "purple", None),
            ("", "Crop resize (85% of the record stretched to full length)", crop_resize(x, scale=(0.85, 0.85))[0, lead], "purple", None),
            ("", "Cycle mask (40-160 ms after every R peak set to zero)", cyc[0, lead], "purple", (x[0, lead] != 0) & (cyc[0, lead] == 0)),
            ("", f"Channel mask (lead {LEAD_NAMES[masked]} set to zero)", chan[0, masked], "purple", None),
            ("f", f"Same ECG with real artifacts from NSTDB (SNR {snr[j]:.1f} dB), lead I", torch.from_numpy(noisy[j, :, lead].astype(np.float32)), "darkred", None),
            ("", "Same ECG with real artifacts from NSTDB, lead V2", torch.from_numpy(noisy[j, :, 7].astype(np.float32)), "darkred", None)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(11, 1.35 * len(rows)), sharex=True)
    ref = x[0, lead, :L].numpy()
    for ax, (letter, title, sig, col, mask) in zip(axes, rows):
        sig = sig[:L].numpy()
        if col == "purple":
            ax.plot(t, ref, color="lightgrey", lw=1.2, label="original")
        if mask is not None:
            m = mask[:L].numpy()
            ax.fill_between(t, ref.min(), ref.max(), where=m, color="red", alpha=0.15, lw=0)
        ax.plot(t, sig, color=col, lw=0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="right", fontsize=8, pad=2)
        if letter:
            ax.text(-0.03, 1.0, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    axes[1].legend(loc="upper left", fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.subplots_adjust(hspace=0.45, top=0.95)
    fig.suptitle(f"Test ECG {D['ids'][i]} (first 5 s): d clean, e the four augmentations, f real artifacts", fontsize=10)
    save(fig, "fig1_ecg_augmentations")


def fig_pretrain():
    h = os.path.join(R.ROOT, "checkpoints", "pretrain", "moco_history.txt")
    if not os.path.exists(h):
        return
    a = np.loadtxt(h, ndmin=2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(a[:, 0], a[:, 1], label="contrastive loss $L_C$")
    ax[0].plot(a[:, 0], a[:, 2], label="distribution divergence $L_D$")
    ax[0].set(xlabel="Epoch", ylabel="Loss", title="MoCo pre-training losses")
    ax[0].legend()
    ax[1].plot(a[:, 0], a[:, 3], color="green")
    ax[1].set(xlabel="Epoch", ylabel="Top-1 accuracy", title="Finding the positive key among 72k negatives")
    for x_ in ax:
        x_.grid(alpha=0.3)
    save(fig, "fig_pretraining_loss")


def make_all(report):
    global R
    R = report
    for f in [fig2_panels, fig2_all, fig1, fig_pretrain]:
        try:
            f()
        except Exception as e:  # a missing experiment must not stop the report
            print(f"{f.__name__} skipped: {type(e).__name__}: {e}")
