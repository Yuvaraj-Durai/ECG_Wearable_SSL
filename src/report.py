"""Tables and figures mirroring Lai et al. (Nat. Commun. 2023) from the saved runs.

  results/RESULTS.md   all tables (markdown)
  figures/             fig1e augmentations, fig2a-g, fig3c (src.cam), pre-training loss

  python -m src.report
"""
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.metrics import precision_recall_curve

from .augment import channel_mask, crop_resize, cycle_mask, freq_dropout
from .datasets import chapman
from .metrics import auroc_auprc, example_metrics, label_metrics
from .models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "results", "runs", "chapman")
FIG = os.path.join(ROOT, "figures")
SEEDS = [0, 1, 2]
ABL = [("dnn", "DNN (34-layer, Hannun)"), ("ssdnn", "Single-scale DNN-18 (k=17)"), ("msdnn", "MSDNN"),
       ("msdnn_pw", "MSDNN with PW"), ("msdnn_aug", "MSDNN with Aug"), ("msdnn_pw_aug", "MSDNN with PW&Aug")]
AUG = [("freq", "Frequency dropout"), ("crop", "Crop resize"), ("cycle", "Cycle mask"), ("channel", "Channel mask")]
LEADS = [("lead-I", "1 (lead I)"), ("lead-holter", "3 (Holter: II, V1, V5)"), ("lead-frank", "3 (Frank: I, aVF, V2)"), ("", "12")]
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
D = None
OUT = []


def md(*lines):
    OUT.extend(lines)


def run(tag):
    j = os.path.join(RUNS, f"{tag}.json")
    if not os.path.exists(j):
        return None
    r = json.load(open(j))
    r.update({k: v for k, v in np.load(os.path.join(RUNS, f"{tag}.npz")).items()})
    return r


def per_class(r, key="p_test"):
    """Per-class AUROC and AUPRC over classes with positives in the test split."""
    y = D["y"][r["idx_test"]]
    c, au, ap = auroc_auprc(y, r[key])
    return c, au, ap


def seeds(prefix):
    return [r for r in (run(f"{prefix}_s{s}") for s in SEEDS) if r is not None]


def ms(v):
    v = np.asarray(v)
    return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def seed_avg_ap(rs, key="p_test"):
    return np.mean([per_class(r, key)[2] for r in rs], 0)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def violin(ax, data, labels, colors=None):
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_alpha(0.35)
        if colors:
            b.set_facecolor(colors[i])
    for i, d in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i + 1, 0.04, len(d)), d, s=6, color="k", alpha=0.6)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)


# ---------------------------------------------------------------------------------------- tables
def table1_2():
    r = run("msdnn_pw_aug_s0")
    if r is None:
        return
    te, va = r["idx_test"], r["idx_val"]
    y = D["y"][te]
    th = r["th"]
    c, au, ap = per_class(r)
    lm = label_metrics(y, r["p_test"], th)
    has_noisy = "p_test_noisy" in r
    ln = label_metrics(y, r["p_test_noisy"], th) if has_noisy else None
    n_all = D["y"].sum(0).astype(int)
    md("## Table 1. Per-term diagnostic performance of MSDNN with PW&Aug (seed 0)", "",
       "Clean = Chapman test split (n = %d); noisy = the same ECGs with real NSTDB artifacts (the stand-in for the paper's "
       "online wearable test). Thresholds: per-class F1-optimal on the validation split." % len(te), "",
       "| Term | Name | Samples (all) | Test pos. | AUROC | AUPRC | F1 | TP | TN | FP | FN | Sen | Spe | F1 |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ci = {k: i for i, k in enumerate(c)}
    rows = []
    for k in range(len(D["classes"])):
        if k not in ci:
            continue
        i = ci[k]
        L = ln if has_noisy else lm
        rows.append([au[i], ap[i], lm["f1"][k], L["tp"][k], L["tn"][k], L["fp"][k], L["fn"][k], L["sen"][k], L["spe"][k], L["f1"][k]])
        md(f"| {D['names'][k]} | {D['full_names'][k]} | {n_all[k]} | {int(y[:, k].sum())} | {au[i]:.3f} | {ap[i]:.3f} | "
           f"{lm['f1'][k]:.3f} | {L['tp'][k]} | {L['tn'][k]} | {L['fp'][k]} | {L['fn'][k]} | {L['sen'][k]:.3f} | "
           f"{L['spe'][k]:.3f} | {L['f1'][k]:.3f} |")
    m = np.array(rows, float).mean(0)
    md(f"| **Average** | | {n_all.mean():.1f} | | **{m[0]:.3f}** | **{m[1]:.3f}** | **{m[2]:.3f}** | {m[3]:.1f} | {m[4]:.1f} | "
       f"{m[5]:.1f} | {m[6]:.1f} | **{m[7]:.3f}** | **{m[8]:.3f}** | **{m[9]:.3f}** |", "",
       "Columns TP-F1 (right block) are on the " + ("noisy" if has_noisy else "clean") + " test set.", "")
    md("## Table 2. Example-based performance (per-ECG averages)", "",
       "| Test set | TP | TN | FP | FN | Sen | Spe | F1 | Acc |", "|---|---|---|---|---|---|---|---|---|")
    for name, key in [("Clean (n = %d)" % len(te), "p_test"), ("Noisy (n = %d)" % len(te), "p_test_noisy")]:
        if key in r:
            e = example_metrics(y, r[key], th)
            md(f"| {name} | {e['tp']:.2f} | {e['tn']:.1f} | {e['fp']:.2f} | {e['fn']:.2f} | {e['sen']:.3f} | {e['spe']:.3f} | {e['f1']:.3f} | {e['acc']:.3f} |")
    md("")


def table3_fig2g():
    r = run("msdnn_pw_aug_s0")
    if r is None:
        return
    k = D["classes"].index("164889003")  # atrial fibrillation
    y, p = D["y"][r["idx_test"], k], r["p_test"][:, k]
    yv, pv = D["y"][r["idx_val"], k], r["p_val"][:, k]
    pr, rc, t = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    be = t[np.argmin(np.abs(pr[:-1] - rc[:-1]))]
    of = t[np.argmax(f1[:-1])]
    # "fine-tuned" point: the paper's cardiologists preferred sensitivity; we take the highest
    # validation threshold that reaches sensitivity >= 0.95
    tv = np.sort(pv[yv == 1])
    ft = tv[int(np.floor(0.05 * len(tv)))]
    ths = sorted(set([0.004, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.97, 0.998, ft, be, of]))
    md("## Table 3. Atrial fibrillation at different thresholds (test split)", "",
       f"Bold: high-sensitivity point (validation sensitivity ≥ 0.95, threshold {ft:.3f}), break-even point ({be:.3f}), "
       f"optimal-F1 point ({of:.3f}).", "",
       "| Threshold | TP | TN | FP | FN | F1 | Sensitivity | Precision | Specificity |", "|---|---|---|---|---|---|---|---|---|")
    for th in ths:
        b = p >= th
        tp, tn, fp, fn = int((b & (y == 1)).sum()), int((~b & (y == 0)).sum()), int((b & (y == 0)).sum()), int((~b & (y == 1)).sum())
        row = (f"{th:.3f} | {tp} | {tn} | {fp} | {fn} | {2 * tp / max(2 * tp + fp + fn, 1):.3f} | {tp / max(tp + fn, 1):.3f} | "
               f"{tp / max(tp + fp, 1):.3f} | {tn / max(tn + fp, 1):.3f}")
        md(f"| **{row.replace(' | ', '** | **')}** |" if th in (ft, be, of) else f"| {row} |")
    md("")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(rc, pr, color="purple")
    ax[0].plot([0, 1], [0, 1], "k", lw=0.8)
    for th, mk, lab in [(be, "o", "Break even point"), (of, "s", "Optimal F1 point"), (ft, "*", "High-sensitivity point")]:
        b = p >= th
        tp = (b & (y == 1)).sum()
        ax[0].scatter(tp / y.sum(), tp / max(b.sum(), 1), marker=mk, s=60, label=lab, zorder=3)
    ax[0].set(xlabel="Sensitivity (recall)", ylabel="PPV (precision)", title="Three operating points on the PR curve of AF")
    ax[0].legend()
    grid = np.linspace(0, 1, 201)
    P = [((p >= g) & (y == 1)).sum() / max((p >= g).sum(), 1) for g in grid]
    Rr = [((p >= g) & (y == 1)).sum() / y.sum() for g in grid]
    F = [2 * a * b / max(a + b, 1e-12) for a, b in zip(P, Rr)]
    ax[1].plot(grid, P, label="Precision"), ax[1].plot(grid, Rr, label="Recall"), ax[1].plot(grid, F, label="F1")
    for th, col, lab in [(ft, "r", "High-sens."), (be, "b", "Break even"), (of, "g", "Optimal F1")]:
        ax[1].axvline(th, color=col, lw=1)
        ax[1].text(th, 0.05, lab, rotation=90, color=col, fontsize=8)
    ax[1].set(xlabel="Threshold", ylabel="Value", title="Precision, recall and F1 versus threshold")
    ax[1].legend()
    savefig(fig, "fig2g_operating_points_af")


def table4_fig2a():
    rs = {k: seeds(k) for k, _ in ABL}
    rs = {k: v for k, v in rs.items() if v}
    if not rs:
        return
    md("## Table 4. Ablation (Fig. 2a): macro metrics on the Chapman test split, mean ± s.d. over seeds", "",
       "| Model | Params | Seeds | AUROC | AUPRC | F1 | AUPRC (noisy test) | ΔAUPRC clean→noisy |", "|---|---|---|---|---|---|---|---|")
    for k, name in ABL:
        if k not in rs:
            continue
        v = rs[k]
        noisy = [r["test_noisy"]["auprc"] for r in v if "test_noisy" in r]
        clean = [r["test"]["auprc"] for r in v]
        drop = f"{np.mean(noisy) - np.mean(clean):+.3f}" if noisy else "–"
        md(f"| {name} | {v[0]['params'] / 1e6:.2f}M | {len(v)} | {ms([r['test']['auroc'] for r in v])} | "
           f"{ms(clean)} | {ms([r['test']['f1'] for r in v])} | {ms(noisy) if noisy else '–'} | {drop} |")
    md("", "Paired-sample t-tests on per-class test AUPRC (seed-averaged), as in the paper's Statistical analysis:", "",
       "| Comparison | mean ΔAUPRC | p-value |", "|---|---|---|")
    pairs = [("dnn", "msdnn"), ("ssdnn", "msdnn"), ("msdnn", "msdnn_pw"), ("msdnn", "msdnn_aug"), ("msdnn", "msdnn_pw_aug"), ("msdnn_aug", "msdnn_pw_aug")]
    names = dict(ABL)
    for a, b in pairs:
        if a in rs and b in rs:
            x, y = seed_avg_ap(rs[a]), seed_avg_ap(rs[b])
            md(f"| {names[a]} vs {names[b]} | {np.mean(y - x):+.4f} | {ttest_rel(x, y).pvalue:.3g} |")
    a, b = run("msdnn_frac0.1_s0"), run("msdnn_pw_frac0.1_s0")
    if a and b:
        x, y = per_class(a)[2], per_class(b)[2]
        md(f"| MSDNN vs MSDNN with PW (10% of training set) | {np.mean(y - x):+.4f} | {ttest_rel(x, y).pvalue:.3g} |")
    md("")
    keys = [k for k, _ in ABL if k in rs]
    fig, ax = plt.subplots(figsize=(9, 4))
    violin(ax, [seed_avg_ap(rs[k]) for k in keys], [names[k].replace(" with ", "\nwith ") for k in keys])
    ax.set(ylabel="AUPRC (per class)", ylim=(0, 1.02), title="Ablation: per-class test AUPRC (seed-averaged)")
    savefig(fig, "fig2a_ablation")


def table5_fig2c():
    base, comb = seeds("msdnn"), seeds("msdnn_aug")
    rs = {k: seeds(f"msdnn_aug-{k}") for k, _ in AUG}
    if not base or not any(rs.values()):
        return
    md("## Table 5. Each augmentation alone (Fig. 2c): test macro AUPRC, mean ± s.d. over seeds", "",
       "| Augmentation | AUROC | AUPRC | AUPRC (noisy test) |", "|---|---|---|---|")
    rows = [("None", base)] + [(n, rs[k]) for k, n in AUG if rs[k]] + [("All four combined", comb)]
    for n, v in rows:
        if v:
            md(f"| {n} | {ms([r['test']['auroc'] for r in v])} | {ms([r['test']['auprc'] for r in v])} | "
               f"{ms([r['test_noisy']['auprc'] for r in v if 'test_noisy' in r])} |")
    md("")
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = [(k, n) for k, n in AUG if rs[k]]
    violin(ax, [[r["test"]["auprc"] for r in rs[k]] for k, _ in ks], [n.replace(" ", "\n") for _, n in ks])
    ax.axhline(np.mean([r["test"]["auprc"] for r in base]), color="tab:orange", label="MSDNN without Aug")
    if comb:
        ax.axhline(np.mean([r["test"]["auprc"] for r in comb]), color="tab:cyan", label="MSDNN with combined Aug")
    ax.set(ylabel="macro AUPRC", xlabel="ECG augmentation methods")
    ax.legend(fontsize=8)
    savefig(fig, "fig2c_augmentations")


def table6_fig2b():
    n_tr = int((D["split"] == "train").sum())
    pts = {}
    for pw in ["", "_pw"]:
        for f in FRACS:
            r = run(f"msdnn{pw}_frac{f}_s0") if f < 1 else run(f"msdnn{pw}_s0")
            if r:
                pts.setdefault(pw, []).append((f, r["test"]["auprc"], r["test"]["auroc"]))
    if not pts:
        return
    md("## Table 6. Training-set size (Fig. 2b): test macro AUPRC / AUROC, random vs pretrained initialisation (seed 0)", "",
       "| Fraction | Training ECGs | MSDNN AUPRC | MSDNN with PW AUPRC | Δ | MSDNN AUROC | MSDNN with PW AUROC |", "|---|---|---|---|---|---|---|")
    a, b = dict((f, (p, q)) for f, p, q in pts.get("", [])), dict((f, (p, q)) for f, p, q in pts.get("_pw", []))
    for f in FRACS:
        if f in a or f in b:
            A, B = a.get(f, (np.nan, np.nan)), b.get(f, (np.nan, np.nan))
            md(f"| {f:.0%} | {int(round(f * n_tr))} | {A[0]:.3f} | {B[0]:.3f} | {B[0] - A[0]:+.3f} | {A[1]:.3f} | {B[1]:.3f} |")
    md("")
    fig, ax = plt.subplots(figsize=(7, 4))
    for pw, lab, c, mk in [("", "MSDNN", "peru", "o"), ("_pw", "MSDNN with PW", "green", "s")]:
        if pw in pts:
            f, p, _ = zip(*pts[pw])
            ax.plot(np.array(f) * n_tr, p, marker=mk, color=c, label=lab)
    ax.set(xlabel="Number of ECGs", ylabel="macro AUPRC")
    ax.legend()
    savefig(fig, "fig2b_training_size")


def fig2d():
    a, b = seeds("msdnn"), seeds("msdnn_aug")
    a, b = [r for r in a if "p_test_noisy" in r], [r for r in b if "p_test_noisy" in r]
    if not a or not b:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [seed_avg_ap(a), seed_avg_ap(a, "p_test_noisy"), seed_avg_ap(b), seed_avg_ap(b, "p_test_noisy")]
    violin(ax, data, ["MSDNN\nclean test", "MSDNN\nnoisy test", "MSDNN with Aug\nclean test", "MSDNN with Aug\nnoisy test"],
           ["tab:orange", "tab:orange", "tab:blue", "tab:blue"])
    ax.set(ylabel="AUPRC (per class)", ylim=(0, 1.02), title="Robustness to real wearable artifacts (NSTDB)")
    savefig(fig, "fig2d_robustness")
    md("## Table 7. Robustness (Fig. 2d): test macro AUPRC on clean and noisy ECGs", "",
       "| Model | Clean | Noisy | Drop |", "|---|---|---|---|")
    for n, v in [("MSDNN", a), ("MSDNN with Aug", b), ("MSDNN with PW&Aug", [r for r in seeds("msdnn_pw_aug") if "p_test_noisy" in r])]:
        if v:
            c_, n_ = [r["test"]["auprc"] for r in v], [r["test_noisy"]["auprc"] for r in v]
            md(f"| {n} | {ms(c_)} | {ms(n_)} | {np.mean(n_) - np.mean(c_):+.3f} |")
    md("", f"Gain from augmentation: clean {np.mean([r['test']['auprc'] for r in b]) - np.mean([r['test']['auprc'] for r in a]):+.3f}, "
       f"noisy {np.mean([r['test_noisy']['auprc'] for r in b]) - np.mean([r['test_noisy']['auprc'] for r in a]):+.3f} AUPRC.", "")


def table8_fig2e():
    rs = [(n, seeds(f"msdnn_pw_aug_{k}" if k else "msdnn_pw_aug")) for k, n in LEADS]
    rs = [(n, v) for n, v in rs if v]
    if len(rs) < 2:
        return
    md("## Table 8. Reduced-lead devices simulated from the 12 leads (Fig. 2e), MSDNN with PW&Aug", "",
       "| Leads | AUROC | AUPRC | F1 |", "|---|---|---|---|")
    for n, v in rs:
        md(f"| {n} | {ms([r['test']['auroc'] for r in v])} | {ms([r['test']['auprc'] for r in v])} | {ms([r['test']['f1'] for r in v])} |")
    md("")
    fig, ax = plt.subplots(figsize=(7, 4))
    violin(ax, [seed_avg_ap(v) for _, v in rs], [n.replace(" (", "\n(") for n, _ in rs])
    ax.set(xlabel="Number of leads", ylabel="AUPRC (per class)", ylim=(0, 1.02))
    savefig(fig, "fig2e_leads")


def fig2f():
    rs = {k: run(f"{k}_s0") for k, _ in ABL}
    rs = {k: v for k, v in rs.items() if v}
    if len(rs) < 2:
        return
    show = [("17338001", "PVC"), ("429622005", "ST depression"), ("713426002", "Incomplete RBBB")]
    show = [(c, n) for c, n in show if c in D["classes"]]
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 4))
    for ax, (code, name) in zip(np.atleast_1d(axes), show):
        k = D["classes"].index(code)
        for key, lab in ABL:
            if key in rs:
                y, p = D["y"][rs[key]["idx_test"], k], rs[key]["p_test"][:, k]
                pr, rc, _ = precision_recall_curve(y, p)
                ap = per_class(rs[key])
                ax.plot(rc, pr, lw=1, label=f"{lab}: {ap[2][list(ap[0]).index(k)]:.4f}")
        ax.set(title=f"Class {name}", xlabel="Sensitivity (recall)", ylabel="PPV (precision)", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=7, loc="lower left")
    savefig(fig, "fig2f_pr_curves")


def table9_cpsc():
    f = os.path.join(ROOT, "results", "cpsc2018.json")
    if not os.path.exists(f):
        return
    res = json.load(open(f))
    md("## Table 9. CPSC2018 (9 classes): 10 cross-validation models, simple voting, held-out test split", "",
       "The paper reports F1 0.839 on the hidden CPSC2018 test set (challenge best 0.837). That set is not public; "
       "our held-out 15% split is scored with the same metric (mean per-class F1).", "",
       "| Initialisation | Vote F1 | Mean-probability F1 | Single model F1 | Mean AUROC | " + " | ".join(
           next(v for v in res.values() if v)["classes"]) + " |",
       "|---|---|---|---|---|" + "---|" * 9)
    for k, n in [("cpsc_aug", "Random + Aug"), ("cpsc_pw_aug", "PW (MoCo) + Aug")]:
        v = res.get(k)
        if v:
            md(f"| {n} | **{v['vote']['f1']:.3f}** | {v['mean']['f1']:.3f} | {v['single_model_f1_mean']:.3f} ± "
               f"{v['single_model_f1_std']:.3f} | {v['mean_auroc']:.3f} | " + " | ".join(f"{x:.3f}" for x in v["vote"]["f1_per_class"]) + " |")
    md("")


def table10_cost():
    torch.set_num_threads(1)
    md("## Table 10. Model size and single-thread CPU inference time per 10 s, 12-lead ECG", "",
       "| Model | Params | ms / ECG |", "|---|---|---|")
    x = torch.randn(1, 12, 5000)
    for m, n in [("dnn", "DNN"), ("ssdnn", "Single-scale DNN-18"), ("msdnn", "MSDNN")]:
        net = build_model(m, len(D["classes"])).eval()
        with torch.no_grad():
            for _ in range(3):
                net(x)
            t = time.perf_counter()
            for _ in range(20):
                net(x)
        md(f"| {n} | {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M | {(time.perf_counter() - t) / 20 * 1e3:.1f} |")
    md("", "The paper: MSDNN 2.13M parameters, < 0.08 s per 15 s recording.", "")


def fig1e():
    nsr = D["y"][:, D["classes"].index("426783006")] == 1  # a normal sinus rhythm test ECG
    i = int(np.where((D["split"] == "test") & nsr & (D["y"].sum(1) == 1))[0][0])
    x = torch.from_numpy(D["X"][i].astype(np.float32).T[None])
    R = torch.from_numpy(D["R"][i][None]).long()
    torch.manual_seed(1)
    views = [("Original (lead I)", x), ("Frequency dropout", freq_dropout(x)), ("Crop resize", crop_resize(x)),
             ("Cycle mask", cycle_mask(x, R)), ("Channel mask (a masked lead)", None)]
    m = channel_mask(x, p=0.5)
    lead = int(torch.where(m[0].abs().sum(1) == 0)[0][0]) if (m[0].abs().sum(1) == 0).any() else 0
    fig, axes = plt.subplots(len(views), 1, figsize=(10, 7), sharex=True)
    t = np.arange(5000) / 500
    for ax, (n, v) in zip(axes, views):
        sig = (m[0, lead] if v is None else v[0, 0]).numpy()
        ax.plot(t, sig, color="purple" if v is not None and n != views[0][0] else "green", lw=0.8)
        ax.set_title(n, fontsize=9, loc="right")
        ax.set_yticks([])
    axes[-1].set_xlabel("time (s)")
    savefig(fig, "fig1e_augmentations")


def fig_pretrain():
    h = os.path.join(ROOT, "checkpoints", "pretrain", "moco_history.txt")
    if not os.path.exists(h):
        return
    a = np.loadtxt(h, ndmin=2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(a[:, 0], a[:, 1], label="contrastive $L_C$")
    ax[0].plot(a[:, 0], a[:, 2], label="divergence $L_D$")
    ax[0].set(xlabel="epoch", ylabel="loss")
    ax[0].legend()
    ax[1].plot(a[:, 0], a[:, 3])
    ax[1].set(xlabel="epoch", ylabel="top-1 key retrieval (72k queue)")
    savefig(fig, "fig_pretraining_loss")


def main():
    global D
    D = chapman()
    os.makedirs(FIG, exist_ok=True)
    md("# Results: reproduction of Lai et al., Nat. Commun. 14:3741 (2023) on public data", "",
       f"Chapman-Shaoxing/Ningbo: {len(D['classes'])} SNOMED-CT terms with ≥ 50 positives; split "
       f"{(D['split'] == 'train').sum()}/{(D['split'] == 'val').sum()}/{(D['split'] == 'test').sum()} (train/val/test). "
       "Pre-training: CODE-15% + Chapman training split. See README.md for every deviation from the paper.", "")
    for f in [table1_2, table3_fig2g, table4_fig2a, table5_fig2c, table6_fig2b, fig2d, table8_fig2e, fig2f,
              table9_cpsc, table10_cost, fig1e, fig_pretrain]:
        try:
            f()
        except Exception as e:  # a missing experiment must not stop the report
            print(f"{f.__name__} skipped: {type(e).__name__}: {e}")
    open(os.path.join(ROOT, "results", "RESULTS.md"), "w").write("\n".join(OUT) + "\n")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
