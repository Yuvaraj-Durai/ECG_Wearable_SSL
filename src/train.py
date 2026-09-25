"""Supervised training / fine-tuning of the multi-label classifier (Lai et al. 2023, Methods).

SGD (momentum 0.9, weight decay 5e-4, batch 128), 100 epochs, one-cycle schedule:
random init  1e-3 -> 1e-2 (epoch 45) -> 1e-3 (epoch 90) -> 1e-6 (epoch 100);
pretrained   the latter half only: 1e-2 -> 1e-3 (epoch 90) -> 1e-6 (epoch 100).
Loss: class-weighted BCE + pairwise ranking. The checkpoint with the lowest validation
loss is kept (the paper used its test set for this; we use a separate validation fold).
All data stay on the GPU; augmentation is done on the GPU per batch.

  python -m src.train --model msdnn --aug all --init checkpoints/pretrain/moco.pt --tag msdnn_pw_aug_s0
  python -m src.train --dataset cpsc2018 --fold 3 --aug all --init ... --tag cpsc_f3
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .augment import OPS, augment
from .data import LEADS, T
from .datasets import chapman, cpsc2018
from .losses import BCERank, class_weights
from .metrics import f1_thresholds, summary
from .models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def lr_at(ep, pretrained, epochs=100):
    """Piecewise-linear one-cycle schedule, ep in [0, epochs)."""
    a, b = 0.45 * epochs, 0.9 * epochs
    if not pretrained and ep < a:
        return 1e-3 + (1e-2 - 1e-3) * ep / a
    if ep < b:
        s = 0 if pretrained else a
        return 1e-2 + (1e-3 - 1e-2) * (ep - s) / (b - s)
    return 1e-3 * (1e-6 / 1e-3) ** ((ep - b) / (epochs - b))


def load_init(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("encoder", sd)
    own = model.state_dict()
    sd = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape and not k.startswith("fc.")}
    missing = [k for k in own if k not in sd]
    model.load_state_dict(sd, strict=False)
    print(f"init from {path}: {len(sd)} tensors loaded, {len(missing)} left random ({missing[:4]}...)", flush=True)


@torch.no_grad()
def predict(model, X, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", torch.bfloat16):
            out.append(torch.sigmoid(model(X[i:i + bs].float()).float()))
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def predict_long(model, X, lengths, win=T, hop=T // 2):
    """Variable-length records: max over sliding 10 s windows (records < 10 s are zero-padded)."""
    model.eval()
    out = []
    for i in range(len(X)):
        n = max(int(lengths[i]), win)
        starts = list(range(0, n - win + 1, hop)) or [0]
        if starts[-1] != n - win:
            starts.append(n - win)
        w = torch.stack([X[i, :, s:s + win] for s in starts]).float()
        with torch.autocast("cuda", torch.bfloat16):
            out.append(torch.sigmoid(model(w).float()).max(0).values)
    return torch.stack(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chapman")
    ap.add_argument("--model", default="msdnn")
    ap.add_argument("--init", default=None)
    ap.add_argument("--aug", default="none", help="none | all | comma list of " + ",".join(OPS))
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of the training set")
    ap.add_argument("--leads", default="all")
    ap.add_argument("--fold", type=int, default=0, help="cpsc2018: CV fold used for validation")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    out_dir = os.path.join(ROOT, "results", "runs", a.dataset)
    ck_dir = os.path.join(ROOT, "checkpoints", a.dataset)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ck_dir, exist_ok=True)
    ops = () if a.aug == "none" else OPS if a.aug == "all" else tuple(a.aug.split(","))
    lead_idx = list(range(12)) if a.leads == "all" else [LEADS.index(l) for l in a.leads.split(",")]

    d = chapman() if a.dataset == "chapman" else cpsc2018()
    y = d["y"]
    if a.dataset == "chapman":
        tr = np.where(d["split"] == "train")[0]
        va = np.where(d["split"] == "val")[0]
        te = np.where(d["split"] == "test")[0]
    else:  # fold -1 = held-out test; the other folds: one validation, the rest training
        f = d["fold"]
        te = np.where(d["test"])[0]
        va = np.where(f == a.fold)[0]
        tr = np.where((f >= 0) & (f != a.fold))[0]
    if a.frac < 1:  # nested subsets: the first frac of one fixed permutation
        tr = np.sort(np.random.default_rng(1234).permutation(tr)[:int(round(a.frac * len(tr)))])

    def to_gpu(idx):
        return torch.from_numpy(np.ascontiguousarray(d["X"][np.sort(idx)][:, :, lead_idx].transpose(0, 2, 1))).to(dev)

    order = lambda idx: np.sort(idx)
    tr, va, te = order(tr), order(va), order(te)
    Xtr, Xva, Xte = to_gpu(tr), to_gpu(va), to_gpu(te)
    Rtr = torch.from_numpy(d["R"][tr]).long().to(dev)
    ytr = torch.from_numpy(y[tr]).to(dev)
    yva = torch.from_numpy(y[va]).to(dev)
    long_rec = a.dataset == "cpsc2018"
    Ltr = torch.from_numpy(d["length"][tr]).long().to(dev) if long_rec else None
    print(f"{a.tag}: train {len(tr)} val {len(va)} test {len(te)} classes {y.shape[1]} leads {len(lead_idx)} aug {ops}", flush=True)

    model = build_model(a.model, y.shape[1], len(lead_idx)).to(dev)
    if a.init:
        load_init(model, a.init)
    fwd = torch.compile(model)  # training steps only (fixed batch shape); evaluation stays eager
    crit = BCERank(class_weights(y[tr])).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    steps = len(tr) // a.bs
    best, best_ep, hist = 1e9, -1, []
    ck = os.path.join(ck_dir, f"{a.tag}.pt")
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        tot = 0.0
        for it in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(ep + it / steps, a.init is not None, a.epochs)
            idx = perm[it * a.bs:(it + 1) * a.bs]
            if long_rec:  # random 10 s window inside the valid part of each record
                L = Ltr[idx].clamp(min=T)
                s = (torch.rand(len(idx), device=dev) * (L - T + 1)).long()
                pos = s[:, None] + torch.arange(T, device=dev)[None]
                x = torch.gather(Xtr[idx], 2, pos[:, None, :].expand(-1, Xtr.shape[1], -1)).float()
                R = Rtr[idx] - s[:, None]
                R = torch.where((Rtr[idx] >= 0) & (R >= 0) & (R < T), R, -1)
            else:
                x, R = Xtr[idx].float(), Rtr[idx]
            if ops:
                x = augment(x, R, ops)
            with torch.autocast("cuda", torch.bfloat16):
                loss = crit(fwd(x), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() if it % 20 == 0 else 0
        # validation loss for checkpoint selection
        model.eval()
        with torch.no_grad():
            vl = 0.0
            if long_rec:
                pv = torch.from_numpy(predict_long(model, Xva, d["length"][va])).to(dev).clamp(1e-6, 1 - 1e-6)
                vl = float(crit(torch.logit(pv), yva))
            else:
                for i in range(0, len(va), 512):
                    with torch.autocast("cuda", torch.bfloat16):
                        vl += float(crit(model(Xva[i:i + 512].float()), yva[i:i + 512])) * len(yva[i:i + 512])
                vl /= len(va)
        hist.append(vl)
        if vl < best:
            best, best_ep = vl, ep
            torch.save(model.state_dict(), ck)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"ep {ep:3d} lr {opt.param_groups[0]['lr']:.2e} train {tot / max(1, steps // 20):.4f} "
                  f"val {vl:.4f} best {best:.4f}@{best_ep} {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(torch.load(ck, weights_only=True))
    if long_rec:
        pva, pte = predict_long(model, Xva, d["length"][va]), predict_long(model, Xte, d["length"][te])
    else:
        pva, pte = predict(model, Xva), predict(model, Xte)
    th = f1_thresholds(y[va], pva)
    res = dict(vars(a), best_epoch=best_ep, val_loss=best, minutes=(time.time() - t0) / 60,
               params=sum(p.numel() for p in model.parameters()),
               val=summary(y[va], pva, th), test=summary(y[te], pte, th), val_hist=hist)
    extra = {}
    noisy = os.path.join(ROOT, "data", "cache", "chapman_test_noisy_X.npy")
    if a.dataset == "chapman" and os.path.exists(noisy):
        Xn = np.load(noisy, mmap_mode="r")
        Xn = torch.from_numpy(np.ascontiguousarray(Xn[:, :, lead_idx].transpose(0, 2, 1))).to(dev)
        extra["p_test_noisy"] = predict(model, Xn)
        res["test_noisy"] = summary(y[te], extra["p_test_noisy"], th)
    np.savez_compressed(os.path.join(out_dir, f"{a.tag}.npz"), p_val=pva, p_test=pte, th=th,
                        idx_val=va, idx_test=te, **extra)
    json.dump(res, open(os.path.join(out_dir, f"{a.tag}.json"), "w"), indent=1)
    print(f"{a.tag} done in {res['minutes']:.1f} min: test AUROC {res['test']['auroc']:.4f} "
          f"AUPRC {res['test']['auprc']:.4f} F1 {res['test']['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
