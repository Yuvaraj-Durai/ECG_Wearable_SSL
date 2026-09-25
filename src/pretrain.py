"""Self-supervised pre-training: MoCo with a distributional divergence loss (Lai et al. 2023, Fig. 3a).

For an anchor ECG x, a key view T(x_k) goes through the momentum encoder (k+), a query view
T(x_q) and N extra query views T(x_q^i) go through the encoder (q, q_i). With a queue of K
negatives, P(q, k) is the (K+1)-way softmax of q against [k+, queue] (Eq. 1).
  L_C = -log P(q, k+)                                     (Eq. 2)
  L_D = sum_i KL(P(q, k) || P(q_i, k)), P(q, k) detached   (Eq. 3)
  L   = L_C + beta * L_D                                  (Eq. 4)
Paper settings: Adam lr 1e-3, momentum 0.9, feature dim 128, tau 0.07, queue 72,000,
batch 360, beta 0.15, N = 3. Pool: CODE-15% (unlabelled) + the Chapman training split.
Shuffled BN for the key encoder (sub-batches) avoids leaking batch statistics on one GPU.

  python -m src.pretrain --epochs 50
"""
import argparse
import copy
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import augment
from .datasets import chapman, load
from .models import build_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "checkpoints", "pretrain")


class Encoder(nn.Module):
    def __init__(self, arch="msdnn", dim=128):
        super().__init__()
        self.net = build_model(arch, 1)
        self.head = nn.Sequential(nn.Linear(self.net.dim, 512), nn.ReLU(inplace=True), nn.Linear(512, dim))

    def forward(self, x):
        return F.normalize(self.head(self.net.embed(x)).float(), dim=1)


class Pool:
    """Random batches from several memory-mapped float16 arrays, prefetched in a thread."""

    def __init__(self, parts, bs, steps, seed=0):
        self.parts, self.bs, self.steps = parts, bs, steps
        self.sizes = np.array([len(i) for _, _, i in parts])
        self.q = queue.Queue(maxsize=6)
        self.rng = np.random.default_rng(seed)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            src = self.rng.choice(len(self.parts), self.bs, p=self.sizes / self.sizes.sum())
            xs, rs = [], []
            for j, (X, R, idx) in enumerate(self.parts):
                n = int((src == j).sum())
                if n == 0:
                    continue
                sel = np.sort(self.rng.choice(idx, n, replace=False))
                xs.append(X[sel])
                rs.append(R[sel])
            x = torch.from_numpy(np.concatenate(xs)).pin_memory()
            r = torch.from_numpy(np.concatenate(rs)).long().pin_memory()
            self.q.put((x, r))

    def get(self, dev):
        x, r = self.q.get()
        return x.to(dev, non_blocking=True).transpose(1, 2).float(), r.to(dev, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=360)
    ap.add_argument("--K", type=int, default=72000)
    ap.add_argument("--m", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--p", type=float, default=0.8, help="probability of each augmentation per view")
    ap.add_argument("--arch", default="msdnn")
    ap.add_argument("--tag", default="moco")
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)

    c15X, c15R, _ = load("code15")
    ch = chapman()
    parts = [(c15X, c15R, np.arange(len(c15X))), (ch["X"], ch["R"], np.where(ch["split"] == "train")[0])]
    n_pool = sum(len(p[2]) for p in parts)
    steps = n_pool // a.bs
    print(f"pre-training pool: {n_pool} ECGs, {steps} steps/epoch", flush=True)
    data = Pool(parts, a.bs, steps)

    enc = Encoder(a.arch).to(dev)
    menc = copy.deepcopy(enc)
    for p in menc.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3, betas=(0.9, 0.999))
    fenc, fmenc = torch.compile(enc), torch.compile(menc)  # fixed batch shapes
    Q = F.normalize(torch.randn(a.K, 128, device=dev), dim=1)
    ptr, start = 0, 0
    ck = os.path.join(OUT, f"{a.tag}_last.pt")
    if os.path.exists(ck):  # resume
        s = torch.load(ck, map_location=dev, weights_only=False)
        enc.load_state_dict(s["enc"]), menc.load_state_dict(s["menc"]), opt.load_state_dict(s["opt"])
        Q, ptr, start = s["Q"], s["ptr"], s["epoch"] + 1
        print("resumed at epoch", start, flush=True)

    t0 = time.time()
    for ep in range(start, a.epochs):
        enc.train()
        menc.train()
        lc_sum = ld_sum = acc_sum = 0.0
        for it in range(steps):
            x, R = data.get(dev)
            B = len(x)
            views = [augment(x, R, p=a.p) for _ in range(a.N + 1)]  # query + N extra views
            xk = augment(x, R, p=a.p)
            with torch.no_grad():  # momentum update, then shuffled-BN key forward
                for pq, pk in zip(enc.parameters(), menc.parameters()):
                    pk.mul_(a.m).add_(pq.detach(), alpha=1 - a.m)
                perm = torch.randperm(B, device=dev)
                with torch.autocast("cuda", torch.bfloat16):
                    kp = torch.cat([fmenc(c) for c in xk[perm].chunk(4)])
                k = torch.empty_like(kp)
                k[perm] = kp
            with torch.autocast("cuda", torch.bfloat16):
                qs = fenc(torch.cat(views)).chunk(a.N + 1)
            logits = [torch.cat([(q * k).sum(1, keepdim=True), q @ Q.T], 1) / a.tau for q in qs]
            lc = F.cross_entropy(logits[0], torch.zeros(B, dtype=torch.long, device=dev))
            target = F.softmax(logits[0].detach(), 1)
            ld = sum(F.kl_div(F.log_softmax(l, 1), target, reduction="batchmean") for l in logits[1:])
            loss = lc + a.beta * ld
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():  # enqueue keys
                n = min(B, a.K - ptr)
                Q[ptr:ptr + n] = k[:n]
                Q[:B - n] = k[n:] if B > n else Q[:0]
                ptr = (ptr + B) % a.K
            if it % 25 == 0:
                lc_sum += lc.item()
                ld_sum += ld.item()
                acc_sum += (logits[0].argmax(1) == 0).float().mean().item()
            if it % 250 == 0:
                print(f"ep {ep} it {it}/{steps} Lc {lc.item():.3f} Ld {ld.item():.3f} "
                      f"top1 {(logits[0].argmax(1) == 0).float().mean().item():.3f} {time.time() - t0:.0f}s", flush=True)
        nlog = len(range(0, steps, 25))
        print(f"== epoch {ep} Lc {lc_sum / nlog:.4f} Ld {ld_sum / nlog:.4f} top1 {acc_sum / nlog:.3f} "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        torch.save(dict(enc=enc.state_dict(), menc=menc.state_dict(), opt=opt.state_dict(), Q=Q, ptr=ptr, epoch=ep), ck + ".tmp")
        os.replace(ck + ".tmp", ck)
        # the downstream initialisation: backbone weights of the query encoder
        torch.save({k[4:]: v for k, v in enc.state_dict().items() if k.startswith("net.")}, os.path.join(OUT, f"{a.tag}.pt"))
        with open(os.path.join(OUT, f"{a.tag}_history.txt"), "a") as f:
            f.write(f"{ep} {lc_sum / nlog:.5f} {ld_sum / nlog:.5f} {acc_sum / nlog:.4f}\n")
    open(os.path.join(OUT, f"{a.tag}.DONE"), "w").write("done\n")


if __name__ == "__main__":
    main()
