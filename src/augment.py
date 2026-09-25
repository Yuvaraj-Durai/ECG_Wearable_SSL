"""The four ECG augmentations of Lai et al. (2023), batched on the GPU.

x: [B, C, T] float tensor, R: [B, MAX_R] long tensor of R-peak indices (-1 = none).

  freq    - frequency dropout: DCT -> zero a random subset of coefficients -> inverse DCT
  crop    - crop resize: random segment resampled back to the full length
  cycle   - cycle mask: zero the same R-relative segment in every heartbeat
  channel - channel mask: zero a few randomly chosen leads

The paper gives no hyper-parameters; the ranges below are ours (see README).
"""
import math

import torch

OPS = ("freq", "crop", "cycle", "channel")
ORDER = ("cycle", "channel", "freq", "crop")  # crop resize moves R peaks, so it runs last
FS = 500
_DCT = {}


def dct_matrix(T, device):
    """Orthonormal DCT-II matrix [T, T] (its transpose is the inverse)."""
    key = (T, str(device))
    if key not in _DCT:
        n = torch.arange(T, device=device, dtype=torch.float64)
        D = torch.cos(math.pi / T * (n[None, :] + 0.5) * n[:, None]) * math.sqrt(2.0 / T)
        D[0] /= math.sqrt(2.0)
        _DCT[key] = D.float()
    return _DCT[key]


def freq_dropout(x, rate=(0.02, 0.1)):
    B, C, T = x.shape
    D = dct_matrix(T, x.device)
    X = x.reshape(B * C, T) @ D.T
    r = torch.empty(B, 1, 1, device=x.device).uniform_(*rate)
    keep = (torch.rand(B, 1, T, device=x.device) >= r).expand(B, C, T).reshape(B * C, T).clone()
    keep[:, 0] = True  # keep the DC term
    return ((X * keep) @ D).reshape(B, C, T)


def crop_resize(x, scale=(0.85, 1.0)):
    """Random crop of scale*T samples, linearly resampled back to T."""
    B, C, T = x.shape
    L = torch.empty(B, 1, device=x.device).uniform_(*scale) * (T - 1)
    s = torch.rand(B, 1, device=x.device) * (T - 1 - L)
    pos = s + L * torch.linspace(0, 1, T, device=x.device)[None]  # [B, T]
    i0 = pos.floor().long().clamp(0, T - 2)
    w = (pos - i0).unsqueeze(1)
    g0 = torch.gather(x, 2, i0.unsqueeze(1).expand(B, C, T))
    g1 = torch.gather(x, 2, (i0 + 1).unsqueeze(1).expand(B, C, T))
    return g0 * (1 - w) + g1 * w


def cycle_mask(x, R, offset=(-0.25, 0.35), width=(0.04, 0.12)):
    """Zero [R + o, R + o + w) in every beat; o and w (seconds) are drawn per record."""
    B, C, T = x.shape
    o = (torch.empty(B, 1, device=x.device).uniform_(*offset) * FS).long()
    w = (torch.empty(B, 1, device=x.device).uniform_(*width) * FS).long().clamp(min=1)
    valid = R >= 0
    a = (R + o).clamp(0, T)
    b = (R + o + w).clamp(0, T)
    d = torch.zeros(B, T + 1, device=x.device)
    d.scatter_add_(1, torch.where(valid, a, T), valid.float())
    d.scatter_add_(1, torch.where(valid, b, T), -valid.float())
    inside = d[:, :T].cumsum(1) > 0
    return x.masked_fill(inside.unsqueeze(1), 0.0)


def channel_mask(x, p=0.25):
    B, C, T = x.shape
    m = torch.rand(B, C, 1, device=x.device) < p
    m[m.all(1, keepdim=True).expand_as(m)] = False  # never mask every lead
    return x.masked_fill(m, 0.0)


def augment(x, R, ops=OPS, p=0.5):
    """Apply each selected operation independently to a random subset (probability p) of the batch."""
    B = x.shape[0]
    for op in [o for o in ORDER if o in ops]:
        sel = torch.rand(B, device=x.device) < p
        if not sel.any():
            continue
        xs = x[sel]
        if op == "freq":
            xs = freq_dropout(xs)
        elif op == "crop":
            xs = crop_resize(xs)
        elif op == "cycle":
            xs = cycle_mask(xs, R[sel])
        elif op == "channel":
            xs = channel_mask(xs)
        x = x.clone()
        x[sel] = xs
    return x
