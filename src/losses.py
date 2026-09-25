"""Multi-label loss of Lai et al.: binary cross-entropy + pairwise ranking (LSEP, Li et al.
CVPR 2017), each class weighted by the inverse of its number of positive recordings."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(y_train):
    """Inverse positive counts, normalised to mean 1."""
    n = torch.as_tensor(y_train.sum(0)).float().clamp(min=1)
    w = 1.0 / n
    return w / w.mean()


def lsep(logits, y):
    """log(1 + sum_{u in pos, v in neg} exp(f_v - f_u)), averaged over samples with >= 1 positive and negative."""
    diff = logits.unsqueeze(1) - logits.unsqueeze(2)  # [B, u, v] = f_v - f_u
    mask = (y.unsqueeze(2) > 0.5) & (y.unsqueeze(1) < 0.5)
    diff = diff.masked_fill(~mask, float("-inf")).flatten(1)
    zero = torch.zeros(len(y), 1, device=y.device, dtype=diff.dtype)
    loss = torch.logsumexp(torch.cat([zero, diff], 1), 1)
    ok = mask.flatten(1).any(1)
    return loss[ok].mean() if ok.any() else loss.sum() * 0


class BCERank(nn.Module):
    def __init__(self, weights=None, rank=True):
        super().__init__()
        self.register_buffer("w", weights if weights is not None else torch.tensor([]))
        self.rank = rank

    def forward(self, logits, y):
        logits = logits.float()
        w = self.w if self.w.numel() else None
        bce = F.binary_cross_entropy_with_logits(logits, y, weight=w)
        return bce + lsep(logits, y) if self.rank else bce
