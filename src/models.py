"""Networks of Lai et al. (Nat. Commun. 2023), Fig. 3b and Methods.

  msdnn - multiscale ResNet18: an MSConv stem and 8 residual modules, each
          MSConv(s2) -> Dropout(0.2) -> MSConv(s1) -> SE, shortcut MaxPool(s2) -> Conv k1;
          MSConv = 4 parallel convolutions (k = 3, 5, 9, 17, N/4 channels each)
          -> concat -> BN -> ReLU; module k has 64 + 16k channels.
  dnn   - baseline: 34-layer network of Hannun et al. (Nat. Med. 2019) with kernel
          width 17 instead of 16 and a squeeze-and-excitation block in every module.
  ssdnn - ablation of the multiscale layer: msdnn with every MSConv replaced by a
          single k = 17 convolution with the same number of channels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean(-1))[..., None]


class MSConv(nn.Module):
    def __init__(self, cin, cout, stride=1, kernels=(3, 5, 9, 17)):
        super().__init__()
        g = cout // len(kernels)
        widths = [g] * (len(kernels) - 1) + [cout - g * (len(kernels) - 1)]
        self.convs = nn.ModuleList(nn.Conv1d(cin, w, k, stride, k // 2, bias=False) for k, w in zip(kernels, widths))
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.relu(self.bn(torch.cat([c(x) for c in self.convs], 1)), inplace=True)


class MSBlock(nn.Module):
    def __init__(self, cin, cout, kernels, drop=0.2):
        super().__init__()
        self.body = nn.Sequential(MSConv(cin, cout, 2, kernels), nn.Dropout(drop), MSConv(cout, cout, 1, kernels), SE(cout))
        self.short = nn.Sequential(nn.MaxPool1d(2, 2, ceil_mode=True), nn.Conv1d(cin, cout, 1, bias=False))

    def forward(self, x):
        y = self.body(x)
        s = self.short(x)
        return y + s[..., :y.shape[-1]]


class MSDNN(nn.Module):
    def __init__(self, n_classes, in_ch=12, kernels=(3, 5, 9, 17), blocks=8):
        super().__init__()
        ch = [64 + 16 * k for k in range(blocks)]
        self.stem = MSConv(in_ch, 64, 1, kernels)
        self.blocks = nn.Sequential(*[MSBlock(ch[max(i - 1, 0)], ch[i], kernels) for i in range(blocks)])
        self.dim = ch[-1]
        self.fc = nn.Linear(self.dim, n_classes)

    def features(self, x):  # [B, C, L] feature map (used for CAM)
        return self.blocks(self.stem(x))

    def embed(self, x):
        return self.features(x).mean(-1)

    def forward(self, x):
        return self.fc(self.embed(x))


class HannunBlock(nn.Module):
    """Pre-activation residual block: (BN-ReLU-Dropout-Conv) x2 + SE, optional subsampling."""

    def __init__(self, cin, cout, stride, first=False, k=17, drop=0.2):
        super().__init__()
        pre = [] if first else [nn.BatchNorm1d(cin), nn.ReLU(inplace=True), nn.Dropout(drop)]
        self.body = nn.Sequential(*pre, nn.Conv1d(cin, cout, k, stride, k // 2, bias=False),
                                  nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.Dropout(drop),
                                  nn.Conv1d(cout, cout, k, 1, k // 2, bias=False), SE(cout))
        self.pool = nn.MaxPool1d(stride, stride, ceil_mode=True) if stride > 1 else nn.Identity()
        self.proj = nn.Conv1d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.body(x)
        return y + self.proj(self.pool(x))[..., :y.shape[-1]]


class DNN(nn.Module):
    """Hannun et al.: conv stem + 16 residual blocks (33 conv layers + dense = 34 layers);
    32 * 2^k filters, k incremented every 4th block; every alternate block subsamples by 2."""

    def __init__(self, n_classes, in_ch=12, k=17):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, 32, k, 1, k // 2, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True))
        layers, c = [], 32
        for i in range(16):
            co = 32 * 2 ** (i // 4)
            layers.append(HannunBlock(c, co, 2 if i % 2 == 1 else 1, first=(i == 0), k=k))
            c = co
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.dim = c
        self.fc = nn.Linear(c, n_classes)

    def features(self, x):
        return self.head(self.blocks(self.stem(x)))

    def embed(self, x):
        return self.features(x).mean(-1)

    def forward(self, x):
        return self.fc(self.embed(x))


def build_model(name, n_classes, in_ch=12):
    if name == "msdnn":
        return MSDNN(n_classes, in_ch)
    if name == "ssdnn":
        return MSDNN(n_classes, in_ch, kernels=(17,))
    if name == "dnn":
        return DNN(n_classes, in_ch)
    raise ValueError(name)


if __name__ == "__main__":
    for m in ["msdnn", "ssdnn", "dnn"]:
        net = build_model(m, 60)
        y = net(torch.randn(2, 12, 5000))
        print(m, tuple(y.shape), f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params")
