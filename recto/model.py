"""A small network: CT layers around a surface (1, 2R+1, H, W) -> per-node offset along the normal (H, W), in voxels.

A 3D stem reads the layer profile at each node with a little in-plane context and folds the layer axis into
channels; a 2D U-Net then spreads evidence along the sheet so a node where the sheet is faint borrows from its
neighbours. Output: offset (voxels) and a log-scale for a Laplace likelihood, used as a confidence."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block2d(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class RectoNet(nn.Module):
    def __init__(self, layers: int = 25, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, (5, 3, 3), padding=(2, 1, 1)), nn.GELU(),
            nn.Conv3d(16, 16, (5, 3, 3), padding=(2, 1, 1)), nn.GELU(),
            nn.Conv3d(16, 8, (3, 1, 1), padding=(1, 0, 0)), nn.GELU())
        c = width
        self.e1 = block2d(8 * layers, c)
        self.e2 = block2d(c, 2 * c)
        self.e3 = block2d(2 * c, 4 * c)
        self.d2 = block2d(4 * c + 2 * c, 2 * c)
        self.d1 = block2d(2 * c + c, c)
        self.head = nn.Conv2d(c, 2, 1)

    def forward(self, x):
        b, _, L, H, W = x.shape
        s = self.stem(x).reshape(b, -1, H, W)
        e1 = self.e1(s)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1], 1))
        o = self.head(d1)
        return o[:, 0], o[:, 1].clamp(-4, 4)


def laplace_nll(pred, logb, target, mask):
    m = mask.float()
    nll = (pred - target).abs() * torch.exp(-logb) + logb
    return (nll * m).sum() / m.sum().clamp(min=1)
