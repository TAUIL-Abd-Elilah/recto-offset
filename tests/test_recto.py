"""Checks on a synthetic stack: a bright sheet with a dark-to-bright rise at a known depth."""
import numpy as np
import torch

from recto import baselines, data
from recto.model import RectoNet, laplace_nll
from recto.stacks import normals, upsample, windows


def sheet_stack(K=24, h=64, w=64, centre=0.0):
    k = np.arange(-K, K + 1, dtype=np.float32)[:, None, None]
    prof = 60 + 120 * np.exp(-0.5 * ((k - centre - 1.5) / 1.5) ** 2)        # bright layer 1.5 vox outward
    return np.broadcast_to(prof, (2 * K + 1, h, w)).astype(np.uint8).copy()


def test_view_is_a_shift_along_the_normal():
    st = sheet_stack()
    f = np.full((64, 64), 3.0, np.float32)
    v = data.view(st, f, R=12)
    assert np.allclose(v[12 - 3], st[24, 0, 0], atol=1e-4)       # view layer k reads stack layer k + f
    assert np.allclose(v[12], st[27, 0, 0], atol=1e-4)
    g = data.view(st, np.full((64, 64), 2.5, np.float32), R=12)
    assert np.allclose(g[12], 0.5 * (st[26, 0, 0].astype(float) + st[27, 0, 0]), atol=0.51)


def test_target_returns_to_the_surface():
    rng = np.random.default_rng(0)
    items = [{'stack': sheet_stack(), 'valid': np.ones((64, 64), bool), 'meta': {}}]
    x, y, m = data.sample(items, rng, crop=64, R=12, A=6.0, p_zero=0.0)
    assert x.shape == (1, 25, 64, 64) and y.shape == (64, 64)
    assert np.abs(y).max() <= 6.0 + 1e-5 and m.all()


def test_field_is_smooth_and_bounded():
    rng = np.random.default_rng(1)
    for _ in range(20):
        f = data.field((64, 64), rng, A=8.0, p_zero=0.0)
        assert np.abs(f).max() <= 8.0 + 1e-6
        assert np.abs(np.diff(f, axis=0)).max() < 2.0              # no node-to-node jumps


def test_ridge_snap_finds_the_pushed_layer():
    st = sheet_stack()
    f = np.full((64, 64), -4.0, np.float32)                         # surface pushed 4 vox inward
    x = data.normalise(data.view(st, f, 12))[None, None]
    off = baselines.snap(x, 'ridge')                                  # offset to the bright layer
    assert np.median(off) == 4 + 2 or np.median(off) == 4 + 1        # layer is ~1.5 vox outward of the surface
    bias = 1.5
    assert abs(np.median(off - bias) - 4.0) <= 0.6


def test_model_shapes_and_loss():
    net = RectoNet(layers=25, width=16)
    x = torch.randn(2, 1, 25, 32, 32)
    p, lb = net(x)
    assert p.shape == (2, 32, 32) and lb.shape == (2, 32, 32)
    loss = laplace_nll(p, lb, torch.zeros(2, 32, 32), torch.ones(2, 32, 32, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)


def test_upsample_and_normals_of_a_plane():
    gi, gj = np.meshgrid(np.arange(6.0), np.arange(5.0), indexing='ij')
    zyx = np.stack([100 + 20 * gi, 50 + 20 * gj, np.full_like(gi, 300.0)], -1)    # plane x = 300
    fz, ok = upsample(zyx, np.ones(gi.shape, bool), 10.0)
    assert ok.all() and np.allclose(fz[10, 10], [120, 70, 300])
    n = normals(fz, ok)
    assert np.allclose(np.abs(n[5, 5]), [0, 0, 1], atol=1e-6)


def test_windows_stay_inside_valid_area():
    ok = np.zeros((300, 300), bool); ok[50:250, 40:260] = True
    ws = windows(ok, 96, 3, np.random.default_rng(0))
    assert ws
    for i0, j0 in ws:
        assert ok[i0:i0 + 96, j0:j0 + 96].mean() >= 0.95
