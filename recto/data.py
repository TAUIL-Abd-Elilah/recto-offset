"""Training pairs: a verified surface pushed off by a smooth random field, and the offset that brings it back.

Sean Johnson's recipe (Discord #general, 14 Oct 2026): take good surfaces, push parts of them off the sheet with a
smooth falloff, and train a model to predict the normal offset back. Here the push is a smooth random field f(i, j)
(voxels along the normal) over a crop of nodes; the model sees the CT layers around the pushed surface,
view[k, i, j] = stack[k + f(i, j), i, j] for k = -R..R (linear interpolation along the stack axis), and the target
is -f: the offset to add along the normal to land on the verified surface. Nothing is re-sampled from the CT.

Field: a constant shift c ~ U(-A, A) plus a Gaussian-smoothed noise field (correlation length L nodes) scaled to a
random peak amplitude; |f| is clipped to A. A fraction of examples is left unpushed (f = 0)."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import ndimage

SPLIT = {'val': [(6000, 7500)], 'test': [(10000, 11500)]}


def split_of(zc: float) -> str:
    for name, bands in SPLIT.items():
        if any(lo <= zc < hi for lo, hi in bands):
            return name
    return 'train'


def load_stacks(root: str, split: str, subdir: str = 'stacks') -> list[dict]:
    """Windows of one split. split 'all' keeps every window (used for other scans, which are never trained on)."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, subdir, '*.npz'))):
        d = np.load(f)
        meta = json.loads(str(d['meta']))
        if split != 'all' and split_of(meta['z_center']) != split:
            continue
        out.append({'stack': d['stack'], 'valid': d['valid'], 'meta': meta, 'file': os.path.basename(f)})
    return out


def field(shape, rng, A=8.0, p_zero=0.1):
    if rng.random() < p_zero:
        return np.zeros(shape, np.float32)
    L = rng.uniform(4, 24)
    g = ndimage.gaussian_filter(rng.standard_normal(shape), L, mode='reflect')
    g /= max(np.abs(g).max(), 1e-6)
    f = rng.uniform(-A, A) * rng.random() + g * rng.uniform(0, A)
    return np.clip(f, -A, A).astype(np.float32)


def view(stack: np.ndarray, f: np.ndarray, R: int) -> np.ndarray:
    """stack (2K+1, h, w) uint8 with the surface at index K; returns (2R+1, h, w) float32 around the pushed surface."""
    K = (stack.shape[0] - 1) // 2
    k = np.arange(-R, R + 1, dtype=np.float32)[:, None, None] + f[None] + K
    k0 = np.clip(np.floor(k).astype(np.int64), 0, stack.shape[0] - 2)
    t = np.clip(k - k0, 0, 1)
    ii, jj = np.meshgrid(np.arange(stack.shape[1]), np.arange(stack.shape[2]), indexing='ij')
    a = stack[k0, ii[None], jj[None]].astype(np.float32)
    b = stack[k0 + 1, ii[None], jj[None]].astype(np.float32)
    return (1 - t) * a + t * b


def normalise(v: np.ndarray) -> np.ndarray:
    m, s = v.mean(), v.std()
    return (v - m) / max(s, 1.0)


def sample(items, rng, crop=64, R=12, A=8.0, p_zero=0.1):
    """One (input (1, 2R+1, crop, crop), target (crop, crop), mask (crop, crop)) example."""
    it = items[rng.integers(len(items))]
    st, va = it['stack'], it['valid']
    h, w = va.shape
    i0, j0 = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
    st, va = st[:, i0:i0 + crop, j0:j0 + crop], va[i0:i0 + crop, j0:j0 + crop]
    if rng.random() < 0.5:                       # in-plane flips / transpose keep the normal direction
        st, va = st[:, ::-1], va[::-1]
    if rng.random() < 0.5:
        st, va = st[:, :, ::-1], va[:, ::-1]
    if rng.random() < 0.5:
        st, va = st.transpose(0, 2, 1), va.T
    f = field((crop, crop), rng, A, p_zero)
    x = normalise(view(np.ascontiguousarray(st), f, R))[None]
    return x.astype(np.float32), (-f).astype(np.float32), np.ascontiguousarray(va)


def fixed_set(items, n: int, seed: int, **kw):
    """A frozen evaluation set: the same crops and fields every time."""
    rng = np.random.default_rng(seed)
    xs, ys, ms = zip(*(sample(items, rng, **kw) for _ in range(n)))
    return np.stack(xs), np.stack(ys), np.stack(ms)
