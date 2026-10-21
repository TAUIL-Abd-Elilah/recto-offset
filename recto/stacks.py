"""CT stacks along the normals of verified patches: the raw material for recto-offset training pairs.

For each kept patch, a few windows of `--win` x `--win` nodes on a `--step`-voxel grid (bilinear upsampling of the
tifxyz grid, which has 20-voxel spacing) are cut where the mesh is complete. At every node the CT is sampled at
offsets k = -K..K voxels along the unit normal (trilinear, 9.6 um voxels), giving stack[k, i, j] (uint8) with the
verified surface at k = 0. Normals point away from the umbilicus (outward), so k > 0 is the outer side.
A training example is then a crop of the stack around a surface pushed off by a smooth random field (see data.py);
no CT is re-sampled for it.
Writes <out>/stacks/<patch>__w<n>.npz with stack (2K+1, win, win) uint8, valid (win, win) bool, zyx (win, win, 3)
float32, normal (win, win, 3) float32, and meta.

  python -m recto.stacks --root C:/VesuviusTemp/recto --win 96 --step 2 --K 24 --windows 2"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tifffile
from scipy import ndimage

from .ct import Volume

UMB_URL = 'https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/umbilicus.json'


def load_tifxyz(d: str):
    x, y, z = (tifffile.imread(os.path.join(d, f'{c}.tif')).astype(np.float64) for c in 'xyz')
    valid = (x > 0) & (y > 0) & (z > 0)
    return np.stack([z, y, x], -1), valid


def upsample(zyx: np.ndarray, valid: np.ndarray, factor: float):
    """Bilinear upsampling of a node grid; a fine node is valid only if its four coarse corners are valid."""
    H, W = valid.shape
    h, w = int((H - 1) * factor) + 1, int((W - 1) * factor) + 1
    gi, gj = np.meshgrid(np.arange(h) / factor, np.arange(w) / factor, indexing='ij')
    i0, j0 = np.clip(np.floor(gi).astype(int), 0, H - 2), np.clip(np.floor(gj).astype(int), 0, W - 2)
    ti, tj = (gi - i0)[..., None], (gj - j0)[..., None]
    q = lambda a, b: zyx[i0 + a, j0 + b]  # noqa: E731
    out = (1 - ti) * (1 - tj) * q(0, 0) + (1 - ti) * tj * q(0, 1) + ti * (1 - tj) * q(1, 0) + ti * tj * q(1, 1)
    ok = valid[i0, j0] & valid[i0 + 1, j0] & valid[i0, j0 + 1] & valid[i0 + 1, j0 + 1]
    return out, ok


def normals(zyx: np.ndarray, ok: np.ndarray) -> np.ndarray:
    du = np.gradient(zyx, axis=0)
    dv = np.gradient(zyx, axis=1)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    n[~ok] = np.nan
    return n


def umbilicus_xy(path: str):
    pts = json.load(open(path))
    pts = pts.get('control_points', pts) if isinstance(pts, dict) else pts
    a = np.array([[p['z'], p['y'], p['x']] if isinstance(p, dict) else [p[2], p[1], p[0]] for p in pts], float)
    a = a[np.argsort(a[:, 0])]
    return lambda z: np.stack([np.interp(z, a[:, 0], a[:, 1]), np.interp(z, a[:, 0], a[:, 2])], -1)


def orient_outward(zyx, n, umb):
    yx = umb(zyx[..., 0])
    radial = np.concatenate([np.zeros(zyx.shape[:-1] + (1,)), zyx[..., 1:] - yx], -1)
    s = np.sign(np.nanmedian(np.sum(radial * n, -1)))
    return n * (s if s != 0 else 1.0)


def windows(ok: np.ndarray, win: int, n: int, rng) -> list:
    """Up to n windows of win x win nodes that are at least 95 % valid, spread over the patch."""
    full = ndimage.uniform_filter(ok.astype(np.float32), win, mode='constant') >= 0.95
    cand = np.argwhere(full)
    out = []
    rng.shuffle(cand)
    for c in cand:
        if all(max(abs(c[0] - o[0]), abs(c[1] - o[1])) >= win for o in out):
            out.append(c)
        if len(out) == n:
            break
    return [(int(c[0]) - win // 2, int(c[1]) - win // 2) for c in out if c[0] >= win // 2 and c[1] >= win // 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--win', type=int, default=96)
    ap.add_argument('--step', type=float, default=2.0)
    ap.add_argument('--K', type=int, default=24)
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args(argv)
    sel = json.load(open(os.path.join(a.root, 'selection.json')))
    up = os.path.join(a.root, 'umbilicus.json')
    if not os.path.exists(up):
        import urllib.request
        urllib.request.urlretrieve(UMB_URL, up)
    umb = umbilicus_xy(up)
    vol = Volume(os.path.join(a.root, 'ct_l2'), workers=a.workers)
    os.makedirs(os.path.join(a.root, 'stacks'), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ks = np.arange(-a.K, a.K + 1, dtype=np.float32)
    patches = sel['patches'][:a.limit] if a.limit else sel['patches']
    for p in patches:
        d = os.path.join(a.root, 'meshes', p['name'])
        zyx, valid = load_tifxyz(d)
        meta = json.load(open(os.path.join(d, 'meta.json')))
        coarse_step = 1.0 / meta['scale'][0]                    # voxels between tifxyz nodes (20)
        fz, fok = upsample(zyx, valid, coarse_step / a.step)
        n = orient_outward(fz, normals(fz, fok), umb)
        for wi, (i0, j0) in enumerate(windows(fok, a.win, a.windows, rng)):
            out = os.path.join(a.root, 'stacks', f'{p["name"]}__w{wi}.npz')
            if os.path.exists(out):
                continue
            Z = fz[i0:i0 + a.win, j0:j0 + a.win]; N = n[i0:i0 + a.win, j0:j0 + a.win]
            ok = fok[i0:i0 + a.win, j0:j0 + a.win] & np.all(np.isfinite(N), -1)
            pts = Z[None] + ks[:, None, None, None] * np.nan_to_num(N)[None]
            vol.prefetch(vol.keys_for(pts[:, ok].reshape(-1, 3)))
            s = vol.sample(pts)
            stack = np.where(np.isfinite(s), np.clip(s, 0, 255), 0).astype(np.uint8)
            ok &= np.all(np.isfinite(s), 0)
            np.savez_compressed(out, stack=stack, valid=ok, zyx=Z.astype(np.float32), normal=N.astype(np.float32),
                                meta=json.dumps({'patch': p['name'], 'split': p['split'], 'z_center': p['z_center'],
                                                 'window': [i0, j0], 'step': a.step, 'K': a.K}))
            print(p['name'][:40], p['split'], 'w', wi, 'valid', round(float(ok.mean()), 3), 'chunks fetched', vol.fetched, flush=True)


if __name__ == '__main__':
    main()
