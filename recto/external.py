"""Stacks from curated segments of another scan, for testing the model off Paris 4 (no training on them).

Default: PHerc. 0139, native 9.362 um scan 20250728140407 (113 keV), the resolution and energy class of the First
Letters scrolls. Its published segments (w023-w059, used for the team's ink labels) carry a mesh on exactly that
volume (`mesh/<id>-on-20250728140407-9.362um.tifxyz`). The scroll axis is not published for it, so normals are
oriented outward from the centre of the non-zero (masked) area of a coarse level slice at the window's height; each
window's mean CT profile is kept so the orientation can be checked (bright sheet just outward of the surface, as on
Paris 4).

  python -m recto.external --root C:/VesuviusTemp/recto --segments w030 w035 w045 ... --windows 2
Writes <root>/ext_stacks/<scroll>/<segment>__w<n>.npz in the same format as recto.stacks."""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request

import numpy as np

from .ct import Volume
from .stacks import load_tifxyz, normals, upsample, windows

B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
# scroll -> (volume path under the scroll, mesh tag used in mesh/<id>-on-<tag>.tifxyz)
SCANS = {'PHerc0139': ('volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr', '20250728140407-9.362um'),
         'PHerc0814': ('volumes/20250804134230-9.362um-1.2m-113keV-masked.zarr', '20250804134230-9.362um'),
         'PHerc0500P2': ('volumes/20250820143440-9.362um-1.2m-113keV-masked.zarr', '20250820143440-9.362um'),
         'PHerc0009B': ('volumes/20250521125136-8.640um-1.2m-116keV-masked.zarr', '20250521125136-8.64um'),
         'PHerc1447': ('volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr', '20250521151220-8.64um')}


def scan_shape(url):
    return tuple(json.loads(urllib.request.urlopen(f'{url}/0/.zarray').read())['shape'])


def segments(scroll: str) -> list[str]:
    xml = urllib.request.urlopen(f'{B}/?list-type=2&prefix={scroll}/segments/&delimiter=/').read().decode()
    return re.findall(rf'<Prefix>{scroll}/segments/([^<]*)/</Prefix>', xml)


def fetch_mesh(scroll, seg, tag, out):
    d = os.path.join(out, seg)
    os.makedirs(d, exist_ok=True)
    stem = seg.split('-')[0]
    for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json'):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f'{B}/{scroll}/segments/{seg}/mesh/{stem}-on-{tag}.tifxyz/{f}', p)
    return d


class Axis:
    """Scroll centre per height from the masked scan's coarse level (centre of the non-zero area)."""
    def __init__(self, vol_url, shape, cache, level=4):
        f = 2 ** level
        self.f = f
        self.v = Volume(cache, url=f'{vol_url}/{level}', shape=tuple(-(-s // f) for s in shape))
        self.memo = {}

    def __call__(self, z: float):
        zi = int(z / self.f)
        if zi not in self.memo:
            H, W = self.v.shape[1:]
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            sl = self.v.sample(np.stack([np.full(yy.shape, zi, float), yy, xx], -1).astype(float))
            m = np.nan_to_num(sl) > 0
            self.memo[zi] = (np.array([yy[m].mean(), xx[m].mean()]) * self.f) if m.any() else None
        return self.memo[zi]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--scroll', default='PHerc0139')
    ap.add_argument('--segments', nargs='*', default=[], help='winding tags (w030) or full segment ids')
    ap.add_argument('--all-segments', action='store_true', help='every segment of the scroll with a mesh on the scan')
    ap.add_argument('--win', type=int, default=96); ap.add_argument('--step', type=float, default=2.0)
    ap.add_argument('--K', type=int, default=24); ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--cleanup', action='store_true', help='delete the CT chunk cache of this scan when done')
    a = ap.parse_args(argv)
    rel, tag = SCANS[a.scroll]
    url = f'{B}/{a.scroll}/{rel}'
    shape = scan_shape(url)
    vol = Volume(os.path.join(a.root, 'ext_ct', a.scroll, '0'), url=f'{url}/0', shape=shape, workers=a.workers)
    axis = Axis(url, shape, os.path.join(a.root, 'ext_ct', a.scroll, 'axis'))
    allsegs = segments(a.scroll)
    out = os.path.join(a.root, 'ext_stacks', a.scroll)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ks = np.arange(-a.K, a.K + 1, dtype=np.float32)
    wanted = allsegs if a.all_segments else a.segments
    for want in wanted:
        cands = [s for s in allsegs if s == want or f'-{want}_' in s]
        if not cands:
            print('no segment', want, flush=True); continue
        seg = sorted(cands)[-1]
        try:
            d = fetch_mesh(a.scroll, seg, tag, os.path.join(a.root, 'ext_meshes', a.scroll))
        except Exception as e:  # noqa: BLE001
            print('no mesh on', tag, seg, str(e)[:80], flush=True); continue
        zyx, valid = load_tifxyz(d)
        meta = json.load(open(os.path.join(d, 'meta.json')))
        f = (1.0 / meta['scale'][0]) / a.step                     # fine nodes per coarse node
        cw = int(np.ceil(a.win / f)) + 2                          # window size on the coarse grid, with a margin
        # choose windows on the coarse grid, then upsample only those crops (huge meshes stay cheap)
        for wi, (ci, cj) in enumerate(windows(valid, cw, a.windows, rng)):
            o = os.path.join(out, f'{seg}__w{wi}.npz')
            if os.path.exists(o):
                continue
            fz, fok = upsample(zyx[ci:ci + cw, cj:cj + cw], valid[ci:ci + cw, cj:cj + cw], f)
            n = normals(fz, fok)
            Z = fz[:a.win, :a.win]; N = n[:a.win, :a.win].copy()
            ok = fok[:a.win, :a.win] & np.all(np.isfinite(N), -1)
            i0, j0 = int(ci * f), int(cj * f)
            if Z.shape[:2] != (a.win, a.win) or ok.mean() < 0.9:
                continue
            c = axis(float(np.nanmedian(Z[..., 0][ok])))
            if c is None:
                continue
            radial = np.concatenate([np.zeros(Z.shape[:-1] + (1,)), Z[..., 1:] - c], -1)
            if np.nanmedian(np.sum(radial * N, -1)[ok]) < 0:
                N = -N
            pts = Z[None] + ks[:, None, None, None] * np.nan_to_num(N)[None]
            vol.prefetch(vol.keys_for(pts[:, ok].reshape(-1, 3)))
            s = vol.sample(pts)
            stack = np.where(np.isfinite(s), np.clip(s, 0, 255), 0).astype(np.uint8)
            ok &= np.all(np.isfinite(s), 0)
            prof = [float(stack[k][ok].mean()) for k in range(stack.shape[0])]
            np.savez_compressed(o, stack=stack, valid=ok, zyx=Z.astype(np.float32), normal=N.astype(np.float32),
                                meta=json.dumps({'patch': seg, 'scroll': a.scroll, 'split': 'external',
                                                 'z_center': float(np.nanmedian(Z[..., 0][ok])), 'window': [i0, j0],
                                                 'step': a.step, 'K': a.K, 'axis_yx': [float(c[0]), float(c[1])],
                                                 'mean_profile': prof}))
            K = a.K
            print(seg[:34], 'w', wi, 'valid', round(float(ok.mean()), 3), 'peak k=%+d' % (int(np.argmax(prof)) - K),
                  'rise k=%+d' % (int(np.argmax(np.gradient(prof))) - K), 'chunks', vol.fetched, flush=True)
    if a.cleanup:
        import shutil
        shutil.rmtree(os.path.join(a.root, 'ext_ct', a.scroll), ignore_errors=True)


if __name__ == '__main__':
    main()
