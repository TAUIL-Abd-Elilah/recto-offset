"""A measured chance level for the bright-layer measure of recto.demo_meshes on PHerc. 0800 (corrects v1.1).

v1.1 compared the meshes with "a random position 38.5 %". That figure is 5/13, the share expected if the brightest
smoothed layer were spread evenly over the 13 positions of the +-6-voxel window; it was not measured, and the layer is
not spread evenly (a smoothed profile often peaks at the window's ends). Here the chance level is measured on the same
12 crops with the same geometry and code as recto.demo_meshes: every node is moved by its own random shift along its
normal, uniform in +-6 voxels or in +-4.5 voxels (about one sheet pitch on this scan), 5 seeds each, and the measure
is taken there. The published meshes and the stored r3 offsets are measured again with the same code, and so is what
recto.apply --min-snr Z would have left of the r3 offsets (the stored offset and confidence maps gated at the mesh
nodes; descriptive, for the README's advice on spiral fits).
Cache only: it never downloads, and stops if a chunk it needs is not in the local cache.

  python -m recto.demo_null --root C:/VesuviusTemp/recto --out results/demo_pherc0800_null.json"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tifffile
from scipy import ndimage

from .ct import Volume
from .demo_meshes import SHAPE, VOL, brightest
from .external import Axis
from .stacks import load_tifxyz, normals, upsample

NULLS = {'uniform6': 6.0, 'uniform4.5': 4.5}
SEEDS = 5
GATES = (1.0, 1.5, 2.0, 3.0)


def cache_only(v: Volume) -> None:
    def nofetch(k):
        p = v._path(k)
        if not (os.path.exists(p) or os.path.exists(p + '.empty')):
            raise SystemExit(f'chunk {k} is not in the local cache {v.cache}; this script never downloads')
    v._fetch = nofetch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='C:/VesuviusTemp/recto')
    ap.add_argument('--demo', default='results/demo_pherc0800.json', help='the published demo: its 12 crops only')
    ap.add_argument('--out', default='results/demo_pherc0800_null.json')
    a = ap.parse_args(argv)
    demo = json.load(open(a.demo))
    vol = Volume(os.path.join(a.root, 'ext_ct', 'PHerc0800', '0'), url=f'{VOL}/0', shape=SHAPE)
    axis = Axis(VOL, SHAPE, os.path.join(a.root, 'ext_ct', 'PHerc0800', 'axis'))
    cache_only(vol); cache_only(axis.v)
    ks = np.arange(-6, 7, dtype=np.float32)
    rows = {}
    for ci, name in enumerate(sorted(demo['meshes'])):
        dst = os.path.join(a.root, 'demo0800', name + '_crop')
        out = dst + '_recto'
        z0, v0 = load_tifxyz(dst)                       # same geometry as recto.demo_meshes
        f0, ok = upsample(z0, v0, 10.0)
        n = normals(f0, ok)
        c = axis(float(np.median(f0[..., 0][ok])))
        radial = np.concatenate([np.zeros(f0.shape[:-1] + (1,)), f0[..., 1:] - np.asarray(c)], -1)
        if np.nanmedian(np.sum(radial * n, -1)[ok]) < 0:
            n = -n
        oc = np.nan_to_num(tifffile.imread(os.path.join(out, 'offset.tif')))
        cc = np.nan_to_num(tifffile.imread(os.path.join(out, 'confidence.tif')))
        H0, W0 = f0.shape[:2]
        up = lambda o: ndimage.zoom(o, 10.0, order=1)[:H0, :W0][::2, ::2]  # noqa: E731
        off = up(oc)
        # recto.apply --min-snr Z on these crops: the stored maps gated at the mesh nodes (exact up to the 8-voxel clip)
        gated = {z: np.where(np.abs(oc) >= z / np.maximum(cc, 1e-12), oc, 0.0) for z in GATES}
        f0, n, ok = f0[::2, ::2], np.nan_to_num(n[::2, ::2]), ok[::2, ::2]
        at = lambda s: vol.sample((f0 + s[..., None] * n)[None] + ks[:, None, None, None] * n[None])  # noqa: E731
        b, aft = at(np.zeros(off.shape)), at(off)
        good = ok & np.all(np.isfinite(b), 0) & np.all(np.isfinite(aft), 0)
        share = lambda s, g: float(np.mean(np.abs(brightest(s[:, g])) <= 2))  # noqa: E731
        r = {'nodes': int(good.sum()), 'published': share(b, good), 'r3': share(aft, good)}
        for z, g in gated.items():
            s = at(up(g))
            r[f'r3_min_snr{z}'] = share(s, good & np.all(np.isfinite(s), 0))
            r[f'r3_min_snr{z}_moved'] = float(np.mean(g != 0))
        for k, w in NULLS.items():
            r[k] = []
            for seed in range(SEEDS):
                u = np.random.default_rng([seed, ci]).uniform(-w, w, off.shape)
                s = at(u)
                g = good & np.all(np.isfinite(s), 0)
                r[k].append(share(s, g))
        rows[name] = r
        print(name, json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, (float, list)) else v) for k, v in r.items()}),
              flush=True)
    wts = np.array([r['nodes'] for r in rows.values()], float)
    pool = lambda vals: float(np.average(vals, weights=wts))  # noqa: E731
    pooled = {k: pool([r[k] for r in rows.values()]) for k in rows[next(iter(rows))] if not isinstance(
        rows[next(iter(rows))][k], list) and k != 'nodes'}
    for k in NULLS:
        per_seed = [pool([r[k][s] for r in rows.values()]) for s in range(SEEDS)]
        pooled[k] = {'mean': float(np.mean(per_seed)), 'min': float(min(per_seed)), 'max': float(max(per_seed)),
                     'per_seed': per_seed}
    chance = pooled['uniform6']['mean']
    pooled['published_minus_chance_points'] = 100 * (pooled['published'] - chance)
    pooled['r3_minus_chance_points'] = 100 * (pooled['r3'] - chance)
    res = {'source': demo['source'], 'ckpt': demo['ckpt'], 'crops': len(rows),
           'measure': 'share of nodes whose brightest smoothed layer within +-6 voxels lies within 2 voxels '
                      '(recto.demo_meshes.brightest)',
           'chance': 'each node moved by its own uniform random shift along its normal (+-6 or +-4.5 voxels), '
                     f'{SEEDS} seeds; pooled over crops weighted by nodes, as in demo_pherc0800.json',
           'v1_1_random_figure': {'value': 5 / 13, 'what_it_is': '5 of the 13 window positions; not measured'},
           'published_demo_pooled': demo['pooled'], 'pooled': pooled, 'per_crop': rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('pooled', json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'per_seed'})
                                for k, v in pooled.items()}))


if __name__ == '__main__':
    main()
