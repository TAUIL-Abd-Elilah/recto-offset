"""Real meshes on a First Letters scroll: pscamillo's spiral-fit windings of PHerc. 0800 (native 8.64 um, 116 keV;
github.com/pscamillo/vesuvius-eligible-meshes) corrected with recto.apply. Descriptive (amendment 3): CT features are
not ground truth.

For each mesh, a fully valid 40 x 40-node crop (800 x 800 voxels, nearest the mesh's median height) is corrected.
At every 2-voxel node, the CT is sampled from -6 to +6 voxels along the outward normal around the surface before and
after correction, smoothed by 1 voxel, and the position of the brightest layer within that half-pitch window is taken
(the per-surface check rodriguescarson ran on eligible-scroll meshes, restricted to +-6 voxels so the neighbouring
sheet does not count). On the Paris 4 verified patches the brightest layer is 0 to +2 voxels from the surface.
  python -m recto.demo_meshes --meshes <repo>/meshes/PHerc0800 --n 12 --ckpt runs/r2/best.pt"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import tifffile
from scipy import ndimage

from . import apply as tool
from .ct import Volume
from .demo_real import N, crop
from .external import Axis
from .stacks import load_tifxyz, normals, upsample

VOL = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/volumes/'
       '20250521135224-8.640um-1.2m-116keV-masked.zarr')
SHAPE = (24298, 9867, 9867)


def brightest(prof):
    """prof (13, n) for k = -6..6 -> position of the brightest smoothed layer (voxels from the surface)."""
    p = ndimage.gaussian_filter1d(prof, 1.0, axis=0)
    return np.argmax(p, axis=0).astype(float) - 6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--meshes', required=True); ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--root', default='C:/VesuviusTemp/recto'); ap.add_argument('--ckpt', default='runs/r2/best.pt')
    ap.add_argument('--out', default='results/demo_pherc0800.json')
    a = ap.parse_args(argv)
    cache = os.path.join(a.root, 'ext_ct', 'PHerc0800', '0')
    vol = Volume(cache, url=f'{VOL}/0', shape=SHAPE, workers=32)
    axis = Axis(VOL, SHAPE, os.path.join(a.root, 'ext_ct', 'PHerc0800', 'axis'))
    dirs = sorted(glob.glob(os.path.join(a.meshes, '*')))
    pick = [dirs[i] for i in np.linspace(0, len(dirs) - 1, a.n).astype(int)]     # spread over the windows, fixed rule
    rows, figs = {}, []
    ks = np.arange(-6, 7, dtype=np.float32); kfig = np.arange(-16, 17, dtype=np.float32)
    for src in pick:
        name = os.path.basename(src)
        dst = os.path.join(a.root, 'demo0800', name + '_crop')
        if not crop(src, dst, None, SHAPE[0]):
            print('no full crop', name); continue
        out = dst + '_recto'
        tool.main([dst, out, '--volume', f'{VOL}/0', '--cache', cache, '--ckpt', a.ckpt, '--device', 'cpu',
                   '--axis-cache', os.path.join(a.root, 'ext_ct', 'PHerc0800', 'axis')])
        z0, v0 = load_tifxyz(dst)
        f0, ok = upsample(z0, v0, 10.0)
        n = normals(f0, ok)
        c = axis(float(np.median(f0[..., 0][ok])))
        radial = np.concatenate([np.zeros(f0.shape[:-1] + (1,)), f0[..., 1:] - np.asarray(c)], -1)
        if np.nanmedian(np.sum(radial * n, -1)[ok]) < 0:
            n = -n
        off = ndimage.zoom(np.nan_to_num(tifffile.imread(os.path.join(out, 'offset.tif'))), 10.0, order=1)[:f0.shape[0], :f0.shape[1]]
        f0, n, ok, off = f0[::2, ::2], np.nan_to_num(n[::2, ::2]), ok[::2, ::2], off[::2, ::2]
        b = vol.sample(f0[None] + ks[:, None, None, None] * n[None])
        f1 = f0 + off[..., None] * n
        aft = vol.sample(f1[None] + ks[:, None, None, None] * n[None])
        good = ok & np.all(np.isfinite(b), 0) & np.all(np.isfinite(aft), 0)
        db, da = brightest(b[:, good]), brightest(aft[:, good])
        rows[name] = {'nodes': int(good.sum()), 'offset_median_abs': float(np.median(np.abs(off[good]))),
                      'offset_p90_abs': float(np.percentile(np.abs(off[good]), 90)),
                      'brightest_before_median': float(np.median(db)), 'brightest_after_median': float(np.median(da)),
                      'within2_of_bright_before': float(np.mean(np.abs(db) <= 2)),
                      'within2_of_bright_after': float(np.mean(np.abs(da) <= 2))}
        print(name, json.dumps({k: round(v, 3) for k, v in rows[name].items()}), flush=True)
        r = f0.shape[0] // 2
        img = vol.sample(f0[r][None] + kfig[:, None, None] * n[r][None])
        figs.append((name, np.nan_to_num(img), off[r], ok[r]))
    tot = {k: float(np.average([v[k] for v in rows.values()], weights=[v['nodes'] for v in rows.values()]))
           for k in ('within2_of_bright_before', 'within2_of_bright_after', 'offset_median_abs')}
    os.makedirs('results', exist_ok=True)
    json.dump({'ckpt': a.ckpt, 'source': 'github.com/pscamillo/vesuvius-eligible-meshes (PHerc0800)', 'pooled': tot,
               'meshes': rows}, open(a.out, 'w'), indent=1)
    print('pooled', json.dumps(tot))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = figs[:4]
    fig, axs = plt.subplots(len(show), 1, figsize=(9, 1.9 * len(show)))
    for ax, (name, img, o, okr) in zip(np.atleast_1d(axs), show):
        ax.imshow(img, cmap='gray', aspect='auto', extent=[0, img.shape[1] * 4, 16.5, -16.5])
        x = np.arange(img.shape[1]) * 4
        ax.plot(x, np.zeros_like(x), 'r-', lw=1, label='published mesh')
        ax.plot(x, np.where(okr, o, np.nan), 'c-', lw=1.2, label='after recto-offset')
        ax.set_ylabel('normal (vox)', fontsize=7); ax.set_title(f'PHerc0800 {name}', fontsize=8); ax.tick_params(labelsize=6)
    np.atleast_1d(axs)[0].legend(fontsize=7, loc='upper right')
    fig.suptitle('PHerc. 0800 (8.64 um, 116 keV): CT along the normals of community spiral-fit meshes (outward = down)', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/demo_pherc0800_sections.png', dpi=110)


if __name__ == '__main__':
    main()
