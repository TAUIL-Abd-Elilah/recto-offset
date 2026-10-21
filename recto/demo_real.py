"""Real meshes: run recto.apply on automatically grown PHerc. 1447 meshes (native 8.64 um, 116 keV) and describe what
it does. Descriptive only (amendment 3): CT features are not ground truth.

For each segment, one 40 x 40-node crop (800 x 800 voxels) where the mesh is complete is corrected with the tool.
Along the original normals at every 2-voxel node, the CT is sampled from -16 to +16 voxels around the surface before
and after; per node we record the signed distance (voxels, + = outward) to the nearest bright layer (local maximum)
and to the nearest dark-to-bright rise going outward. On Paris 4 verified patches these sit at about +1 and -2.
Writes results/demo_pherc1447.json and figures/demo_pherc1447_sections.png.
  python -m recto.demo_real --root C:/VesuviusTemp/recto --ckpt runs/r2/best.pt"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tifffile
from scipy import ndimage

from . import apply as tool
from .ct import Volume
from .external import B, SCANS, fetch_mesh, scan_shape
from .stacks import load_tifxyz, normals, upsample

SEGS = ['20250502180708-auto_grown_20250502160708188', '20250502182142-auto_grown_20250502161324419',
        '20250502183138-auto_grown_20250502162038685', '20250502184658-auto_grown_20250502163923577',
        '20250502205333-auto_grown_20250502181030065', '20250703025628-auto_grown_20250703025628283']
N = 40


def crop(src, dst, rng, zmax):
    """A fully valid N x N crop, at least 200 voxels inside the scan in z, nearest the mesh's median height."""
    zyx, valid = load_tifxyz(src)
    inside = valid & (zyx[..., 0] > 200) & (zyx[..., 0] < zmax - 200)
    full = ndimage.minimum_filter(inside.astype(np.uint8), N, mode='constant', origin=-(N // 2)) > 0
    c = np.argwhere(full[:inside.shape[0] - N + 1, :inside.shape[1] - N + 1])
    if not len(c):
        return False
    zmed = np.median(zyx[..., 0][valid])
    i0, j0 = (int(v) for v in c[np.argmin(np.abs(zyx[c[:, 0], c[:, 1], 0] - zmed))])
    assert valid[i0:i0 + N, j0:j0 + N].all()
    os.makedirs(dst, exist_ok=True)
    for k, ax in enumerate('zyx'):
        tifffile.imwrite(os.path.join(dst, f'{ax}.tif'), zyx[i0:i0 + N, j0:j0 + N, k].astype(np.float32))
    m = json.load(open(os.path.join(src, 'meta.json'))); m['uuid'] = os.path.basename(dst)
    json.dump(m, open(os.path.join(dst, 'meta.json'), 'w'))
    return True


def features(prof):
    """prof (33, ...) along the normal, k=-16..16 -> signed distance from k=0 to nearest bright peak and outward rise."""
    K = 16
    p = ndimage.gaussian_filter1d(prof, 1.0, axis=0)
    g = np.gradient(p, axis=0)
    k = np.arange(-K, K + 1)[:, None]
    pk = (p == ndimage.maximum_filter1d(p, 5, axis=0)) & (p > np.percentile(p, 50, axis=0))
    rs = (g == ndimage.maximum_filter1d(g, 5, axis=0)) & (g > 0)
    def nearest(mask):
        d = np.where(mask, k, 999)
        return np.take_along_axis(d, np.abs(d).argmin(0)[None], 0)[0].astype(float)
    return nearest(pk.reshape(33, -1)), nearest(rs.reshape(33, -1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--ckpt', default='runs/r2/best.pt')
    ap.add_argument('--out', default='results/demo_pherc1447.json')
    a = ap.parse_args(argv)
    rel, tag = SCANS['PHerc1447']
    url = f'{B}/PHerc1447/{rel}'
    cache = os.path.join(a.root, 'ext_ct', 'PHerc1447', '0')
    vol = Volume(cache, url=f'{url}/0', shape=scan_shape(url), workers=32)
    rng = np.random.default_rng(0)
    rows, figs = {}, []
    for seg in SEGS:
        try:
            src = fetch_mesh('PHerc1447', seg, tag, os.path.join(a.root, 'ext_meshes', 'PHerc1447'))
        except Exception as e:  # noqa: BLE001
            print('skip', seg, str(e)[:60]); continue
        dst = os.path.join(a.root, 'demo1447', seg[:14] + '_crop')
        if not crop(src, dst, rng, vol.shape[0]):
            continue
        out = dst + '_recto'
        tool.main([dst, out, '--volume', f'{url}/0', '--cache', cache, '--ckpt', a.ckpt, '--device', 'cpu'])
        z0, v0 = load_tifxyz(dst)
        f0, ok = upsample(z0, v0, 10.0)
        n = normals(f0, ok)
        off = ndimage.zoom(np.nan_to_num(tifffile.imread(os.path.join(out, 'offset.tif'))), 10.0, order=1)[:f0.shape[0], :f0.shape[1]]
        # orient as the tool did (sign of the stored offset is along the tool's outward normal): recompute outward
        from .external import Axis
        c = Axis(url, vol.shape, os.path.join(a.root, 'ext_ct', 'PHerc1447', 'axis'))(float(np.nanmedian(f0[..., 0])))
        radial = np.concatenate([np.zeros(f0.shape[:-1] + (1,)), f0[..., 1:] - np.asarray(c)], -1)
        if np.nanmedian(np.sum(radial * n, -1)[ok]) < 0:
            n = -n
        f0, n, ok, off = f0[::2, ::2], np.nan_to_num(n[::2, ::2]), ok[::2, ::2], off[::2, ::2]
        ks = np.arange(-16, 17, dtype=np.float32)
        before = vol.sample(f0[None] + ks[:, None, None, None] * n[None])
        after = vol.sample((f0 + off[..., None] * n)[None] + ks[:, None, None, None] * n[None])
        pb, rb = features(np.nan_to_num(before)); pa, ra = features(np.nan_to_num(after))
        m = ok.ravel()
        rows[seg] = {'nodes': int(m.sum()), 'offset_median_abs': float(np.median(np.abs(off[ok]))),
                     'offset_p90_abs': float(np.percentile(np.abs(off[ok]), 90)),
                     'peak_before_median': float(np.median(pb[m])), 'peak_after_median': float(np.median(pa[m])),
                     'rise_before_median': float(np.median(rb[m])), 'rise_after_median': float(np.median(ra[m])),
                     'within2_of_peak_before': float(np.mean(np.abs(pb[m]) <= 2)), 'within2_of_peak_after': float(np.mean(np.abs(pa[m]) <= 2))}
        print(seg[:24], json.dumps({k: round(v, 3) for k, v in rows[seg].items()}), flush=True)
        r = f0.shape[0] // 2
        figs.append((seg, np.nan_to_num(before[:, r, :]), off[r], ok[r]))
    os.makedirs('results', exist_ok=True)
    json.dump({'ckpt': a.ckpt, 'segments': rows}, open(a.out, 'w'), indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(figs), 1, figsize=(9, 1.9 * len(figs)))
    for ax, (seg, img, o, okr) in zip(np.atleast_1d(axs), figs):
        ax.imshow(img, cmap='gray', aspect='auto', extent=[0, img.shape[1] * 4, 16.5, -16.5])
        x = np.arange(img.shape[1]) * 4
        ax.plot(x, np.zeros_like(x), 'r-', lw=1, label='mesh as grown')
        ax.plot(x, np.where(okr, o, np.nan), 'c-', lw=1.2, label='after recto-offset')
        ax.set_ylabel('normal (vox)', fontsize=7); ax.set_title(seg[:40], fontsize=8); ax.tick_params(labelsize=6)
    np.atleast_1d(axs)[0].legend(fontsize=7, loc='upper right')
    fig.suptitle('PHerc. 1447, native 8.64 um: CT along the normals of automatically grown meshes', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); os.makedirs('figures', exist_ok=True)
    fig.savefig('figures/demo_pherc1447_sections.png', dpi=110)


if __name__ == '__main__':
    main()
