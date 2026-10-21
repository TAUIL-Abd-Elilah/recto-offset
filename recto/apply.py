"""Move a tifxyz surface along its normals onto its sheet with the trained recto-offset model.

  python -m recto.apply <in.tifxyz> <out.tifxyz> --volume <zarr level-0 URL or local folder> \
      [--ckpt weights/recto_offset_r2.pt] [--umbilicus umbilicus.json] [--max-shift 8] [--device cuda]

Works on any scan stored like the Vesuvius Challenge open-data volumes (zarr v2, uint8, uncompressed 128^3 chunks,
"/" separator): an https URL of a volume level, or a local copy of one. The mesh must be in that level's voxel frame.

What it does:
1. upsamples the mesh grid to 2-voxel node spacing (the model's training spacing) and computes normals, oriented away
   from the scroll axis (the umbilicus if given, otherwise the centre of the non-zero area of a coarse level of the
   masked scan);
2. samples the CT from -12 to +12 voxels along each normal, in overlapping 64 x 64-node tiles;
3. predicts the offset along the normal (voxels) and its confidence, blends the tiles, and moves every node of the
   original grid by the offset (clipped to --max-shift; nodes where the model is unsure keep their position).
Writes out.tifxyz (x/y/z.tif + meta.json) plus offset.tif and confidence.tif on the original grid.

Validated on PHerc. Paris 4 (9.6 um), PHerc. 0139 (9.362 um) and the ink of PHerc. 0841 (9.366 um); see the README
for where it is and is not reliable."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request

import numpy as np
import tifffile
import torch
from scipy import ndimage

from .ct import Volume
from .external import Axis
from .model import RectoNet
from .stacks import load_tifxyz, normals, umbilicus_xy, upsample


def open_volume(src: str, cache: str | None):
    if os.path.isdir(src):
        meta = json.load(open(os.path.join(src, '.zarray')))
        v = Volume(cache=src, url='file://' + src, shape=meta['shape'])
        v._fetch = lambda k: None                      # read local chunks in place, never download
        return v, meta['shape']
    meta = json.loads(urllib.request.urlopen(src.rstrip('/') + '/.zarray').read())
    assert meta.get('compressor') is None and meta['dtype'] in ('|u1', 'uint8'), 'only uncompressed uint8 levels'
    return Volume(cache or os.path.join(os.path.expanduser('~'), '.cache', 'recto-offset'), url=src.rstrip('/'),
                  shape=meta['shape']), meta['shape']


def predict_offsets(net, stack, valid, tile=64, overlap=16, device='cpu'):
    """stack (2R+1, H, W) float on the 2-voxel node grid -> offset (H, W), log-scale (H, W)."""
    from . import data
    L, H, W = stack.shape
    acc = np.zeros((H, W), np.float64); lb = np.zeros((H, W), np.float64); wsum = np.zeros((H, W), np.float64)
    w1 = np.hanning(tile + 2)[1:-1]; wt = np.outer(w1, w1) + 1e-3
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)) + ([H - tile] if H > tile else [])
    xs = list(range(0, max(W - tile, 0) + 1, step)) + ([W - tile] if W > tile else [])
    for y in sorted(set(max(0, v) for v in ys)):
        for x in sorted(set(max(0, v) for v in xs)):
            crop = np.zeros((L, tile, tile), np.float32); ok = np.zeros((tile, tile), bool)
            c = stack[:, y:y + tile, x:x + tile]; crop[:, :c.shape[1], :c.shape[2]] = c
            ok[:c.shape[1], :c.shape[2]] = valid[y:y + tile, x:x + tile]
            if ok.mean() < 0.05:
                continue
            fill = crop[:, ok].mean() if ok.any() else 0.0
            crop[:, ~ok] = fill
            with torch.no_grad():
                p, b = net(torch.from_numpy(data.normalise(crop)[None, None]).to(device))
            p, b = p[0].cpu().numpy(), b[0].cpu().numpy()
            hh, ww = c.shape[1], c.shape[2]
            acc[y:y + hh, x:x + ww] += (p * wt)[:hh, :ww]; lb[y:y + hh, x:x + ww] += (b * wt)[:hh, :ww]
            wsum[y:y + hh, x:x + ww] += wt[:hh, :ww]
    with np.errstate(invalid='ignore'):
        return acc / wsum, lb / wsum


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mesh'); ap.add_argument('out')
    ap.add_argument('--volume', required=True)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--ckpt', default=os.path.join(here, 'weights', 'recto_offset_r2.pt'))
    ap.add_argument('--umbilicus'); ap.add_argument('--cache')
    ap.add_argument('--max-shift', type=float, default=8.0)
    ap.add_argument('--min-confidence', type=float, default=0.0,
                    help='keep nodes in place where the predicted Laplace scale exceeds 1/min-confidence voxels')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location='cpu'); args = ck['args']
    net = RectoNet(layers=2 * args['R'] + 1, width=args['width']); net.load_state_dict(ck['model']); net.eval().to(a.device)
    R = args['R']
    vol, shape = open_volume(a.volume, a.cache)
    zyx, valid = load_tifxyz(a.mesh)
    meta = json.load(open(os.path.join(a.mesh, 'meta.json')))
    coarse = 1.0 / meta['scale'][0]
    f = coarse / 2.0
    fz, fok = upsample(zyx, valid, f)
    n = normals(fz, fok)
    # orient outward
    if a.umbilicus:
        umb = umbilicus_xy(a.umbilicus)
        centre = lambda z: umb(np.atleast_1d(z))[0]  # noqa: E731
    else:
        base = a.volume.rstrip('/').rsplit('/', 1)[0]
        if os.path.isdir(a.volume):
            if not os.path.isdir(os.path.join(base, '4')):
                raise SystemExit('local volume without level 4: pass --umbilicus')
            centre = Axis(base, shape, os.path.join(base, '4'))
            centre.v._fetch = lambda k: None
        else:
            centre = Axis(base, shape, a.cache or os.path.join(os.path.expanduser('~'), '.cache', 'recto-offset', 'axis'))
    zmid = float(np.nanmedian(fz[..., 0][fok]))
    c = centre(zmid)
    radial = np.concatenate([np.zeros(fz.shape[:-1] + (1,)), fz[..., 1:] - np.asarray(c)], -1)
    if np.nanmedian(np.sum(radial * n, -1)[fok]) < 0:
        n = -n
    ok = fok & np.all(np.isfinite(n), -1)
    ks = np.arange(-R, R + 1, dtype=np.float32)
    H, W = ok.shape
    stack = np.zeros((2 * R + 1, H, W), np.float32)
    for y in range(0, H, 128):                     # sample in row blocks to bound memory
        Z = fz[y:y + 128]; N = np.nan_to_num(n[y:y + 128]); m = ok[y:y + 128]
        pts = Z[None] + ks[:, None, None, None] * N[None]
        if m.any():
            vol.prefetch(vol.keys_for(pts[:, m].reshape(-1, 3)))
        s = vol.sample(pts)
        ok[y:y + 128] &= np.all(np.isfinite(s), 0)
        stack[:, y:y + 128] = np.nan_to_num(np.clip(s, 0, 255))
    off, logb = predict_offsets(net, stack, ok, device=a.device)
    off = ndimage.gaussian_filter(np.nan_to_num(off), 1.0)
    conf = np.exp(-np.nan_to_num(logb, nan=4.0))
    keep = ok & (conf >= a.min_confidence)
    off = np.where(keep, np.clip(off, -a.max_shift, a.max_shift), 0.0)
    # back to the original grid: sample the fine maps at the coarse nodes
    gi, gj = np.meshgrid(np.arange(zyx.shape[0]) * f, np.arange(zyx.shape[1]) * f, indexing='ij')
    gi, gj = np.clip(np.round(gi).astype(int), 0, H - 1), np.clip(np.round(gj).astype(int), 0, W - 1)
    o = off[gi, gj]; nn = np.nan_to_num(n[gi, gj]); cc = conf[gi, gj]
    new = zyx + o[..., None] * nn
    new[~valid] = -1
    os.makedirs(a.out, exist_ok=True)
    for i, ax in enumerate('zyx'):
        tifffile.imwrite(os.path.join(a.out, f'{ax}.tif'), new[..., i].astype(np.float32))
    tifffile.imwrite(os.path.join(a.out, 'offset.tif'), np.where(valid, o, np.nan).astype(np.float32))
    tifffile.imwrite(os.path.join(a.out, 'confidence.tif'), np.where(valid, cc, np.nan).astype(np.float32))
    meta = dict(meta)
    meta.setdefault('tags', {})
    meta['tags']['recto_offset'] = {'ckpt': os.path.basename(a.ckpt), 'step': ck.get('step'), 'max_shift': a.max_shift,
                                    'median_abs_offset': float(np.median(np.abs(o[valid]))) if valid.any() else None}
    meta['uuid'] = os.path.basename(os.path.normpath(a.out))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta['tags']['recto_offset']), flush=True)


if __name__ == '__main__':
    main()
