"""Move a tifxyz surface along its normals onto its sheet with the trained recto-offset model.

  python -m recto.apply <in.tifxyz> <out.tifxyz> --volume <zarr level-0 URL or local folder> \
      [--ckpt weights/recto_offset_r2.pt] [--umbilicus umbilicus.json] [--max-shift 8] [--device cuda] \
      [--min-snr 0] [--iterations 1] [--cache <dir>] [--axis-cache <dir>]

Works on any scan stored like the Vesuvius Challenge open-data volumes (zarr v2, uint8, uncompressed 128^3 chunks,
"/" separator): an https URL of a volume level, or a local copy of one. The mesh must be in that level's voxel frame.

What it does:
1. upsamples the mesh grid to 2-voxel node spacing (the model's training spacing) and computes normals, oriented away
   from the scroll axis (the umbilicus if given, otherwise the centre of the non-zero area of a coarse level of the
   masked scan);
2. samples the CT from -12 to +12 voxels along each normal, in overlapping 64 x 64-node tiles;
3. predicts the offset along the normal (voxels) and its confidence, blends the tiles, and moves every node of the
   original grid by the offset (clipped to --max-shift; nodes where the model is unsure keep their position).
   --min-snr Z keeps a node in place unless |offset| >= Z * b, where b is the predicted Laplace scale (voxels;
   b = 1 / confidence); 0 (default) moves every node. --iterations N re-samples the CT at the corrected surface
   (along the same normals) and adds the next prediction, N passes in all; the gate applies to each pass.
Writes out.tifxyz (x/y/z.tif + meta.json) plus offset.tif and confidence.tif (of the last pass) on the original grid,
and gate.tif (number of passes in which the node moved; 0 = kept in place) when --min-snr is set.
CT chunks are cached under --cache (default: a folder per volume in ~/.cache/recto-offset); the coarse level read for
the scroll axis goes to its own folder (--axis-cache, default <cache>/axis), never among the level-0 chunks.

Validated on PHerc. Paris 4 (9.6 um), PHerc. 0139 (9.362 um) and the ink of PHerc. 0841 (9.366 um); see the README
for where it is and is not reliable."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request

import numpy as np
import tifffile
import torch
from scipy import ndimage

from .ct import Volume
from .external import Axis
from .model import RectoNet
from .stacks import load_tifxyz, normals, umbilicus_xy, upsample


def default_cache(url: str) -> str:
    """One chunk-cache folder per volume level, so two scans never share chunk keys."""
    path = urllib.parse.urlparse(url).path.strip('/')
    tag = re.sub(r'[^A-Za-z0-9._-]+', '_', path)[-80:]
    return os.path.join(os.path.expanduser('~'), '.cache', 'recto-offset',
                        f'{tag}_{hashlib.sha1(url.encode()).hexdigest()[:8]}')


def axis_cache_dir(cache: str, axis_cache: str | None = None) -> str:
    """Cache for the coarse level read by the axis estimate. Default <cache>/axis: chunk keys under <cache> are integer
    folders (<cz>/<cy>/<cx>), so the coarse level can never land on a level-0 key."""
    return axis_cache or os.path.join(cache, 'axis')


def open_volume(src: str, cache: str | None):
    if os.path.isdir(src):
        meta = json.load(open(os.path.join(src, '.zarray')))
        v = Volume(cache=src, url='file://' + src, shape=meta['shape'])
        v._fetch = lambda k: None                      # read local chunks in place, never download
        return v, meta['shape']
    src = src.rstrip('/')
    meta = json.loads(urllib.request.urlopen(src + '/.zarray').read())
    assert meta.get('compressor') is None and meta['dtype'] in ('|u1', 'uint8'), 'only uncompressed uint8 levels'
    return Volume(cache or default_cache(src), url=src, shape=meta['shape']), meta['shape']


def load_net(ckpt: str, device: str):
    ck = torch.load(ckpt, map_location='cpu'); args = ck['args']
    net = RectoNet(layers=2 * args['R'] + 1, width=args['width']); net.load_state_dict(ck['model']); net.eval().to(device)
    return net, ck


def axis_centre(volume: str, shape, umbilicus: str | None, cache: str, axis_cache: str | None = None):
    """z -> (y, x) of the scroll axis: the umbilicus if given, else the centre of the non-zero area of level 4."""
    if umbilicus:
        umb = umbilicus_xy(umbilicus)
        return lambda z: umb(np.atleast_1d(z))[0]
    base = volume.rstrip('/').rsplit('/', 1)[0]
    if os.path.isdir(volume):
        if not os.path.isdir(os.path.join(base, '4')):
            raise SystemExit('local volume without level 4: pass --umbilicus')
        centre = Axis(base, shape, os.path.join(base, '4'))
        centre.v._fetch = lambda k: None
        return centre
    return Axis(base, shape, axis_cache_dir(cache, axis_cache))


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


def sample_stack(vol, fz, n, ok, R, shift=None):
    """CT at k = -R..R voxels along the normals n of the node grid fz (H, W, 3), around fz + shift * n.
    Returns stack (2R+1, H, W) float32 and ok narrowed to nodes whose samples all lie inside the volume."""
    ks = np.arange(-R, R + 1, dtype=np.float32)
    H, W = ok.shape
    ok = ok.copy()
    stack = np.zeros((2 * R + 1, H, W), np.float32)
    rows = max(1, min(128, 65536 // max(W, 1)))     # sample in row blocks to bound memory
    for y in range(0, H, rows):
        Z = fz[y:y + rows]; N = np.nan_to_num(n[y:y + rows]); m = ok[y:y + rows]
        if shift is not None:
            Z = Z + shift[y:y + rows, :, None] * N
        pts = Z[None] + ks[:, None, None, None] * N[None]
        pts[:, ~m] = np.nan                         # nodes outside the mesh read nothing (no stray chunk fetches)
        if m.any():
            vol.prefetch(vol.keys_for(pts[:, m].reshape(-1, 3)))
        s = vol.sample(pts)
        ok[y:y + rows] &= np.all(np.isfinite(s), 0)
        stack[:, y:y + rows] = np.nan_to_num(np.clip(s, 0, 255))
    return stack, ok


def correct(net, vol, fz, n, ok, R, max_shift=8.0, min_confidence=0.0, min_snr=0.0, iterations=1, device='cpu'):
    """Offsets along the outward normals n of the 2-voxel node grid fz (H, W, 3).
    Returns offset (H, W; 0 where a node keeps its position), confidence (1 / Laplace scale; from the last pass that
    could read the node), moved (passes in which the node moved) and ok (nodes the first pass could read)."""
    total = np.zeros(ok.shape, np.float64)
    conf = np.zeros(ok.shape, np.float64)
    moved = np.zeros(ok.shape, np.uint8)
    ok0 = None
    for it in range(max(1, iterations)):
        stack, okk = sample_stack(vol, fz, n, ok if ok0 is None else ok0, R, shift=total if it else None)
        if ok0 is None:
            ok0 = okk
        off, logb = predict_offsets(net, stack, okk, device=device)
        off = ndimage.gaussian_filter(np.nan_to_num(off), 1.0)
        c = np.exp(-np.nan_to_num(logb, nan=4.0))
        keep = okk & (c >= min_confidence) & (np.abs(off) >= min_snr / np.maximum(c, 1e-12))
        conf = c if it == 0 else np.where(okk, c, conf)
        total = np.where(keep, np.clip(total + off, -max_shift, max_shift), total)
        moved += keep.astype(np.uint8)
    return np.where(ok0, total, 0.0), conf, moved, ok0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mesh'); ap.add_argument('out')
    ap.add_argument('--volume', required=True)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--ckpt', default=os.path.join(here, 'weights', 'recto_offset_r2.pt'))
    ap.add_argument('--umbilicus', help='scroll axis: umbilicus.json, or a CSV of z,y,x rows (ScrollFiesta axis.csv)')
    ap.add_argument('--cache', help='level-0 chunk cache (default: a folder per volume in ~/.cache/recto-offset)')
    ap.add_argument('--axis-cache', help='chunk cache for the coarse level read by the axis estimate (default <cache>/axis)')
    ap.add_argument('--max-shift', type=float, default=8.0)
    ap.add_argument('--min-confidence', type=float, default=0.0,
                    help='keep nodes in place where the predicted Laplace scale exceeds 1/min-confidence voxels')
    ap.add_argument('--min-snr', type=float, default=0.0,
                    help='keep a node in place unless |offset| >= min-snr * predicted Laplace scale (0 = off)')
    ap.add_argument('--iterations', type=int, default=1, help='passes; each re-reads the CT at the corrected surface')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    net, ck = load_net(a.ckpt, a.device)
    R = ck['args']['R']
    vol, shape = open_volume(a.volume, a.cache)
    zyx, valid = load_tifxyz(a.mesh)
    meta = json.load(open(os.path.join(a.mesh, 'meta.json')))
    coarse = 1.0 / meta['scale'][0]
    f = coarse / 2.0
    fz, fok = upsample(zyx, valid, f)
    n = normals(fz, fok)
    # orient outward
    centre = axis_centre(a.volume, shape, a.umbilicus, vol.cache, a.axis_cache)
    zmid = float(np.nanmedian(fz[..., 0][fok]))
    c = centre(zmid)
    radial = np.concatenate([np.zeros(fz.shape[:-1] + (1,)), fz[..., 1:] - np.asarray(c)], -1)
    if np.nanmedian(np.sum(radial * n, -1)[fok]) < 0:
        n = -n
    ok = fok & np.all(np.isfinite(n), -1)
    H, W = ok.shape
    off, conf, moved, ok = correct(net, vol, fz, n, ok, R, a.max_shift, a.min_confidence, a.min_snr, a.iterations,
                                   a.device)
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
    tag = {'ckpt': os.path.basename(a.ckpt), 'step': ck.get('step'), 'max_shift': a.max_shift,
           'min_snr': a.min_snr, 'iterations': a.iterations,
           'median_abs_offset': float(np.median(np.abs(o[valid]))) if valid.any() else None}
    if a.min_snr > 0:
        g = moved[gi, gj]
        tifffile.imwrite(os.path.join(a.out, 'gate.tif'), np.where(valid, g, 0).astype(np.uint8))
        tag['moved_share'] = float((g[valid] > 0).mean()) if valid.any() else None
    meta = dict(meta)
    meta.setdefault('tags', {})
    meta['tags']['recto_offset'] = tag
    meta['uuid'] = os.path.basename(os.path.normpath(a.out))
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(tag), flush=True)


if __name__ == '__main__':
    main()
