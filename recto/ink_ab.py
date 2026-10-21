"""Ink A/B: does a recto-corrected surface read ink better than the surface it came from?

Held-out text: the three labelled PHerc. 0841 segments (no public ink model was trained on PHerc. 0841), on its
native 9.366 um, 113 keV scan. For each segment, `--tiles` non-overlapping 512 x 512 px tiles with the most
labelled ink are chosen from the labels alone. In each tile the CT is sampled every voxel along the outward normal of
the segment's own 9.366 um mesh (k = -24..24), at 1-px node spacing, so a tile pixel is a pixel of the published
surface volume and of its human labels (`D9_DATA/<seg>/human_al.npy`, built by the 9um-reader-letter-shape scripts).

Arms (each a 21-layer surface volume around a surface, layers in the published volume's order):
  A  surface as published
  B  A moved by the recto model's offset
  C  A pushed off by a smooth field (peak 4 voxels, seeded per tile)
  D  C moved by the recto model's offset (the model sees only C)
  E  C moved by the best model-free snap (nearest dark-to-bright rise, bias from Paris 4 training windows)
The recto model runs on the stack 2x down-sampled in-plane (its training spacing); offsets are up-sampled and
smoothed (sigma 2 px). Ink: Reader V3 and Reader v2 through the team's ink_9um inference (as in
9um-reader-letter-shape). Score: human-label AUC on each tile mosaic (positives label >= 0.5; negatives label < 0.05
within 214 px of ink), per segment and in total, with a paired block bootstrap over 128-px blocks.

Stages: build (CPU) -> arms (CPU) -> infer (GPU) -> score (CPU).
  python -m recto.ink_ab build|arms|infer|score --root C:/VesuviusTemp/recto --d9 C:/VesuviusTemp/d9v2_test"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import zarr
from scipy import ndimage

from . import baselines, data
from .ct import Volume
from .external import Axis, B
from .model import RectoNet
from .stacks import load_tifxyz, normals, upsample

SEGS = ['20260220213127-w00', '20260220214732-auto_grown_20260220144552896',
        '20260221022814-auto_grown_20260220174252405']
VOL = 'PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr'
SHAPE = (19400, 7703, 7703)
TAG = '20250821151531-9.366um'
T, K, RI = 512, 24, 10
P = 'D:/Competition/Vesuvius progress prizes'
READERS = {'reader_v3': f'{P}/_fl/ckpt/reader_v3/reader-v3.pth', 'reader_v2': f'{P}/_fl/ckpt/reader_v2/reader-v2-step040000.pth'}


def out_dir(root):
    return os.path.join(root, 'ink_ab')


def pick_tiles(h, valid, n):
    """Top-n non-overlapping T x T tiles by labelled ink pixels (labels only)."""
    H, W = h.shape
    ink = np.nan_to_num(h) >= 0.5
    best = []
    cand = [(int(ink[y:y + T, x:x + T].sum()), y, x) for y in range(0, H - T + 1, T // 2) for x in range(0, W - T + 1, T // 2)
            if valid[y:y + T, x:x + T].mean() > 0.9]
    for s, y, x in sorted(cand, reverse=True):
        if s == 0:
            break
        if all(abs(y - b[1]) >= T or abs(x - b[2]) >= T for b in best):
            best.append((s, y, x))
        if len(best) == n:
            break
    return [(y, x) for _, y, x in best]


def build(a):
    vol = Volume(os.path.join(a.root, 'ext_ct', 'PHerc0841', '0'), url=f'{B}/{VOL}/0', shape=SHAPE, workers=32)
    axis = Axis(f'{B}/{VOL}', SHAPE, os.path.join(a.root, 'ext_ct', 'PHerc0841', 'axis'))
    from .external import fetch_mesh
    ks = np.arange(-K, K + 1, dtype=np.float32)
    for seg in SEGS:
        od = os.path.join(out_dir(a.root), seg); os.makedirs(od, exist_ok=True)
        d = fetch_mesh('PHerc0841', seg, TAG, os.path.join(a.root, 'ext_meshes', 'PHerc0841'))
        zyx, valid = load_tifxyz(d)
        fz, fok = upsample(zyx, valid, 20.0)                     # 1 node per published-volume pixel
        dd = os.path.join(a.d9, f'PHerc0841__{seg}')
        h = np.load(os.path.join(dd, 'human_al.npy')).astype(np.float32)
        sv = zarr.open_array(os.path.join(dd, 'sv.zarr'), mode='r')
        H, W = min(h.shape[0], fz.shape[0]), min(h.shape[1], fz.shape[1])
        svmid = np.asarray(sv[sv.shape[0] // 2, :H, :W])
        tiles = pick_tiles(h[:H, :W], (svmid > 0) & fok[:H, :W], a.tiles)
        nall = normals(fz, fok)
        json.dump({'tiles': tiles}, open(os.path.join(od, 'tiles.json'), 'w'))
        for ti, (y, x) in enumerate(tiles):
            o = os.path.join(od, f'tile{ti}.npz')
            if os.path.exists(o):
                continue
            Z = fz[y:y + T, x:x + T]; ok = fok[y:y + T, x:x + T]
            n = nall[y:y + T, x:x + T]
            c = axis(float(np.nanmedian(Z[..., 0][ok])))
            radial = np.concatenate([np.zeros(Z.shape[:-1] + (1,)), Z[..., 1:] - c], -1)
            sign = 1.0 if np.nanmedian(np.sum(radial * n, -1)[ok]) >= 0 else -1.0
            n = n * sign
            ok &= np.all(np.isfinite(n), -1)
            pts = Z[None] + ks[:, None, None, None] * np.nan_to_num(n)[None]
            vol.prefetch(vol.keys_for(pts[:, ok].reshape(-1, 3)))
            s = vol.sample(pts)
            stack = np.where(np.isfinite(s), np.clip(s, 0, 255), 0).astype(np.uint8)
            ok &= np.all(np.isfinite(s), 0)
            # which way the published volume's layers run: correlate its layers with ours at the same pixels
            svt = np.asarray(sv[:, y:y + T, x:x + T]).astype(np.float32)
            corr = {}
            for direction in (1, -1):
                for shift in range(-6, 7):
                    rs = []
                    for j in range(svt.shape[0]):
                        kk = K + direction * (j - svt.shape[0] // 2) + shift
                        if 0 <= kk < stack.shape[0]:
                            m = ok & (svt[j] > 0)
                            rs.append(np.corrcoef(svt[j][m][::7], stack[kk][m][::7].astype(np.float32))[0, 1])
                    corr[(direction, shift)] = float(np.nanmean(rs))
            (direction, shift), r = max(corr.items(), key=lambda kv: kv[1])
            np.savez_compressed(o, stack=stack, valid=ok, label=h[y:y + T, x:x + T], yx=[y, x], sv_direction=direction,
                                sv_shift=shift, sv_corr=r, normal_sign=sign)
            print(seg[:24], 'tile', ti, (y, x), 'ink px', int((h[y:y + T, x:x + T] >= 0.5).sum()),
                  'sv order', direction, 'shift', shift, 'r %.3f' % r, 'chunks', vol.fetched, flush=True)


def model_offsets(net, stack, R, device):
    """Recto model on the stack at 2-px spacing; returns a 1-px offset map (voxels along the outward normal)."""
    s2 = stack[:, ::2, ::2]
    x = data.normalise(data.view(np.ascontiguousarray(s2), np.zeros(s2.shape[1:], np.float32), R))[None, None]
    with torch.no_grad():
        p, _ = net(torch.from_numpy(x.astype(np.float32)).to(device))
    p = p[0].cpu().numpy()
    up = ndimage.zoom(p, 2, order=1)[:stack.shape[1], :stack.shape[2]]
    return ndimage.gaussian_filter(up, 2)


def arms(a):
    ck = torch.load(a.ckpt, map_location='cpu'); args = ck['args']
    net = RectoNet(layers=2 * args['R'] + 1, width=args['width']); net.load_state_dict(ck['model']); net.eval()
    tr = data.load_stacks(a.root, 'train')
    cx, cy, cm = data.fixed_set(tr, 200, seed=7, crop=args['crop'], R=args['R'], A=args['A'])
    bias_edge = baselines.calibrate(cx, cy, cm, 'edge')
    rec = {'ckpt': a.ckpt, 'bias_edge': bias_edge, 'tiles': {}}
    for seg in SEGS:
        od = os.path.join(out_dir(a.root), seg)
        tiles = json.load(open(os.path.join(od, 'tiles.json')))['tiles']
        mos = {k: np.zeros((2 * RI + 1, T, T * len(tiles)), np.uint8) for k in 'ABCDE'}
        lab = np.zeros((T, T * len(tiles)), np.float32); val = np.zeros((T, T * len(tiles)), bool)
        for ti in range(len(tiles)):
            d = np.load(os.path.join(od, f'tile{ti}.npz'))
            st, ok = d['stack'], d['valid']
            rng = np.random.default_rng(1000 + 17 * ti + len(seg))
            f0 = np.zeros(st.shape[1:], np.float32)
            g = data.field((T // 2, T // 2), rng, A=4.0, p_zero=0.0)
            push = ndimage.zoom(g / max(np.abs(g).max(), 1e-6) * 4.0, 2, order=1)[:T, :T].astype(np.float32)
            pA = model_offsets(net, st, args['R'], 'cpu')
            pushed_view = data.view(st, push, args['R'] + 0)                 # stack around the pushed surface
            s2 = pushed_view[:, ::2, ::2]
            with torch.no_grad():
                pp, _ = net(torch.from_numpy(data.normalise(s2)[None, None].astype(np.float32)))
            pC = ndimage.gaussian_filter(ndimage.zoom(pp[0].numpy(), 2, order=1)[:T, :T], 2)
            xe = data.normalise(s2)[None, None]
            pE = ndimage.zoom(baselines.snap(xe, 'edge', bias_edge)[0], 2, order=0)[:T, :T]
            pos = {'A': f0, 'B': pA, 'C': push, 'D': push + pC, 'E': push + pE}
            for k, f in pos.items():
                v = data.view(st, f.astype(np.float32), RI)                  # outward order, layer RI = surface
                if int(d['sv_direction']) == -1:
                    v = v[::-1]
                mos[k][:, :, ti * T:(ti + 1) * T] = np.clip(v, 0, 255).astype(np.uint8)
            lab[:, ti * T:(ti + 1) * T] = d['label']; val[:, ti * T:(ti + 1) * T] = ok
            rec['tiles'][f'{seg}/{ti}'] = {k: float(np.median(np.abs(f[ok]))) for k, f in pos.items()}
            rec['tiles'][f'{seg}/{ti}'].update({'model_vs_push_median_err': float(np.median(np.abs((push + pC) - 0)[ok])),
                                                'sv_corr': float(d['sv_corr']), 'sv_shift': int(d['sv_shift'])})
        for k, m in mos.items():
            z = zarr.open_array(os.path.join(od, f'arm_{k}.zarr'), mode='w', shape=m.shape, chunks=(m.shape[0], 128, 128),
                                dtype='u1', zarr_format=2, compressor=None, fill_value=0)
            z[:] = m
        np.save(os.path.join(od, 'label.npy'), lab); np.save(os.path.join(od, 'valid.npy'), val)
        print(seg[:24], 'arms written', json.dumps({t: v for t, v in rec['tiles'].items() if t.startswith(seg)})[:300], flush=True)
    json.dump(rec, open(os.path.join(out_dir(a.root), 'arms.json'), 'w'), indent=1)


def infer(a):
    env = dict(os.environ, PYTHONPATH=f'{P}/villa/_worktrees/ink9um/vesuvius/src')
    for seg in SEGS:
        od = os.path.join(out_dir(a.root), seg)
        for k in 'ABCDE':
            for name, ck in READERS.items():
                o = os.path.join(od, f'pred_{k}_{name}.tif')
                if os.path.exists(o):
                    continue
                subprocess.check_call([sys.executable, '-m', 'vesuvius.ink_detection.inference.infer', os.path.join(od, f'arm_{k}.zarr'),
                                       ck, o, '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8',
                                       '--direction', 'forward', '--no-compile'], env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
                print('inferred', seg[:24], k, name, flush=True)


def auc(score, pos, neg):
    sp, sn = score[pos], score[neg]
    if sp.size < 50 or sn.size < 50:
        return None
    allv = np.concatenate([sp, sn]); r = allv.argsort(kind='mergesort').argsort() + 1
    return float((r[:sp.size].sum() - sp.size * (sp.size + 1) / 2) / (sp.size * sn.size))


def score(a):
    import tifffile
    out = {'per_segment': {}, 'pooled': {}, 'bootstrap': {}}
    pooled = {}
    for seg in SEGS:
        od = os.path.join(out_dir(a.root), seg)
        lab = np.load(os.path.join(od, 'label.npy')); val = np.load(os.path.join(od, 'valid.npy'))
        near = ndimage.distance_transform_edt(lab < 0.5) <= 214
        pos, neg = val & (lab >= 0.5), val & near & (lab < 0.05)
        out['per_segment'][seg] = {}
        for name in READERS:
            for k in 'ABCDE':
                p = tifffile.imread(os.path.join(od, f'pred_{k}_{name}.tif')).astype(np.float32)
                p = p[:lab.shape[0], :lab.shape[1]]
                out['per_segment'][seg][f'{name}|{k}'] = auc(p, pos, neg)
                pooled.setdefault(f'{name}|{k}', []).append((p, pos, neg))
    rng = np.random.default_rng(0)
    for key, parts in pooled.items():
        out['pooled'][key] = float(np.mean([auc(p, ps, ng) for p, ps, ng in parts]))
    # paired block bootstrap (128-px blocks) for B-A and D-C, Reader V3 and v2
    for name in READERS:
        for x, y in (('B', 'A'), ('D', 'C'), ('E', 'C'), ('D', 'E')):
            blocks = []
            for seg in SEGS:
                od = os.path.join(out_dir(a.root), seg)
                lab = np.load(os.path.join(od, 'label.npy')); val = np.load(os.path.join(od, 'valid.npy'))
                near = ndimage.distance_transform_edt(lab < 0.5) <= 214
                px = tifffile.imread(os.path.join(od, f'pred_{x}_{name}.tif')).astype(np.float32)[:lab.shape[0], :lab.shape[1]]
                py = tifffile.imread(os.path.join(od, f'pred_{y}_{name}.tif')).astype(np.float32)[:lab.shape[0], :lab.shape[1]]
                for by in range(0, lab.shape[0], 128):
                    for bx in range(0, lab.shape[1], 128):
                        sl = (slice(by, by + 128), slice(bx, bx + 128))
                        blocks.append((px[sl], py[sl], (val & (lab >= 0.5))[sl], (val & near & (lab < 0.05))[sl]))
            diffs = []
            for _ in range(200):
                idx = rng.integers(0, len(blocks), len(blocks))
                cat = lambda i: np.concatenate([blocks[j][i].ravel() for j in idx])  # noqa: E731
                pxx, pyy, ps, ng = cat(0), cat(1), cat(2).astype(bool), cat(3).astype(bool)
                ax, ay = auc(pxx, ps, ng), auc(pyy, ps, ng)
                if ax is not None and ay is not None:
                    diffs.append(ax - ay)
            out['bootstrap'][f'{name}|{x}-{y}'] = {'mean': float(np.mean(diffs)), 'ci95': [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    json.dump(out, open(os.path.join(a.out), 'w'), indent=1)
    print(json.dumps({'pooled': out['pooled'], 'bootstrap': out['bootstrap']}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'arms', 'infer', 'score'])
    ap.add_argument('--root', required=True); ap.add_argument('--d9', default='C:/VesuviusTemp/d9v2_test')
    ap.add_argument('--tiles', type=int, default=4); ap.add_argument('--ckpt', default='runs/r1/best.pt')
    ap.add_argument('--out', default='results/ink_ab.json')
    a = ap.parse_args(argv)
    {'build': build, 'arms': arms, 'infer': infer, 'score': score}[a.stage](a)


if __name__ == '__main__':
    main()
