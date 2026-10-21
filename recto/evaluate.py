"""Held-out evaluation: the network against model-free snaps on frozen, pushed-off test crops.

  python -m recto.evaluate --root C:/VesuviusTemp/recto --ckpt runs/r1/best.pt --out results/test.json
Test windows come only from the held-out z band (data.SPLIT['test']); baselines are calibrated on training windows.
Pushes are drawn at fixed peak amplitudes (0, 2, 4, 6, 8 voxels; smooth fields as in training) so the table shows
where each method breaks. Metrics per node on valid nodes: mean / median absolute error, share within 1 and 2 voxels
(Hari Seldon's bar for a usable surface: within ~2 voxels of the recto), 95th percentile."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import baselines, data
from .model import RectoNet
from .train import metrics, predict


def pushed_set(items, n, seed, amp, crop=64, R=12):
    rng = np.random.default_rng(seed)
    xs, ys, ms = [], [], []
    for _ in range(n):
        it = items[rng.integers(len(items))]
        h, w = it['valid'].shape
        i0, j0 = rng.integers(0, h - crop + 1), rng.integers(0, w - crop + 1)
        st = it['stack'][:, i0:i0 + crop, j0:j0 + crop]; va = it['valid'][i0:i0 + crop, j0:j0 + crop]
        if amp == 0:
            f = np.zeros((crop, crop), np.float32)
        else:
            f = data.field((crop, crop), rng, A=amp, p_zero=0.0)
            f = f / max(np.abs(f).max(), 1e-6) * amp * np.sign(rng.standard_normal())   # peak exactly amp
        xs.append(data.normalise(data.view(np.ascontiguousarray(st), f, R))[None]); ys.append(-f); ms.append(va)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), np.stack(ms)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test'); ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--stacks', default='stacks', help="subdir of root; e.g. ext_stacks/PHerc0139 with --split all")
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location='cpu')
    args = ck['args']
    net = RectoNet(layers=2 * args['R'] + 1, width=args['width']); net.load_state_dict(ck['model']); net.to(a.device)
    train = data.load_stacks(a.root, 'train'); test = data.load_stacks(a.root, a.split, a.stacks)
    cx, cy, cm = data.fixed_set(train, 200, seed=7, crop=args['crop'], R=args['R'], A=args['A'])
    bias = {k: baselines.calibrate(cx, cy, cm, k) for k in ('ridge', 'edge')}
    res = {'ckpt': a.ckpt, 'step': ck.get('step'), 'split': a.split, 'stacks': a.stacks, 'windows': len(test),
           'patches': len({t['meta']['patch'] for t in test}), 'bias': bias, 'by_amplitude': {}}
    for amp in (0, 2, 4, 6, 8):
        x, y, m = pushed_set(test, a.n, seed=2000 + amp, amp=amp, crop=args['crop'], R=args['R'])
        p, _ = predict(net, x, a.device)
        row = {'model': metrics(p - y, m), 'zero': metrics(-y, m)}
        for k in ('ridge', 'edge'):
            row[k] = metrics(baselines.snap(x, k, bias[k]) - y, m)
        res['by_amplitude'][str(amp)] = row
        print(amp, {k: (round(v['median'], 2), round(v['within2'], 3)) for k, v in row.items()}, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
