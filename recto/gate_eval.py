"""The apply-time gate (--min-snr) and the second pass (--iterations 2), scored on the frozen crops of recto.evaluate.

  python -m recto.gate_eval --root C:/VesuviusTemp/recto --out results/v12_gating.json [--threads 4] [--check-only]

Same windows, crops, fields and seeds as recto.evaluate (200 crops per push, seed 2000 + push), weights r2 and r3,
Z in {0, 1, 1.5, 2, 3}: a node keeps its position unless |offset| >= Z * b, b the predicted Laplace scale. Share
within 1 voxel at rest and within 2 voxels at pushes of 4-12 voxels. The second pass reads the crop again at the
corrected surface (data.view at the pushed field plus the first, gated offset) and adds its own gated offset. As in
recto.evaluate, crops are predicted whole (64 x 64 nodes), without the 1-node smoothing apply.py adds, and offsets are
not clipped.

Choice rule (PREREG.md amendment 4, written before this script was run): per model, Z = the smallest grid value whose
one-pass gate keeps at least 85 % of nodes within 1 voxel at rest on PHerc. 0814 (validation).
First, Z = 0 with one pass must reproduce every published number in results/*.json (within 1 and within 2 voxels,
every push) within 0.1 points; if it does not, the script writes what it found and stops."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from . import baselines, data
from .evaluate import pushed_set
from .model import RectoNet
from .train import metrics, predict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANS = {'Paris4_val': ('stacks', 'val', 'validation: r2 checkpoint and r2/r3 choice'),
         'PHerc0814': ('ext_stacks/PHerc0814', 'all', 'validation: r2/r3 choice; the Z rule'),
         'Paris4_test': ('stacks', 'test', 'held-out test band (r3 is its second look)'),
         'PHerc0139': ('ext_stacks/PHerc0139', 'all', 'cross-scan test for r2; training scan for r3 (not held out)'),
         'PHerc0009B': ('ext_stacks/PHerc0009B', 'all', 'held-out test scan')}
PUBLISHED = {('Paris4_val', 'r2'): ['r2_val.json'],
             ('Paris4_test', 'r2'): ['final_test.json', 'v2_paris4_test_r2_large.json'],
             ('Paris4_test', 'r3'): ['v2_paris4_test.json'],
             ('PHerc0139', 'r2'): ['final_pherc0139.json'],
             ('PHerc0009B', 'r2'): ['v2_pherc0009b_r2.json'],
             ('PHerc0009B', 'r3'): ['v2_pherc0009b.json']}
AMPS = [0, 4, 6, 8, 10, 12]
ZS = [0.0, 1.0, 1.5, 2.0, 3.0]
RULE_SCAN, RULE_SHARE = 'PHerc0814', 0.85


def load(tag):
    ck = torch.load(os.path.join(HERE, 'weights', f'recto_offset_{tag}.pt'), map_location='cpu')
    net = RectoNet(layers=2 * ck['args']['R'] + 1, width=ck['args']['width']); net.load_state_dict(ck['model'])
    return net.eval(), ck['args']


def gate(p, lb, z):
    return np.where(np.abs(p) >= z * np.exp(lb), p, 0.0).astype(np.float32)


def row(err, m, moved):
    r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics(err, m).items()}
    r['moved'] = round(float(moved[m].mean()), 4)
    return r


def published(scan, tag, amps):
    """{amp: {'within1', 'within2'}} of the model rows in the published results files."""
    out = {}
    for f in PUBLISHED.get((scan, tag), []):
        d = json.load(open(os.path.join(HERE, 'results', f)))
        for amp, r in d['by_amplitude'].items():
            if int(amp) in amps:
                out[int(amp)] = {'file': f, 'within1': r['model']['within1'], 'within2': r['model']['within2']}
    return out


def reproduce(items, net, args, scan, tag, n):
    pub = published(scan, tag, AMPS)
    rows = {}
    for amp, ref in sorted(pub.items()):
        x, y, m = pushed_set(items, n, seed=2000 + amp, amp=amp, crop=args['crop'], R=args['R'])
        p, _ = predict(net, x, 'cpu')
        r = metrics(p - y, m)
        d = max(abs(r['within1'] - ref['within1']), abs(r['within2'] - ref['within2'])) * 100
        rows[amp] = {'file': ref['file'], 'within1': round(r['within1'], 4), 'within2': round(r['within2'], 4),
                     'published_within1': round(ref['within1'], 4), 'published_within2': round(ref['within2'], 4),
                     'max_abs_diff_points': round(d, 3)}
        print('reproduce', scan, tag, amp, rows[amp], flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--models', nargs='+', default=['r2', 'r3'])
    ap.add_argument('--check-only', action='store_true', help='only the reproduction check')
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    nets = {t: load(t) for t in a.models}
    res = {'protocol': 'recto.evaluate crops and fields (seed 2000 + push); a node keeps its position unless '
                       '|offset| >= Z * predicted Laplace scale; it2 = second pass read at the corrected surface; '
                       'crops predicted whole, no smoothing, no clipping',
           'rule': f'per model, Z = smallest of {ZS} with one-pass within-1 share at rest >= {RULE_SHARE} on {RULE_SCAN}',
           'amps': AMPS, 'z_grid': ZS, 'n_crops': a.n, 'reproduction': {}, 'scans': {}}
    items_of = {s: data.load_stacks(a.root, split, sub) for s, (sub, split, _) in SCANS.items()}
    # 1. Z = 0, one pass, must reproduce the published numbers
    bad = []
    for (scan, tag) in PUBLISHED:
        if tag not in nets:
            continue
        rows = reproduce(items_of[scan], *nets[tag], scan, tag, a.n)
        res['reproduction'][f'{scan}/{tag}'] = rows
        bad += [f'{scan}/{tag}/{amp}' for amp, r in rows.items() if r['max_abs_diff_points'] > 0.1]
    res['reproduction_ok'] = not bad
    if bad or a.check_only:
        res['reproduction_failed'] = bad
        json.dump(res, open(a.out, 'w'), indent=1)
        print('reproduction', 'FAILED: ' + ', '.join(bad) if bad else 'ok', flush=True)
        sys.exit(2 if bad else 0)
    # 2. model-free snaps, calibrated on training windows exactly as recto.evaluate
    train = data.load_stacks(a.root, 'train')
    args0 = next(iter(nets.values()))[1]
    cx, cy, cm = data.fixed_set(train, 200, seed=7, crop=args0['crop'], R=args0['R'], A=8.0)
    bias = {k: baselines.calibrate(cx, cy, cm, k) for k in ('ridge', 'edge')}
    del train, cx, cy, cm
    res['snap_bias'] = bias
    # 3. the gate grid, one and two passes
    for scan, (sub, split, role) in SCANS.items():
        items = items_of[scan]
        out = res['scans'][scan] = {'role': role, 'windows': len(items), 'snap': {}, 'models': {}}
        for tag, (net, args) in nets.items():
            R = args['R']
            mres = out['models'][tag] = {'it1': {str(z): {} for z in ZS}, 'it2': {str(z): {} for z in ZS}}
            for amp in AMPS:
                x, y, m, sts = pushed_set(items, a.n, seed=2000 + amp, amp=amp, crop=args['crop'], R=R, raw=True)
                key = 'within1' if amp == 0 else 'within2'
                if str(amp) not in out['snap']:
                    s = {k: metrics(baselines.snap(x, k, bias[k]) - y, m)[key] for k in ('ridge', 'edge')}
                    out['snap'][str(amp)] = {'zero': round(metrics(-y, m)[key], 4), 'ridge': round(s['ridge'], 4),
                                             'edge': round(s['edge'], 4), 'best_snap': round(max(s.values()), 4)}
                p1, lb1 = predict(net, x, 'cpu')
                del x
                for z in ZS:
                    g1 = gate(p1, lb1, z)
                    mres['it1'][str(z)][str(amp)] = row(g1 - y, m, g1 != 0)
                    x2 = np.stack([data.normalise(data.view(st, -yy + gg, R))[None] for st, yy, gg in zip(sts, y, g1)])
                    p2, lb2 = predict(net, x2.astype(np.float32), 'cpu')
                    del x2
                    g2 = gate(p2, lb2, z)
                    mres['it2'][str(z)][str(amp)] = row(g1 + g2 - y, m, (g1 != 0) | (g2 != 0))
                print(scan, tag, amp, key, 'Z:', ' '.join(f"{z}={mres['it1'][str(z)][str(amp)][key]:.3f}/"
                                                          f"{mres['it2'][str(z)][str(amp)][key]:.3f}" for z in ZS),
                      f'snap {out["snap"][str(amp)]["best_snap"]:.3f}', f'{time.time() - t0:.0f} s', flush=True)
            json.dump(res, open(a.out + '.tmp', 'w'), indent=1)
    # 4. the rule
    res['choice'] = {}
    for tag in nets:
        rest = {str(z): res['scans'][RULE_SCAN]['models'][tag]['it1'][str(z)]['0']['within1'] for z in ZS}
        ok = [z for z in ZS if rest[str(z)] >= RULE_SHARE]
        res['choice'][tag] = {'Z': ok[0] if ok else None, f'within1_at_rest_{RULE_SCAN}': rest}
    res['summary'] = {scan: {tag: {f'{it} Z={z}': {amp: r['within1' if amp == '0' else 'within2']
                                                   for amp, r in res['scans'][scan]['models'][tag][it][str(z)].items()}
                                   for it in ('it1', 'it2') for z in ZS}
                             for tag in nets} for scan in SCANS}
    res['sec'] = round(time.time() - t0)
    json.dump(res, open(a.out, 'w'), indent=1)
    if os.path.exists(a.out + '.tmp'):
        os.remove(a.out + '.tmp')
    print('choice', json.dumps(res['choice']), flush=True)


if __name__ == '__main__':
    main()
