"""Model-free controls for recto-offset's PHerc. 0800 real-mesh result (results/demo_pherc0800.json, v1.2 at 9f868fa).

Same 12 crops, same nodes, same outward-normal rule, same smoothing and the same measure as recto.demo_meshes
(recto.demo_meshes.brightest: brightest 1-voxel-smoothed layer within +-6 voxels along the normal; share of nodes
where it lies within 2 voxels of the surface; pooled over crops weighted by nodes). Geometry code follows
recto.demo_null, which reproduces the published numbers exactly.

Cases, each a shift s (voxels) of every node along its outward normal before the profile is read:
  published            s = 0
  r3                   s = stored r3 offset (demo0800/<name>_crop_recto/offset.tif, the published run)
  r2                   s = r2 offset, recto.apply --ckpt weights/recto_offset_r2.pt --device cpu, run here from the
                       local CT cache only (outputs in D:/PrizeTemp/recto0800_ctrl/<name>_crop_recto_r2)
  constant_<c>         s = c for every node; requested grid {-2,-1,-0.5,+0.5,+1,+mean(r3),+1.5,+2,+2.5,+3}, plus an
                       added sweep to -6..+6 (the requested grid's best value sits at its edge on several crops)
  per_crop_median_r3   s = that crop's median r3 offset, for every node of the crop
  r3_permuted          s = the crop's r3 offsets randomly permuted across its measured nodes (5 seeds): same offset
                       distribution per crop, no per-node information
CPU only (CUDA hidden). Cache only: nothing is downloaded. recto.apply (r2) runs on a volume whose _fetch raises on a
chunk missing from the local cache, and urllib is disabled. The measuring volume reads a missing chunk as NaN: that
node becomes unreadable for that case and is counted (a first run that raised instead stopped on crop z6864_w080, see
control_0800_run1_aborted_uncached_chunk.log). Results are given on each case's readable nodes and on one common node
set per crop. Only the 12 crops named in results/demo_pherc0800.json are used.

  C:/Users/PC/miniconda3/envs/vesuvius/python.exe D:/PrizeTemp/recto0800_ctrl/control_0800.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'                # never the GPU
import json
import sys
import time
import urllib.request

import numpy as np
import psutil
import tifffile
from scipy import ndimage

REPO = 'D:/Competition/Vesuvius progress prizes/_public/recto-offset'
ROOT = 'C:/VesuviusTemp/recto'
OUT = 'D:/PrizeTemp/recto0800_ctrl'
CACHE = os.path.join(ROOT, 'ext_ct', 'PHerc0800', '0')
AXIS = os.path.join(ROOT, 'ext_ct', 'PHerc0800', 'axis')
RSS_LIMIT = 2.4 * 2 ** 30
SEEDS = 5
sys.path.insert(0, REPO)

import torch  # noqa: E402
torch.set_num_threads(4)
from recto import apply as tool  # noqa: E402
from recto.ct import Volume  # noqa: E402
from recto.demo_meshes import SHAPE, VOL, brightest  # noqa: E402
from recto.external import Axis  # noqa: E402
from recto.stacks import load_tifxyz, normals, upsample  # noqa: E402


# ---------- cache only ----------
def _nofetch(self, k):
    p = self._path(k)
    if not (os.path.exists(p) or os.path.exists(p + '.empty')):
        raise SystemExit(f'chunk {k} is not in the local cache {self.cache}; this control never downloads')


def _nonet(*a, **k):
    raise SystemExit(f'network access attempted ({a[:1]}); this control never downloads')


Volume._fetch = _nofetch
urllib.request.urlopen = _nonet
urllib.request.urlretrieve = _nonet
tool.open_volume = lambda src, cache: (Volume(cache, url=src, shape=SHAPE), SHAPE)   # skips the .zarray request

NANCHUNK = np.full((128, 128, 128), np.nan, np.float32)


class CacheOnlyVolume(Volume):
    """Measuring volume: a chunk missing from the local cache reads as NaN (never fetched); the missing keys are kept."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.missing = set()

    def chunk(self, k):
        if k not in self.mem:
            p = self._path(k)
            if not (os.path.exists(p) or os.path.exists(p + '.empty')):
                self.missing.add(tuple(int(x) for x in k))
                return NANCHUNK
        return super().chunk(k)


def nfiles(d):
    return sum(len(f) for _, _, f in os.walk(d))


def rss_check(where):
    r = psutil.Process().memory_info().rss
    if r > RSS_LIMIT:
        raise SystemExit(f'RSS {r / 2 ** 30:.2f} GB > limit at {where}')
    return r


t0 = time.time()
demo = json.load(open(os.path.join(REPO, 'results', 'demo_pherc0800.json')))
names = sorted(demo['meshes'])
assert len(names) == 12, names
files0 = {'level0': nfiles(CACHE), 'axis': nfiles(AXIS)}
print('cache files before', files0, flush=True)

# ---------- pass 1: geometry, offsets, the published / r3 node set ----------
vol = CacheOnlyVolume(CACHE, url=f'{VOL}/0', shape=SHAPE)
axis = Axis(VOL, SHAPE, AXIS)
ks = np.arange(-6, 7, dtype=np.float32)
geo = {}
for ci, name in enumerate(names):
    dst = os.path.join(ROOT, 'demo0800', name + '_crop')
    z0, v0 = load_tifxyz(dst)                               # as recto.demo_meshes / recto.demo_null
    f0, ok = upsample(z0, v0, 10.0)
    n = normals(f0, ok)
    c = axis(float(np.median(f0[..., 0][ok])))
    radial = np.concatenate([np.zeros(f0.shape[:-1] + (1,)), f0[..., 1:] - np.asarray(c)], -1)
    if np.nanmedian(np.sum(radial * n, -1)[ok]) < 0:
        n = -n
    H0, W0 = f0.shape[:2]
    up = lambda o, H0=H0, W0=W0: ndimage.zoom(np.nan_to_num(o), 10.0, order=1)[:H0, :W0][::2, ::2]  # noqa: E731
    off3 = up(tifffile.imread(os.path.join(dst + '_recto', 'offset.tif')))
    f0, n, ok = f0[::2, ::2], np.nan_to_num(n[::2, ::2]), ok[::2, ::2]
    g = {'f0': f0, 'n': n, 'off3': off3, 'up': up}
    at = lambda s: vol.sample((f0 + s[..., None] * n)[None] + ks[:, None, None, None] * n[None])  # noqa: E731
    b, a3 = at(np.zeros(off3.shape)), at(off3)
    g['good'] = good = ok & np.all(np.isfinite(b), 0) & np.all(np.isfinite(a3), 0)
    g['s0'] = float(np.mean(np.abs(brightest(b[:, good])) <= 2))
    g['s3'] = float(np.mean(np.abs(brightest(a3[:, good])) <= 2))
    print('pass1', name, round(g['s0'], 4), round(g['s3'], 4), flush=True)
    geo[name] = g
    rss_check('pass1 ' + name)
assert not vol.missing, f'published / r3 need uncached chunks {vol.missing}'
w1 = np.array([geo[k]['good'].sum() for k in names], float)
repro = {'published': [float(np.average([geo[k]['s0'] for k in names], weights=w1)),
                       demo['pooled']['within2_of_bright_before']],
         'r3': [float(np.average([geo[k]['s3'] for k in names], weights=w1)), demo['pooled']['within2_of_bright_after']]}
repro_ok = all(abs(a - b) <= 0.002 for a, b in repro.values())
print('reproduction (this run, published json)', repro, 'ok' if repro_ok else 'FAILED', flush=True)
if not repro_ok:
    json.dump({'what': 'reproduction failed; nothing else computed', 'reproduction': repro},
              open(os.path.join(OUT, 'control_pherc0800.json'), 'w'), indent=1)
    raise SystemExit('r3 / published did not reproduce within 0.2 points: stopping')

# ---------- r2 on CPU from the cache (same call as recto.demo_meshes, only --ckpt differs) ----------
r2_ckpt = os.path.join(REPO, 'weights', 'recto_offset_r2.pt')
for name in names:
    dst = os.path.join(ROOT, 'demo0800', name + '_crop')
    out2 = os.path.join(OUT, name + '_crop_recto_r2')
    if os.path.exists(os.path.join(out2, 'offset.tif')):
        continue
    t1 = time.time()
    tool.main([dst, out2, '--volume', f'{VOL}/0', '--cache', CACHE, '--ckpt', r2_ckpt, '--device', 'cpu',
               '--axis-cache', AXIS])
    print('r2', name, f'{time.time() - t1:.0f}s', flush=True)
    rss_check('r2 ' + name)
for name in names:
    geo[name]['off2'] = geo[name].pop('up')(tifffile.imread(os.path.join(OUT, name + '_crop_recto_r2', 'offset.tif')))

mean_r3 = float(np.mean(np.concatenate([g['off3'][g['good']] for g in geo.values()])))
consts = sorted([-2.0, -1.0, -0.5, 0.5, 1.0, float(np.round(mean_r3, 2)), 1.5, 2.0, 2.5, 3.0])    # requested grid
extra = [-6.0, -5.0, -4.0, -3.0, -2.5, -1.5, 3.5, 4.0, 4.5, 5.0, 6.0]     # added sweep
print('mean r3 offset (signed, measured nodes)', mean_r3, 'constants', consts, 'extra', extra, flush=True)


def ck(c):
    return f'constant_{c:+g}'


def positions(g, s):
    """Brightest-layer position (voxels from the shifted surface) at the measured nodes; NaN where unreadable."""
    prof = vol.sample((g['f0'] + s[..., None] * g['n'])[None] + ks[:, None, None, None] * g['n'][None])[:, g['good']]
    fin = np.all(np.isfinite(prof), 0)
    pos = np.full(prof.shape[1], np.nan, np.float32)
    pos[fin] = brightest(prof[:, fin])
    return pos


# ---------- pass 2: every case ----------
pos = {}            # name -> case -> positions (float32, measured nodes)
per_crop = {}
for ci, name in enumerate(names):
    g = geo[name]
    good, off3 = g['good'], g['off3']
    sh = {'published': np.zeros(off3.shape), 'r3': off3, 'r2': g['off2']}
    for c in consts + extra:
        sh[ck(c)] = np.full(off3.shape, c)
    med = float(np.median(off3[good]))
    sh['per_crop_median_r3'] = np.full(off3.shape, med)
    for seed in range(SEEDS):
        s = off3.copy()
        s[good] = np.random.default_rng([seed, ci]).permutation(off3[good])
        sh[f'r3_permuted_seed{seed}'] = s
    miss0 = len(vol.missing)
    pos[name] = {k: positions(g, s) for k, s in sh.items()}
    common = np.all([np.isfinite(p) for p in pos[name].values()], 0)
    row = {'nodes': int(good.sum()), 'common_nodes': int(common.sum()), 'r3_offset_median': med,
           'r3_offset_mean': float(np.mean(off3[good])), 'r3_offset_median_abs': float(np.median(np.abs(off3[good]))),
           'r3_offset_abs_gt6_share': float(np.mean(np.abs(off3[good]) > 6)),
           'r2_offset_median': float(np.median(g['off2'][good])),
           'r2_offset_median_abs': float(np.median(np.abs(g['off2'][good]))),
           'uncached_chunks_hit': len(vol.missing) - miss0,
           'unreadable_nodes': {k: int(np.sum(~np.isfinite(p))) for k, p in pos[name].items() if np.any(~np.isfinite(p))},
           'readable': {k: int(np.isfinite(p).sum()) for k, p in pos[name].items()},
           'share': {k: float(np.mean(np.abs(p[np.isfinite(p)]) <= 2)) for k, p in pos[name].items()},
           'share_common': {k: float(np.mean(np.abs(p[common]) <= 2)) for k, p in pos[name].items()}}
    per_crop[name] = row
    print(name, json.dumps({k: round(v, 4) for k, v in row['share'].items()}), 'unreadable', row['unreadable_nodes'],
          f'rss {rss_check(name) / 2 ** 20:.0f} MB', flush=True)

files1 = {'level0': nfiles(CACHE), 'axis': nfiles(AXIS)}
cases = list(per_crop[names[0]]['share'])
req_keys = ['published'] + [ck(c) for c in consts]                       # constant shifts incl. c = 0, requested grid
sweep_keys = sorted(req_keys + [ck(c) for c in extra], key=lambda k: 0.0 if k == 'published' else float(k[9:]))


def summarise(field):
    """field 'share': each case on its own readable nodes, pooled with weights = that case's readable nodes per crop.
    field 'share_common': one node set per crop (readable in every case), weights = common nodes per crop."""
    def w(k, case):
        return per_crop[k]['readable'][case] if field == 'share' else per_crop[k]['common_nodes']

    def pool(case):
        return float(np.average([per_crop[k][field][case] for k in names], weights=[w(k, case) for k in names]))
    pooled = {k: pool(k) for k in cases if not k.startswith('r3_permuted')}
    ps = [pool(f'r3_permuted_seed{s}') for s in range(SEEDS)]
    pooled['r3_permuted'] = {'mean': float(np.mean(ps)), 'min': min(ps), 'max': max(ps), 'per_seed': ps}
    best_req = max(req_keys, key=lambda k: pooled[k])
    best_sweep = max(sweep_keys, key=lambda k: pooled[k])
    crops, diffs = {}, {}
    for k in names:
        s = per_crop[k][field]
        perm = [s[f'r3_permuted_seed{i}'] for i in range(SEEDS)]
        orq = max(req_keys, key=lambda c: s[c])
        ors = max(sweep_keys, key=lambda c: s[c])
        crops[k] = {'published': s['published'], 'r3': s['r3'], 'r2': s['r2'],
                    'best_constant_requested_grid': [orq, s[orq]], 'best_constant_sweep': [ors, s[ors]],
                    'per_crop_median_r3': s['per_crop_median_r3'], 'r3_permuted_mean': float(np.mean(perm)),
                    'r3_permuted_max': max(perm)}
        diffs[k] = {'published': s['r3'] - s['published'],
                    'best_pooled_constant_requested_grid': s['r3'] - s[best_req],
                    'best_pooled_constant_sweep': s['r3'] - s[best_sweep],
                    'best_constant_this_crop_requested_grid': s['r3'] - s[orq],
                    'best_constant_this_crop_sweep': s['r3'] - s[ors],
                    'per_crop_median_r3': s['r3'] - s['per_crop_median_r3'],
                    'r3_permuted_mean': s['r3'] - float(np.mean(perm)), 'r3_permuted_max': s['r3'] - max(perm),
                    'r2_minus_best_constant_this_crop_sweep': s['r2'] - s[ors]}
    wr = [w(k, 'r3') for k in names]
    oracle_req = float(np.average([crops[k]['best_constant_requested_grid'][1] for k in names], weights=wr))
    oracle_sweep = float(np.average([crops[k]['best_constant_sweep'][1] for k in names], weights=wr))
    return {'pooled': pooled,
            'best_pooled_constant_requested_grid': [best_req, pooled[best_req]],
            'best_pooled_constant_sweep': [best_sweep, pooled[best_sweep]],
            'pooled_best_constant_per_crop_oracle_requested_grid': oracle_req,
            'pooled_best_constant_per_crop_oracle_sweep': oracle_sweep,
            'r3_minus_points_pooled': {
                'published': 100 * (pooled['r3'] - pooled['published']),
                'best_pooled_constant_requested_grid': 100 * (pooled['r3'] - pooled[best_req]),
                'best_pooled_constant_sweep': 100 * (pooled['r3'] - pooled[best_sweep]),
                'best_constant_per_crop_oracle_requested_grid': 100 * (pooled['r3'] - oracle_req),
                'best_constant_per_crop_oracle_sweep': 100 * (pooled['r3'] - oracle_sweep),
                'per_crop_median_r3': 100 * (pooled['r3'] - pooled['per_crop_median_r3']),
                'r3_permuted_mean': 100 * (pooled['r3'] - pooled['r3_permuted']['mean'])},
            'crops_where_r3_beats': {c: int(sum(diffs[k][c] > 0 for k in names)) for c in diffs[names[0]]},
            'per_crop_table': crops,
            'per_crop_r3_minus_points': {k: {c: 100 * v for c, v in d.items()} for k, d in diffs.items()}}


own = summarise('share')
com = summarise('share_common')
repro['published_pass2'] = own['pooled']['published']
repro['r3_pass2'] = own['pooled']['r3']

hist = {}
for case in ['published', 'r3', 'r2', own['best_pooled_constant_sweep'][0], 'per_crop_median_r3', 'r3_permuted_seed0']:
    p = np.concatenate([pos[k][case] for k in names])
    p = p[np.isfinite(p)].astype(int) + 6
    hist[case] = np.bincount(p, minlength=13).tolist()

res = {'what': __doc__,
       'repo_commit': '9f868fa', 'crops': names, 'measured_nodes_total': int(sum(per_crop[k]['nodes'] for k in names)),
       'common_nodes_total': int(sum(per_crop[k]['common_nodes'] for k in names)),
       'reproduction': {'pooled_this_run_vs_published_json': repro, 'tolerance': 0.002, 'ok': repro_ok},
       'mean_r3_offset_signed': mean_r3, 'constants_requested': consts, 'constants_added': extra,
       'own_readable_nodes': own,
       'common_node_set': com,
       'brightest_position_histogram_pooled': {'positions': list(range(-6, 7)), 'counts': hist},
       'per_crop': per_crop,
       'notes': ['own_readable_nodes: every case on the demo node set minus the nodes whose shifted profile needs a CT '
                 'chunk missing from the local cache (per_crop.unreadable_nodes; never fetched). common_node_set: one '
                 'node set per crop, readable in every case.',
                 'best constants (pooled or per crop) are chosen on this measure itself: optimistic baselines. '
                 'per_crop_median_r3 uses r3 output but no per-node information.',
                 'constants_added extend the requested grid to -6..+6 because the requested grid\'s best value sits at '
                 'its edge on several crops; *_requested_grid figures use only the requested constants and c = 0.'],
       'uncached_chunks_hit_total': sorted(vol.missing),
       'ct_cache_files_before': files0, 'ct_cache_files_after': files1, 'ct_chunks_fetched': vol.fetched + axis.v.fetched,
       'device': 'cpu (CUDA_VISIBLE_DEVICES=-1)', 'seconds': round(time.time() - t0, 1),
       'peak_rss_mb': round(psutil.Process().memory_info().peak_wset / 2 ** 20)}
json.dump(res, open(os.path.join(OUT, 'control_pherc0800.json'), 'w'), indent=1)


def brief(d):
    return {k: d[k] for k in d if k not in ('per_crop_table', 'per_crop_r3_minus_points')}


print(json.dumps({'reproduction': res['reproduction'], 'own': brief(own), 'common': brief(com),
                  'common_nodes_total': res['common_nodes_total'], 'uncached_chunks': len(vol.missing),
                  'files': [files0, files1], 'fetched': res['ct_chunks_fetched'], 'seconds': res['seconds'],
                  'peak_rss_mb': res['peak_rss_mb']}, indent=1), flush=True)
