"""Model-free baselines for the recto.obj demo on the ScrollFiesta PHerc0139 atlas: every node moved by the same constant
along its outward normal, and every chart moved by its own median r2 offset; same measure as recto.demo_obj
(recto.demo_meshes.brightest, within 2 voxels), on a seeded subsample of 200,000 of the measured nodes, with the
ScrollFiesta mesh and both recto.obj outputs measured on the same subsample."""
import json, sys, time
import numpy as np
sys.path.insert(0, r'D:/Competition/Vesuvius progress prizes/_public/recto-offset')
from recto import apply as tool, obj
from recto.demo_obj import profiles, share
from recto.stacks import umbilicus_xy
R = 'D:/PrizeTemp/recto-sf'
MESH = 'D:/Competition/Vesuvius progress prizes/_sf/output/canonical_best/pherc0139_4x5x5/atlas/atlas_bake.obj'
VOL = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/'
       '20250728140407-9.362um-1.2m-113keV-masked.zarr/0')
t0 = time.time()
zyx, uv, F, FT = obj.read_obj(MESH)
du = 2.0 / obj.uv_scale(zyx, uv, F, FT)
X, lab, uv0 = obj.rasterise(zyx, uv, F, FT, du)
n = obj.chart_normals(X, lab)
ok = (lab >= 0) & np.all(np.isfinite(n), -1)
n = obj.orient(X, n, lab, umbilicus_xy(f'{R}/sf_axis_0139.csv'))
after = {}
for lb, p in (('r2', f'{R}/atlas_bake_recto.obj'), ('r2_min_snr2', f'{R}/atlas_bake_recto_snr2.obj')):
    z1, uv1, F1, FT1 = obj.read_obj(p)
    X1, lab1, _ = obj.rasterise(z1, uv1, F1, FT1, du)
    ok &= lab1 == lab
    after[lb] = X1
I, J = np.nonzero(ok)
print('nodes', len(I), flush=True)
L = lab[I, J]
off_r2 = np.sum((after['r2'][I, J] - X[I, J]) * n[I, J], -1)
order = np.argsort(L, kind='stable'); ls, st = np.unique(L[order], return_index=True)
med = np.zeros(L.max() + 1)
for l_, s in zip(ls, np.split(off_r2[order], st[1:])):
    med[l_] = np.median(s)
sub = np.sort(np.random.default_rng(0).choice(len(I), 200000, replace=False))
I, J, L, off_r2 = I[sub], J[sub], L[sub], off_r2[sub]
P, N = X[I, J], n[I, J]
Pa = {lb: X1[I, J] for lb, X1 in after.items()}
del X, n, lab, ok, after
vol, _ = tool.open_volume(VOL, f'{R}/ct')
f0 = vol.fetched
res = {}
pos0, _ = profiles(vol, P, N)
good = np.isfinite(pos0)
cases = {lb: Q for lb, Q in Pa.items()}
cases['per_chart_median_r2_offset'] = P + med[L][:, None] * N
for c in (-2.0, -1.0, -0.5, 0.5, 1.0, float(np.round(np.mean(off_r2), 2)), 1.5, 2.0, 2.5, 3.0):
    cases[f'constant_{c:+g}'] = P + c * N
pos = {}
for k, Q in cases.items():
    pos[k], _ = profiles(vol, Q, N)
    good &= np.isfinite(pos[k])
    print(k, round(time.time() - t0), flush=True)
res = {'what': __doc__, 'subsample_nodes': int(good.sum()), 'seed': 0, 'scrollfiesta_mesh': share(pos0, good),
       'mean_r2_offset_on_subsample': float(np.mean(off_r2)),
       'cases': {k: share(p, good) for k, p in pos.items()}, 'ct_chunks_fetched': vol.fetched - f0,
       'seconds': round(time.time() - t0, 1)}
try:
    import psutil; res['peak_rss_mb'] = round(psutil.Process().memory_info().peak_wset / 2 ** 20)
except Exception:
    pass
json.dump(res, open(f'{R}/demo_sf0139_baselines.json', 'w'), indent=1)
print(json.dumps(res, indent=1), flush=True)
