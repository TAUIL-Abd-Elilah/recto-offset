"""An OBJ mesh (ScrollFiesta's export) before and after recto.obj, measured as in recto.demo_meshes and recto.demo_null.

  python -m recto.demo_obj in.obj --after out.obj [--after out_snr2.obj] --volume <zarr level-0 URL or folder> \
      --cache <dir> --umbilicus axis.csv [--node-level] [--ckpt weights/recto_offset_r2.pt] --out demo.json --fig demo.png

Descriptive (amendment 3): CT features are not ground truth.
1. The input OBJ is rasterised exactly as recto.obj does it (UV scale, 2-voxel nodes, charts, normals oriented away
   from the axis per chart). Each corrected OBJ has the same UVs, so it is rasterised onto the same grid (same UV
   origin and node spacing): every node has a position X before and X' after. Its offset is (X' - X) . n along the
   normal n of the input grid. Only nodes valid (same chart) in every grid are measured.
2. At every node, the CT is sampled from -6 to +6 voxels along n around X (before) and X' (after), smoothed by 1
   voxel, and the brightest layer is taken (recto.demo_meshes.brightest). Measure: share of nodes where it lies
   within 2 voxels of the surface (also within 1). Chance: every node moved by its own uniform random shift in +-6
   voxels along n, 5 seeds (as recto.demo_null). Also by chart size (nodes per chart) and by distance from the axis.
3. --node-level runs the model again on the same grid (recto.obj.correct_strips, the same call recto.obj makes) and
   measures the node offsets themselves (X + offset * n), before they are interpolated onto the OBJ's vertices; the
   --min-snr gates of --gates are applied to those offsets as recto.apply does (exact for one pass, except nodes whose
   offset hit the --max-shift clip while Z x b > max-shift; counted).
The figure shows the CT from -16 to +16 voxels along the normals on one line of nodes through the middle of each of
the --fig-charts largest charts (along the chart's longer grid axis), the input mesh at 0 and the corrected OBJs.
CT chunks come from --cache (fetched if missing; the count is reported)."""
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

from . import apply as tool
from . import obj
from .demo_meshes import brightest
from .stacks import umbilicus_xy

KS = np.arange(-6, 7, dtype=np.float32)
KFIG = np.arange(-16, 17, dtype=np.float32)
SEEDS = 5
NULL = 6.0
SIZES = ((0, 1000), (1000, 2500), (2500, 5000), (5000, None))         # nodes per chart
RADII = ((0, 100), (100, 200), (200, 300), (300, None))               # voxels from the axis


def profiles(vol, P, N, block=50000):
    """Brightest smoothed layer within +-6 voxels along N around P (NaN where a sample is outside the scan) and the
    smoothed CT at the surface."""
    pos = np.full(len(P), np.nan); val = np.full(len(P), np.nan)
    for b in range(0, len(P), block):
        p, nn = P[b:b + block], N[b:b + block]
        pts = p[None] + KS[:, None, None] * nn[None]
        fin = np.all(np.isfinite(pts), (0, 2))
        if fin.any():
            vol.prefetch(vol.keys_for(pts[:, fin].reshape(-1, 3)))
        s = vol.sample(pts)
        g = np.all(np.isfinite(s), 0)
        pos[b:b + block][g] = brightest(s[:, g])
        val[b:b + block][g] = ndimage.gaussian_filter1d(s[:, g], 1.0, axis=0)[6]
    return pos, val


def share(pos, m, w=2):
    return float(np.mean(np.abs(pos[m]) <= w)) if m.any() else None


def summary(off, m):
    a = np.abs(off[m])
    return {'median_abs_offset': float(np.median(a)), 'p90_abs_offset': float(np.percentile(a, 90)),
            'mean_offset': float(np.mean(off[m])), 'moved_share_abs_gt_0.1': float(np.mean(a > 0.1))}


def histogram(pos, m):
    """Share of nodes per brightest-layer position -6..6."""
    return (np.bincount((pos[m] + 6).astype(int), minlength=13) / max(int(m.sum()), 1)).round(4).tolist()


def figure_lines(vol, X, n, lab, ok, sizes, after, k):
    """One line of nodes through the middle of each of the k largest charts: CT image and offsets of each OBJ."""
    out = []
    for big in np.argsort(sizes)[::-1][:k]:
        m = ok & (lab == big)
        ii, jj = np.nonzero(m)
        along_i = (ii.max() - ii.min()) >= (jj.max() - jj.min())
        if along_i:
            c = max(range(int(np.percentile(jj, 40)), int(np.percentile(jj, 60)) + 1), key=lambda j: m[:, j].sum())
            sel = np.flatnonzero(m[:, c]); li = np.arange(sel.min(), sel.max() + 1); lj = np.full(len(li), c)
        else:
            c = max(range(int(np.percentile(ii, 40)), int(np.percentile(ii, 60)) + 1), key=lambda i: m[i].sum())
            sel = np.flatnonzero(m[c]); lj = np.arange(sel.min(), sel.max() + 1); li = np.full(len(lj), c)
        lm = m[li, lj]
        Pl, Nl = np.nan_to_num(X[li, lj]), np.nan_to_num(n[li, lj])
        pts = Pl[None] + KFIG[:, None, None] * Nl[None]
        pts[:, ~lm] = np.nan                                           # nodes off the chart read nothing
        img = vol.sample(pts)
        offs = {lb: np.where(lm, np.sum((np.nan_to_num(X1[li, lj]) - Pl) * Nl, -1), np.nan) for lb, X1 in after.items()}
        out.append({'chart': int(big), 'chart_nodes': int(sizes[big]), 'line': f'{"column" if along_i else "row"} {c}',
                    'nodes_on_line': int(lm.sum()), 'img': img, 'offs': offs, 'lm': lm,
                    'centre_zyx': [round(float(v)) for v in Pl[lm][len(Pl[lm]) // 2]]})
    return out


def plot(lines, step, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(lines), 1, figsize=(9, 1.9 * len(lines) + 0.4), squeeze=False)
    styles = [('c-', 1.2), ('y--', 1.0), ('m:', 1.0)]
    for ax, L in zip(axs[:, 0], lines):
        img, lm = L['img'], L['lm']
        ax.imshow(np.nan_to_num(img), cmap='gray', aspect='auto', extent=[0, img.shape[1] * step, 16.5, -16.5])
        x = np.arange(img.shape[1]) * step
        ax.plot(x, np.where(lm, 0.0, np.nan), 'r-', lw=1, label='ScrollFiesta mesh')
        for (lb, o), (st, lw) in zip(L['offs'].items(), styles):
            ax.plot(x, o, st, lw=lw, label=f'after recto.obj ({lb})')
        ax.set_ylabel('normal (vox)', fontsize=7); ax.tick_params(labelsize=6)
        ax.set_title(f'chart {L["chart"]} ({L["chart_nodes"]} nodes), grid {L["line"]}, centre zyx {L["centre_zyx"]}; '
                     'voxels along the line', fontsize=8)
    axs[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'{title}: CT along the normals of the {len(lines)} largest charts (outward = down)', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mesh')
    ap.add_argument('--after', action='append', default=[], help='a corrected OBJ (repeat)')
    ap.add_argument('--label', action='append', default=[], help='a name per --after')
    ap.add_argument('--volume', required=True); ap.add_argument('--cache')
    ap.add_argument('--umbilicus', required=True, help='axis CSV (z,y,x rows) or umbilicus.json, as given to recto.obj')
    ap.add_argument('--order', choices=['zyx', 'xyz'], default='zyx')
    ap.add_argument('--step', type=float, default=2.0)
    ap.add_argument('--node-level', action='store_true')
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--ckpt', default=os.path.join(here, 'weights', 'recto_offset_r2.pt'))
    ap.add_argument('--gates', type=float, nargs='*', default=[2.0])
    ap.add_argument('--max-shift', type=float, default=8.0); ap.add_argument('--rows', type=int, default=2048)
    ap.add_argument('--title', default='OBJ mesh')
    ap.add_argument('--out', required=True); ap.add_argument('--fig'); ap.add_argument('--fig-charts', type=int, default=4)
    a = ap.parse_args(argv)
    labels = a.label + [os.path.splitext(os.path.basename(p))[0] for p in a.after[len(a.label):]]
    t0 = time.time()

    zyx, uv, F, FT = obj.read_obj(a.mesh, a.order)
    du = a.step / obj.uv_scale(zyx, uv, F, FT)
    X, lab, uv0 = obj.rasterise(zyx, uv, F, FT, du)
    n = obj.chart_normals(X, lab)
    ok = (lab >= 0) & np.all(np.isfinite(n), -1)
    axis = umbilicus_xy(a.umbilicus)
    n = obj.orient(X, n, lab, axis)
    nodes_rasterised, grid = int((lab >= 0).sum()), list(lab.shape)
    del zyx
    after = {}
    for lb, p in zip(labels, a.after):
        z1, uv1, F1, FT1 = obj.read_obj(p, a.order)
        assert np.array_equal(uv1, uv) and np.array_equal(FT1, FT), f'{p}: UVs differ from the input mesh'
        X1, lab1, uv01 = obj.rasterise(z1, uv1, F1, FT1, du)
        assert lab1.shape == lab.shape and np.allclose(uv01, uv0), f'{p}: grid differs'
        ok &= (lab1 == lab)
        after[lb] = X1
        del z1, lab1
    vol, shape = tool.open_volume(a.volume, a.cache)
    fetched0 = vol.fetched

    node, res_extra = {}, {}
    if a.node_level:
        net, ck = tool.load_net(a.ckpt, 'cpu')
        ns = SimpleNamespace(rows=a.rows, max_shift=a.max_shift, min_confidence=0.0, min_snr=0.0, iterations=1,
                             device='cpu')
        off, conf, moved, okr = obj.correct_strips(net, vol, X, n, ok, ck['args']['R'], ns)
        ok &= okr
        node['r2'] = off
        for z in a.gates:
            node[f'r2_min_snr{z:g}'] = np.where(np.abs(off) >= z / np.maximum(conf, 1e-12), off, 0.0)
        res_extra['gate_clip_ambiguous_nodes'] = {
            f'{z:g}': int(np.sum(ok & (np.abs(off) >= a.max_shift - 1e-9) & (z / np.maximum(conf, 1e-12) > a.max_shift)))
            for z in a.gates}
        node_conf = conf
        del net, moved, okr

    # node arrays (valid nodes only); the grids are freed before the CT is read
    I, J = np.nonzero(ok)
    P, N, L = X[I, J], n[I, J], lab[I, J]
    sizes = np.bincount(L)
    S = sizes[L]
    yx = axis(P[:, 0])
    radial = np.concatenate([np.zeros((len(P), 1)), P[:, 1:] - yx], -1)
    Rad = np.linalg.norm(radial[:, 1:], axis=1)
    inward = float(np.mean(np.sum(radial * N, -1) < 0))
    res_off = {lb: np.sum((X1[I, J] - P) * N, -1) for lb, X1 in after.items()}
    tang = {lb: float(np.median(np.linalg.norm(X1[I, J] - P - res_off[lb][:, None] * N, axis=1))) for lb, X1 in after.items()}
    Pa = {lb: X1[I, J] for lb, X1 in after.items()}
    for k, o in node.items():
        res_off['node_' + k] = o[I, J]
    if a.node_level:
        node_scale = 1 / np.maximum(node_conf[I, J], 1e-12)
    lines = figure_lines(vol, X, n, lab, ok, sizes, after, a.fig_charts) if a.fig else []
    del X, n, lab, ok, after, node
    if a.node_level:
        del off, conf, node_conf

    before_pos, before_val = profiles(vol, P, N)
    good = np.isfinite(before_pos)
    res_pos, res_val = {}, {}
    for lb in res_off:
        Q = Pa[lb] if lb in Pa else P + res_off[lb][:, None] * N
        res_pos[lb], res_val[lb] = profiles(vol, Q, N)
        good &= np.isfinite(res_pos[lb])
    del Pa
    nulls = []
    for seed in range(SEEDS):
        u = np.random.default_rng([seed, 0]).uniform(-NULL, NULL, len(P))
        nulls.append(profiles(vol, P + u[:, None] * N, N)[0])

    def null_share(m, w=2):
        v = [share(p, m & np.isfinite(p), w) for p in nulls]
        return {'mean': float(np.mean(v)), 'min': float(min(v)), 'max': float(max(v)), 'per_seed': v}

    on0 = np.abs(before_pos) <= 2
    pooled = {'nodes': int(good.sum()), 'charts': int(len(np.unique(L[good]))), 'before': share(before_pos, good),
              'chance_uniform6': null_share(good), 'before_within1': share(before_pos, good, 1),
              'chance_uniform6_within1': null_share(good, 1)['mean'],
              'ct_at_surface_before': float(np.mean(before_val[good])),
              'brightest_before_median': float(np.median(before_pos[good])),
              'brightest_hist_before': histogram(before_pos, good),
              'brightest_hist_chance_seed0': histogram(nulls[0], good & np.isfinite(nulls[0])),
              'normals_pointing_to_axis_share': inward,
              'distance_from_axis_vox': {'median': float(np.median(Rad[good])), 'min': float(Rad[good].min()),
                                         'max': float(Rad[good].max())}}
    pooled['before_minus_chance_points'] = 100 * (pooled['before'] - pooled['chance_uniform6']['mean'])
    rows = {}
    cb = np.bincount(L[good], weights=on0[good], minlength=len(sizes)); cn = np.bincount(L[good], minlength=len(sizes))
    for lb in res_pos:
        on1 = np.abs(res_pos[lb]) <= 2
        r = rows[lb] = {}
        if lb in tang:
            r['tangential_median_vox'] = tang[lb]
        r.update({'after': share(res_pos[lb], good), 'after_within1': share(res_pos[lb], good, 1),
                  **summary(res_off[lb], good),
                  'ct_at_surface_after': float(np.mean(res_val[lb][good])),
                  'brightest_after_median': float(np.median(res_pos[lb][good])),
                  'brightest_hist_after': histogram(res_pos[lb], good),
                  'on_to_off': float(np.mean(on0[good] & ~on1[good])), 'off_to_on': float(np.mean(~on0[good] & on1[good])),
                  'on_to_on': float(np.mean(on0[good] & on1[good]))})
        r['after_minus_before_points'] = 100 * (r['after'] - pooled['before'])
        r['after_minus_chance_points'] = 100 * (r['after'] - pooled['chance_uniform6']['mean'])
        ca = np.bincount(L[good], weights=on1[good], minlength=len(sizes))
        live = cn >= 100
        d = 100 * (ca[live] - cb[live]) / cn[live]
        r['charts_ge100_nodes'] = {'charts': int(live.sum()), 'worse_by_5_points_or_more': int(np.sum(d <= -5)),
                                   'better_by_5_points_or_more': int(np.sum(d >= 5)),
                                   'median_change_points': float(np.median(d)),
                                   'nodes_in_charts_worse_by_5': int(cn[live][d <= -5].sum())}
    if a.node_level:
        res_extra['moved_share'] = {k[5:]: float(np.mean(np.abs(res_off[k][good]) > 0)) for k in res_off if k.startswith('node_')}
        res_extra['median_predicted_scale_vox'] = float(np.median(node_scale[good]))

    def split(key, bounds, v):
        out = {}
        for lo, hi in bounds:
            m = good & (v >= lo) & ((v < hi) if hi else True)
            if not m.any():
                continue
            b = {'nodes': int(m.sum()), 'charts': int(len(np.unique(L[m]))), 'before': share(before_pos, m),
                 'chance_uniform6_mean': null_share(m)['mean']}
            for lb in res_pos:
                b[lb] = share(res_pos[lb], m); b[lb + '_median_abs_offset'] = float(np.median(np.abs(res_off[lb][m])))
            out[f'{lo}-{hi if hi else "max"} {key}'] = b
        return out

    res = {'mesh': a.mesh, 'after': dict(zip(labels, a.after)), 'volume': a.volume, 'umbilicus': a.umbilicus,
           'grid': grid, 'step_vox': a.step,
           'uv_scale_vox_per_unit': float(a.step / du), 'nodes_rasterised': nodes_rasterised,
           'measure': 'share of nodes whose brightest smoothed layer within +-6 voxels along the outward normal lies '
                      'within 2 voxels of the surface (recto.demo_meshes.brightest); nodes every 2 voxels',
           'chance': f'each node moved by its own uniform random shift in +-{NULL:g} voxels along its normal, {SEEDS} '
                     'seeds (as recto.demo_null)',
           'pooled': pooled, 'corrected': rows, 'by_chart_size': split('nodes/chart', SIZES, S),
           'by_distance_from_axis': split('vox from axis', RADII, Rad)}
    if a.node_level:
        res['node_level'] = {'ckpt': os.path.basename(a.ckpt), **res_extra}
    for lb, p in zip(labels, a.after):
        z = os.path.splitext(p)[0] + '.offsets.npz'
        if os.path.exists(z):
            d = np.load(z); o = d['offset']; h = np.isfinite(o)
            rows[lb]['vertices'] = {'with_offset': int(h.sum()), 'of': int(len(o)),
                                    'median_abs_offset': float(np.median(np.abs(o[h]))),
                                    'p90_abs_offset': float(np.percentile(np.abs(o[h]), 90)),
                                    'moved_share': float(np.mean(d['moved'][h] > 0))}
    if lines:
        plot(lines, a.step, a.title, a.fig)
        res['figure'] = {'path': a.fig, 'panels': [{k: v for k, v in L_.items() if k not in ('img', 'offs', 'lm')}
                                                   for L_ in lines]}
    res['ct_chunks_fetched_here'] = vol.fetched - fetched0
    res['seconds'] = round(time.time() - t0, 1)
    try:
        import psutil
        res['peak_rss_mb'] = round(psutil.Process().memory_info().peak_wset / 2 ** 20)
    except Exception:
        pass
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({'pooled': {k: v for k, v in pooled.items() if not isinstance(v, (dict, list))},
                      'chance': pooled['chance_uniform6']['mean'],
                      'corrected': {k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
                                    for k, v in rows.items()}}), flush=True)
    return res


if __name__ == '__main__':
    main()
