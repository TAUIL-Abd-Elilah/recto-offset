"""recto-offset on an OBJ mesh with texture coordinates (ScrollFiesta's export): rasterise by UV, correct, move vertices.

  python -m recto.obj in.obj out.obj --volume <zarr level-0 URL or local folder> [--umbilicus axis.csv or .json] \
      [--ckpt weights/recto_offset_r2.pt] [--min-snr 0] [--iterations 1] [--max-shift 8] [--step 2] [--order zyx] \
      [--tifxyz grid.tifxyz] [--dry-run]

Input as ScrollFiesta writes it (src/common/obj_io.h): "v z y x" in level-0 voxels of the scan, "vt u v" in
voxel-sized pixels of the flattened surface, faces "f a/a b/b c/c" (polygons are split into fans). --order xyz reads
"v x y z" (the VC3D convention) instead.

1. The UV scale (median 3D edge length per UV unit) is measured and the faces are rasterised by their UVs onto a
   grid with --step voxels between nodes (2, the model's training spacing and recto.apply's internal grid). A node
   inside a face gets the barycentric 3D position; a node covered by two UV charts (or by a fold, two positions more
   than 1 voxel apart) is left out.
2. Normals from the grid, with neighbours from the same UV chart (connected faces) only; each chart is oriented away
   from the scroll axis (--umbilicus, else the centre of the masked scan as in recto.apply).
3. The correction of recto.apply on that grid (recto.apply.correct: same sampling, model, smoothing, clip, gate and
   passes), in strips of --rows nodes along the grid's long axis, each read with 64 nodes of context.
4. Every vertex moves by the displacement (offset x normal) interpolated bilinearly at its UV from the nodes of its own
   chart; the OBJ is written back with only the "v" lines of moved vertices changed.
Also writes <out>.offsets.npz with, per vertex in file order: offset (voxels along the outward normal; NaN where no
node of its chart is near), confidence (1 / predicted Laplace scale), moved (passes in which it moved) and
displacement (in the file's coordinate order). --tifxyz also writes the rasterised grid as a tifxyz (x/y/z.tif, -1
outside the mesh, meta scale 1/step), which recto.apply reads too (for one-chart meshes only: recto.apply orients the
whole grid at once and takes normals across chart borders). --dry-run stops after rasterising and prints the grid
size and the CT chunks a run would read (no CT access).

On an atlas the 2D part of the model also sees neighbouring charts across UV gaps narrower than its tiles; charts
packed edge to edge are a regime it was never tested in."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tifffile
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import apply as tool

BOX = 16                                         # faces whose UV box spans up to BOX nodes per axis go vectorised


def read_obj(path: str, order: str = 'zyx'):
    """-> vertices (nv, 3) in z, y, x order, uv (nvt, 2), faces (nf, 3) vertex indices, face uv indices (nf, 3)."""
    V, T, F, FT = [], [], [], []
    for line in open(path):
        if line.startswith('v '):
            V.append(line.split()[1:4])
        elif line.startswith('vt '):
            T.append(line.split()[1:3])
        elif line.startswith('f '):
            c = [t.split('/') for t in line.split()[1:]]
            vi = [int(t[0]) for t in c]
            ti = [int(t[1]) if len(t) > 1 and t[1] else int(t[0]) for t in c]
            for k in range(1, len(c) - 1):
                F.append((vi[0], vi[k], vi[k + 1])); FT.append((ti[0], ti[k], ti[k + 1]))
    if not T:
        raise SystemExit(f'{path}: no texture coordinates (vt); the UVs are what the grid is built from')
    V, T = np.array(V, float), np.array(T, float)
    F, FT = np.array(F, np.int64).reshape(-1, 3), np.array(FT, np.int64).reshape(-1, 3)
    F = np.where(F < 0, len(V) + F, F - 1); FT = np.where(FT < 0, len(T) + FT, FT - 1)
    return (V[:, ::-1].copy() if order == 'xyz' else V), T, F, FT


def uv_scale(zyx, uv, F, FT) -> float:
    """Median 3D edge length per UV unit."""
    r = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        d3 = np.linalg.norm(zyx[F[:, a]] - zyx[F[:, b]], axis=1); d2 = np.linalg.norm(uv[FT[:, a]] - uv[FT[:, b]], axis=1)
        r.append(d3[d2 > 1e-9] / d2[d2 > 1e-9])
    return float(np.median(np.concatenate(r)))


def charts(nvt: int, FT) -> np.ndarray:
    """Chart label per uv index: connected components of faces sharing texture coordinates."""
    e = np.concatenate([FT[:, [0, 1]], FT[:, [1, 2]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nvt, nvt))
    return connected_components(g, directed=False)[1]


def rasterise(zyx, uv, F, FT, du: float):
    """Grid node (i, j) sits at uv0 + (i, j) * du. Returns positions (H, W, 3; NaN outside), chart label (H, W; -1
    outside or ambiguous), uv0."""
    used = np.unique(FT)
    uv0 = np.floor(uv[used].min(0) / du) * du
    g = (uv[FT] - uv0) / du                                         # (nf, 3, 2) in node units
    P = zyx[F]                                                      # (nf, 3, 3)
    lab_f = charts(len(uv), FT)[FT[:, 0]]
    H, W = (np.floor(g.reshape(-1, 2).max(0)).astype(int) + 2)
    pos = np.full((H, W, 3), np.nan); lab = np.full((H, W), -1, np.int64); bad = np.zeros((H, W), bool)
    lo = np.ceil(g.min(1) - 1e-9).astype(int); hi = np.floor(g.max(1) + 1e-9).astype(int)
    span = hi - lo + 1
    a, e1, e2 = g[:, 0], g[:, 1] - g[:, 0], g[:, 2] - g[:, 0]
    d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    den = d00 * d11 - d01 ** 2
    live = np.abs(den) > 1e-12

    def put(fi, I, J):
        p = np.stack([I, J], -1) - a[fi]
        d20, d21 = (p * e1[fi]).sum(1), (p * e2[fi]).sum(1)
        v = (d11[fi] * d20 - d01[fi] * d21) / den[fi]; w = (d00[fi] * d21 - d01[fi] * d20) / den[fi]
        ins = (v >= -1e-9) & (w >= -1e-9) & (v + w <= 1 + 1e-9)
        fi, I, J, v, w = fi[ins], I[ins], J[ins], v[ins], w[ins]
        X = (1 - v - w)[:, None] * P[fi, 0] + v[:, None] * P[fi, 1] + w[:, None] * P[fi, 2]
        L = lab_f[fi]
        new = lab[I, J] < 0
        pos[I[new], J[new]] = X[new]; lab[I[new], J[new]] = L[new]
        # another chart, or the same chart folded over itself, at the same node: ambiguous
        np.logical_or.at(bad, (I, J), (lab[I, J] != L) | (np.linalg.norm(pos[I, J] - X, axis=1) > 1.0))

    small = live & (span[:, 0] <= BOX) & (span[:, 1] <= BOX)
    for oi in range(BOX):
        for oj in range(BOX):
            fi = np.flatnonzero(small & (oi < span[:, 0]) & (oj < span[:, 1]))
            if len(fi):
                put(fi, lo[fi, 0] + oi, lo[fi, 1] + oj)
    for f in np.flatnonzero(live & ~small):
        I, J = np.meshgrid(np.arange(lo[f, 0], hi[f, 0] + 1), np.arange(lo[f, 1], hi[f, 1] + 1), indexing='ij')
        put(np.full(I.size, f), I.ravel(), J.ravel())
    lab[bad] = -1; pos[bad] = np.nan
    return pos, lab, uv0


def chart_normals(X, lab):
    """Unit normals du x dv from central differences, one-sided where a neighbour is outside the node's chart."""
    H, W = lab.shape
    lp = np.pad(lab, 1, constant_values=-1); Xp = np.pad(X, ((1, 1), (1, 1), (0, 0)), constant_values=np.nan)
    d = []
    for di, dj in ((1, 0), (0, 1)):
        nxt, prv = Xp[1 + di:H + 1 + di, 1 + dj:W + 1 + dj], Xp[1 - di:H + 1 - di, 1 - dj:W + 1 - dj]
        sf = (lp[1 + di:H + 1 + di, 1 + dj:W + 1 + dj] == lab)[..., None]
        sb = (lp[1 - di:H + 1 - di, 1 - dj:W + 1 - dj] == lab)[..., None]
        d.append(np.where(sf & sb, (nxt - prv) / 2, np.where(sf, nxt - X, np.where(sb, X - prv, np.nan))))
    n = np.cross(d[0], d[1])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    n[lab < 0] = np.nan
    return n


def orient(X, n, lab, centre):
    """Flip each chart so its median normal points away from the axis; centre(z (m,)) -> (m, 2)."""
    ok = (lab >= 0) & np.all(np.isfinite(n), -1)
    radial = np.concatenate([np.zeros((int(ok.sum()), 1)), X[ok][:, 1:] - centre(X[ok][:, 0])], -1)
    dot, L = np.sum(radial * n[ok], -1), lab[ok]
    order = np.argsort(L, kind='stable')
    ls, starts = np.unique(L[order], return_index=True)
    flip = np.zeros(int(lab.max()) + 2, bool)
    for l_, s in zip(ls, np.split(dot[order], starts[1:])):
        flip[l_] = np.nanmedian(s) < 0
    return np.where(flip[lab][..., None] & (lab >= 0)[..., None], -n, n)


def centres(volume, shape, umbilicus, cache, axis_cache):
    c = tool.axis_centre(volume, shape, umbilicus, cache, axis_cache)
    if umbilicus:
        from .stacks import umbilicus_xy
        return umbilicus_xy(umbilicus)
    def per_node(z):                                               # the coarse axis per level-4 slice
        zi = (np.asarray(z) // 16).astype(int)
        u, inv = np.unique(zi, return_inverse=True)
        cs = [c(v * 16 + 8) for v in u]
        if any(v is None for v in cs):
            raise SystemExit('no scan at some heights of the mesh to estimate the axis: pass --umbilicus')
        return np.array(cs, float)[inv]
    return per_node


def chunks_needed(X, n, ok, K: float, chunk: int = 128) -> set:
    """Chunk keys read when sampling -K..K voxels along the normals (both signs, so orientation does not matter)."""
    ks = np.arange(-np.ceil(K), np.ceil(K) + 1)
    P, N = X[ok], np.nan_to_num(n[ok])
    keys = set()
    for s in range(0, len(P), 65536):
        q = P[s:s + 65536][None] + ks[:, None, None] * N[s:s + 65536][None]
        f = np.floor(q.reshape(-1, 3)).astype(np.int64)
        lo, hi = f // chunk + (1 << 20), (f + 1) // chunk + (1 << 20)
        for cz in (lo[:, 0], hi[:, 0]):                            # all eight trilinear corners
            for cy in (lo[:, 1], hi[:, 1]):
                for cx in (lo[:, 2], hi[:, 2]):
                    keys |= set(np.unique((cz << 42) | (cy << 21) | cx).tolist())
    m, o = (1 << 21) - 1, 1 << 20
    return {((k >> 42) - o, ((k >> 21) & m) - o, (k & m) - o) for k in keys}


def vertex_maps(uv, lab_vt, uv0, du, lab, fields):
    """Bilinear interpolation of fields (list of (H, W[, c]) arrays) at each uv, from nodes of the uv's own chart; the
    nearest such node within 2 nodes where no corner qualifies. Returns the interpolated fields and a found mask."""
    H, W = lab.shape
    g = (uv - uv0) / du
    i0, j0 = np.floor(g[:, 0]).astype(int), np.floor(g[:, 1]).astype(int)
    ti, tj = g[:, 0] - i0, g[:, 1] - j0
    acc = [np.zeros((len(uv),) + f.shape[2:]) for f in fields]; wsum = np.zeros(len(uv))
    for di, dj, w in ((0, 0, (1 - ti) * (1 - tj)), (1, 0, ti * (1 - tj)), (0, 1, (1 - ti) * tj), (1, 1, ti * tj)):
        I, J = np.clip(i0 + di, 0, H - 1), np.clip(j0 + dj, 0, W - 1)
        m = (i0 + di >= 0) & (i0 + di < H) & (j0 + dj >= 0) & (j0 + dj < W) & (lab[I, J] == lab_vt) & (lab_vt >= 0)
        wm = w * m
        for k, f in enumerate(fields):
            acc[k] += (wm.reshape(-1, *([1] * (f.ndim - 2)))) * np.nan_to_num(f[I, J])
        wsum += wm
    found = wsum > 1e-9
    out = [a / np.maximum(wsum, 1e-9).reshape(-1, *([1] * (a.ndim - 1))) for a in acc]
    miss = np.flatnonzero(~found & (lab_vt >= 0))
    if len(miss):
        best = np.full(len(miss), np.inf); bi = np.zeros(len(miss), int); bj = np.zeros(len(miss), int)
        for di in range(-2, 4):
            for dj in range(-2, 4):
                I, J = i0[miss] + di, j0[miss] + dj
                inb = (I >= 0) & (I < H) & (J >= 0) & (J < W)
                Ic, Jc = np.clip(I, 0, H - 1), np.clip(J, 0, W - 1)
                d = (g[miss, 0] - I) ** 2 + (g[miss, 1] - J) ** 2
                m = inb & (lab[Ic, Jc] == lab_vt[miss]) & (d < best)
                best[m], bi[m], bj[m] = d[m], Ic[m], Jc[m]
        hit = np.isfinite(best)
        for k, f in enumerate(fields):
            out[k][miss[hit]] = np.nan_to_num(f[bi[hit], bj[hit]])
        found[miss[hit]] = True
    return out, found


def correct_strips(net, vol, X, n, ok, R, a, margin=64):
    """recto.apply.correct over strips of a.rows nodes along the grid's long axis (each read with a margin of context),
    so a large atlas fits in memory."""
    t = ok.shape[1] > ok.shape[0]
    if t:
        X, n, ok = X.swapaxes(0, 1), n.swapaxes(0, 1), ok.swapaxes(0, 1)
    H = ok.shape[0]
    off, conf = np.zeros(ok.shape), np.zeros(ok.shape)
    moved, okr = np.zeros(ok.shape, np.uint8), np.zeros(ok.shape, bool)
    for s in range(0, H, a.rows):
        lo, hi, e = max(0, s - margin), min(H, s + a.rows + margin), min(H, s + a.rows)
        if not ok[s:e].any():
            continue
        r = tool.correct(net, vol, np.nan_to_num(X[lo:hi]), n[lo:hi], ok[lo:hi], R, a.max_shift, a.min_confidence,
                         a.min_snr, a.iterations, a.device)
        for full, part in zip((off, conf, moved, okr), r):
            full[s:e] = part[s - lo:e - lo]
    return tuple(v.swapaxes(0, 1) if t else v for v in (off, conf, moved, okr))


def write_tifxyz(d, X, lab, step, source):
    os.makedirs(d, exist_ok=True)
    for k, ax in enumerate('zyx'):
        tifffile.imwrite(os.path.join(d, f'{ax}.tif'), np.where(lab >= 0, X[..., k], -1).astype(np.float32))
    json.dump({'format': 'tifxyz', 'type': 'seg', 'scale': [1.0 / step, 1.0 / step], 'uuid': os.path.basename(
        os.path.normpath(d)), 'source': os.path.basename(source)}, open(os.path.join(d, 'meta.json'), 'w'), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mesh'); ap.add_argument('out')
    ap.add_argument('--volume', help='zarr level-0 URL or local folder (not needed with --dry-run)')
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--ckpt', default=os.path.join(here, 'weights', 'recto_offset_r2.pt'))
    ap.add_argument('--umbilicus', help='scroll axis: umbilicus.json, or a CSV of z,y,x rows (ScrollFiesta axis.csv)')
    ap.add_argument('--cache'); ap.add_argument('--axis-cache')
    ap.add_argument('--order', choices=['zyx', 'xyz'], default='zyx', help='coordinate order of the "v" lines')
    ap.add_argument('--step', type=float, default=2.0, help='node spacing of the grid, voxels')
    ap.add_argument('--max-shift', type=float, default=8.0)
    ap.add_argument('--min-confidence', type=float, default=0.0)
    ap.add_argument('--min-snr', type=float, default=0.0)
    ap.add_argument('--iterations', type=int, default=1)
    ap.add_argument('--rows', type=int, default=2048, help='grid nodes per strip along the long axis (memory)')
    ap.add_argument('--tifxyz', help='also write the rasterised grid as a tifxyz folder')
    ap.add_argument('--dry-run', action='store_true', help='rasterise only; print the grid and the CT chunks needed')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    zyx, uv, F, FT = read_obj(a.mesh, a.order)
    scale = uv_scale(zyx, uv, F, FT)
    du = a.step / scale
    X, lab, uv0 = rasterise(zyx, uv, F, FT, du)
    n = chart_normals(X, lab)
    ok = (lab >= 0) & np.all(np.isfinite(n), -1)
    info = {'vertices': len(zyx), 'faces': len(F), 'uv_scale_vox_per_unit': round(scale, 4), 'grid': list(lab.shape),
            'nodes': int(ok.sum()), 'charts': int(len(np.unique(lab[lab >= 0])))}
    if a.tifxyz:
        write_tifxyz(a.tifxyz, X, lab, a.step, a.mesh)
    if a.dry_run:
        net_R = torch.load(a.ckpt, map_location='cpu')['args']['R']
        reach = net_R + (a.max_shift if a.iterations > 1 else 0)
        keys = chunks_needed(X, n, ok, reach)
        lo, hi = np.nanmin(X[ok], 0), np.nanmax(X[ok], 0)
        info.update({'reach_vox': reach, 'chunks_128': len(keys), 'bytes': len(keys) * 128 ** 3,
                     'bbox_zyx': [[round(float(v)) for v in lo], [round(float(v)) for v in hi]]})
        print(json.dumps(info), flush=True)
        return info
    if not a.volume:
        raise SystemExit('--volume is required unless --dry-run')
    net, ck = tool.load_net(a.ckpt, a.device)
    vol, shape = tool.open_volume(a.volume, a.cache)
    n = orient(X, n, lab, centres(a.volume, shape, a.umbilicus, vol.cache, a.axis_cache))
    off, conf, moved, ok = correct_strips(net, vol, X, n, ok, ck['args']['R'], a)
    lab = np.where(ok, lab, -1)
    D = off[..., None] * np.nan_to_num(n)
    lab_vt = charts(len(uv), FT)
    (d_vt, o_vt, c_vt, m_vt), found = vertex_maps(uv, lab_vt, uv0, du, lab,
                                                  [D, off, np.where(ok, conf, 0.0), moved.astype(float)])
    # vertices take the mean over the texture coordinates they appear with (one, unless the mesh has UV seams)
    corner_v, corner_t = F.ravel(), FT.ravel()
    pairs = np.unique(np.stack([corner_v, corner_t], 1), axis=0)
    pv, pt = pairs[:, 0], pairs[:, 1]
    fv = found[pt].astype(float)
    cnt = np.bincount(pv, weights=fv, minlength=len(zyx))
    mean = lambda x: np.stack([np.bincount(pv, weights=fv * x[pt][:, k], minlength=len(zyx)) for k in range(x.shape[1])],  # noqa: E731
                              1) / np.maximum(cnt, 1)[:, None]
    disp = mean(d_vt)
    off_v, conf_v = mean(o_vt[:, None])[:, 0], mean(c_vt[:, None])[:, 0]
    moved_v = np.round(mean(m_vt[:, None])[:, 0]).astype(np.uint8)
    hit = cnt > 0
    off_v[~hit] = np.nan; conf_v[~hit] = np.nan; disp[~hit] = 0
    new = zyx + disp
    # write back: only the "v" lines of vertices that moved change
    file_disp = disp[:, ::-1] if a.order == 'xyz' else disp
    file_new = new[:, ::-1] if a.order == 'xyz' else new
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    vi = 0
    with open(a.mesh, newline='') as src, open(a.out, 'w', newline='') as dst:
        for line in src:
            if line.startswith('v '):
                if np.any(file_disp[vi] != 0):
                    t = line.split()
                    eol = '\r\n' if line.endswith('\r\n') else '\n'
                    line = 'v ' + ' '.join(f'{c:.3f}' for c in file_new[vi]) + ''.join(' ' + s for s in t[4:]) + eol
                vi += 1
            dst.write(line)
    stem = os.path.splitext(a.out)[0]
    np.savez_compressed(stem + '.offsets.npz', offset=off_v.astype(np.float32), confidence=conf_v.astype(np.float32),
                        moved=moved_v, displacement=file_disp.astype(np.float32))
    info.update({'ckpt': os.path.basename(a.ckpt), 'step': ck.get('step'), 'max_shift': a.max_shift, 'min_snr': a.min_snr,
                 'iterations': a.iterations, 'vertices_with_offset': int(hit.sum()),
                 'median_abs_offset': float(np.nanmedian(np.abs(off_v[hit]))) if hit.any() else None,
                 'moved_share': float((moved_v[hit] > 0).mean()) if hit.any() else None})
    print(json.dumps(info), flush=True)
    return info


if __name__ == '__main__':
    main()
