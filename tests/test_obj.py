"""recto.obj: UV rasterisation, and an end-to-end run on a synthetic OBJ (ScrollFiesta conventions: v z y x, vt in
voxel units, f a/a b/b c/c) pushed off a sheet of the synthetic scan used in test_apply."""
import json
import os

import numpy as np
from test_apply import write_volume

from recto import obj


def test_rasterise_is_exact_and_drops_overlaps():
    # chart A: the UV square [0, 10]^2, mapped affinely; chart B: a triangle overlapping A's corner, elsewhere in 3D
    uv = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [15, 5], [5, 15]], float)
    affine = lambda u, v: np.stack([100 + v, 200 + u, 300 + 0.5 * u], -1)  # noqa: E731
    zyx = np.concatenate([affine(uv[:4, 0], uv[:4, 1]), [[500, 500, 500], [500, 510, 500], [510, 500, 500]]])
    F = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
    X, lab, uv0 = obj.rasterise(zyx, uv, F, F, 1.0)
    assert np.allclose(uv0, 0) and np.allclose(X[2, 3], affine(2.0, 3.0))
    assert lab[2, 3] >= 0 and lab[12, 6] >= 0 and lab[2, 3] != lab[12, 6]       # only A, only B
    assert lab[7, 6] == -1 and np.all(np.isnan(X[7, 6]))                         # covered by both: left out
    n = obj.chart_normals(X, lab)
    e = np.cross([0, 1, 0.5], [1, 0, 0]); e /= np.linalg.norm(e)                # d/du x d/dv of the affine map
    assert np.allclose(n[2, 3], e, atol=1e-6)


def write_plane_obj(path, x_plane, order='zyx', n=9, s=10.0):
    gz, gy = np.meshgrid(20 + s * np.arange(n), 20 + s * np.arange(n), indexing='ij')
    lines = ['# synthetic plane\n']
    for z, y in zip(gz.ravel(), gy.ravel()):
        lines.append('v %.2f %.2f %.2f\n' % ((z, y, x_plane) if order == 'zyx' else (x_plane, y, z)))
    for z, y in zip(gz.ravel(), gy.ravel()):
        lines.append('vt %.4f %.4f\n' % (y - 20, z - 20))                      # u along y, v along z, voxel units
    idx = np.arange(n * n).reshape(n, n) + 1
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            lines.append(f'f {a}/{a} {b}/{b} {c}/{c}\n'); lines.append(f'f {b}/{b} {d}/{d} {c}/{c}\n')
    open(path, 'w').writelines(lines)
    return lines


def test_obj_moves_a_pushed_mesh_back(tmp_path):
    vol = str(tmp_path / 'vol.zarr')
    sheet = write_volume(vol)
    umb = tmp_path / 'umb.json'
    json.dump({'control_points': [{'z': 0, 'y': 64, 'x': -500}, {'z': 200, 'y': 64, 'x': -500}]}, open(umb, 'w'))
    res = {}
    for order in ('zyx', 'xyz'):
        src, out = str(tmp_path / f'in_{order}.obj'), str(tmp_path / f'out_{order}.obj')
        lines = write_plane_obj(src, sheet + 4.0, order)
        info = obj.main([src, out, '--volume', vol, '--umbilicus', str(umb), '--device', 'cpu', '--order', order])
        new = open(out).readlines()
        assert len(new) == len(lines)
        assert [ln for ln in new if not ln.startswith('v ')] == [ln for ln in lines if not ln.startswith('v ')]
        V = np.array([ln.split()[1:4] for ln in new if ln.startswith('v ')], float)
        res[order] = V if order == 'zyx' else V[:, ::-1]
        side = np.load(os.path.splitext(out)[0] + '.offsets.npz')
        assert np.isfinite(side['offset']).mean() > 0.9 and info['nodes'] > 1000
    V = res['zyx']
    assert np.median(np.abs(V[:, 2] - sheet)) < 4.0                  # closer to the sheet than it started
    assert np.abs(V[:, :2] - V[:, :2].round()).max() < 0.01           # moved along the normal (x) only
    assert np.allclose(res['zyx'], res['xyz'], atol=1e-3)             # the coordinate order is a file convention


def test_strips_stitch_back_in_place(monkeypatch):
    calls = []

    def fake(net, vol, fz, n, ok, R, *args):
        calls.append(fz.shape[0])
        return fz[..., 0].copy(), fz[..., 1].copy(), ok.astype(np.uint8), ok.copy()
    monkeypatch.setattr(obj.tool, 'correct', fake)
    H, W = 30, 500                                                    # long axis second: strips run along it
    gi, gj = np.meshgrid(np.arange(H, dtype=float), np.arange(W, dtype=float), indexing='ij')
    X = np.stack([gi, gj, np.zeros((H, W))], -1)
    ok = np.ones((H, W), bool); ok[:, 128:192] = False
    a = type('A', (), {'rows': 64, 'max_shift': 8.0, 'min_confidence': 0.0, 'min_snr': 0.0, 'iterations': 1,
                       'device': 'cpu'})
    off, conf, moved, okr = obj.correct_strips(None, None, X, np.zeros_like(X), ok, 12, a, margin=8)
    assert np.array_equal(off[ok], X[..., 0][ok]) and np.array_equal(conf[ok], X[..., 1][ok]) and np.array_equal(okr, ok)
    assert max(calls) == 64 + 2 * 8 and len(calls) == 7               # 8 strips of 64, the one inside the gap skipped


def test_dry_run_counts_chunks(tmp_path):
    src = str(tmp_path / 'p.obj'); write_plane_obj(src, 78.0)
    info = obj.main([src, str(tmp_path / 'o.obj'), '--dry-run'])
    assert info['chunks_128'] == 1 and info['bytes'] == 128 ** 3 and abs(info['uv_scale_vox_per_unit'] - 1) < 1e-6
