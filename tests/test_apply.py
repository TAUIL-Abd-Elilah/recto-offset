"""End-to-end check of recto.apply on a synthetic local volume: flat sheets, a mesh pushed off one of them."""
import json
import os

import numpy as np
import tifffile

from recto import apply

C = 128


def write_volume(root, shape=(128, 128, 256), pitch=14, first=60):
    os.makedirs(root, exist_ok=True)
    json.dump({'shape': list(shape), 'chunks': [C, C, C], 'dtype': '|u1', 'compressor': None, 'zarr_format': 2,
               'dimension_separator': '/', 'fill_value': 0, 'order': 'C', 'filters': None},
              open(os.path.join(root, '.zarray'), 'w'))
    x = np.arange(shape[2], dtype=np.float32)
    prof = 60 + np.zeros_like(x)
    for s in range(first, shape[2] - 10, pitch):              # sheets: recto surface at x = s, bright 1.5 vox outward
        prof += 120 * np.exp(-0.5 * ((x - s - 1.5) / 1.6) ** 2)
    vol = np.broadcast_to(np.clip(prof, 0, 255).astype(np.uint8), shape)
    rng = np.random.default_rng(0)
    vol = np.clip(vol + rng.normal(0, 4, shape), 0, 255).astype(np.uint8)
    for cx in range(shape[2] // C):
        d = os.path.join(root, '0', '0'); os.makedirs(d, exist_ok=True)
        vol[:, :, cx * C:(cx + 1) * C].tofile(os.path.join(d, str(cx)))
    return first + pitch                                        # a sheet in the middle


def write_mesh(d, x_plane, n=9):
    os.makedirs(d, exist_ok=True)
    gz, gy = np.meshgrid(20 + 10.0 * np.arange(n), 20 + 10.0 * np.arange(n), indexing='ij')
    tifffile.imwrite(os.path.join(d, 'z.tif'), gz.astype(np.float32))
    tifffile.imwrite(os.path.join(d, 'y.tif'), gy.astype(np.float32))
    tifffile.imwrite(os.path.join(d, 'x.tif'), np.full_like(gz, x_plane, dtype=np.float32))
    json.dump({'format': 'tifxyz', 'scale': [0.1, 0.1], 'type': 'seg', 'uuid': 'synthetic'}, open(os.path.join(d, 'meta.json'), 'w'))


def test_apply_moves_a_pushed_mesh_back_towards_its_sheet(tmp_path):
    vol = str(tmp_path / 'vol.zarr')
    sheet = write_volume(vol)
    umb = tmp_path / 'umb.json'
    json.dump({'control_points': [{'z': 0, 'y': 64, 'x': -500}, {'z': 200, 'y': 64, 'x': -500}]}, open(umb, 'w'))
    for push in (-4.0, 4.0):
        mesh, out = str(tmp_path / f'm{push}'), str(tmp_path / f'o{push}')
        write_mesh(mesh, sheet + push)
        apply.main([mesh, out, '--volume', vol, '--umbilicus', str(umb), '--device', 'cpu'])
        x_new = tifffile.imread(os.path.join(out, 'x.tif'))
        off = tifffile.imread(os.path.join(out, 'offset.tif'))
        assert np.all(np.isfinite(off)) and np.abs(off).max() <= 8.0
        before, after = abs(push), float(np.median(np.abs(x_new - sheet)))
        assert after < before, (push, before, after)            # closer to the sheet than it started
        assert os.path.exists(os.path.join(out, 'confidence.tif'))
        assert json.load(open(os.path.join(out, 'meta.json')))['tags']['recto_offset']['ckpt'].endswith('.pt')


def write_ramp_volume(root, shape=(128, 128, 256)):
    """Intensity = x (voxels), no noise: a sample k voxels along +x reads exactly x + k."""
    os.makedirs(root, exist_ok=True)
    json.dump({'shape': list(shape), 'chunks': [C, C, C], 'dtype': '|u1', 'compressor': None, 'zarr_format': 2,
               'dimension_separator': '/', 'fill_value': 0, 'order': 'C', 'filters': None},
              open(os.path.join(root, '.zarray'), 'w'))
    vol = np.broadcast_to(np.clip(np.arange(shape[2]), 0, 255).astype(np.uint8), shape)
    d = os.path.join(root, '0', '0'); os.makedirs(d, exist_ok=True)
    for cx in range(shape[2] // C):
        np.ascontiguousarray(vol[:, :, cx * C:(cx + 1) * C]).tofile(os.path.join(d, str(cx)))


def test_gate_and_second_pass(tmp_path, monkeypatch):
    """--min-snr keeps a node unless |offset| >= Z * scale, per pass; --iterations re-reads the CT at the moved surface."""
    vol = str(tmp_path / 'ramp.zarr'); write_ramp_volume(vol)
    umb = tmp_path / 'umb.json'
    json.dump({'control_points': [{'z': 0, 'y': 64, 'x': -500}, {'z': 200, 'y': 64, 'x': -500}]}, open(umb, 'w'))
    mesh = str(tmp_path / 'm'); write_mesh(mesh, 100.0)
    passes = [(2.0, 0.5), (1.0, 2.0)]                        # (offset, Laplace scale) returned by pass 1, pass 2
    stacks = []

    def fake(net, stack, valid, device='cpu'):
        off, b = passes[len(stacks) % 2]
        stacks.append(stack.copy())
        return np.full(valid.shape, off), np.full(valid.shape, np.log(b))
    monkeypatch.setattr(apply, 'predict_offsets', fake)

    def run(name, *extra):
        stacks.clear()
        out = str(tmp_path / name)
        apply.main([mesh, out, '--volume', vol, '--umbilicus', str(umb), '--device', 'cpu', *extra])
        x = tifffile.imread(os.path.join(out, 'x.tif'))
        gate = os.path.join(out, 'gate.tif')
        return x, (tifffile.imread(gate) if os.path.exists(gate) else None)

    x, gate = run('default')                                  # v1.1 behaviour: one pass, every node moves
    assert np.allclose(x, 102.0) and gate is None and len(stacks) == 1
    x, gate = run('two', '--iterations', '2')
    assert np.allclose(x, 103.0) and len(stacks) == 2
    ok = stacks[0][12] > 0
    assert np.allclose((stacks[1] - stacks[0])[:, ok], 2.0, atol=1e-3)      # pass 2 read 2 voxels further out
    x, gate = run('z1', '--iterations', '2', '--min-snr', '1')  # pass 1: 2 >= 1 * 0.5 moves; pass 2: 1 < 1 * 2 stays
    assert np.allclose(x, 102.0) and np.all(gate == 1)
    x, gate = run('z5', '--iterations', '2', '--min-snr', '5')  # nothing passes: the mesh is left as it was
    assert np.allclose(x, 100.0) and np.all(gate == 0)
    assert np.allclose((stacks[1] - stacks[0])[:, ok], 0.0, atol=1e-3)


def test_gate_with_the_model(tmp_path):
    vol = str(tmp_path / 'vol.zarr')
    sheet = write_volume(vol)
    umb = tmp_path / 'umb.json'
    json.dump({'control_points': [{'z': 0, 'y': 64, 'x': -500}, {'z': 200, 'y': 64, 'x': -500}]}, open(umb, 'w'))
    mesh, out = str(tmp_path / 'm'), str(tmp_path / 'o')
    write_mesh(mesh, sheet + 4.0)
    apply.main([mesh, out, '--volume', vol, '--umbilicus', str(umb), '--device', 'cpu', '--min-snr', '2',
                '--iterations', '2'])
    off = tifffile.imread(os.path.join(out, 'offset.tif')); gate = tifffile.imread(os.path.join(out, 'gate.tif'))
    conf = tifffile.imread(os.path.join(out, 'confidence.tif'))
    assert np.all(np.isfinite(off)) and np.abs(off).max() <= 8.0 and np.all(np.isfinite(conf))
    assert np.all(off[gate == 0] == 0) and gate.max() <= 2
    tag = json.load(open(os.path.join(out, 'meta.json')))['tags']['recto_offset']
    assert tag['min_snr'] == 2 and tag['iterations'] == 2 and 0 <= tag['moved_share'] <= 1


def test_axis_cache_never_lands_on_level0_chunks(tmp_path):
    """The axis estimate reads level 4; with --cache its chunks must not be stored under level-0 keys (v1.1 bug)."""
    base = tmp_path / 'scan.zarr'
    sheet = write_volume(str(base / '0'))
    l4 = base / '4'; os.makedirs(l4 / '0' / '0', exist_ok=True)
    json.dump({'shape': [8, 8, 16], 'chunks': [C, C, C], 'dtype': '|u1', 'compressor': None, 'zarr_format': 2,
               'dimension_separator': '/', 'fill_value': 0, 'order': 'C', 'filters': None}, open(l4 / '.zarray', 'w'))
    c4 = np.zeros((C, C, C), np.uint8); c4[:8, :8, :16] = 200
    c4.tofile(l4 / '0' / '0' / '0')
    cache = tmp_path / 'cache'
    mesh, out = str(tmp_path / 'm'), str(tmp_path / 'o')
    write_mesh(mesh, sheet + 4.0)
    apply.main([mesh, out, '--volume', base.as_uri() + '/0', '--cache', str(cache), '--device', 'cpu'])
    level0 = [p for p in cache.rglob('*') if p.is_file() and all(s.isdigit() for s in p.relative_to(cache).parts)]
    assert level0
    for p in level0:                                          # every cached level-0 chunk is the level-0 chunk
        assert p.read_bytes() == (base / '0' / p.relative_to(cache)).read_bytes(), p
    assert (cache / 'axis' / '0' / '0' / '0').read_bytes() == c4.tobytes()


def test_sample_stack_reads_only_mesh_nodes():
    """Nodes outside the mesh (ok False; recto.obj passes them at the origin) must not make the volume read chunks."""
    class Vol:
        def prefetch(self, keys): pass
        def keys_for(self, p): return set()
        def sample(self, pts):
            self.pts = pts
            return np.where(np.all(np.isfinite(pts), -1), 100.0, np.nan).astype(np.float32)
    vol = Vol()
    fz = np.zeros((4, 5, 3)); fz[:2] = 50.0
    n = np.tile([0.0, 0.0, 1.0], (4, 5, 1)); n[2:] = np.nan
    ok = np.zeros((4, 5), bool); ok[:2] = True
    stack, okr = apply.sample_stack(vol, fz, n, ok, 2)
    assert np.array_equal(okr, ok) and np.all(stack[:, ok] == 100) and np.all(stack[:, ~ok] == 0)
    assert not np.isfinite(vol.pts[:, ~ok]).any()
