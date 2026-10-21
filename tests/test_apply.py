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
