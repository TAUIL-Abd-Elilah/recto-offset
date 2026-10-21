"""Read-only access to the PHerc. Paris 4 CT at 9.6 um (level 2 of the 2.4 um masked scan) through a disk chunk cache.

The OME-Zarr level is uint8, shape (Z, Y, X) = (18946, 8174, 8174), uncompressed 128^3 chunks with "/" separators.
Chunks are fetched once over HTTP and kept under <cache>/<cz>/<cy>/<cx>; a missing chunk (outside the scan) reads
as zeros. `Volume.sample(zyx)` returns trilinear samples (float32) at fractional voxel positions."""
from __future__ import annotations

import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PARIS4_L2 = ('https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/'
             '20260411134726-2.400um-0.2m-78keV-masked.zarr/2')
SHAPE = (18946, 8174, 8174)
C = 128


class Volume:
    def __init__(self, cache: str, url: str = PARIS4_L2, shape=SHAPE, chunk: int = C, workers: int = 8):
        self.cache, self.url, self.shape, self.c, self.workers = cache, url, tuple(shape), chunk, workers
        self.mem: dict[tuple, np.ndarray] = {}
        self.lock = threading.Lock()
        self.fetched = 0

    def _path(self, k):
        return os.path.join(self.cache, *map(str, k))

    def _fetch(self, k) -> None:
        p = self._path(k)
        if os.path.exists(p) or os.path.exists(p + '.empty'):
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for i in range(4):
            try:
                with urllib.request.urlopen(f'{self.url}/{k[0]}/{k[1]}/{k[2]}', timeout=120) as r:
                    data = r.read()
                if len(data) != self.c ** 3:
                    raise IOError(f'chunk {k}: {len(data)} bytes')
                with open(p + '.part', 'wb') as f:
                    f.write(data)
                os.replace(p + '.part', p)
                with self.lock:
                    self.fetched += 1
                return
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    open(p + '.empty', 'w').close()
                    return
                if i == 3:
                    raise
            except Exception:
                if i == 3:
                    raise

    def prefetch(self, keys) -> None:
        keys = [k for k in set(keys) if all(0 <= k[i] < -(-self.shape[i] // self.c) for i in range(3))]
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(self._fetch, keys))

    def chunk(self, k) -> np.ndarray | None:
        a = self.mem.get(k)
        if a is None:
            p = self._path(k)
            if not os.path.exists(p):
                self._fetch(k)
            if os.path.exists(p):
                a = np.fromfile(p, np.uint8).reshape(self.c, self.c, self.c)
            else:
                a = np.zeros((1, 1, 1), np.uint8)       # outside the scan
            if len(self.mem) > 256:
                self.mem.clear()
            self.mem[k] = a
        return a

    def keys_for(self, zyx: np.ndarray, margin: int = 1) -> set:
        """Chunk keys touched by trilinear samples at zyx (N, 3)."""
        lo = np.floor(zyx).astype(np.int64) // self.c
        hi = (np.floor(zyx).astype(np.int64) + margin) // self.c
        ks = set(map(tuple, lo)) | set(map(tuple, hi))
        return ks

    def sample(self, zyx: np.ndarray) -> np.ndarray:
        """Trilinear samples at zyx (..., 3) in voxel units; NaN where any corner lies outside the volume."""
        shp = zyx.shape[:-1]
        p = zyx.reshape(-1, 3).astype(np.float64)
        out = np.full(len(p), np.nan, np.float32)
        ok = np.all(np.isfinite(p), 1) & np.all(p >= 0, 1) & np.all(p < np.array(self.shape) - 1, 1)
        idx = np.flatnonzero(ok)
        if not len(idx):
            return out.reshape(shp)
        f = np.floor(p[idx]).astype(np.int64)
        t = (p[idx] - f).astype(np.float32)
        acc = np.zeros(len(idx), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    q = f + np.array([dz, dy, dx])
                    w = ((t[:, 0] if dz else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1]) * (t[:, 2] if dx else 1 - t[:, 2]))
                    kk = q // self.c
                    order = np.lexsort((kk[:, 2], kk[:, 1], kk[:, 0]))
                    ks, starts = np.unique(kk[order], axis=0, return_index=True)
                    bounds = list(starts) + [len(order)]
                    v = np.empty(len(idx), np.float32)
                    for j, k in enumerate(map(tuple, ks)):
                        sel = order[bounds[j]:bounds[j + 1]]
                        a = self.chunk(k)
                        if a.shape[0] == 1:
                            v[sel] = 0.0
                        else:
                            r = q[sel] - np.array(k) * self.c
                            v[sel] = a[r[:, 0], r[:, 1], r[:, 2]]
                    acc += w * v
        out[idx] = acc
        return out.reshape(shp)
