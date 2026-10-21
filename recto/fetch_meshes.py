"""Pick a reproducible, z-stratified sample of PHerc. Paris 4 verified patches and download their tifxyz meshes.

  python -m recto.fetch_meshes --out C:/VesuviusTemp/recto --sample 2500 --per-band 12 --seed 0

1. lists verified_patches/ (spiral_datasets/PHercParis4 on dl.ash2txt.org) and draws `--sample` names at random
   (seeded), downloading only their meta.json (bbox, area);
2. splits z into bands of `--band` voxels (level-2 frame, 9.6 um) and keeps up to `--per-band` patches per band with
   area >= `--min-cm2`, largest first, so every height of the scroll is represented;
3. downloads x.tif, y.tif, z.tif and meta.json of the kept patches into <out>/meshes/<name>/.
Writes <out>/metas.jsonl (all sampled metas) and <out>/selection.json (kept names, bands, split).
Split: whole z bands, so train / val / test patches never share a band (see SPLIT below)."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = 'https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/verified_patches'
# z bands (level-2 voxels) reserved for evaluation; everything else trains
SPLIT = {'val': [(6000, 7500)], 'test': [(10000, 11500)]}


def get(url: str, path: str | None = None, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + '.part', 'wb') as f:
                    f.write(data)
                os.replace(path + '.part', path)
            return data
        except Exception:
            if i == tries - 1:
                raise
    raise RuntimeError(url)


def split_of(zc: float) -> str:
    for name, bands in SPLIT.items():
        if any(lo <= zc < hi for lo, hi in bands):
            return name
    return 'train'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample', type=int, default=2500)
    ap.add_argument('--band', type=int, default=500)
    ap.add_argument('--per-band', type=int, default=12)
    ap.add_argument('--min-cm2', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    names = sorted(set(re.findall(r'href="([^"?/][^"]*)/"', get(URL + '/').decode())))
    random.Random(a.seed).shuffle(names)
    pick = names[:a.sample]
    mp = os.path.join(a.out, 'metas.jsonl')
    have = {}
    if os.path.exists(mp):
        for line in open(mp):
            m = json.loads(line); have[m['name']] = m

    def meta(n):
        if n in have:
            return have[n]
        try:
            m = json.loads(get(f'{URL}/{n}/meta.json'))
        except Exception as e:  # noqa: BLE001
            return {'name': n, 'error': str(e)[:100]}
        return {'name': n, 'bbox': m.get('bbox'), 'area_cm2': m.get('area_cm2'), 'scale': m.get('scale'),
                'tags': list((m.get('tags') or {}).keys())}

    with ThreadPoolExecutor(16) as ex:
        metas = list(ex.map(meta, pick))
    with open(mp, 'w') as f:
        for m in metas:
            f.write(json.dumps(m) + '\n')
    ok = [m for m in metas if m.get('bbox') and (m.get('area_cm2') or 0) >= a.min_cm2]
    bands: dict[int, list] = {}
    for m in ok:
        zc = 0.5 * (m['bbox'][0][2] + m['bbox'][1][2])
        bands.setdefault(int(zc // a.band), []).append(m)
    keep = []
    for b in sorted(bands):
        for m in sorted(bands[b], key=lambda m: -m['area_cm2'])[:a.per_band]:
            zc = 0.5 * (m['bbox'][0][2] + m['bbox'][1][2])
            keep.append({'name': m['name'], 'z_center': zc, 'band': b, 'area_cm2': m['area_cm2'], 'split': split_of(zc)})

    def dl(k):
        for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json'):
            p = os.path.join(a.out, 'meshes', k['name'], f)
            if not os.path.exists(p):
                get(f'{URL}/{k["name"]}/{f}', p)
        return k['name']

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(dl, keep))
    summary = {s: {'patches': sum(k['split'] == s for k in keep), 'cm2': round(sum(k['area_cm2'] for k in keep if k['split'] == s), 1)}
               for s in ('train', 'val', 'test')}
    json.dump({'args': vars(a), 'split_bands': SPLIT, 'summary': summary, 'patches': keep},
              open(os.path.join(a.out, 'selection.json'), 'w'), indent=1)
    print(json.dumps(summary), f'sampled {len(metas)}, usable {len(ok)}, kept {len(keep)}', flush=True)


if __name__ == '__main__':
    main()
