"""Model choice for amendment 3: lowest mean absolute error on the combined fixed validation set.
  python -m recto.select --root C:/VesuviusTemp/recto --ckpts runs/r2/best.pt runs/r3/best.pt --out runs/final.txt"""
import argparse
import json

import numpy as np
import torch

from . import data
from .model import RectoNet
from .train import metrics, predict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--ckpts', nargs='+', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args(argv)
    val = data.load_stacks(a.root, 'val') + data.load_stacks(a.root, 'all', 'ext_stacks/PHerc0814')
    x, y, m = data.fixed_set(val, 256, seed=1000, crop=64, R=12, A=12.0)
    res = {}
    for c in a.ckpts:
        ck = torch.load(c, map_location='cpu'); args = ck['args']
        net = RectoNet(layers=2 * args['R'] + 1, width=args['width']); net.load_state_dict(ck['model']); net.to(a.device)
        p, _ = predict(net, x, a.device)
        res[c] = metrics(p - y, m)
    best = min(res, key=lambda c: res[c]['mae'])
    json.dump({'val_windows': len(val), 'results': res, 'final': best}, open(a.out.replace('.txt', '.json'), 'w'), indent=1)
    open(a.out, 'w').write(best + '\n')
    print(json.dumps({c: round(v['mae'], 4) for c, v in res.items()}), 'final', best)


if __name__ == '__main__':
    main()
