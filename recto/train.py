"""Train RectoNet on pushed-off verified surfaces; keep the checkpoint with the lowest validation error.

  python -m recto.train --root C:/VesuviusTemp/recto --out runs/r1 --steps 20000 [--device cuda]
Writes <out>/args.json, train.jsonl (loss and validation metrics), best.pt, last.pt."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import data
from .model import RectoNet, laplace_nll


def metrics(err: np.ndarray, mask: np.ndarray) -> dict:
    e = np.abs(err[mask])
    return {'mae': float(e.mean()), 'median': float(np.median(e)), 'within1': float((e <= 1).mean()),
            'within2': float((e <= 2).mean()), 'p95': float(np.percentile(e, 95)), 'nodes': int(e.size)}


@torch.no_grad()
def predict(net, x: np.ndarray, device, bs: int = 32):
    net.eval()
    out, conf = [], []
    for i in range(0, len(x), bs):
        p, lb = net(torch.from_numpy(x[i:i + bs]).to(device))
        out.append(p.cpu().numpy()); conf.append(lb.cpu().numpy())
    net.train()
    return np.concatenate(out), np.concatenate(conf)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--crop', type=int, default=64)
    ap.add_argument('--R', type=int, default=12); ap.add_argument('--A', type=float, default=8.0)
    ap.add_argument('--width', type=int, default=32); ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--val-n', type=int, default=256); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--aug', action='store_true', help='intensity augmentation (data.augment)')
    ap.add_argument('--extra-train', nargs='*', default=[], help='stack subdirs used whole for training, e.g. ext_stacks/PHerc0139')
    ap.add_argument('--extra-val', nargs='*', default=[], help='stack subdirs used whole for validation')
    ap.add_argument('--extra-weight', type=float, default=0.5, help='share of training crops drawn from --extra-train')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train = data.load_stacks(a.root, 'train'); val = data.load_stacks(a.root, 'val')
    extra = [w for s in a.extra_train for w in data.load_stacks(a.root, 'all', s)]
    for s in a.extra_val:
        val += data.load_stacks(a.root, 'all', s)
    kw = dict(crop=a.crop, R=a.R, A=a.A)
    vx, vy, vm = data.fixed_set(val, a.val_n, seed=1000, **kw)
    print(f'train windows {len(train)} + extra {len(extra)}, val windows {len(val)}, device {a.device}', flush=True)
    net = RectoNet(layers=2 * a.R + 1, width=a.width).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    log = open(os.path.join(a.out, 'train.jsonl'), 'a')
    best, t0 = 1e9, time.time()
    for step in range(1, a.steps + 1):
        xs, ys, ms = zip(*(data.sample(extra if extra and rng.random() < a.extra_weight else train, rng, aug=a.aug, **kw)
                           for _ in range(a.batch)))
        x = torch.from_numpy(np.stack(xs)).to(a.device)
        y = torch.from_numpy(np.stack(ys)).to(a.device)
        m = torch.from_numpy(np.stack(ms)).to(a.device)
        p, lb = net(x)
        loss = laplace_nll(p, lb, y, m)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            vp, _ = predict(net, vx, a.device)
            mv = metrics(vp - vy, vm)
            rec = {'step': step, 'loss': float(loss), 'val': mv, 'sec': round(time.time() - t0)}
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(json.dumps(rec), flush=True)
            torch.save({'model': net.state_dict(), 'args': vars(a), 'step': step}, os.path.join(a.out, 'last.pt'))
            if mv['mae'] < best:
                best = mv['mae']
                torch.save({'model': net.state_dict(), 'args': vars(a), 'step': step, 'val': mv}, os.path.join(a.out, 'best.pt'))


if __name__ == '__main__':
    main()
