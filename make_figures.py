"""Figures for the README from the ink A/B outputs (C:/VesuviusTemp/recto/ink_ab by default).
  python make_figures.py [--root C:/VesuviusTemp/recto]"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

SEG = '20260220214732-auto_grown_20260220144552896'
ARMS = [('label', 'human ink labels'), ('A', 'A  published surface'), ('C', 'C  pushed off (peak 4 vox)'),
        ('D', 'D  C + recto model'), ('E', 'E  C + edge snap')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='C:/VesuviusTemp/recto'); a = ap.parse_args()
    d = os.path.join(a.root, 'ink_ab', SEG)
    res = json.load(open('results/ink_ab.json'))['per_segment'][SEG]
    lab = np.load(os.path.join(d, 'label.npy'))
    t0, t1 = 0, 512                                           # first tile of this segment (chosen from labels)
    fig, axs = plt.subplots(1, len(ARMS), figsize=(3.2 * len(ARMS), 3.6))
    for ax, (k, title) in zip(axs, ARMS):
        if k == 'label':
            im = lab[:, t0:t1]
        else:
            im = tifffile.imread(os.path.join(d, f'pred_{k}_reader_v3.tif')).astype(np.float32)[:, t0:t1]
            title += f'\nAUC {res["reader_v3|" + k]:.3f} (segment)'
        ax.imshow(im, cmap='gray'); ax.set_title(title, fontsize=9); ax.axis('off')
    fig.suptitle('Held-out PHerc. 0841 (native 9.366 um), Reader V3 read at each surface', fontsize=10)
    os.makedirs('figures', exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig('figures/ink_ab_pherc0841.png', dpi=110)
    print('figures/ink_ab_pherc0841.png')


if __name__ == '__main__':
    main()
