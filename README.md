# recto-offset

A small model that moves a surface mesh along its normals back onto its sheet, and the label generator it was
trained with. Labels: PHerc. Paris 4 verified patches pushed off their sheet on purpose, with the offset back as
the target, following Sean Johnson's suggestion (Vesuvius Challenge Discord, #general, 14 October 2026). One command
corrects any tifxyz mesh on any open-data scan:

```bash
python -m recto.apply in.tifxyz out.tifxyz --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0
```

## Results (21 October 2026; test and bars fixed beforehand in [PREREG.md](PREREG.md))

**1. Held-out PHerc. Paris 4 (z 10000-11500, never used for training or choosing the model).** Surfaces pushed off
by a smooth field of the given peak size, share of nodes within 2 voxels of the verified surface after correction:

| push | recto model | best model-free snap | leave in place |
|---|---|---|---|
| 4 voxels | **86.0 %** | 58.1 % | 51.5 % |
| 6 voxels | **77.9 %** | 51.9 % | 30.5 % |
| 8 voxels | **70.2 %** | 42.5 % | 24.0 % |
| none (a good surface) | 93.0 % within 1 voxel | | 100 % |

Both pre-set bars pass: at least 10 points over the best snap at 4-8 voxels (it is 26-28), and at least 90 % of a
good surface left within 1 voxel.

**2. Another scan: PHerc. 0139 at its native 9.362 µm, 113 keV** (the scan type of the First Letters scrolls; 19
curated segments, 38 windows, nothing from this scroll used anywhere in training). Pushed surfaces: 63.8 / 58.5 /
50.7 % within 2 voxels at 4 / 6 / 8 voxels, against 51.5 / 39.3 / 32.5 % for the best snap (+12 to +19 points;
bar 1 passes). On the curated surfaces as published, the model moves whole regions by about 1 voxel (45 % of
nodes stay within 1 voxel; bar 2 fails). The ink test below asks whether those moves matter.

**3. Ink: held-out PHerc. 0841 at 9.366 µm** (no public ink model was trained on PHerc. 0841). 12 tiles chosen from
the labels only; Reader V3 read at each surface; human-label AUC, paired block bootstrap (95 % interval):

| comparison | Reader V3 AUC change | Reader v2 |
|---|---|---|
| pushed surface corrected by the model vs left pushed | **+0.016** [+0.003, +0.028] | -0.003 [-0.013, +0.008] |
| model vs edge snap, same pushed surface | **+0.065** [+0.034, +0.090] | +0.051 [+0.022, +0.080] |
| edge snap vs left pushed | -0.049 [-0.080, -0.019] | -0.055 [-0.084, -0.028] |
| model on the published surfaces vs as published | +0.006 [-0.005, +0.017] | +0.000 [-0.009, +0.011] |

Snapping to the CT edge costs ink; the model does not, recovers part of what a bad surface loses, and leaves the
ink of published surfaces unchanged within noise (slightly up on 2 of 3 segments). So its ~1-voxel moves on curated
native-scan surfaces (result 2) do not hurt what an ink model reads.

![PHerc. 0841: labels and Reader V3 reads at the published, pushed, model-corrected and snapped surfaces](figures/ink_ab_pherc0841.png)

**Where to use it.** On meshes that may sit off their sheet (automatic traces, spiral-fit windings, meshes moved
between scans): it brings them back better than any CT snap tried here. On curated meshes it is not needed and may
move them by about a voxel without changing their ink.

## How

1. `recto/fetch_meshes.py`: a seeded, z-stratified sample of 246 verified patches (6,000 names drawn, up to 12
   patches of at least 1 cm² per 500-slice band).
2. `recto/stacks.py`: two windows of 96 x 96 nodes per patch at 2-voxel spacing; the CT from -24 to +24 voxels along
   the outward normal (9.6 µm level of the 2.4 µm scan). On the verified patches the CT's bright layer lies 0-2 voxels
   outward and the steepest dark-to-bright rise 1-4 voxels inward of the surface.
3. `recto/data.py`: a pair is a crop around the surface pushed off by a smooth random field f (constant part plus
   smoothed noise, up to 8 voxels); the target is -f. No CT is re-sampled for a pair. Training adds generic
   intensity augmentation (gamma, contrast, blur, noise).
4. `recto/model.py`: a 3D stem over the 25 layers and a 2D U-Net over the sheet (542k parameters); offset plus a
   Laplace scale (confidence). 40,000 steps, one RTX 3090, 1.7 h.
5. `recto/evaluate.py`, `recto/baselines.py`, `recto/external.py`, `recto/ink_ab.py`: the three tests above.
6. `recto/apply.py`: the tool. Weights: `weights/recto_offset_r2.pt` (2 MB).

Dataset: `data/manifest.json` lists every window (scan, patch, position, split); the stacks are rebuilt from the
open data with `fetch_meshes` + `stacks` + `external` (CT is not redistributed). Tests: `python -m pytest tests`
(8, including an end-to-end run of `apply` on a synthetic scan).

## Related work

- villa: `lasagna`/`snap_surf` and Erwin Nieuwlaar's `snap_meshes.py` snap to the m7 surface *prediction*, not to
  the CT; `scrollprize/fiber_strip_small_v1` (Sean Johnson, 14 Oct 2026) predicts the sheet's normal offset around a
  traced fiber (Paris 4), so it needs fibers first. This model needs only the mesh and the scan.
- ScrollFiesta (Hari Seldon): `RectoRefine` moves a mesh to the outward dark-to-bright boundary of the CT; Hari set
  the "within about 2 voxels of the recto" bar this work uses.
- Per-tile depth re-centring before ink: tarikcankorkmaz measured an oracle gain (0.785 to 0.896 AUC) with no
  label-free selector; AndreasHad04's label-free depth filter (villa #1898); inkdx's window-scale snap. The ink test
  here is a label-free per-node correction of the surface itself.

## Limits

- Truth on Paris 4 is the verified patch, not an independent recto measurement; the patches sit about 2 voxels
  outward of the CT's dark-to-bright rise. The model reproduces that convention.
- Trained on Paris 4 at 9.6 µm only; on the native 9.36 µm scan it moves good surfaces by about a voxel (result 2).
- Pushes of up to 8 voxels; a surface on the wrong sheet (one pitch or more away) is a sheet switch, not something
  this model is meant to fix.
- The ink test covers one scroll, 12 tiles and two readers; the recovery gain is clear for Reader V3 and absent for
  Reader v2.

## Credits

Sean Johnson (the pushed-surface recipe), Hari Seldon (the recto definition and the 2-voxel bar), the Vesuvius
Challenge team (verified patches, scans, Reader-compatible ink inference), Domenico Russo (Reader v2, Reader V3).
Data: Vesuvius Challenge open data (CC BY-NC 4.0). Code and weights: MIT.

Developed with AI assistance under my direction.
