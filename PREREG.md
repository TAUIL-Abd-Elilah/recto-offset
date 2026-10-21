# Pre-registration (written 21 October 2026, before any model was trained)

**Question.** Can a small network trained only on CT around pushed-off PHerc. Paris 4 verified patches bring a
surface back onto its sheet better than snapping to the CT itself?

**Data.** 246 verified patches from `spiral_datasets/PHercParis4/verified_patches`, a seeded random sample of 6,000
names (`recto/fetch_meshes.py --sample 6000 --seed 0`), up to 12 patches of at least 1 cm² per 500-slice z band.
CT: level 2 (9.6 µm) of `20260411134726-2.400um-0.2m-78keV-masked.zarr`. Split by whole z bands (level-2 slices):
validation 6000–7500, **test 10000–11500**, training everything else. Test windows are not used for training,
model selection or baseline calibration.

**Truth.** The verified patch is offset 0. Measured before training on the first windows: the CT's brightest layer
is 0–2 voxels outward of the patch and the steepest dark-to-bright rise 1–4 voxels inward. The final release states
this convention and the measured offset over all test windows.

**Pushes.** Smooth random fields (constant part plus Gaussian-smoothed noise, correlation 4–24 nodes at 2-voxel
node spacing), scored at fixed peak amplitudes 0, 2, 4, 6 and 8 voxels (`recto/evaluate.py`, seeds 2000 + amplitude).

**Baselines** (model-free, bias calibrated on training windows): leave in place; nearest bright layer along the
normal; nearest dark-to-bright rise.

**Primary metric.** Share of valid test nodes within 2 voxels of the verified surface after correction.

**Bar for calling it a success** (all on the test band, the checkpoint chosen on validation only):
1. at amplitudes 4, 6 and 8 voxels, the network beats the best baseline on the within-2-voxel share by at least
   10 percentage points and on the median absolute error;
2. at amplitude 0 (an already-good surface) the network keeps at least 90 % of nodes within 1 voxel, so it does
   not damage good surfaces.
If 1 fails, the result is reported as a negative. If 2 fails, the model is not released for use on good surfaces.

**Out of scope for this test.** Scrolls other than Paris 4 (a later, separately stated check on native 9 µm scans);
whether better placement raises ink-detection scores.

## Amendment 1 (21 October 2026, before any PHerc. 0139 result): cross-scan check

**Data.** Published PHerc. 0139 segments (w025-w059) on the native 9.362 µm, 113 keV scan 20250728140407, the
resolution and energy class of the First Letters scrolls (`recto/external.py`). Two windows per segment where the
mesh exists on that scan. The scroll axis is estimated from the masked scan (centre of the non-zero area of a
level-4 slice); orientation was checked on the first windows (bright layer 0-3 voxels outward, as on Paris 4).
Nothing from PHerc. 0139 is used for training, model selection or baseline calibration.

**Bar.** The same as the main test, applied unchanged with the checkpoint chosen on Paris 4 validation: at
amplitudes 4, 6 and 8 voxels the network beats the best baseline on the within-2-voxel share by at least
10 percentage points and on the median error; at amplitude 0 at least 90 % of nodes stay within 1 voxel.
