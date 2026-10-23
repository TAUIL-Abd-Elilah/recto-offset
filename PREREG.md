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

## Amendment 2 (21 October 2026, before any ink was read on these tiles): ink A/B on PHerc. 0841

**Question.** Does moving a surface with the recto model change what an ink model reads at it?

**Data.** The three labelled PHerc. 0841 segments (no public ink model trained on PHerc. 0841), their 9.366 µm
meshes and native 113 keV scan; per segment the 4 non-overlapping 512 x 512 px tiles with the most labelled ink,
chosen from the labels only (`recto/ink_ab.py`). Labels: the 20260918 human labels aligned to the published surface
volume (as in 9um-reader-letter-shape).

**Arms.** A published surface; B = A moved by the model; C = A pushed off by a smooth field of peak 4 voxels
(seeded per tile); D = C moved by the model (it sees only C); E = C moved by the best model-free snap (nearest
dark-to-bright rise). Readers: Reader V3 (primary) and Reader v2, stock ink_9um inference, forward. Checkpoint:
the one chosen on Paris 4 validation.

**Outcomes, fixed now.** Primary: Reader V3 human-label AUC over the 12 tiles (mean of the three segments),
paired block bootstrap over 128-px blocks.
1. *Recovery*: D - C > 0 with the 95 % interval above 0, and D - E > 0 with the interval above 0 (the model
   recovers ink lost to a bad surface, better than a snap).
2. *Published surfaces*: B - A reported with its interval. Above 0: the model improves curated surfaces. Within
   +-0.005: it leaves their ink unchanged. Below 0 with the interval below 0: it harms them, and the release says so.

## Amendment 3 (21 October 2026, before r3 is trained): native-scan training and a new held-out scan

**Change.** r3 adds curated segments on native 9.362 µm scans to training (PHerc. 0139 and PHerc. 0500P2, half of
the training crops), pushes up to 12 voxels (was 8), and keeps the intensity augmentation. PHerc. 0139 thereby
leaves the test set; its r1/r2 results stay reported as they were.

**Validation (model choice).** Paris 4 validation band plus all PHerc. 0814 windows (native 9.362 µm). The final
model is whichever of r2 and r3 has the lower mean absolute error on this combined fixed set (256 crops, seed 1000,
pushes up to 12 voxels). Nothing else decides.

**New held-out scan.** PHerc. 0009B on its native 8.64 µm, 116 keV scan, the scan class of most First Letters
scrolls; never used for training or choice. Bars as before: at pushes of 4, 6 and 8 voxels at least 10 points over
the best model-free snap on the within-2-voxel share and a lower median error; at rest at least 90 % of nodes within
1 voxel. Pushes of 10 and 12 voxels are reported without a bar. The Paris 4 test band is scored again with the final
model (a second look at that band, stated as such).

**Real meshes (descriptive, no bar).** Automatically grown PHerc. 1447 meshes on its 8.64 µm scan: offsets the model
applies, and before/after distance of the surface to the CT's nearest dark-to-bright rise and bright layer, plus
cross-sections. A CT feature is not ground truth; this shows what the tool does on meshes it was built for.

**Amendment 3b (21 October 2026, before r3 is trained).** Most native-scan segments are automatically grown meshes,
not hand-curated, so their position is noisier truth than the Paris 4 verified patches. Native training and
validation windows are kept only if their mean CT profile matches the Paris 4 convention (`data.consistent`: within
+-6 voxels, brightest layer 1 inward to 3 outward, steepest dark-to-bright rise 4 inward to 1 outward). The filter is
never applied to the PHerc. 0009B test windows.

## Amendment 4 (23 October 2026, after the v1.1 results): an optional gate and a second pass, and a correction

**What was seen first.** After v1.1 was published, a CPU exploration on the existing stacks (no retraining, no new
data) scored apply-time variants on every evaluation set, including the held-out PHerc. 0009B scan, the Paris 4 test
band and PHerc. 0139: a confidence gate, smoothing, dead bands, constant shifts, a second pass, and reading the CT at
another scale. The gate and the second pass were kept because of what that exploration showed. Every gated or
two-pass number on PHerc. 0009B, PHerc. 0139 and the Paris 4 test band is therefore post hoc, not a pre-registered
test. The exploration also saw the r2 gate numbers on PHerc. 0814.

**Gate.** `recto.apply --min-snr Z`: a node keeps its position unless |offset| >= Z * b, with b the predicted Laplace
scale in voxels (b = 1 / confidence). Z = 0 (the default) is the v1.1 tool. Grid: Z in {0, 1, 1.5, 2, 3}.

**Choice rule (written before `recto/gate_eval.py` was run).** Per model (r2, r3), Z = the smallest grid value whose
one-pass gate keeps at least 85 % of nodes within 1 voxel at rest on PHerc. 0814 (validation: all 38 windows, the
same crops and seeds as `recto.evaluate` uses on the other scans). If no grid value does, the gate is not recommended
for that model. The rule makes the choice mechanical; it is not blind, since the r2 numbers on PHerc. 0814 were seen
in the exploration.

**Second pass.** `recto.apply --iterations 2` reads the CT again at the corrected surface and adds the next
prediction. No rule picks it; it is reported for every Z, and any recommendation for it is a post hoc reading.

**Reproduction first.** Z = 0 with one pass must reproduce the published numbers in `results/*.json` within 0.1
points at every push, or no gate number is reported.

**Outcome of the rule (after the run).** Z = 0 reproduced every published number within 0.002 points. On PHerc. 0814
at rest, within 1 voxel for Z = 0 / 1 / 1.5 / 2 / 3: r2 45.0 / 62.6 / 76.0 / 85.6 / 97.2 %, r3 44.4 / 65.2 / 76.7 /
84.0 / 94.2 %. So Z = 2 for r2 and Z = 3 for r3 (`results/v12_gating.json`). Every other number in that file
(the test scans, the second pass, the other Z) is post hoc. In particular r3 at Z = 3 leaves 90.0 % of nodes within
1 voxel at rest on PHerc. 0009B, the level of the amendment-3 bar, but this is not a pass of that bar: the gate was
chosen after those results were seen, and at 4 and 6 voxel pushes the gated r3 falls below the best baseline there
(50.4 vs 51.5 %, 34.7 vs 35.6 %).

**Correction (the PHerc. 0800 demo of v1.1).** v1.1 compared the published PHerc. 0800 meshes with "a random position
38.5 %". That figure is 5/13, the share expected if the brightest smoothed layer were spread evenly over the 13
positions of the +-6-voxel window; it was never measured. Measured with the demo's own code on the same 12 crops,
every node moved by its own uniform random shift along its normal (+-6 voxels, 5 seeds; `recto/demo_null.py`,
`results/demo_pherc0800_null.json`), chance is 32.9 % (32.8-32.9 % over seeds; +-4.5-voxel shifts give the same).
The published meshes (32.9 %) are at chance, not below it, and r3 (39.5 %) is 6.6 points above chance. This stays
descriptive (amendment 3): a CT feature, not ground truth.
