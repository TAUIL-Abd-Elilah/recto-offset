# recto-offset

A small model that moves a surface mesh along its normals back onto its sheet in the CT, trained on CT around
PHerc. Paris 4 verified patches that were pushed off their sheet on purpose.

**Status (21 October 2026): work in progress.** The data pipeline, the model, three model-free baselines and the
held-out evaluation are written and tested on synthetic stacks; no model has been trained yet. The test and its
pass bar were fixed before any training: [PREREG.md](PREREG.md).

**Why.** Ink models read the papyrus at the surface, so a surface a few voxels off the recto reads the wrong layer.
Snapping to the nearest bright layer can jump to the next sheet, and moving meshes by hand does not scale. Sean
Johnson suggested training a model that predicts the normal offset back onto the recto, with labels made by pushing
good surfaces off the sheet with a smooth falloff (Vesuvius Challenge Discord, #general, 14 October 2026). This
repository builds those labels and that model.

**How.**
1. `recto/fetch_meshes.py`: a seeded, z-stratified sample of verified patches (tifxyz).
2. `recto/stacks.py`: on windows of each patch (2-voxel node grid), the CT sampled every voxel from -24 to +24
   voxels along the outward normal (9.6 µm level of the 2.4 µm scan).
3. `recto/data.py`: a training pair is a crop of a stack around the surface pushed off by a smooth random field f;
   the target is -f. No CT is re-sampled for a pair.
4. `recto/model.py`: a 3D stem over the layers and a 2D U-Net over the sheet (542k parameters), Laplace likelihood
   (offset plus confidence).
5. `recto/baselines.py`, `recto/evaluate.py`: leave in place, nearest bright layer, nearest dark-to-bright rise;
   scored on a held-out z band at fixed push amplitudes.

Tests: `python -m pytest tests`.

Data: Vesuvius Challenge open data (CC BY-NC 4.0); this repository stores no CT. Code: MIT.

Developed with AI assistance under my direction.
