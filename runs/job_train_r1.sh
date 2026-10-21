#!/usr/bin/env bash
# GPU queue job (prize fork): first recto-offset training run + held-out evaluation. ~30-60 min on the 3090.
# Exits at once (rc=3) if the CT stacks are not built yet, so it never holds the GPU idle.
R="D:/Competition/Vesuvius progress prizes/_public/recto-offset"
[ -f C:/VesuviusTemp/recto/build_stacks.done ] || { echo "stacks not ready; skipped"; exit 3; }
cd "$R" || exit 4
PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe
$PY -u -m recto.train --root C:/VesuviusTemp/recto --out runs/r1 --steps 20000 --batch 16 --device cuda && \
\
$PY -u -m recto.evaluate --root C:/VesuviusTemp/recto --ckpt runs/r1/best.pt --out results/r1_val.json --split val --device cuda
rc=$?
touch runs/r1.done
exit $rc
