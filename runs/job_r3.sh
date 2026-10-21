#!/usr/bin/env bash
# GPU queue job (prize fork), ~2-2.5 h: r3 (native-scan data, pushes to 12 vox), model choice r2 vs r3 on the
# combined validation set (amendment 3), then the held-out PHerc0009B scan and the Paris 4 test band with the final
# model. Exits at once if the native stacks are not built, so it never holds the GPU idle.
R="D:/Competition/Vesuvius progress prizes/_public/recto-offset"
ROOT=C:/VesuviusTemp/recto
PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe
cd "$R" || exit 4
[ -f $ROOT/build_native.done ] || { echo "native stacks missing"; exit 3; }
[ -f runs/r3/best.pt ] || $PY -u -m recto.train --root $ROOT --out runs/r3 --steps 40000 --batch 16 --aug --A 12 \
    --extra-train ext_stacks/PHerc0139 ext_stacks/PHerc0500P2 --extra-val ext_stacks/PHerc0814 --device cuda || exit 5
$PY -u -m recto.select --root $ROOT --ckpts runs/r2/best.pt runs/r3/best.pt --out runs/final3.txt || exit 6
F=$(head -1 runs/final3.txt)
$PY -u -m recto.evaluate --root $ROOT --ckpt $F --out results/v2_pherc0009b.json --split all --stacks ext_stacks/PHerc0009B --amps 0 2 4 6 8 10 12 --device cuda
$PY -u -m recto.evaluate --root $ROOT --ckpt $F --out results/v2_paris4_test.json --split test --amps 0 2 4 6 8 10 12 --device cuda
touch runs/r3.done
