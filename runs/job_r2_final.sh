#!/usr/bin/env bash
# GPU queue job (prize fork), ~2-2.5 h: r2 training (intensity augmentation, 40k steps), final-model choice on
# Paris 4 validation only, the one run on the held-out Paris 4 test band, the PHerc0139 cross-scan check with the
# final model, and the PHerc0841 ink A/B (skipped if its CPU build is not finished). Never waits on the GPU.
R="D:/Competition/Vesuvius progress prizes/_public/recto-offset"
ROOT=C:/VesuviusTemp/recto
PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe
cd "$R" || exit 4
[ -f $ROOT/build_stacks.done ] || { echo "stacks missing"; exit 3; }
[ -f runs/r2/best.pt ] || $PY -u -m recto.train --root $ROOT --out runs/r2 --steps 40000 --batch 16 --aug --device cuda || exit 5
$PY -u -m recto.evaluate --root $ROOT --ckpt runs/r2/best.pt --out results/r2_val.json --split val --device cuda
# final model = lower validation MAE (best.pt records it); nothing else decides
FINAL=$($PY -c "
import torch
v={r: torch.load(f'runs/{r}/best.pt',map_location='cpu')['val']['mae'] for r in ('r1','r2')}
print(min(v,key=v.get))")
echo "final=$FINAL" | tee runs/final.txt
$PY -u -m recto.evaluate --root $ROOT --ckpt runs/$FINAL/best.pt --out results/final_test.json --split test --device cuda
$PY -u -m recto.evaluate --root $ROOT --ckpt runs/$FINAL/best.pt --out results/final_pherc0139.json --split all --stacks ext_stacks/PHerc0139 --device cuda
if [ -f $ROOT/ink_ab/BUILD_DONE ]; then
  $PY -u -m recto.ink_ab arms --root $ROOT --ckpt runs/$FINAL/best.pt && \
  $PY -u -m recto.ink_ab infer --root $ROOT && \
  $PY -u -m recto.ink_ab score --root $ROOT --out results/ink_ab.json
else
  echo "ink A/B build not finished; skipped"
fi
touch runs/r2_final.done
