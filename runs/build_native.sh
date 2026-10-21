#!/usr/bin/env bash
# Detached CPU/network job: native-scan stacks for r3 (train 0500P2, val 0814, test 0009B; 0139 exists already).
cd "D:/Competition/Vesuvius progress prizes/_public/recto-offset"
L=C:/VesuviusTemp/recto/build_native.log
PY=C:/Users/PC/miniconda3/envs/vesuvius/python.exe
for s in PHerc0814 PHerc0009B; do
  echo "$(date '+%F %T') START $s" >> $L
  $PY -u -m recto.external --root C:/VesuviusTemp/recto --scroll $s --all-segments --windows 2 --cleanup >> $L 2>&1
  echo "$(date '+%F %T') END $s rc=$? $(ls C:/VesuviusTemp/recto/ext_stacks/$s 2>/dev/null | wc -l) windows" >> $L
done
touch C:/VesuviusTemp/recto/build_native.done
