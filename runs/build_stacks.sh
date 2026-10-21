#!/usr/bin/env bash
# Detached CPU/network job: CT stacks for every selected verified patch (2 windows each). Log + .done marker.
cd "D:/Competition/Vesuvius progress prizes/_public/recto-offset"
L=C:/VesuviusTemp/recto/build_stacks.log
echo "$(date '+%F %T') START" >> $L
C:/Users/PC/miniconda3/envs/vesuvius/python.exe -u -m recto.stacks --root C:/VesuviusTemp/recto --win 96 --step 2 --K 24 --windows 2 >> $L 2>&1
echo "$(date '+%F %T') END rc=$? $(ls C:/VesuviusTemp/recto/stacks | wc -l) stacks" >> $L
touch C:/VesuviusTemp/recto/build_stacks.done
