#!/bin/sh
# tools/run_autotune.sh — driver for the NPU tiling autotuner (tools/autotune.c).
#
# The autotune binary sweeps core-count (1..N) in-process and bit-exact-gates every config; the
# M/N/K-tile knobs are per-process env (run_i8 caches them at first call), so this driver re-execs
# the binary once per env-combo and aggregates the [AUTOTUNE] lines. Each line is:
#   [AUTOTUNE] shape=.. K=.. N=.. cores=.. STM=.. STN=.. KT=.. us=.. gops=.. status=OK|WRONG|WEDGE
# The "best" per shape = min us among status=OK rows. Run on the board:  sudo sh tools/run_autotune.sh
# Env: M (default 256), REPS (default 5), KTILES (default "0 512 1024 1792"), TO (per-run timeout s).
set -e
M=${M:-256}; REPS=${REPS:-5}; TO=${TO:-240}
KTILES=${KTILES:-"0 512 1024 1792"}
OUT=${OUT:-autotune_results.txt}
: > "$OUT"
NSHAPE=5
i=0
while [ $i -lt $NSHAPE ]; do
  for kt in $KTILES; do
    # KT=0 means default (no K-tile, full-K path); skip KT>0 for the down shape (K>4096 already K-split)
    if [ "$kt" != "0" ] && [ $i -eq 4 ]; then continue; fi
    env_kt=""; [ "$kt" != "0" ] && env_kt="ORK_KTILE=$kt"
    sudo env AT_SHAPE=$i $env_kt timeout -s INT $TO ./autotune "$M" "$REPS" 2>/dev/null \
      | grep '^\[AUTOTUNE\]' | tee -a "$OUT" || true
  done
  i=$((i+1))
done
echo "=== best OK config per shape (min us) ==="
awk '/status=OK/{
  for(f=1;f<=NF;f++){split($f,a,"=");k[a[1]]=a[2]}
  s=k["shape"]; u=k["us"]+0
  if(!(s in bu)||u<bu[s]){bu[s]=u; bl[s]=$0}
} END{for(s in bl)print bl[s]}' "$OUT"
