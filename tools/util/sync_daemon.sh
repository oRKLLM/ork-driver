#!/bin/bash
# Bidirectional polling rsync daemon (macOS <-> board), 2s cadence, persistent SSH master.
# Push (Mac source-of-truth -> board build dir): src/ include/ tools/ Makefile examples/ docs/
# Pull (board results -> Mac outbox):            ~/ork-outbox/ -> scratchpad/ork-outbox/
set -u
REPO="/Users/michael/Dev/llama.cpp/ggml/src/ggml-ork/ork-driver"
OUTBOX_LOCAL="/private/tmp/claude-501/-Users-michael-Dev-llama-cpp/f1b95628-9dd9-4e1c-92c6-3188a4774cb1/scratchpad/ork-outbox"
SSHOPT="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-board-%r -o ControlPersist=120s -o ConnectTimeout=8"
mkdir -p "$OUTBOX_LOCAL"
echo "[sync] daemon start $(date '+%H:%M:%S')  repo=$REPO"
while true; do
  # --no-t (NOT plain -a): rsync -a PRESERVES mtimes, so a file restored from git — or any file whose
  # content changed but whose timestamp went BACKWARDS — lands OLDER than the object built from it, and
  # make/cmake silently skips the rebuild. You then measure a stale binary while believing you changed
  # something, and the build still prints success. That cost four wrong conclusions on 2026-08-23 (a
  # phantom "W4A4 regression", a phantom 8% offline-vs-NPU divergence, and two precision results derived
  # from them). Giving the destination the CURRENT time makes "content changed" always imply "rebuild".
  rsync -az --no-t --timeout=15 -e "ssh $SSHOPT" \
    --include='src/***' --include='include/***' --include='tools/***' \
    --include='examples/***' --include='docs/***' \
    --include='Makefile' --exclude='*' \
    "$REPO/" board:ork-driver/ 2>/dev/null && P=ok || P=FAIL
  rsync -az --no-t --timeout=15 -e "ssh $SSHOPT" board:ork-outbox/ "$OUTBOX_LOCAL/" 2>/dev/null && Q=ok || Q=FAIL
  echo "[sync] $(date '+%H:%M:%S') push=$P pull=$Q"
  sleep 2
done
