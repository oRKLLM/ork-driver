#!/bin/sh
# check_registry.sh — BUILD-TIME gate for OPS_REGISTRY.md (wired into `make`).
#
# Turns the "a status with no probe is a red flag" convention from something a human
# has to NOTICE while reading (runtime) into something the build ENFORCES (compile time):
#
#   1. Every probe/test the registry cites must exist (a tools/ or examples/ source, an
#      npu.c probe function, or a Makefile target) — no fabricated/renamed/stale evidence.
#   2. Every op symbol the registry names must exist in the sources — no stale rows for
#      renamed/removed ops.
#   3. THE RED FLAG: every table row with a hard status (PROVEN/PARTIAL/DEAD) must carry
#      evidence — the name of a real probe/test file, a `make test`/`replay`/`gtest`/`ppl`
#      reference, or an explicit `(no ... probe)` acknowledgment. A hard status backed by
#      nothing fails. WIP / diagnostic / legend rows are exempt.
#
# Runs anywhere (pure grep/awk, no NPU, no python). Exit 0 = clean, 1 = registry drift.
set -eu
REG=OPS_REGISTRY.md
[ -f "$REG" ] || { echo "check-registry: $REG missing"; exit 1; }
fail=0

SRC="src/npu.c include/ork_npu.h"
GGML="../llama.cpp/ggml/src/ggml-ork/ggml-ork.cpp"
[ -f "$GGML" ] && SRC="$SRC $GGML"

# The real testable artifacts: every tools/ + examples/ source basename. This is the
# ground truth for "a probe/test that exists". Used by checks 1 and 3.
arts=$(ls tools examples 2>/dev/null | sed -nE 's/\.(c|sh)$//p' | sort -u)
arts_re=$(printf '%s' "$arts" | paste -sd'|' -)

# --- 1) cited probes must resolve to a real artifact -------------------------------------
# A "cited probe" = a backtick/prose token shaped like a probe name.
# (exclude ork_-prefixed matches: those are API functions, not probe files — check 2 covers them)
probes=$(grep -oE '[a-z0-9_]+_(probe|test|check|stress|bench)|(probe|test)_[a-z0-9_]+' "$REG" | grep -vE '^ork_' | sort -u)
for p in $probes; do
  printf '%s\n' "$arts" | grep -qx "$p" && continue          # real tools/examples file
  grep -qE "\b$p\b" src/npu.c 2>/dev/null && continue          # internal probe function
  grep -qE "^$p:" Makefile 2>/dev/null && continue             # make target
  echo "check-registry: FAIL — cited probe '$p' has no tools/ or examples/ source, npu.c fn, or make target"
  fail=1
done

# --- 2) named op symbols must exist (exclude probe/test tokens caught above) --------------
ops=$(grep -oE 'ork_(npu|mm|dyn|submit|ppu)_[a-z0-9_]+|set_[a-z0-9]+_(out8?|silu32?|fp16in)|run_chain_i8_impl' "$REG" \
      | grep -vE '_(probe|test|check|stress|bench)$' | sort -u)
for o in $ops; do
  hit=0
  for f in $SRC; do grep -qE "\b$o\b" "$f" 2>/dev/null && { hit=1; break; }; done
  [ "$hit" = 1 ] || { echo "check-registry: FAIL — op '$o' named in registry not found in sources"; fail=1; }
done

# --- 3) the red flag: a hard status must be backed by evidence ---------------------------
awk -F'|' -v arts="$arts_re" '
  /^\| / && /PROVEN|PARTIAL|DEAD/ {
    name=$2; gsub(/^ +| +$/,"",name)
    if (name ~ /^\*\*(PROVEN|PARTIAL|DEAD|WIP)\*\*$/) next          # legend row
    if ($0 ~ ("(" arts ")")) next                                   # names a real probe/test file
    if ($0 ~ /make test|replay|gtest|[Pp][Pp][Ll]|no [a-z ]*probe/) next
    print "check-registry: FAIL — hard status with NO probe evidence (red flag): " name
    bad=1
  }
  END { exit bad ? 3 : 0 }
' "$REG" || fail=1

[ "$fail" = 0 ] && echo "check-registry: OK — every status is probe-anchored; all cited probes/ops exist"
exit $fail
