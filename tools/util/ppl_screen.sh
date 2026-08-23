#!/bin/sh
# ppl_screen.sh — TWO-TIER perplexity comparison for quantization work.
#
# WHY TWO TIERS. A full-text PPL run is ~15 min per arm on this board, which is fine as a release gate and
# useless as an iteration loop: a 2x2 experiment costs an hour, and you stop running experiments you should
# run. Most quantization questions ("did this scale change help? is fp32 safe?") only need to know whether a
# change is neutral, better, or broken — and that separates in a few hundred tokens.
#
#   SCREEN (default) : 1 window of 256 tokens, ~30 s/arm. The iteration loop.
#   FULL (--full)    : the whole text in 512-token windows, ~15 min/arm. The release-candidate gate.
#
# WHAT THE SCREEN IS AND IS NOT. Every arm scores the IDENTICAL token sequence with deterministic models, so
# the PAIRED difference between arms is exact — there is no sampling noise between them. What a short run
# cannot tell you is whether that difference GENERALISES: a 0.5% gap on 255 tokens may be 2% on 6132, or
# nothing. So use the screen to rank and to catch breakage, and re-run --full before changing a default or
# publishing a number. A screen result is a reason to keep going, never a release claim.
#
# Usage:  sudo tools/util/ppl_screen.sh [--full] <model.gguf> <text> <label=pack> [label=pack ...]
# e.g.    sudo tools/util/ppl_screen.sh ~/m.gguf ~/ppl_text.txt fp64=~/q4_fp64.orkpack fp32=~/q4_fp32.orkpack
#
# Env passed through to every arm (so all arms share it): ORK_QUANT, ORK_MIXED_W4A4, ORK_I4_NOCLIP, ...
set -eu

# 256 tokens costs ~40 s and tracked the full 6132-token run to within 0.2% on the pack measured here,
# which is worth the extra seconds over a shorter window. ORK_PPL_SCREEN_W=192 gets ~30 s if you want it.
W=${ORK_PPL_SCREEN_W:-256}; UB=$W; MAXW=1; TIER="SCREEN"
TMO=900
if [ "${1:-}" = "--full" ]; then W=512; UB=512; MAXW=0; TIER="FULL"; TMO=5400; shift; fi
[ $# -ge 3 ] || { sed -n '2,26p' "$0"; exit 2; }

MODEL=$1; TEXT=$2; shift 2
# $HOME is /root under sudo (which this must run under — the NPU needs it), so resolve the INVOKING
# user's home instead of the effective one, or the default path points at a tree that does not exist.
HOMEDIR=$HOME
[ -n "${SUDO_USER:-}" ] && HOMEDIR=$(getent passwd "$SUDO_USER" | cut -d: -f6)
BIN=${ORK_PPL_BIN:-$HOMEDIR/llama.cpp/build/bin/ork_ppl}
GUARD=$(dirname "$0")/npu_guard.sh
[ -x "$BIN" ] || { echo "no ork_ppl at $BIN (set ORK_PPL_BIN)"; exit 2; }

# PROVENANCE. A stale binary is the single most expensive failure mode this harness has: rsync -a preserves
# mtimes, cmake then skips the rebuild, and you measure OLD code while believing you changed something —
# silently, because the build reports success. On 2026-08-23 that produced a phantom regression and three
# derived results that all had to be retracted. So every run states WHICH binary produced it: build time and
# a content hash of the backend library. If two runs disagree and these lines differ, the code differs.
BIN_LIB=$(ls "$(dirname "$BIN")"/libggml-ork.so 2>/dev/null | head -1)
printf '== BINARY %s  built %s' "$BIN" "$(date -r "$BIN" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '?')"
[ -n "$BIN_LIB" ] && printf '  | libggml-ork %s built %s' \
    "$(sha256sum "$BIN_LIB" 2>/dev/null | cut -c1-12)" "$(date -r "$BIN_LIB" '+%H:%M:%S' 2>/dev/null || echo '?')"
echo
echo "== $TIER: window=$W ubatch=$UB maxwin=$MAXW  model=$(basename "$MODEL")  text=$(basename "$TEXT")"
for spec in "$@"; do
    label=${spec%%=*}; pack=${spec#*=}
    [ -f "$pack" ] || { printf '%-10s : MISSING %s\n' "$label" "$pack"; continue; }
    s=$(date +%s)
    # One arm per invocation: the NPU is single-stream, and the guard serialises against other agents.
    # Tier-dependent timeout. 900 s was fine for a ~40 s screen and silently killed every --full arm: a
    # full run on a 42 KB text is 883-920 s, i.e. right at the limit, so both arms died with no output and
    # no error anyone would read as "too slow". Give --full real headroom.
    out=$("$GUARD" -- env ORK_ORKPACK_PATH="$pack" timeout "$TMO" \
          "$BIN" "$MODEL" "$TEXT" "$W" "$UB" "$MAXW" 2>&1 | grep -E '^\[ork_ppl\] PPL' || true)
    e=$(date +%s)
    if [ -z "$out" ]; then printf '%-10s : NO RESULT (see the run output)\n' "$label"
    else printf '%-10s : %s  [%ss]\n' "$label" "$(echo "$out" | sed 's/\[ork_ppl\] //')" "$((e-s))"; fi
done
[ "$TIER" = SCREEN ] && echo "== screen only — re-run with --full before changing a default or quoting a number"
exit 0
