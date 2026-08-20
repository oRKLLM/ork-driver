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
#   6. HEADER PLACEMENT: npu/<mod>/<mod>.h is private to its folder; only npu/<mod>/*.c may include
#      it. Keeps the "inside the folder = private, beside it = interface" rule from eroding.
#   5. NO DANGLING DECLARATIONS: every ork_*/orkd_* function PROTOTYPED in a header must have
#      an implementation. A header can promise a symbol nothing defines — the caller gets a
#      link error, and if nothing in-tree calls it the gap is invisible. That is exactly how
#      ork_gptq_i4 shipped declared-but-unimplemented on main for months (fixed 61ccb48).
#
# Runs anywhere (pure grep/awk, no NPU, no python). Exit 0 = clean, 1 = registry drift.
set -eu
REG=OPS_REGISTRY.md
[ -f "$REG" ] || { echo "check-registry: $REG missing"; exit 1; }
fail=0

# All library sources — globbed, NOT a hardcoded src/npu.c, so these gates keep working as the monolith
# is split into src/npu/*.c (and deeper). Unmatched globs expand to themselves; the greps just miss them.
LIBSRC=$(ls src/*.c src/*/*.c src/*/*/*.c 2>/dev/null || true)   # || true: set -e, and the deeper globs may not match yet
SRC="$LIBSRC include/ork_npu.h"
GGML="../llama.cpp/ggml/src/ggml-ork/ggml-ork.cpp"
[ -f "$GGML" ] && SRC="$SRC $GGML"

# The real testable artifacts: every tools/ + examples/ source basename. This is the
# ground truth for "a probe/test that exists". Used by checks 1 and 3.
# Recursive, and .cpp counts: tools/re/ holds the llama-API harnesses (ork_bench, ork_ppl, the *_diff_probe
# differential oracles). The old non-recursive .c/.sh-only listing could not see ANY of them, so citations to
# them silently failed check 1 (or, worse, resolved to a same-basename file elsewhere).
arts=$(find tools examples -type f \( -name '*.c' -o -name '*.cpp' -o -name '*.sh' \) 2>/dev/null \
       | sed -E 's|.*/||; s/\.(c|cpp|sh)$//' | sort -u)
arts_re=$(printf '%s' "$arts" | paste -sd'|' -)

# --- 1) cited probes must resolve to a real artifact -------------------------------------
# A "cited probe" = a backtick/prose token shaped like a probe name.
# (exclude ork_-prefixed matches: those are API functions, not probe files — check 2 covers them)
probes=$(grep -oE '[a-z0-9_]+_(probe|test|check|stress|bench)|(probe|test)_[a-z0-9_]+' "$REG" | grep -vE '^ork_' | sort -u)
for p in $probes; do
  printf '%s\n' "$arts" | grep -qx "$p" && continue          # real tools/examples file
  grep -qE "\b$p\b" $LIBSRC 2>/dev/null && continue           # internal probe function
  grep -qE "^$p:" Makefile 2>/dev/null && continue             # make target
  echo "check-registry: FAIL — cited probe '$p' has no tools/ or examples/ source, npu.c fn, or make target"
  fail=1
done

# --- 2) named op symbols must exist (exclude probe/test tokens caught above) --------------
# NOTE: the precision split prefixed internal symbols orki_*, so the internal-op patterns must allow it —
# otherwise `orki_set_i8_silu32` in the registry still matches the bare `set_i8_silu32` here and the
# source lookup then fails on the word boundary.
ops=$(grep -oE '(orki_)?ork_(npu|mm|dyn|submit|ppu)_[a-z0-9_]+|(orki_)?set_[a-z0-9]+_(out8?|silu32?|fp16in)|(orki_)?run_chain_i8_impl' "$REG" \
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

# --- 4) regcmd -> op binding: every REGCMD_* base template in npu.c must bind to an op in ork_ops.c -------
# Enforces "0 ops not exported": a regcmd byte template that isn't declared as the implementation of some
# ork_op (in ORK_REGCMD_BIND) is an orphan — an op the SDK can't name. New regcmd with no binding => fail.
if [ -f src/ork_ops.c ]; then
  regcmds=$(grep -hoE '\bREGCMD_[A-Z0-9_]+\b' $LIBSRC 2>/dev/null | grep -vE '_N$' | sort -u)
  for rc in $regcmds; do
    grep -qE "\"$rc\"" src/ork_ops.c && continue
    echo "check-registry: FAIL — regcmd '$rc' has no op binding in src/ork_ops.c (ORK_REGCMD_BIND) — orphan regcmd / unexported op"
    fail=1
  done
fi

# --- 5) no dangling declarations: a header prototype with no implementation ---------------
# A line is a PROTOTYPE iff (after comment-stripping) it carries no "{" and ends in ";". Anything
# else carrying "name(" is a definition — including K&R-wrapped ones whose "{" lands on a later
# line, and one-liner bodies. The name is taken from before the FIRST "(" on the line, so a body
# that calls another ork_* function is not mistaken for its own declaration. `static` in a header
# is a header-local DEFINITION (it satisfies includers), so those count as defined, not declared;
# typedefs and preprocessor lines are skipped (a struct tag is not a function).
DVD_PFX='orkd?_[a-z0-9_]*'
DVD_NAME="s/^[^(]*[^A-Za-z0-9_](${DVD_PFX})[ \t]*\\(.*/\\1/p"
DVD_STRIP='s:/\\*[^*]*\\*/::g; s://.*::; s/[ \t]*$//'
# Declarations wrap across lines (72 of the 294 public prototypes do, e.g. ork_ssm_scan_f32), and a
# per-LINE rule is blind to every one of them -- which is how the ssm lift slipped past this gate. Join
# each logical statement onto one line first: accumulate until a ";" or "{" closes it.
# Strip block comments ACROSS lines, then join each logical statement onto one line. Both are needed:
# 72 of the 294 public prototypes wrap (ork_ssm_scan_f32 among them) so a per-line rule is blind to them,
# and a per-line comment strip leaves multi-line /* */ prose to be joined INTO the statement (which put
# words like "ork_w" in front of a paren and produced 52 phantom failures when tried without this).
dvd_clean() { awk '
  { line=$0; out=""
    while (length(line)) {
      if (inc) { i=index(line,"*/"); if(i==0){line=""} else {line=substr(line,i+2); inc=0} }
      else     { i=index(line,"/*"); if(i==0){out=out line; line=""} else {out=out substr(line,1,i-1); line=substr(line,i+2); inc=1} }
    }
    sub(/\/\/.*/,"",out)
    l = l " " out
    if (index(out,";") || index(out,"{")) { print l; l="" }
  } END { if (l!="") print l }'; }
dvd_decl=$(cat include/*.h src/*.h 2>/dev/null | dvd_clean | sed 's/^/ /' \
  | grep -vE '^ *#' | grep -vE '[^A-Za-z0-9_](static|typedef)[[:space:]]' \
  | grep -E "[^A-Za-z0-9_]${DVD_PFX}[[:space:]]*\\(" | grep -vE '\\{' | grep -E ';[[:space:]]*$' \
  | sed -nE "$DVD_NAME" | sort -u)
dvd_def=$( { cat $LIBSRC 2>/dev/null | sed 's/^/ /' | sed "$DVD_STRIP" \
             | grep -E "^ [A-Za-z_][A-Za-z0-9_ *]*[^A-Za-z0-9_]${DVD_PFX}[[:space:]]*\\(" \
             | grep -vE '^[^{]*;$' ;
           cat include/*.h src/*.h 2>/dev/null | sed 's/^/ /' | sed "$DVD_STRIP" \
             | grep -E '[^A-Za-z0-9_]static[[:space:]]' \
             | grep -E "[^A-Za-z0-9_]${DVD_PFX}[[:space:]]*\\(" ; } \
         | sed -nE "$DVD_NAME" | sort -u)
for d in $dvd_decl; do
  printf '%s\n' "$dvd_def" | grep -qx "$d" && continue
  echo "check-registry: FAIL — '$d' is declared in a header but defined nowhere in src/ — dangling declaration (implement it, or stub it like src/ork_gptq.c)"
  fail=1
done

# --- 6) header placement: a subtree header must stay PRIVATE to its folder ------------------------
# npu/<mod>/<mod>.h is private (only npu/<mod>/*.c may include it); npu/*.h is tree-wide. If the
# scaffold needs a module symbol the declaration belongs in internal.h, not an include of the module's
# private header — that is what keeps core.h beside core/ meaningful rather than arbitrary.
for h in src/npu/*/*.h; do
  mod=$(basename "$(dirname "$h")"); base=$(basename "$h" .h)
  [ "$mod" = "$base" ] || continue
  bad=$(grep -rl "include \"npu/$mod/$base.h\"" src 2>/dev/null | grep -v "^src/npu/$mod/" || true)
  if [ -n "$bad" ]; then
    echo "check-registry: FAIL — private subtree header npu/$mod/$base.h included from outside its folder:"
    echo "$bad" | sed 's/^/    /'
    fail=1
  fi
done

[ "$fail" = 0 ] && echo "check-registry: OK — status probe-anchored; probes/ops exist; every regcmd bound to an op; no dangling declarations; subtree headers private"
exit $fail
