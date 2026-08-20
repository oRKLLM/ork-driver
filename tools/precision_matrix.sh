#!/bin/sh
# precision_matrix.sh — generate the capability x precision matrix from the source tree.
#
# WHY GENERATED: the matrix is derived from what the code defines, so a hand-maintained table goes wrong
# the first time someone adds a function. `make matrix` regenerates it; paste into README.md.
#
# HOW IT CLASSIFIES: a symbol serves a precision if its NAME carries that dtype token (i8/f16/i4/i16),
# which under the dtype-FIRST convention sits right after the ork_/orki_ prefix. The capability
# patterns below therefore allow an optional segment there (^ork_((i8|i4a8|i4|nf4|f16|i16)_)?mm_pack), so they match
# both ork_i8_mm_pack and any untagged sibling. Anchoring them to ^ork_mm_ silently emptied the whole
# table during the rename — the generator reported "—" everywhere rather than failing.
# otherwise by the module folder it lives in. That is right where the dtype is in the name, and WRONG for
# shared implementations serving several dtypes under one name — those are listed in tools/precision_overrides.tsv
# and render with a dagger. Each cites the sharing symbol, which check_registry check 8 verifies still exists.
#
# Portable sh + grep/sed only: the board runs mawk and the Mac runs BSD awk, so no gawk extensions and
# (AGENTS section 2) no Python.
set -eu
cd "$(dirname "$0")/.."

# Shared implementations the name-based rule cannot see live in a DATA FILE, not here, so that
# check_registry.sh check 8 can verify every cited symbol still exists (see that file's header).
OVERRIDES=$(grep -v "^#" tools/precision_overrides.tsv | sed "/^$/d")

SRC=$(ls src/npu.c src/npu/*.c src/npu/*/*.c 2>/dev/null)
TMP=${TMPDIR:-/tmp}/orkmatrix.$$; mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT

# every top-level definition: "name<TAB>file"
# One definition per line as "name<TAB>file". Two traps, both hit in practice:
#   - the name must be the FIRST identifier before a "(", not the last: a one-line function that CALLS
#     another (seq_disp_ewmul_i8 -> ork_npu_ewmul_i8) otherwise records the CALLEE as defined here, which
#     put 32 phantom symbols in the map and mis-filed capabilities into npu.c;
#   - a knob accessor (`int ork_f16_colsplit(void){ ...getenv... }`) is a PREDICATE, not an implementation.
#     Counting it made fp16 claim its own multicore path when the real one is i8/colsplit.c, silently
#     replacing a correct dagger with a wrong plain tick. Drop any definition line that reads getenv.
for f in $SRC; do
  grep -E '^(static +)?[A-Za-z_][A-Za-z0-9_ *]*\(' "$f" 2>/dev/null \
    | grep -v ';[[:space:]]*$' \
    | grep -v 'getenv' \
    | sed -E "s/^[^(]*[ *]([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\(.*/\1/" \
    | sed "s|\$|	$f|"
done | sort -u > "$TMP/syms"

for d in i8 f16 i4 i16; do
  case $d in
    i8)  pat='(^|_)i8($|_)|out8' ;;
    f16) pat='(^|_)(f16|fp16)($|_)' ;;
    i4)  pat='(^|_)(i4|i4a8|nf4)($|_)' ;;
    i16) pat='(^|_)i16($|_)|out16' ;;
  esac
  { cut -f1 "$TMP/syms" | grep -E "$pat" || true
    grep "	src/npu/$d/" "$TMP/syms" | cut -f1 || true ; } | sort -u > "$TMP/$d"
done

printf '| capability | i8 | f16 | i4 | i16 |\n|---|:--:|:--:|:--:|:--:|\n'
notes=''
while IFS='	' read -r cap re; do
  [ -n "$cap" ] || continue
  row="| $cap |"
  for d in i8 f16 i4 i16; do
    if grep -qE "$re" "$TMP/$d" 2>/dev/null; then mark='✅'
    elif printf '%s\n' "$OVERRIDES" | grep -q "^$cap	$d	"; then mark='✅†'
    else mark='—'; fi
    row="$row $mark |"
  done
  printf '%s\n' "$row"
done <<CAPS
regcmd synth	^orki_((i8|i4a8|i4|nf4|f16|i16)_)?synth
output stage (requant)	^orki_((i8|i4a8|i4|nf4|f16|i16)_)?set_.*(out8|out16|_?out\$|fp16in)
fused-act output stage	^orki_((i8|i4a8|i4|nf4|f16|i16)_)?set_.*silu
pack weights	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?mm_pack
load / .orkpack persist	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?mm_load|^ork_((i8|i4a8|i4|nf4|f16|i16)_)?w_dump
zero-copy import / adopt	import|adopt
quantise from f32	quant_chan|pack_.*f32|chan_scales
run — single core	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?mm_run\$|^orki_((i8|i4a8|i4|nf4|f16|i16)_)?run\$
run — multicore	mc_db|_mc\$|colsplit|multicore
run — HW chain	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?mm_run_chain|run_chain_impl|bch_db
run — async stream	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?mm_run_stream|stream_worker
run — NONBLOCK doorbell	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?dyn_
batched GEMM (bmm)	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?bmm
fused matmul+activation	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?mm_run_.*(silu|act)
SDP activations	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?npu_(silu|gelu|rsqrt|exp)|act_lut
elementwise mul	ewmul
elementwise add	^ork_((i8|i4a8|i4|nf4|f16|i16)_)?npu_add
per-channel multiply	perchan
slice-and-dice tiles	slice_(pack|run)
M-fold chain	fold
MoE expert coalescing	experts
resident KV	kv_
probes / RE replay	probe|replay|benchmark
regcmd fuzz hooks	fuzz
CAPS
printf '\n'
printf '%s\n' "$OVERRIDES" | while IFS="	" read -r c d sym n; do
  [ -n "$c" ] && printf '† **%s / %s** — %s (`%s`)\n' "$d" "$c" "$n" "$sym"
done
printf '\nfunctions per precision: '
for d in i8 f16 i4 i16; do printf '%s=%s  ' "$d" "$(wc -l < "$TMP/$d" | tr -d ' ')"; done
printf '\n'
