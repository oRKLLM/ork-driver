#!/bin/sh
# orkpack_info — print the format version of one or more .orkpack files, and whether this tree can read them.
#
# WHY THIS EXISTS. A pack whose version predates the reader is marked STALE, and stale means REGENERATE:
# pointing the runtime at an old pack does not fail, it silently OVERWRITES it. Measured 2026-08-24: 53 packs
# totalling 220 GiB on the validated board were pre-v6, several of them 15-17 GiB artifacts costing hours to
# rebuild, and a run came seconds from destroying one. Check before you point anything at an old pack.
#
# The footer is the LAST 32 bytes and the magic is its LAST field, not its first -- which is easy to get
# wrong from a hex dump (it was, twice). Layout, little-endian:
#     index_off u64 | n_entries u32 | version u32 | ork_fmt u32 | quant_sig u32 | magic[8] "ORKPK01\0"
#
# Usage:  tools/util/orkpack_info.sh <pack> [pack...]
#         tools/util/orkpack_info.sh ~/*.orkpack        # survey a directory
# Read-only. Never opens a pack for writing.
[ $# -ge 1 ] || { echo "usage: $0 <pack.orkpack> [...]"; exit 2; }

# READABLE_MIN/MAX track the reader's accepted range (additive versions stay readable).
python3 - "$@" <<'PY'
import struct, os, sys
READABLE_MIN, READABLE_MAX = 6, 8
risk = 0
print("%-46s %8s %5s %8s %10s  %s" % ("pack", "GiB", "ver", "entries", "quant_sig", "status"))
for f in sys.argv[1:]:
    if not os.path.isfile(f):
        print("%-46s %8s %5s %8s %10s  %s" % (os.path.basename(f), "-", "-", "-", "-", "NOT FOUND")); continue
    sz = os.path.getsize(f)
    if sz < 32:
        print("%-46s %8.2f %5s %8s %10s  %s" % (os.path.basename(f), sz/1073741824, "-", "-", "-", "TOO SMALL")); continue
    with open(f, "rb") as fh:
        fh.seek(sz - 32); t = fh.read(32)
    idx, nent, ver, fmt, sig = struct.unpack("<QIIII", t[:24])
    if t[24:31] != b"ORKPK01":
        print("%-46s %8.2f %5s %8s %10s  %s" % (os.path.basename(f), sz/1073741824, "?", "?", "?",
              "NO MAGIC — not an orkpack, or truncated")); risk += 1; continue
    if READABLE_MIN <= ver <= READABLE_MAX:
        st = "ok — loads"
    else:
        st = "STALE for this reader -> WOULD BE REGENERATED (destructive)"; risk += 1
    print("%-46s %8.2f %5d %8d %#10x  %s" % (os.path.basename(f), sz/1073741824, ver, nent, sig, st))
if risk:
    print("\n%d file(s) at risk. Archive them, or run with ORK_ORKPACK_CLOBBER=1 only if you mean to discard them." % risk)
    print("Packs over ORK_ORKPACK_MAX_REGEN_MB (default 2048) now refuse to regenerate rather than overwrite.")
sys.exit(1 if risk else 0)
PY
