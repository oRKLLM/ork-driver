# tools/util — board workflow helpers (Mac-side)

Convenience tooling for long board sessions. Neither is required to build or use ork-driver; they
just remove friction when iterating against the RK3588 board over SSH (host alias `board`).

## `board` — SSH command wrapper (no nested-quote hell)

```sh
tools/util/board -c 'make test'           # single command (cd's into ~/ork-driver first)
tools/util/board tools/util/probe.sh      # run a local script file on the board
tools/util/board <<'EOF'                   # pipe a script via stdin
  make fold_tiler && sudo ./fold_tiler 72
EOF
```

Reached via the SSH host alias `board` (override with `BOARD_HOST=...`). Add to PATH
(`ln -s "$PWD/tools/util/board" ~/bin/board`) to call it as `board`. Authoring board-side scripts as
files and piping them avoids the escaping pitfalls of `ssh board '...'`.

## `sync_daemon.sh` — bidirectional polling rsync (Mac ⇄ board)

2 s-cadence rsync over a persistent SSH master: pushes Mac source-of-truth
(`src/ include/ tools/ examples/ docs/ Makefile`, no `--delete`) → `board:ork-driver/`, and pulls
`board:~/ork-outbox/` → a local outbox. Launch once in the background; then just build/run on the
board without per-edit hand-syncing. Put board-side artifacts you want back (dumps, logs, captured
regcmds) in `~/ork-outbox/`. Source-of-truth stays the Mac; the board is build+run only (consistent
with the "no macOS binary transfers" rule).

```sh
tools/util/sync_daemon.sh &     # edit REPO/OUTBOX_LOCAL paths at the top for your checkout
```

## `npu_guard.sh` — run this before anything that touches the NPU

    sudo tools/util/npu_guard.sh                 # check only: exit 0 = free, non-zero = DO NOT START
    sudo tools/util/npu_guard.sh -- make test    # take the lock, re-check, then run
    ORK_NPU_LOCK_WAIT=300 sudo -E tools/util/npu_guard.sh -- make test   # queue instead of failing

The RK3588 NPU is single-stream: two concurrent direct-NPU processes wedge the IOMMU, and recovering costs
a reboot or a power-cycle. The guard checks three things — who holds the render node (via `/proc/*/fd`, so
it sees the device rather than a process name), whether NPU utilisation is idle, and whether a fault storm
is already in progress (a wedge in progress looks idle by utilisation alone).

`--` additionally holds an `flock` for the lifetime of the command, which closes the check-then-launch race.
Without it two sessions can both pass the check and then collide.

Origin: written by a parallel session as `/tmp/npu_guard.sh`; landed here so it survives a reboot and every
session gates on the same check. The lock wrapper was added on top; the detection logic is unchanged.

## `ppl_screen.sh` — two-tier perplexity comparison (board-side)

Quantization work needs a fast comparator far more often than it needs a publishable number. A full-text
PPL run is ~15 min per arm here, so a 2x2 experiment costs an hour — and an hour-long loop is one you stop
running. This splits the question in two:

| tier | config | cost | use |
|---|---|---|---|
| `SCREEN` (default) | 1 window x 256 tok | ~30 s/arm | the iteration loop: rank changes, catch breakage |
| `--full` | whole text, 512-tok windows | ~15 min/arm | the release-candidate gate |

```sh
sudo tools/util/ppl_screen.sh ~/model.gguf ~/ppl_text.txt fp64=~/a.orkpack fp32=~/b.orkpack
sudo tools/util/ppl_screen.sh --full ~/model.gguf ~/ppl_text.txt cand=~/rc.orkpack
```

**Read the screen correctly.** All arms score the identical token sequence with deterministic models, so the
PAIRED difference between arms is exact — no sampling noise between them. What a 255-token run cannot tell
you is whether that difference GENERALISES to the full text. So: screen to rank and to catch breakage,
`--full` before you change a default or quote a number. A screen result is a reason to keep going, never a
release claim.

Runs one arm at a time under `npu_guard.sh` (the NPU is single-stream and the board is shared).

## `orkpack_info.sh` — check a pack's format version BEFORE pointing the runtime at it

A pack older than the reader is marked *stale*, and stale means **regenerate**: opening an old pack does not
fail, it silently **overwrites** it. On 2026-08-24 the validated board held 53 pre-v6 packs totalling
220 GiB — several 15-17 GiB artifacts costing hours to rebuild — and a run came seconds from destroying one.

```sh
tools/util/orkpack_info.sh ~/*.orkpack     # survey; exit 1 if any are at risk
```

Read-only. Prints version, entry count, quant_sig and whether this tree can load each pack. The footer is
the last 32 bytes with the magic as its LAST field, which is easy to misread from a hex dump — this encodes
it once so nobody has to.

Packs above `ORK_ORKPACK_MAX_REGEN_MB` (default 2048) now refuse to regenerate; `ORK_ORKPACK_CLOBBER=1`
overrides when discarding is genuinely intended.
