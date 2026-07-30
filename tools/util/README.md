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
