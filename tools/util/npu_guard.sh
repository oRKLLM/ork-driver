#!/bin/bash
# npu_guard.sh — assert the NPU is idle and unclaimed BEFORE running anything on it.
#
# ORIGIN: written by a parallel Claude session working this board, as /tmp/npu_guard.sh. Landed here
# essentially verbatim (one stray character removed from a comment) because /tmp does not survive a reboot
# and every session sharing this board should gate on the SAME check. The detection logic below is theirs.
#
# WHY IT EXISTS, concretely: the RK3588 NPU is single-stream, and two concurrent direct-NPU processes wedge
# the IOMMU. A name-based `pgrep` is NOT sufficient to detect this — that is how the incident this file
# prevents actually happened. One session polled `pgrep -x make`, but the other session's suite runs its
# recipe as `/bin/sh -c fail=0; for t in ...`, so make never matched, the board read as free, and the two
# runs overlapped: three `RKNPU: switch iommu domain time out` faults, and an orphaned orkd holding a domain
# that blocked the other run's domain switches until it was SIGTERMed.
#
# Checking the RENDER NODE via /proc/*/fd is the fix: it sees whoever actually has the device open,
# whatever the process is called.
#
#   sudo tools/util/npu_guard.sh                    # check only; exit 0 = free, non-zero = DO NOT START
#   sudo tools/util/npu_guard.sh -- make test       # take the lock, re-check, then run (see --lock below)
#   sudo tools/util/npu_guard.sh --since-uptime N   # only count RKNPU faults newer than N seconds uptime
#
# THE LOCK (added on top of the original): checking and then launching is a RACE — the window between
# "guard says free" and "my submit starts" is exactly where the incident above landed. With `--`, the guard
# holds an flock for the lifetime of the command, so a second session's guard blocks instead of racing.
# Sessions that only ever call the plain check still get the old behaviour.

LOCK=/tmp/ork-npu.lock
if [ "$1" = "--" ]; then
    shift
    [ $# -gt 0 ] || { echo "npu_guard: -- needs a command"; exit 2; }
    exec 9>"$LOCK" || { echo "npu_guard: cannot open $LOCK"; exit 2; }
    if ! flock -w "${ORK_NPU_LOCK_WAIT:-0}" 9; then
        echo "GUARD BLOCKED: another session holds $LOCK (set ORK_NPU_LOCK_WAIT=<seconds> to wait)"; exit 1
    fi
    "$0" || exit $?          # run the checks below with the lock held
    echo "guard: lock held, running: $*"
    "$@"; rc=$?
    echo "guard: released $LOCK"
    exit $rc
fi
rc=0
# 1) definitive: who has the render node open (name-based pgrep misses renamed binaries)
holders=""
for p in /proc/[0-9]*; do
  pid=${p#/proc/}
  [ "$pid" = "$$" ] && continue
  if ls -l "$p/fd" 2>/dev/null | grep -q 'renderD12[89]'; then
    holders="$holders $pid($(cat "$p/comm" 2>/dev/null))"
  fi
done
if [ -n "$holders" ]; then echo "GUARD FAIL: render node held by:$holders"; rc=1; else echo "guard: render node unclaimed"; fi

# 2) NPU utilisation must be idle
load=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null)
busy=$(echo "$load" | grep -oE '[0-9]+%' | tr -d '%' | sort -rn | head -1)
echo "guard: NPU load = ${load:-unavailable}"
[ -n "$busy" ] && [ "$busy" -gt 5 ] && { echo "GUARD FAIL: NPU busy (${busy}%)"; rc=1; }

# 3) no in-flight fault storm (a wedge in progress looks idle by utilisation alone)
since=0; [ "$1" = "--since-uptime" ] && since=$2
now=$(cut -d' ' -f1 /proc/uptime)
recent=$(dmesg 2>/dev/null | grep -E 'job abort|job timeout|job commit failed' \
         | sed -n 's/^\[ *\([0-9.]*\)\].*/\1/p' | awk -v s="$since" -v n="$now" '$1>s && (n-$1)<120' | wc -l)
echo "guard: RKNPU faults in the last 120s = $recent"
[ "$recent" -gt 0 ] && { echo "GUARD WARN: recent NPU faults — a wedge may be in progress"; rc=1; }

[ $rc -eq 0 ] && echo "GUARD OK — NPU is free" || echo "GUARD BLOCKED"
exit $rc
