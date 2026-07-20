#!/bin/sh
# dom_teardown_hunt.sh — test whether the cross-domain wedge is a TEARDOWN artifact.
#
# Hypothesis: ork_npu_free (and the SIGTERM cleanup) destroy buffers + per-domain anchors ACROSS multiple
# IOMMU domains then close(fd) — a cross-domain teardown that can leave the kernel IOMMU stuck for the NEXT
# process (every observed wedge surfaced at the next process's first MEM_CREATE). This runs a minimal
# multi-domain process (init -> pack d1 -> loop[run d1 + xdom pack d2] -> clean ork_npu_free) many times and
# checks dmesg AFTER EACH exit. If the wedge appears after a run that ITSELF exited clean (rc=0), the wedge was
# created by that process's TEARDOWN (the next invocation would then fail on its first pack). settle OFF.
#   sudo sh tools/dom_teardown_hunt.sh [invocations]
set -u
export ORK_DOM_SETTLE_US=0
export ORK_MM_TIMEOUT=3000
export ORK_BCREATE_TRACE=/tmp/td_bcreate.log
N=${1:-300}
dmesg -C 2>/dev/null || true
for k in $(seq 1 "$N"); do
    ./dom_race_stress 200 >/tmp/td.log 2>&1
    rc=$?
    if dmesg 2>/dev/null | grep -q 'switch iommu domain time out'; then
        if [ "$rc" -eq 0 ]; then
            echo "*** WEDGE after invocation $k which EXITED CLEAN (rc=0) => its TEARDOWN wedged the IOMMU ***"
        else
            echo "*** invocation $k FAILED (rc=$rc) with the wedge already present => invocation $((k-1))'s TEARDOWN wedged it ***"
        fi
        dmesg 2>/dev/null | grep -iE 'switch iommu|gem_object' | tail
        echo "--- last run tail ---"; tail -3 /tmp/td.log
        exit 1
    fi
    [ "$rc" -ne 0 ] && { echo "invocation $k rc=$rc but no dmesg wedge — tail:"; tail -3 /tmp/td.log; exit 2; }
    [ $((k%25)) -eq 0 ] && echo "  $k clean multi-domain lifecycles"
done
echo "TEARDOWN HUNT: $N clean multi-domain process lifecycles, NO wedge"
