#!/bin/sh
# dom_wedge_hunt.sh — hunt the rare cross-domain IOMMU "switch iommu domain time out" wedge.
#
# The wedge did NOT reproduce in isolated tiny-op loops (~1.44M ops clean). This attacks the untested
# dimensions with the SETTLE OFF (ORK_DOM_SETTLE_US=0 — so the insurance settle can't mask it):
#   - OP SIZE sweep: a bigger op has a longer compute + longer RETIREMENT tail, which should widen the
#     switch-idle-wait race window (the prime untested lever). tiny -> medium -> large.
#   - the cross-domain bcreate path (dom_pack_stress).
#   - domain-0 work interleaved (quant) so the kernel IOMMU sees N->0->N transitions across processes.
# Stops on the FIRST dmesg wedge and dumps the presubmit trace tail (the culprit submit). Run under sudo on
# the board:  sudo sh tools/dom_wedge_hunt.sh [rounds]
set -u
export ORK_DOM_SETTLE_US=0
export ORK_MM_TIMEOUT=3000
export ORK_PRESUBMIT_TRACE=/tmp/hunt_presub.log
ROUNDS=${1:-30}
dmesg -C 2>/dev/null || true
rm -f /tmp/hunt_presub.log 2>/dev/null || true

check(){   # $1 = label of the workload just run
    if dmesg 2>/dev/null | grep -q 'switch iommu domain time out'; then
        echo "*** WEDGE reproduced after: $1 (round $r) ***"
        dmesg 2>/dev/null | grep -iE 'switch iommu|gem_object' | tail
        echo "--- presubmit trace tail (culprit submit) ---"
        tail -4 /tmp/hunt_presub.log 2>/dev/null || echo '(no trace)'
        exit 1
    fi
}

for r in $(seq 1 "$ROUNDS"); do
    echo "== round $r/$ROUNDS =="
    ORK_NO_REF=1 ./dom_switch_stress 20000 8   512  64   >/dev/null 2>&1; check "dom_switch_stress tiny (8/512/64)"
    ORK_NO_REF=1 ./dom_switch_stress 8000  64  2048 512  >/dev/null 2>&1; check "dom_switch_stress med (64/2048/512)"
    ORK_NO_REF=1 ./dom_switch_stress 3000  256 4096 2048 >/dev/null 2>&1; check "dom_switch_stress LARGE (256/4096/2048)"
    ORK_NO_REF=1 ./dom_switch_stress 5000  128 2048 1024 >/dev/null 2>&1; check "dom_switch_stress big (128/2048/1024)"
    ./dom_pack_stress 40000 >/dev/null 2>&1; check "dom_pack_stress"
    ./quant >/dev/null 2>&1; check "quant (domain 0, cross-process N->0->N)"
done
echo "HUNT: $ROUNDS rounds, mixed sizes + pack + dom0, settle OFF — NO wedge reproduced"
