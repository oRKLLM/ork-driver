#!/bin/bash
# tools/i4_clk_envelope.sh — is the int4 BCHAIN geometry envelope a function of the DRAM clock?
#
# WHY. orki_i4_hcap() and the bank-width N-tile Wb are constants measured on ONE board. A second
# RK3588 accepts the submit and never starts the task at geometries the first one runs. This probe
# answers which clock the envelope actually follows, by capping the NPU devfreq and then the DMC
# devfreq while running one shape known to fail and one known to pass.
#
# Measured 2026-10-10 on 10.3.0.235 (ROCK 5B+, LPDDR5): the bad shape fails at EVERY NPU frequency
# (1000/800/600/400 MHz) and PASSES as soon as DMC drops 2400 -> 1968, with the correct checksum. So
# the DRAM clock is the variable and the NPU clock is not. 10.3.0.236 (LPDDR4X 2112 MHz) sits below
# the threshold, which is why the table held until a board with faster memory existed. Full write-up:
# wiki Exp-2026-10-09-The-Domain-Reference-Leak.
#
# Run it on any NEW board before trusting the measured geometry there:
#   make i4_hcap_probe && sudo tools/util/npu_guard.sh -- bash tools/i4_clk_envelope.sh
#
# ORK_I4_NOLADDER=1 is set deliberately: the geometry ladder in orki_i4_run_bchain_db exists to HIDE
# this failure, so leaving it on would make every row pass and measure nothing.
set -u
NPU=/sys/class/devfreq/fdab0000.npu
DMC=/sys/class/devfreq/dmc
BAD=${BAD:-"2048 64 16"}      # K N M — the table's H=8 at K=2048
GOOD=${GOOD:-"512 64 16"}     # control: expected to pass everywhere

run() { timeout -s TERM 150 env ORK_I4_NOLADDER=1 ORK_MM_TIMEOUT=1500 ./i4_hcap_probe $1 2>/dev/null | tail -1; }
row() { echo -n "  BAD  : "; run "$BAD"; echo -n "  GOOD : "; run "$GOOD"; }

[ -x ./i4_hcap_probe ] || { echo "build it first: make i4_hcap_probe"; exit 2; }

echo "### baseline npu=$(cat $NPU/cur_freq) dmc=$(cat $DMC/cur_freq)"; row
for f in 800000000 600000000 400000000; do
    echo "$f" | sudo tee $NPU/max_freq >/dev/null
    echo "### npu max_freq=$f (cur=$(cat $NPU/cur_freq))"; row
done
echo "1000000000" | sudo tee $NPU/max_freq >/dev/null

# userspace governor for the DMC sweep, and a 2400 arm INSIDE it so the governor switch itself is
# controlled for -- otherwise "1968 passes" could just be "changing the governor passes".
GOV=$(cat $DMC/governor)
echo userspace | sudo tee $DMC/governor >/dev/null
for d in 2400000000 1968000000 1320000000 2400000000; do
    echo "$d" | sudo tee $DMC/userspace/set_freq >/dev/null; sleep 1
    echo "### dmc=$(cat $DMC/cur_freq) (userspace) npu=$(cat $NPU/cur_freq)"; row
done
echo "$GOV" | sudo tee $DMC/governor >/dev/null
echo "### restored dmc=$(cat $DMC/governor)/$(cat $DMC/cur_freq) npu_max=$(cat $NPU/max_freq)"
