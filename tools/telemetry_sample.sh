#!/bin/bash
# tools/telemetry_sample.sh — run a command and sample NPU per-core load + DMC (RAM bandwidth)
# load while it runs, then report avg/max. The decode bottleneck on this stack is idle time, not
# bandwidth or compute, so tok/s alone is misleading — always sample utilization alongside it.
#
#   sudo tools/telemetry_sample.sh "./bench 28 64 16 i8"
#   sudo tools/telemetry_sample.sh "./rkllm_bench /path/librkllmrt.so /path/model.rkllm 128"
#
# NPU load:  /sys/kernel/debug/rknpu/load  -> "Core0: X%, Core1: Y%, Core2: Z%" (root-only)
# RAM BW:    /sys/class/devfreq/dmc/load   -> "<load>@<freq>Hz"
# Confirm DDR is at performance first: cat /sys/class/devfreq/dmc/cur_freq (want 2112000000 on RK3588).
set -u
[ $# -ge 1 ] || { echo "usage: sudo $0 \"<command to run>\""; exit 1; }
log=$(mktemp); : > "$log"
eval "$1" >/tmp/ts_cmd.out 2>&1 &
bp=$!
while kill -0 "$bp" 2>/dev/null; do
  n=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null)
  d=$(cat /sys/class/devfreq/dmc/load 2>/dev/null)
  c0=$(echo "$n" | grep -oE 'Core0: *[0-9]+' | grep -oE '[0-9]+$')
  c1=$(echo "$n" | grep -oE 'Core1: *[0-9]+' | grep -oE '[0-9]+$')
  c2=$(echo "$n" | grep -oE 'Core2: *[0-9]+' | grep -oE '[0-9]+$')
  echo "${c0:-0} ${c1:-0} ${c2:-0} ${d%@*}" >> "$log"
done
wait "$bp"
echo "--- command output (tok/s etc.) ---"; grep -hiE 'tok/s|DECODE|PREFILL|^decode|^prefill' /tmp/ts_cmd.out | head
echo "--- utilization while running ---"
awk '{n++; for(i=1;i<=4;i++){s[i]+=$i; if($i>m[i])m[i]=$i}}
  END{ if(!n){print "  no samples"; exit}
       printf "  NPU  C0 avg/max=%d/%d  C1=%d/%d  C2=%d/%d\n", s[1]/n,m[1], s[2]/n,m[2], s[3]/n,m[3];
       printf "  RAM bandwidth  avg/max=%d/%d %%   (%d samples)\n", s[4]/n,m[4], n }' "$log"
rm -f "$log"
