#!/bin/bash
# Is the int4 BCHAIN H envelope a TIMING envelope (clock-dependent) or a capacity one?
# Cap the NPU devfreq, then DDR, and re-run one known-bad shape and one known-good control.
cd /home/michael/ork-driver
NPU=/sys/class/devfreq/fdab0000.npu
DMC=/sys/class/devfreq/dmc

run() { timeout -s TERM 150 env ORK_MM_TIMEOUT=1500 ./i4_hcap_probe "$@" 2>/dev/null | tail -1; }

echo "### baseline npu=$(cat $NPU/cur_freq) dmc=$(cat $DMC/cur_freq)"
echo -n "  BAD  K2048 N64 M16 : "; run 2048 64 16
echo -n "  GOOD K512  N64 M16 : "; run 512 64 16

for f in 800000000 600000000 400000000; do
  echo "$f" | sudo tee $NPU/max_freq >/dev/null
  echo "### npu max_freq=$f (cur=$(cat $NPU/cur_freq))"
  echo -n "  BAD  K2048 N64 M16 : "; run 2048 64 16
  echo -n "  GOOD K512  N64 M16 : "; run 512 64 16
done
echo "1000000000" | sudo tee $NPU/max_freq >/dev/null

for d in 1968000000 1320000000; do
  echo userspace | sudo tee $DMC/governor >/dev/null 2>&1
  echo "$d" | sudo tee $DMC/userspace/set_freq >/dev/null 2>&1
  echo "### dmc=$(cat $DMC/cur_freq) npu=$(cat $NPU/cur_freq)"
  echo -n "  BAD  K2048 N64 M16 : "; run 2048 64 16
  echo -n "  GOOD K512  N64 M16 : "; run 512 64 16
done
echo performance | sudo tee $DMC/governor >/dev/null
echo "### restored dmc=$(cat $DMC/cur_freq) npu_max=$(cat $NPU/max_freq)"
