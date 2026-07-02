#!/usr/bin/env bash
# bench_monitored.sh — run any command (e.g. llama-bench) while sampling RK3588 resource use, then print
# the AVG and PEAK of each resource over the run. Turns a bare tok/s number into an attributable one:
# you SEE whether RAM bandwidth / NPU / GPU / CPU was the wall. Sources match oRKLLM's dashboard (monitor.js):
#   RAM    : /proc/meminfo  (MemTotal-MemAvailable)/MemTotal
#   RAM BW : /sys/class/devfreq/dmc/load  ("<pct>@<freq>Hz")  <- DMC utilisation = RAM-bandwidth pressure
#   NPU    : /sys/kernel/debug/rknpu/load ("Core0: X%, ...")  mean across cores   [needs root]
#   GPU    : /sys/class/devfreq/*.gpu/load ("<pct>@<freq>Hz", Mali)
#   CPU    : /proc/stat aggregate busy% (100% = all cores saturated)
#   SWAP   : /proc/meminfo (SwapTotal-SwapFree)/SwapTotal
#
# Usage:
#   sudo ./tools/bench_monitored.sh [--interval-ms N] [--csv FILE] [--label STR] -- <command...>
# Isolate DECODE so the summary is decode-specific (llama-bench decode-only: -p 0 -n N):
#   sudo env ORK_GROUP=chain ./tools/bench_monitored.sh --label chain -- \
#        ./build/bin/llama-bench -m MODEL.gguf -p 0 -n 128 -t 4 -r 2
# NPU sampling needs root (debugfs) — run the whole thing under sudo so sampler + workload both see it.
set -u

INTERVAL_MS=200; CSV=""; LABEL="run"; SAMPLER_CPUS="0-3"   # RK3588/RK3576 little cores (A55/A53) = cpu0-3
while [ $# -gt 0 ]; do
  case "$1" in
    --interval-ms)  INTERVAL_MS="$2";  shift 2 ;;
    --csv)          CSV="$2";          shift 2 ;;
    --label)        LABEL="$2";        shift 2 ;;
    --sampler-cpus) SAMPLER_CPUS="$2"; shift 2 ;;   # cpu list for the sampler (keep it off the workload's cores)
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "error: no command after --" >&2; exit 2; }
[ -n "$CSV" ] || CSV="$(mktemp /tmp/bench_mon_XXXX.csv)"
STATE="$(mktemp /tmp/bench_mon_cpu_XXXX)"
SLEEP=$(awk "BEGIN{printf \"%.3f\", $INTERVAL_MS/1000}")

# Overall CPU busy% via /proc/stat aggregate-line deltas (prev in $STATE).
cpu_tick() {
  awk -v state="$STATE" '
    BEGIN{ if((getline line < state)>0){split(line,a," ");pt=a[1];pi=a[2]} close(state) }
    /^cpu /{ idle=$5+$6; tot=0; for(i=2;i<=NF;i++)tot+=$i
             dt=tot-pt; di=idle-pi; busy=(dt>0)?100*(dt-di)/dt:0
             if(busy<0)busy=0; if(busy>100)busy=100
             printf "%d %d", tot, idle > state; close(state)
             printf "%.0f", busy; exit }' /proc/stat
}

sampler() {
  cpu_tick >/dev/null 2>&1   # prime the delta baseline
  echo "ts_ms,ram,rambw,npu,gpu,cpu,swap" > "$CSV"
  local gpudir; gpudir="$(ls -d /sys/class/devfreq/*gpu* 2>/dev/null | head -1)"
  while :; do
    local ts ram rambw npu gpu cpu swap n0 n1 n2 mt ma st sf
    ts=$(date +%s%3N)
    # RAM
    mt=$(awk '/^MemTotal/{print $2}' /proc/meminfo); ma=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    ram=$(( mt>0 ? (100*(mt-ma))/mt : 0 ))
    # SWAP
    st=$(awk '/^SwapTotal/{print $2}' /proc/meminfo); sf=$(awk '/^SwapFree/{print $2}' /proc/meminfo)
    swap=$(( st>0 ? (100*(st-sf))/st : 0 ))
    # RAM BW (DMC utilisation %)
    rambw="$(cat /sys/class/devfreq/dmc/load 2>/dev/null | grep -oE '^[0-9]+')"; rambw=${rambw:-0}
    # NPU (mean across cores)
    read -r n0 n1 n2 <<<"$(grep -oE '[0-9]+%' /sys/kernel/debug/rknpu/load 2>/dev/null | tr -d '%' | tr '\n' ' ')"
    n0=${n0:-0}; n1=${n1:-0}; n2=${n2:-0}; npu=$(( (n0+n1+n2)/3 ))
    # GPU
    gpu=0; [ -n "$gpudir" ] && gpu="$(cat "$gpudir/load" 2>/dev/null | grep -oE '^[0-9]+')"; gpu=${gpu:-0}
    # CPU
    cpu="$(cpu_tick)"; cpu=${cpu:-0}
    echo "$ts,$ram,$rambw,$npu,$gpu,$cpu,$swap" >> "$CSV"
    sleep "$SLEEP"
  done
}

sampler & SAMPLER_PID=$!
# Pin the sampler (and the cat/awk/grep it forks, which inherit affinity) to the little cores so it can't
# steal cycles from the -t N big-core workload or the big-core-pinned NPU-driver threads — otherwise the
# monitor perturbs the very numbers it reports.
taskset -cp "$SAMPLER_CPUS" "$SAMPLER_PID" >/dev/null 2>&1 || echo "[monitor] warn: taskset unavailable, sampler not pinned" >&2
cleanup(){ kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "[monitor] label=$LABEL interval=${INTERVAL_MS}ms csv=$CSV sampler_cpus=$SAMPLER_CPUS" >&2
echo "[monitor] === WORKLOAD START ===" >&2
"$@"; RC=$?
echo "[monitor] === WORKLOAD END (rc=$RC) ===" >&2
cleanup; trap - EXIT INT TERM

# Summary: AVG and PEAK for each of the 6 resources.
awk -F, -v label="$LABEL" '
  NR==1{ for(i=2;i<=NF;i++) name[i]=$i; next }
  { n++; for(i=2;i<=NF;i++){ v=$i+0; sum[i]+=v; if(n==1||v>mx[i])mx[i]=v } }
  END{
    if(n==0){ print "[monitor] no samples"; exit }
    printf "\n===== RESOURCE USE (%s, %d samples) =====\n", label, n
    printf "%-8s %8s %8s\n", "metric", "avg%", "peak%"
    for(i=2;i<=NF;i++) printf "%-8s %8.1f %8.0f\n", name[i], sum[i]/n, mx[i]
    print  "==========================================="
  }' "$CSV"
echo "[monitor] csv: $CSV" >&2
rm -f "$STATE"
exit $RC
