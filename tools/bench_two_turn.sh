#!/usr/bin/env bash
set -e

# NOTE: do NOT stop orkllm here. Idle, it sits on the little cores with no NPU
# handle (zero contention) and keeps the telemetry dashboard live; this bench runs
# llama-server directly and never spawns an orkllm worker. Stopping it was the
# wrong lever (it also killed the dashboard) — established policy: leave orkllm up.

# --- Guaranteed performance-governor pinning ----------------------------------
# A benchmark on a parked DDR governor reads ~half the real decode, so we pin DDR
# + the big cores, then VERIFY
# by reading the values back and ABORT if any isn't 'performance' — a non-pinned
# run must never be mistaken for a valid number. Originals are restored on exit.
DMC_GOV=/sys/class/devfreq/dmc/governor
BIG_CORES="4 5 6 7"            # RK3588 A76 cluster; -t 4 runs here
ORIG_DMC_GOV=""
declare -A ORIG_CPU_GOV

pin_and_verify_governors() {
    echo "== Pinning CPU/DDR governors to performance =="
    ORIG_DMC_GOV=$(cat "$DMC_GOV" 2>/dev/null || true)
    local c g
    for c in $BIG_CORES; do
        ORIG_CPU_GOV[$c]=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true)
    done

    echo performance | sudo tee "$DMC_GOV" >/dev/null
    for c in $BIG_CORES; do
        echo performance | sudo tee /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor >/dev/null
    done
    sleep 1

    local fail=0
    local dmc; dmc=$(cat "$DMC_GOV" 2>/dev/null || true)
    [ "$dmc" = "performance" ] || { echo "ERROR: DDR (dmc) governor is '$dmc', expected 'performance'." >&2; fail=1; }
    local dmc_freq; dmc_freq=$(cat /sys/class/devfreq/dmc/cur_freq 2>/dev/null || true)
    echo "  dmc: governor=$dmc cur_freq=$dmc_freq"
    [ "$dmc_freq" = "2112000000" ] || echo "  WARNING: dmc cur_freq is not 2112000000 (RK3588 max) — decode may be DDR-bound." >&2
    for c in $BIG_CORES; do
        g=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || true)
        [ "$g" = "performance" ] || { echo "ERROR: cpu$c governor is '$g', expected 'performance'." >&2; fail=1; }
    done
    if [ "$fail" -ne 0 ]; then
        echo "ERROR: governor pinning failed verification — refusing to benchmark a non-pinned machine." >&2
        exit 1
    fi
    echo "  verified: dmc + cpu[$BIG_CORES] pinned to performance"
}

restore_governors() {
    local c
    [ -n "$ORIG_DMC_GOV" ] && { echo "$ORIG_DMC_GOV" | sudo tee "$DMC_GOV" >/dev/null 2>&1 || true; }
    for c in $BIG_CORES; do
        [ -n "${ORIG_CPU_GOV[$c]:-}" ] && { echo "${ORIG_CPU_GOV[$c]}" | sudo tee /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor >/dev/null 2>&1 || true; }
    done
}

pin_and_verify_governors

# Configuration
SERVER_BIN="${LLAMA_SERVER_BIN:-./llama-server}"
PORT=8085
HOST="127.0.0.1"
URL="http://$HOST:$PORT"
QWEN_1_7B="/var/lib/orkllm/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-UD-Q8_K_XL.gguf"
EAGLE_0_6B="/var/lib/orkllm/models/Netsnake/Qwen3-0.6B-Q4_0-GGUF/qwen3-0.6b-q4_0.gguf"

# Baselines
BASELINE_DECODE=12.5
BASELINE_DECODE_EAGLE=19.0
MAX_DECODE_DROP_RATIO=0.5

start_server() {
    local model=$1
    local draft=$2
    local extra_args=""
    if [ -n "$draft" ]; then
        extra_args="-md $draft"
        echo "Starting server with model: $model and draft: $draft"
    else
        echo "Starting server with model: $model"
    fi

    $SERVER_BIN -m "$model" -c 2048 -t 4 --port $PORT $extra_args > server.log 2>&1 &
    SERVER_PID=$!

    # Wait for health
    local i=0
    while [ $i -lt 60 ]; do
        local health_out=$(curl -s "$URL/health" || true)
        echo "Health out: $health_out"
        if echo "$health_out" | grep -q '"status".*"ok"'; then
            echo "Server is healthy."
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    echo "Server failed to start."
    cat server.log
    kill $SERVER_PID
    exit 1
}
cleanup() { stop_server; restore_governors; }
trap cleanup EXIT

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        SERVER_PID=""
        sleep 3 # Wait for port to be freed
    fi
}
run_turn() {
    local prompt="$1"
    local response=$(curl -s -X POST "$URL/completion" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"n_predict\": 128, \"cache_prompt\": true, \"stream\": false, \"temperature\": 0.0}")
    
    echo "$response"
}

extract_timing() {
    local response="$1"
    local field="$2"
    # Basic extraction using grep and sed to avoid requiring jq if unavailable
    echo "$response" | tr -d '\000' | grep -o "\"$field\": *[0-9.]*" | cut -d':' -f2 | tr -d ' '
}

extract_content() {
    local response="$1"
    echo "$response" | tr -d '\000' | grep -o '"content": "[^"]*"' | sed 's/"content": "//' | sed 's/"$//'
}

run_benchmark() {
    local model_type="$1"
    local expected_baseline="$2"
    
    echo "Running Turn 1..."
    local turn1_prompt="<|im_start|>user\nPlease tell me a short story about a brave knight.<|im_end|>\n<|im_start|>assistant\n"
    local res1=$(run_turn "$turn1_prompt")
    
    local prefill1=$(extract_timing "$res1" "prompt_per_second")
    local decode1=$(extract_timing "$res1" "predicted_per_second")
    local content=$(extract_content "$res1")
    
    echo "Response 1: $res1"
    
    if [ -z "$decode1" ]; then
        echo "Failed to get timings for Turn 1"
        exit 1
    fi
    
    echo "Turn 1: Prefill: $prefill1 tok/s | Decode: $decode1 tok/s"
    
    echo "Running Turn 2..."
    local turn2_prompt="${turn1_prompt}${content}<|im_end|>\n<|im_start|>user\nWhat was the knight's horse named?<|im_end|>\n<|im_start|>assistant\n"
    local res2=$(run_turn "$turn2_prompt")
    
    local prefill2=$(extract_timing "$res2" "prompt_per_second")
    local decode2=$(extract_timing "$res2" "predicted_per_second")
    
    if [ -z "$decode2" ]; then
        echo "Failed to get timings for Turn 2"
        exit 1
    fi
    
    echo "Turn 2: Prefill: $prefill2 tok/s | Decode: $decode2 tok/s"
    
    # Calculate ratio (using awk for floating point math)
    local drop_ratio=$(awk "BEGIN {print $decode2 / $decode1}")
    echo "Decode Turn 2 / Turn 1 ratio: ${drop_ratio}x"
    
    # Check drop
    local drop_check=$(awk "BEGIN {print ($drop_ratio < $MAX_DECODE_DROP_RATIO) ? 1 : 0}")
    if [ "$drop_check" -eq 1 ]; then
        echo "ERROR: Significant performance degradation in Turn 2 decode ($drop_ratio x). Likely KV-cache append/M-scheduling issue."
        stop_server
        exit 1
    fi
    
    # Check baseline
    local baseline_check=$(awk "BEGIN {print ($decode2 < $expected_baseline) ? 1 : 0}")
    if [ "$baseline_check" -eq 1 ]; then
        echo "ERROR: Decode performance ($decode2) is below baseline threshold ($expected_baseline tok/s) for $model_type."
        stop_server
        exit 1
    fi
    echo "$model_type PASS."
}

# 1. Normal configuration
start_server "$QWEN_1_7B" ""
run_benchmark "Base Model (No Draft)" "$BASELINE_DECODE"
stop_server
echo ""

# 2. Speculative decoding configuration
start_server "$QWEN_1_7B" "$EAGLE_0_6B"
run_benchmark "Base Model + Eagle Draft" "$BASELINE_DECODE_EAGLE"
stop_server

echo "All configurations passed."
