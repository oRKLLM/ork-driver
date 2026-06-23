#!/usr/bin/env bash
set -e

sudo systemctl stop orkllm || true

# Set maximum performance governors
echo "performance" | sudo tee /sys/class/devfreq/dmc/governor || true
for i in 4 5 6 7; do
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor || true
done

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

    ORK_VERBOSE=1 $SERVER_BIN -m "$model" -c 2048 -t 4 --port $PORT $extra_args > server.log 2>&1 &
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
trap stop_server EXIT

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
