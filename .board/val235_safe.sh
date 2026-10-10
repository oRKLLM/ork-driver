#!/bin/bash
# make test's list MINUS `i4` and `test_slice_rescue` — the two that drove the hard wedge on this
# kernel. Everything else, in the same order, under the guard.
cd /home/michael/ork-driver
echo "=== ORK-SAFE-SUITE-START $1 ===" | sudo tee /dev/kmsg >/dev/null
fail=0
ORKD_BIN=$PWD/orkd; export ORKD_BIN
for t in "test_api_parity" "test_spine" "test_activations" "test_matmul" "test_bmm" "quant" \
         "perplexity_i4" "layer" "decode" "model 1" "model 12" "test_speed" "test_chain_i4" \
         "test_gptq" "test_i4_dump_cpu" "test_cpu_gemm" "test_offline_load" "test_f16_load" \
         "test_grouped_i8" "test_i4_gemm" "test_sn3" "test_affinity" "test_stream_interleave" \
         "test_mm_i8_out8" "test_silu_native" "test_ewmul_i8" "test_ewmul_f16" "test_ewmul_i16" \
         "test_silu" "test_add" "test_gelu" "test_ssd_chunk" "test_ssd_chunk_npu" \
         "test_mode_transition" "test_f16colsplit" "chain_xition_probe" "test_bmm_fused" \
         "chainrr_conc_probe"; do
  echo "== $t"
  sudo -E timeout -k 15 600 ./$t || { echo "!! FAILED: $t"; fail=1; }
done
for t in "orkd_probe mm" "orkd_ring_probe" "orkd_dom_api"; do
  echo "== $t"
  sudo -E timeout -k 15 600 ./$t || { echo "!! FAILED: $t"; fail=1; }
done
echo "== orkd_seq_probe (Path B)"
sudo -E env ORK_USE_ORKD=1 timeout -k 15 600 ./orkd_seq_probe >/tmp/ork_seq.log 2>&1 || { echo "!! FAILED: orkd_seq_probe"; fail=1; }
tail -4 /tmp/ork_seq.log
echo "== reaping orkd (SIGTERM; never -9)"
sudo pkill -TERM -x orkd 2>/dev/null; sleep 3
echo "SAFE-SUITE fail=$fail"
echo "=== ORK-SAFE-SUITE-END $1 ===" | sudo tee /dev/kmsg >/dev/null
