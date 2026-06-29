# ork-driver — userspace regcmd matmul library for the Rockchip NPU.
# Build on a Rockchip board (needs /dev/dri/cardN + the in-tree rknpu DRM driver), or
# cross-compile for aarch64. No external dependencies (libc + the kernel DRM uABI only).
CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -Wall -Iinclude -Isrc -pthread   # -pthread: multi-core path uses worker threads

# Stamp the build with a short git hash WHEN git + a repo are present (workstation builds). On the
# board (no git) this is empty and ork_npu_version() returns the bare semver — no failure either way.
GIT_HASH := $(shell git rev-parse --short=7 HEAD 2>/dev/null)
ifneq ($(GIT_HASH),)
CFLAGS += -DORK_GIT_HASH=\"$(GIT_HASH)\"
endif
PREFIX  ?= /usr/local
CORE    := src/npu.c src/soc.c src/soc/rk3588.c src/soc/rk3576.c src/neon_activations.c
EXAMPLES := test_matmul quant i4 layer decode model llama2 bench perplexity_i4 test_baseline test_registers test_layouts test_speed test_chain_i4 test_activations
TESTS    :=

all: $(EXAMPLES) $(TESTS)

$(EXAMPLES): %: examples/%.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

$(TESTS): %: %.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE diagnostic (NOT in `make test`): probes whether the captured PPU LUT/PWL regcmd can be
# driven STANDALONE. NEGATIVE RESULT on RK3588 — the PPU does not activate from an isolated
# replay (output buffer comes back unwritten). Kept as a runnable ground-truth probe; exits
# nonzero by design. See wiki Exp-2026-06-24-PPU-LUT-Silicon-Verification.
test_ppu_lut: examples/test_ppu_lut.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# --- library for embedding in other projects (e.g. llama.cpp-rockchip, FFI bindings) ---
lib: libork_npu.a libork_npu.so

libork_npu.a: $(CORE:.c=.o)              # static — link directly, no runtime .so dependency
	$(AR) rcs $@ $^
libork_npu.so: $(CORE)                   # shared — dynamic link / FFI from Python, Node, Rust, ...
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(CORE)
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# comparison tool: benchmark the closed librkllmrt (dlopen'd at runtime, no build dep).
# Not in `all`/`test` — it needs a .rkllm model + the runtime present on the board.
rknpu_bench: tools/rknpu_bench.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

# RE/calibration probe: sweeps K to find this SoC's single-submit K-tile ceiling (int8).
# Not in `all`/`test` — it intentionally wedges the NPU past the cap (recoverable).
ksubmit_probe: tools/ksubmit_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE/calibration probe: ork-current vs rkllm-captured M-tile mode, bit-exact + effective GOPS.
mtile_probe: tools/mtile_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Tiling AUTOTUNER: per-shape tiling-config search (cores x M/N/K-tile), bit-exact-gated + GOPS.
autotune: tools/autotune.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Streaming/persist diagnostics (board only). dma_probe is standalone (raw rknpu ioctls): measures the
# NPU's ~4 GiB IOVA window. stream_probe proves ork_mm_free reclaims IOVA. persist_probe proves the
# .orkpack dump/load roundtrip is byte-identical and far faster than packing.
dma_probe: tools/dma_probe.c
	$(CC) $(CFLAGS) -o $@ $<
# domain_probe is standalone (raw rknpu ioctls): proves >4 GiB residence by splitting buffers across
# multiple IOMMU domains (iommu_domain_id field) — single domain caps ~4 GiB, 8 domains reach ~32 GiB.
domain_probe: tools/domain_probe.c
	$(CC) $(CFLAGS) -o $@ $<
stream_probe: tools/stream_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
persist_probe: tools/persist_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
# Zero-copy weight IMPORT + fixed-slot streaming validation/measurement (board only; needs dma-heap).
zerocopy_import_test: tools/zerocopy_import_test.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
# int4 prefetch-inflate streaming probe: does a background fill+map of slot N+1 hide behind submit(N)?
stream_prefetch_probe: tools/stream_prefetch_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
# domain_correct: multi-domain int8 run is bit-exact vs CPU ref (per-weight IOMMU domain threading).
domain_correct: tools/domain_correct.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
# grouped_i4_leak: pack/free a grouped-int4 weight in a loop; RSS/IOVA must stay flat (ork_mm_free reclaim).
grouped_i4_leak: tools/grouped_i4_leak.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE probe: hunt the weight stride register for in-place K-slicing of a full-K buffer.
slice_probe: tools/slice_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE probe: derive the w4a16 (int4 weight x fp16 activation) regcmd by sweeping vs a CPU reference
# (the runtime won't expose int4 on this board; the NPU does it — we emit the regcmd directly).
i4_probe: tools/i4_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# load a real GGUF Q4_K model, dequant a weight, run it on the NPU (requantized to int8/W8A8).
gguf_q4k: tools/gguf_q4k.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Tier 4a: does a Hadamard rotation make int4 (W4A4) accurate? Pure CPU math (no NPU) — compares the
# int4 matmul quant error vs the fp32 reference, plain vs Hadamard-rotated. See ROADMAP Tier 4.
hadamard_int4: tools/hadamard_int4.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# Tier 4a on a REAL weight tensor: dequant a Q4_K weight from a GGUF and run the same int4/Hadamard
# error comparison on real model weights (random data is worst-case for quant — this removes that).
hadamard_real: tools/hadamard_real.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# Tier 4b: time PER-CHANNEL int4 (full-K single submit) vs grouped int4 (K/G submits) vs int8 —
# does per-channel int4 reach int8 speed (kill the grouped submit explosion)? Needs the NPU.
int4_bench: tools/int4_bench.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Tier 4b RE: probe whether the int4 regcmd does multi-M in one submit + brute-force the output layout.
i4_multim_probe: tools/i4_multim_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Tier 4b RE: exhaustive register fuzzer for undocumented int4 multi-M row execution.
i4_multim_fuzz: tools/i4_multim_fuzz.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# find the int8 prefill correctness bug: exact int8 matmul vs CPU ref across real model shapes/M/cores.
prefill_check: tools/prefill_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE/calibration probe: sweeps registers to find vector/PPU instructions
vec_fuzz: tools/vec_fuzz.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Diagnostic (NOT in `make test`): anatomy of the post-NPU-fence latency gap (submit/bsync/copy + NEON consume).
fence_probe: tools/fence_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Diagnostic (NOT in `make test`): speculative-decoding economics — M-batch amortization of the GEMV
# submit floor + the CPU accept/reject gate cost. Mock data (structural overhead, not model quality).
test_speculative_bridge: tools/test_speculative_bridge.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE diagnostic (NOT in `make test`): fuller-slice replay of the captured RKNN Sigmoid PPU op.
# Reconstructs the full 5-buffer topology from examples/sigmoid_slice.h (extracted from the SDK
# trace) to test whether populating the LUT/PWL config buffers (handles 4 & 5) makes the PPU write.
slice_replay: tools/slice_replay.c $(CORE) examples/sigmoid_slice.h
	$(CC) $(CFLAGS) -Iexamples -o $@ $< $(CORE) -lm

# RE probe: does batching tasks per RKNPU_SUBMIT amortize the per-matmul submit-latency floor?
batch_probe: tools/batch_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# apples-to-apples: SAME int8 matmul via ork-driver vs the closed RKNN matmul API (librknnrt).
# Not in `all`/`test` — needs librknnrt.so + rknn_matmul_api.h present. Point RKNN_DIR at them.
#   make rknn_vs_ork RKNN_DIR=/tmp/rknn && sudo env LD_LIBRARY_PATH=/tmp/rknn ./rknn_vs_ork [iters]
RKNN_DIR ?= /tmp/rknn
rknn_vs_ork: tools/rknn_vs_ork.c $(CORE)
	$(CC) $(CFLAGS) -I$(RKNN_DIR) -o $@ $< $(CORE) -L$(RKNN_DIR) -lrknnrt -lm

# diagnostic: why does large-M (prefill) multi-core barely scale? per-core copy/submit/acc split.
mc_prof: tools/mc_prof.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic: time a batched chain single-core vs cross-core fan-out (ORK_CHAIN_MC). Throughput only.
chain_bench: tools/chain_bench.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic: static gated "exhaustive MoE ring" — all experts in one chained submit, per-token zero-mask
test_exhaustive_moe: tools/test_exhaustive_moe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic: isolate ork_mm_pack_i8_f32 (NEON f32->int8) vs pack_i8 on identical logical weights
pack_f32_probe: tools/pack_f32_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic (P5.3): can a CPU prefetch thread hide per-slice streaming prep (int4->int8 inflate + tile
# into a reused DMA buffer) behind the NPU computing the previous slice? Splits prep into inflate vs tile.
prefetch_headroom: tools/prefetch_headroom.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic: verify (bit-identical tiles + exact matmul) + bench the DIRECT int4->int8-tiled inflate
# (ORK_DIRECT_I4, no f32 round-trip) vs the int4->f32->tile path, UNIFORM and NF4.
direct_i4_test: tools/direct_i4_test.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic (Phase-5 M0 gate): is a sparse-MoE expert FFN GEMM on the NPU (DIRECT int4-NF4 inflate +
# cacheable tiled buf, single + chained) now competitive with a NEON int8 CPU GEMM? vs the old f32 repack.
moe_expert_probe: tools/moe_expert_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# adversarial verify of the 21ms-floor claim: LFM2.5 expert dims at M=1, RESIDENT int8 weights,
# run_i8 vs run_chain_i8 vs run_stream_i8 vs CPU NEON i8.
moe_m1_probe: tools/moe_m1_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# FEASIBILITY: does an NPU expert submit OVERLAP with CPU expert compute at M=1, and does a
# concurrent split beat the CPU-fused MoE? (LFM2.5/Qwen3.6 decode shapes; crossing isolated.)
moe_concurrent_probe: tools/moe_concurrent_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm -lpthread

# BATCHED (M>1) extension of moe_concurrent_probe: at what batch size M does NPU||CPU distributed
# expert execution beat the CPU-fused MoE? Sweeps M=1..128, best NPU/CPU split per M; crossing tracked.
# -march=native: enable ARM SDOT (asimddp) so the CPU baseline (T_cpu_all) uses the same int8 dot kernel
# llama.cpp uses on RK3588 — a FAIR baseline, not the slower vmull/vpadal fallback.
moe_batched_probe: tools/moe_batched_probe.c $(CORE)
	$(CC) $(CFLAGS) -march=native -o $@ $< $(CORE) -lm -lpthread

# diagnostic (P5.3 follow-on): does filling the resident NPU dma_buf from DISK (mmap warm/cold, pread cold)
# add a penalty vs RAM, and how much cheaper is the pre-tiled int8 fill than the int4 inflate+tile path?
disk_stream_bench: tools/disk_stream_bench.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic: can we break the ~1.06 GB/s weight dma_buf fill wall? A/B write-combine (0x401) vs cacheable
# (0x403) vs NEON streaming-store fills, each with an NPU-read correctness check vs a CPU int8 reference.
dmabuf_fill_probe: tools/dmabuf_fill_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic: Tier 4a per-group Hadamard for W4A4, validated end-to-end ON THE NPU (plain vs rotated RMS)
hadamard_i4_npu: tools/hadamard_i4_npu.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# install the public header + both libs (override PREFIX=/path as needed)
install: libork_npu.a libork_npu.so
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m644 libork_npu.a libork_npu.so $(DESTDIR)$(PREFIX)/lib/
	install -m644 include/ork_npu.h $(DESTDIR)$(PREFIX)/include/

# --- tests: the examples ARE the tests (each self-validates vs a CPU reference and exits
# 0/nonzero). Run them on the board; a wall timeout catches an NPU hang. The llama2 test
# needs a model and is skipped when absent. ---
MODEL ?= stories15M.bin
# per-test wall timeout (s) — catches an NPU hang. test_matmul's full shape + ChainPrefill sweep
# is ~3m15s, so the wall must exceed it; still bounds a genuine hang. Override: make test TEST_TIMEOUT=120
TEST_TIMEOUT ?= 360
test: $(EXAMPLES) $(TESTS)
	@fail=0; \
	for t in "test_activations" "test_matmul" "quant" "i4" "perplexity_i4" "layer" "decode" "model 1" "model 12" "test_speed" "test_chain_i4"; do \
	  echo "== $$t"; timeout $(TEST_TIMEOUT) sudo ./$$t || fail=1; done; \
	if [ -f "$(MODEL)" ]; then echo "== llama2 $(MODEL)"; timeout $(TEST_TIMEOUT) sudo ./llama2 "$(MODEL)" 6 || fail=1; \
	  else echo "== llama2 SKIP (no $(MODEL))"; fi; \
	if [ $$fail -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "TESTS FAILED"; exit 1; fi

bench-llama:
	@echo "== Running two-turn conversation integration benchmark via llama-server =="
	@LLAMA_SERVER_BIN=$(HOME)/llama.cpp/build/bin/llama-server tools/bench_two_turn.sh

clean:
	rm -f $(EXAMPLES) $(TESTS) rknpu_bench vec_fuzz test_ppu_lut libork_npu.a libork_npu.so src/*.o src/soc/*.o

.PHONY: all lib install test clean

attn_cost: tools/attn_cost.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

attn_cost7b: tools/attn_cost7b.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
