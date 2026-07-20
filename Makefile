# ork-driver — userspace regcmd matmul library for the Rockchip NPU.
# Build on a Rockchip board (needs /dev/dri/cardN + the in-tree rknpu DRM driver), or
# cross-compile for aarch64. No external dependencies (libc + the kernel DRM uABI only).
CC      ?= cc
# Use ccache when present: CORE is compiled once into shared .o objects (below), so ccache
# content-hashes them — clean builds, branch switches, and the board's several ork-driver
# checkouts all hit the cache instead of recompiling the ~9.7k-line npu.c. Disable: NO_CCACHE=1.
ifndef NO_CCACHE
CC := $(strip $(shell command -v ccache 2>/dev/null) $(CC))
endif
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
# Compile CORE ONCE into shared objects, so an npu.c edit recompiles it once (not per-example).
# The make-test build path (examples/tests/chain_xition_probe) and the libs link these; the
# special-flag perf tools (-fopenmp / -march=native / RKNN) keep compiling CORE inline.
COBJ    := $(CORE:.c=.o) src/orkd_client.o   # orkd client shim (Path B: npu.c transparently routes through orkd under ORK_USE_ORKD). Not in CORE/ATTEST — it's RPC transport, not an NPU-output source.
# Board-validation attestation: `make test` (on ALL PASS) records a hash of the sources that determine
# the NPU output + the test goldens; CI `make check-attest` (no NPU) fails if the tree differs — a catch
# that the commit was board-validated before push. Excludes include/ork_npu.h (the version-bump bot edits
# only that header, so it must not invalidate the attest).
ATTEST_FILE := tests/sbc_attest.txt
ATTEST_SRCS := $(CORE) examples/test_matmul.c examples/quant.c examples/test_sn3.c examples/model.c
EXAMPLES := test_matmul quant i4 layer decode model llama2 bench perplexity_i4 test_baseline test_registers test_layouts test_speed test_chain_i4 test_sn3 test_activations test_affinity test_stream_interleave test_mm_i8_out8 test_silu_native test_ewmul_i8 test_ewmul_f16 test_ewmul_i16 test_silu test_add test_gelu test_bmm test_ssd_chunk test_ssd_chunk_npu test_mode_transition test_bmm_fused
TESTS    :=

all: $(EXAMPLES) $(TESTS)

$(EXAMPLES): %: examples/%.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm

$(TESTS): %: %.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm

# RE diagnostic (NOT in `make test`): probes whether the captured PPU LUT/PWL regcmd can be
# driven STANDALONE. NEGATIVE RESULT on RK3588 — the PPU does not activate from an isolated
# replay (output buffer comes back unwritten). Kept as a runnable ground-truth probe; exits
# nonzero by design. See wiki Exp-2026-06-24-PPU-LUT-Silicon-Verification.
test_ppu_lut: examples/test_ppu_lut.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# --- library for embedding in other projects (e.g. llama.cpp-rockchip, FFI bindings) ---
lib: libork_npu.a libork_npu.so

libork_npu.a: $(COBJ)                    # static — link directly, no runtime .so dependency
	$(AR) rcs $@ $^
libork_npu.so: $(COBJ)                   # shared — dynamic link / FFI from Python, Node, Rust, ...
	$(CC) $(CFLAGS) -shared -o $@ $^
%.o: %.c                                 # CORE objects are -fPIC so one set serves executables + the .so
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<

# comparison tool: benchmark the closed librkllmrt (dlopen'd at runtime, no build dep).
# Not in `all`/`test` — it needs a .rkllm model + the runtime present on the board.
rknpu_bench: tools/rknpu_bench.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

# Phase-1 validation of ork_submit_seq: mixed int8(HW-doorbell)+fp16(SW-chain) sequence, bit-exact across
# many runs + HW<->SW transitions. Board NPU op — sudo env ORK_MM_TIMEOUT=3000 timeout 300 ./test_submit_seq
test_submit_seq: tools/test_submit_seq.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm

# int4 NONBLOCK-doorbell viability probe: ork_dyn_i4_probe (NONBLOCK + int16-sentinel) vs the blocking
# ork_mm_run_chain_i4 reference. Board NPU op — sudo env ORK_MM_TIMEOUT=3000 timeout 200 ./i4_doorbell_probe
i4_doorbell_probe: tools/i4_doorbell_probe.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm

# A1: int4 Sn>1 (wide-N N-tiled) on the doorbell — bit-exact vs CPU + blocking ref. Board tool, not in make test.
i4_sn_probe: tools/i4_sn_probe.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm

sdp_chain_probe: tools/sdp_chain_probe.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm

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

# Pack-time SiLU calibrator: search the fused-output register space (R/cfg4068/bias) vs a silu ref.
silu_calibrate: tools/silu_calibrate.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# validate the raised fused-SiLU M-tile cap (mg_max*64) by self-consistency vs forced mc=64.
fused_mtile_check: tools/fused_mtile_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# validate the INT32-output fused SiLU (silu emitted at int32 precision vs int8) against CPU silu.
silu32_check: tools/silu32_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# decide if a fused fp16 gate+SiLU beats baseline int8-gate + CPU-silu (validated ops, no wedge risk).
f16_gate_bench: tools/f16_gate_bench.c $(CORE)
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(CORE) -lm

# smoke-test the fp16 fused gate+SiLU primitive (runs? silu-shaped?).
silu_f16_check: tools/silu_f16_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# validate int8 JIT-inflate to fp16 (emulated W8A16) is bit-exact vs a direct fp16 pack.
jit_inflate_check: tools/jit_inflate_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# accuracy of the fp16 gate+SiLU (fused-LUT vs plain-matmul+CPU-silu) vs a CPU fp32 reference.
f16_gate_acc: tools/f16_gate_acc.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# reproduce the int8<->fp16 mode-switch instability (blocker b) in isolation, up an M ladder.
mode_switch_probe: tools/mode_switch_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE the CNA precision field (0x100c): map INT8/INT16/FP16 encoding by fuzzing proc/in precision bits.
i16_precision_probe: tools/i16_precision_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# ork-native fused SiLU: build ork's own silu LUT for its own matmul program (correct silu, ~1 int8).
silu_native: tools/silu_native.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Matmul-level benchmark: fused-SiLU output stage vs plain matmul + the CPU-silu handoff it replaces.
silu_bench: tools/silu_bench.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# EW-mul (SwiGLU dual-input) board diagnostic: runs the spliced 126-reg fused EW-mul op + dumps numerics.
ewmul_probe: tools/ewmul_probe.c $(CORE)
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

# calibration: drive the closed RKNN matmul API with B_quant_type per-channel(1) vs per-group(2) to
# probe whether per-K-group dequant is a single hardware submit. Needs librknnrt (RKNN_DIR). Not in all/test.
pgquant_capture: tools/pgquant_capture.c
	$(CC) $(CFLAGS) -I$(RKNN_DIR) -o $@ $< -L$(RKNN_DIR) -lrknnrt -lm

# LD_PRELOAD shim: decode rknpu SUBMIT task_number/subcore tasks (monolithic kernel vs PC-chained tasks).
submit_introspect.so: tools/submit_introspect.c
	$(CC) -shared -fPIC -O2 -o $@ $< -ldl

# diagnostic: why does large-M (prefill) multi-core barely scale? per-core copy/submit/acc split.
mc_prof: tools/mc_prof.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# decode-pipeline validator: CPU int4 bulk || NPU int8 share overlapped — aggregate win at M=1? Not in all/test.
# -march=native: ork_native_cpu.h uses vdotq_s32 (dotprod ISA), which the default CFLAGS march may not enable.
hybrid_decode_probe: tools/hybrid_decode_probe.c $(CORE)
	$(CC) $(CFLAGS) -march=native -Iinclude -o $@ $< $(CORE) -lm -lpthread

# per-op doorbell progress probe: can the host see per-op chain completions by polling? Not in all/test.
doorbell_id_probe: tools/doorbell_id_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm -lpthread

# diagnostic: ceiling of within-backend CPU/NPU overlap (quant/deq pipelined behind NPU run). Not in all/test.
overlap_probe: tools/overlap_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# diagnostic: decode async-overlap hypothesis — real 7B M=1 matmuls, sync vs async-pipelined w/ CPU prep. Not in all/test.
async_decode_probe: tools/async_decode_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

# diagnostic: M-sweep over real 7B decode projections — does per-verified-token cost amortize for M>1
# batched verify (spec-decode)? 1-core vs 3-core scaling per M. Not in all/test.
batch_verify_probe: tools/batch_verify_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

# diagnostic: per-submit DMA cache-maintenance (bsync) cost across output sizes — bounds the
# cacheability/DISABLE_FLUSH lever (API lead #3). Not in all/test.
bsync_cost_probe: tools/bsync_cost_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

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
# per-test wall timeout (s) — catches an NPU hang. Tests are golden-checksum'd now (full `make test`
# ~33s), but the wall must still exceed a cold model/regen run; bounds a genuine hang. Override: TEST_TIMEOUT=120
TEST_TIMEOUT ?= 360
test: $(EXAMPLES) $(TESTS) chain_xition_probe
	@fail=0; \
	for t in "test_activations" "test_matmul" "test_bmm" "quant" "i4" "perplexity_i4" "layer" "decode" "model 1" "model 12" "test_speed" "test_chain_i4" "test_sn3" "test_affinity" "test_stream_interleave" "test_mm_i8_out8" "test_silu_native" "test_ewmul_i8" "test_ewmul_f16" "test_ewmul_i16" "test_silu" "test_add" "test_gelu" "test_ssd_chunk" "test_ssd_chunk_npu" "test_mode_transition" "chain_xition_probe" "test_bmm_fused"; do \
	  echo "== $$t"; timeout $(TEST_TIMEOUT) sudo ./$$t || fail=1; done; \
	if [ -f "$(MODEL)" ]; then echo "== llama2 $(MODEL)"; timeout $(TEST_TIMEOUT) sudo ./llama2 "$(MODEL)" 6 || fail=1; \
	  else echo "== llama2 SKIP (no $(MODEL))"; fi; \
	if [ $$fail -eq 0 ]; then echo "ALL TESTS PASSED"; \
	  { echo "# Auto-written by 'make test' on ALL TESTS PASSED — the hashed sources were board-validated together."; \
	    echo "# CI 'make check-attest' fails if the tree hash differs: run 'make test' on the SBC + commit this file."; \
	    echo "CORE_SHA=$$(cat $(ATTEST_SRCS) | sha256sum | cut -c1-64)"; } > $(ATTEST_FILE); \
	  echo "wrote $(ATTEST_FILE)"; \
	  else echo "TESTS FAILED"; exit 1; fi

# CI gate (no NPU): prove the commit was board-validated. `make test` on the SBC writes tests/sbc_attest.txt
# with a hash of ATTEST_SRCS; this recomputes it and fails on a mismatch — i.e. the NPU-output-determining
# sources (or goldens) changed since the last passing board `make test`. Also flags 0/placeholder goldens.
check-attest:
	@have=$$(grep '^CORE_SHA=' $(ATTEST_FILE) 2>/dev/null | cut -d= -f2); \
	want=$$(cat $(ATTEST_SRCS) | sha256sum | cut -c1-64); \
	if [ -z "$$have" ]; then echo "check-attest: no $(ATTEST_FILE) — run 'make test' on the SBC + commit it"; exit 1; fi; \
	if [ "$$have" != "$$want" ]; then \
	  echo "check-attest: FAIL — NPU-output sources/goldens changed since the last board-validated 'make test'."; \
	  echo "  committed CORE_SHA=$$have"; echo "  working-tree =$$want"; \
	  echo "  => run 'make test' on the SBC (RK3588) and commit $(ATTEST_FILE)."; exit 1; fi; \
	echo "check-attest: attest OK ($$have)"; \
	ph=0; \
	if grep -nE '(check|check_i8|one)\(c[^)]*, *0\)' examples/test_matmul.c examples/quant.c examples/test_sn3.c 2>/dev/null; then ph=1; fi; \
	if grep -nE 'case [0-9]+: *return 0;' examples/model.c 2>/dev/null; then ph=1; fi; \
	if [ $$ph -ne 0 ]; then echo "check-attest: FAIL — placeholder (0) golden found; regen on the SBC"; exit 1; fi; \
	echo "check-attest: no placeholder goldens — OK"

# Quick subset via the same pass/fail harness — builds and runs only the named examples:
#   make test-only T="test_ewmul_i8 test_ewmul_f16 test_ewmul_i16"     (a name may carry args, e.g. T="model 1")
# Prereqs (bare example names) build via the EXAMPLES pattern rule; then each is run under sudo+timeout.
test-only: $(filter $(EXAMPLES) $(TESTS),$(T))
	@if [ -z "$(T)" ]; then echo 'usage: make test-only T="test_ewmul_i8 test_ewmul_f16 test_ewmul_i16"'; exit 2; fi; \
	fail=0; for t in $(T); do echo "== $$t"; timeout $(TEST_TIMEOUT) sudo ./$$t || fail=1; done; \
	if [ $$fail -eq 0 ]; then echo "SUBSET PASSED"; else echo "SUBSET FAILED"; exit 1; fi

# A-suite (orkd "first client" milestone): run the ROUTABLE subset of the test suite through ONE orkd daemon.
# Each example is UNCHANGED and self-validates vs its golden/CPU reference — only the NPU-access path changes
# (direct ioctl -> orkd RPC) via ORK_USE_ORKD=1. ONE daemon is spun up, survives the whole run (a large
# ORKD_IDLE_MS bridges the inter-test gaps so it does NOT idle-reap between examples), and is torn down after;
# the daemon's PID is asserted stable across the run to prove a single persistent daemon serviced every test.
#   ROUTABLE = every NPU op the example issues is on the orkd RPC surface: matmul (int8 quant/test_speed,
#   fp16 layer/decode/model) + SDP (ewmul i8/f16). CPU-only test_activations rides along.
#   OUT OF SCOPE (stay under plain `make test`, direct NPU): internal entrypoints not yet on the RPC surface —
#   run_stream_* (test_matmul/test_stream_interleave/test_affinity), int4 chain/stream/grouped/i4a8
#   (test_chain_i4/i4/perplexity_i4/test_matmul), bmm + ssm (test_bmm*/test_ssd_chunk_npu/test_mode_transition),
#   int16 SDP + gelu/exp/rsqrt (test_ewmul_i16/test_silu/test_add/test_gelu), and the probe/perf tools.
# Run on the SBC, one at a time, no concurrent make: sudo make test-orkd  (or via ssh run_in_background).
ORKD_SUITE := test_activations quant layer decode model test_ewmul_i8 test_ewmul_f16
test-orkd: $(ORKD_SUITE) orkd
	@echo "== A-suite: routable test subset through ONE orkd (first-client milestone) =="; \
	sudo pkill -x orkd 2>/dev/null; sleep 1; \
	echo "-- spinning up one orkd (idle-reap bridged; ORKD_IDLE_MS=600000)"; \
	sudo env ORKD_IDLE_MS=600000 ./orkd; sleep 1; \
	pid0=$$(pgrep -x orkd | head -1); \
	if [ -z "$$pid0" ]; then echo "orkd failed to start"; exit 1; fi; \
	echo "-- orkd up, pid=$$pid0"; \
	fail=0; \
	for t in "test_activations" "quant" "layer" "decode" "model 1" "model 12" "test_ewmul_i8" "test_ewmul_f16"; do \
	  echo "== [orkd] $$t"; timeout $(TEST_TIMEOUT) sudo env ORK_USE_ORKD=1 ORKD_BIN=$$PWD/orkd ./$$t || fail=1; \
	  pidn=$$(pgrep -x orkd | head -1); \
	  if [ "$$pidn" != "$$pid0" ]; then echo "FAIL: orkd pid changed ($$pid0 -> $$pidn) — daemon did not persist across the run"; fail=1; fi; \
	done; \
	echo "-- tearing down orkd (SIGTERM)"; sudo pkill -TERM -x orkd 2>/dev/null; sleep 2; \
	if pgrep -x orkd >/dev/null; then echo "WARN: orkd survived SIGTERM — forcing"; sudo pkill -x orkd; fi; \
	if [ $$fail -eq 0 ]; then echo "A-SUITE (orkd) PASSED — routable subset self-validated through one persistent daemon"; \
	  else echo "A-SUITE (orkd) FAILED"; exit 1; fi

bench-llama:
	@echo "== Running two-turn conversation integration benchmark via llama-server =="
	@LLAMA_SERVER_BIN=$(HOME)/llama.cpp/build/bin/llama-server tools/bench_two_turn.sh

clean:
	rm -f $(EXAMPLES) $(TESTS) rknpu_bench vec_fuzz test_ppu_lut libork_npu.a libork_npu.so src/*.o src/soc/*.o

.PHONY: all lib install test clean check-attest

attn_cost: tools/attn_cost.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

attn_cost7b: tools/attn_cost7b.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE: dump one single-core int8 ork matmul's regcmd (ORK_TRACE=1) for the weight-residence diff vs rknn.
ork_trace_mm: tools/ork_trace_mm.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

# RE: reproduce the mixed-K cross-matmul state interaction (layer/model fail at cbuf raise).
seq_check: tools/seq_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

# diagnostic: fast SAMPLED correctness for shapes whose full O(M*N*K) CPU ref takes minutes (wide-K,
# e.g. ffn_down M=256 K=18944) — the slow ref masks NPU health and trips test/probe timeouts. Not in all/test.
sparse_check: tools/sparse_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

# diagnostic: fp16 matmul correctness at a chosen (M,K,N,cores), sampled. Used to validate the fp16
# M-scheduler row ceiling (sched=1 miscomputes >8 rows at K>=2048; single-core vs multi-core). Not in all/test.
fp16_sparse_check: tools/fp16_sparse_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

# RE: round-robin single-core (run_stream) vs barrier multi-core, at R=32 (cbuf reinstated).
rr_experiment: tools/rr_experiment.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

# RE: reproduce the fp16 layer/model failure at cbuf raise (multi-core fp16 matmul + CPU ref).
fp16_check: tools/fp16_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lpthread -lm

# calibrate the fp16 fused SiLU (measure R/idx(gate), build curve, validate).
silu_f16_calib: tools/silu_f16_calib.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# sweep the bounded-Gmax fp16 fused-SiLU builder across the in-model gate range (bulk vs full error).
f16_gmax_sweep: tools/f16_gmax_sweep.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# probe whether the fp16 fused-SiLU index can reach the UPPER LUT bank (idx>514) under any register config.
f16_bank_probe: tools/f16_bank_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# --- RE tooling for cross-NPU bring-up (board-only diagnostics; not in all/test) ---
# NPU on-chip SRAM: total/free size query + allocation-usability probe. See README
# "Enabling the NPU on-chip SRAM" and wiki NPU-on-chip-SRAM.
sram_probe: tools/sram_probe.c
	$(CC) $(CFLAGS) -o $@ $< -lm
sram_alloc_probe: tools/sram_alloc_probe.c
	$(CC) $(CFLAGS) -o $@ $< -lm
sram_query: tools/sram_query.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# constraint-guided template-preserving fuzzer for norm ops (standalone regcmd).
test_norm: examples/test_norm.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# round-robin cross-core decode probe.
rr_decode_probe: tools/rr_decode_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# IOMMU-domain concurrency probe (NPU || CPU overlap).
domain_concur_probe: tools/domain_concur_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# layer-pipeline probe.
layer_pipeline_probe: tools/layer_pipeline_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# on-die fused-activation harness (PPU/SDP chain RE).
test_fused_activation: tests/test_fused_activation.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# on-NPU norm: fused reduce+rsqrt (ork_mm_build_f16_rsqrt_lut) validated vs CPU 1/sqrt. Board+PPU only.
test_rsqrt_lut: tools/re/test_rsqrt_lut.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# tools/re toolkit: IOMMU domain-switch probe + int4 CPU-reference check.
dom_switch_probe: tools/re/dom_switch_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
softmax_replay: tools/re/softmax_replay.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

i4cpu_check: tools/re/i4cpu_check.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# first real int16 matmul via the fp16 run path + 0x100c=INT16 fuzz-override, vs a CPU int16 reference.
i16_matmul_test: tools/i16_matmul_test.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# is the on-NPU int16 SiLU accurate enough to fix the gmax-gate coherence (vs CPU fp32 silu)?
i16_silu_acc: tools/i16_silu_acc.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# where does the on-NPU int16 activation op wedge (M vs N vs M*N)? — to size act_lut_i16 internal tiling
i16_shape_probe: tools/i16_shape_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# PHASE 0 chained-FFN: does the NPU walk a heterogeneous matmul(0xd)->int16-silu(0x18) 2-task chain in one submit?
chain_probe: tools/chain_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# CHAIN ASSEMBLER increment 1: data-connected matmul(int16-out)->silu bridge in one PC-chain
chain_gatesilu_probe: tools/chain_gatesilu_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# CHAIN ASSEMBLER increment 1 (corrected): [gate*silu -> up] one submit via run_chain_i8 + set_i8_silu
chain_gu_silu_probe: tools/chain_gu_silu_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# PHASE 1 chained-FFN RE: does an int8 matmul emit correct int16 via set_i16_out (the on-NPU requant handoff)?
i16out_probe: tools/i16out_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

bmm_probe: tools/bmm_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# Phase-1 GATE G1: decisive fusion micro-bench — fused mixed chunk chain (1 submit) vs the same ops as
# N separate submits, at a floor-dominated shape. Proves fusion clears the submit floor before the big build.
ssd_fusion_bench: tools/ssd_fusion_bench.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

floor_decomp: tools/floor_decomp.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

mode_probe: tools/mode_probe.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

chain_xition_probe: tools/chain_xition_probe.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

# orkd — the NPU daemon (single owner + submit queue for many client processes). First increment is the
# lifecycle skeleton (flock single-instance, accept loop, ref-count, idle-reap); NPU stubbed, so it needs no
# COBJ yet. When the submit path is wired it will link $(COBJ). Not in `all`/`test` until it has a client.
orkd: src/orkd.c src/orkd_proto.h $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

# orkd_probe — daemon lifecycle validation (auto-spawn + connect + core count + ping); board tool, not in test.
orkd_probe: tools/orkd_probe.c src/orkd_client.c src/orkd_client.h src/orkd_proto.h
	$(CC) $(CFLAGS) -o $@ $< src/orkd_client.c -lpthread

# test_orkd — first orkd client: int8 matmuls THROUGH the daemon, self-validated vs CPU ref. Pure client
# (links orkd_client, not COBJ). Standalone, NOT in `make test` (would contend with direct-NPU examples).
test_orkd: examples/test_orkd.c src/orkd_client.c src/orkd_client.h src/orkd_proto.h
	$(CC) $(CFLAGS) -o $@ $< src/orkd_client.c -lpthread

# orkd_ring_probe — A-ring validation + latency bench (socket RPC vs the shared-memory ring, bit-exact + us/op).
# Pure client (links orkd_client, not COBJ). Board tool, not in `make test`.
orkd_ring_probe: tools/orkd_ring_probe.c src/orkd_client.c src/orkd_client.h src/orkd_proto.h src/orkd_ring.h src/orkd_shm.h
	$(CC) $(CFLAGS) -o $@ $< src/orkd_client.c -lpthread

# test_orkd_transparent — Path B: NORMAL ork_npu API, bit-exact direct vs routed-through-orkd (ORK_USE_ORKD).
# Links COBJ (which now includes orkd_client.o for the transparent route). Board tool, not in `make test`.
test_orkd_transparent: examples/test_orkd_transparent.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

# Multi-tenant orkd proof: forks N client processes sharing one daemon (board tool, not in `make test`).
# sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./test_orkd_multi [nclients] [iters]
test_orkd_multi: tools/test_orkd_multi.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

# Multi-consumer proof: ONE process, TWO orkd connections, interleaved (board tool, not in `make test`).
# sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./test_orkd_2conn [iters]
test_orkd_2conn: tools/test_orkd_2conn.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

# Multi-consumer SEQ proof: two connections, each submits a grouped op-sequence (board tool, not in `make test`).
# sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./test_orkd_2conn_seq [iters]
test_orkd_2conn_seq: tools/test_orkd_2conn_seq.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

# mc_miss_repro — tight repro of the multi-core doorbell-miss flake (stop-on-first-miss, [MC-DIAG]). Board tool.
mc_miss_repro: tools/mc_miss_repro.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

i4_xition_probe: tools/i4_xition_probe.c $(COBJ)
	$(CC) $(CFLAGS) -o $@ $< $(COBJ) -lm -lpthread

ssd_layer_bench: tools/ssd_layer_bench.c $(CORE)
	$(CC) $(CFLAGS) -O3 -march=native -o $@ $< $(CORE) -lm -lpthread

ssd_cumba_exp: tools/ssd_cumba_exp.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

softmax_probe: tools/softmax_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

softmax_cost: tools/softmax_cost.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

rope_probe: tools/rope_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

two_stream_f16_probe: tools/two_stream_f16_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

softmax_reduce_probe: tools/softmax_reduce_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

sdp_max_fuzz: tools/sdp_max_fuzz.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

max_reduce_probe: tools/max_reduce_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

bs_scale_probe: tools/bs_scale_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

tnorm_probe: tools/tnorm_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

chain_mm_perchan_probe: tools/chain_mm_perchan_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

chain_mm_perchan_f16_probe: tools/chain_mm_perchan_f16_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

f16_mm_f16out_probe: tools/f16_mm_f16out_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

mm_perchan_f16_fused_probe: tools/mm_perchan_f16_fused_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

mul_perchan_f16_contig_probe: tools/mul_perchan_f16_contig_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

reshape_probe: tools/reshape_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

requant_i32_probe: tools/requant_i32_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

mm_perchan_f16_probe: tools/mm_perchan_f16_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

mm_perchan_f16_diag_probe: tools/mm_perchan_f16_diag_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

stridedA_bmm_probe: tools/stridedA_bmm_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

perchan_bench: tools/perchan_bench.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

doorbell_prof: tools/doorbell_prof.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

overlap_prof: tools/overlap_prof.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# split-tensor CPU+NPU decode probe: NEON sdot CPU GEMV needs armv8.2 +dotprod (A76).
split_matmul_probe: tools/split_matmul_probe.c $(CORE)
	$(CC) $(CFLAGS) -march=armv8.2-a+dotprod -o $@ $< $(CORE) -lm -lpthread

# MoE expert-split CPU+NPU probe (NEON sdot needs +dotprod).
split_expert_probe: tools/split_expert_probe.c $(CORE)
	$(CC) $(CFLAGS) -march=armv8.2-a+dotprod -o $@ $< $(CORE) -lm -lpthread

# per-op NPU-vs-CPU profiler (NEON sdot needs +dotprod).
op_profile: tools/op_profile.c $(CORE)
	$(CC) $(CFLAGS) -march=armv8.2-a+dotprod -o $@ $< $(CORE) -lm -lpthread

qkv_chain_probe: tools/qkv_chain_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# CPU-only int4-vs-int8 memory-bound GEMV (no NPU). NEON +dotprod.
cpu_i4_vs_i8: tools/cpu_i4_vs_i8.c
	$(CC) $(CFLAGS) -march=armv8.2-a+dotprod -o $@ $< -lpthread

jit_i4_i8_probe: tools/jit_i4_i8_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# static-ID doorbell probe: can one address show the current op ID? Not in all/test.
static_id_probe: tools/static_id_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm -lpthread

# RE probe: does the PC sequencer read chain descriptors from DRAM at exec-time (steerable) or pre-cache? Not in all/test.
steer_probe: tools/steer_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# dynamic-submit API test: begin/progress/halt/end (NONBLOCK chain, per-op doorbell, mid-flight early-exit). Not in all/test.
ork_dyn_test: tools/ork_dyn_test.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# P1b/G1: N-tiling (Sn>1) on the doorbell — bit-exact vs CPU int32 ref, column-varying weights
ork_dyn_ntile_test: tools/ork_dyn_ntile_test.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# fp16-doorbell priming/persistence probe (int8-primes-fp16 hypothesis for the mixed-precision async pipeline)
ork_dyn_f16_interleave: tools/ork_dyn_f16_interleave.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# fp16 doorbell-vs-chain regcmd/submit dump-and-diff (RE: crack fp16 HW-chaining). Not in all/test.
f16_dumpdiff: tools/f16_dumpdiff.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm -lpthread

# SRAM-vs-DRAM contention probe: is NPU-on-SRAM a separate memory path from CPU-on-DRAM? (partition test)
sram_bw_probe: tools/sram_bw_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm -lpthread

# submit-queue chunk-pipeline bench: CPU‖NPU decode-split overlap. Not in all/test.
ork_dyn_queue_bench: tools/ork_dyn_queue_bench.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# persistent-job spin keep-alive + mid-flight redirect probe. Not in all/test.
dyn_spin_probe: tools/dyn_spin_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# precompiled-program cache (regime A) vs synth-every-call bench. Not in all/test.
ork_pc_bench: tools/ork_pc_bench.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

ork_pc_sram_probe: tools/ork_pc_sram_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

ork_dyn_spin_test: tools/ork_dyn_spin_test.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

ork_dyn_spin_diag: tools/ork_dyn_spin_diag.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

risky_dump_test: tools/risky_dump_test.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
