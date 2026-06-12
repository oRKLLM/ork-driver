# ork-driver — userspace regcmd matmul library for the Rockchip NPU.
# Build on a Rockchip board (needs /dev/dri/cardN + the in-tree rknpu DRM driver), or
# cross-compile for aarch64. No external dependencies (libc + the kernel DRM uABI only).
CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -Wall -Iinclude -Isrc -pthread   # -pthread: multi-core path uses worker threads
PREFIX  ?= /usr/local
CORE    := src/npu.c src/soc.c src/soc/rk3588.c src/soc/rk3576.c
EXAMPLES := test_matmul quant i4 layer decode model llama2 bench

all: $(EXAMPLES)

$(EXAMPLES): %: examples/%.c $(CORE)
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
rkllm_bench: tools/rkllm_bench.c
	$(CC) $(CFLAGS) -o $@ $< -ldl

# RE/calibration probe: sweeps K to find this SoC's single-submit K-tile ceiling (int8).
# Not in `all`/`test` — it intentionally wedges the NPU past the cap (recoverable).
ksubmit_probe: tools/ksubmit_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE probe: hunt the weight stride register for in-place K-slicing of a full-K buffer.
slice_probe: tools/slice_probe.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# RE probe: derive the w4a16 (int4 weight x fp16 activation) regcmd by sweeping vs a CPU reference
# (the runtime won't expose int4 on this board; the NPU does it — we emit the regcmd directly).
i4_probe: tools/i4_probe.c $(CORE)
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
test: $(EXAMPLES)
	@fail=0; \
	for t in "test_matmul" "quant" "layer" "decode" "model 1" "model 12"; do \
	  echo "== $$t"; timeout 120 sudo ./$$t || fail=1; done; \
	if [ -f "$(MODEL)" ]; then echo "== llama2 $(MODEL)"; timeout 120 sudo ./llama2 "$(MODEL)" 6 || fail=1; \
	  else echo "== llama2 SKIP (no $(MODEL))"; fi; \
	if [ $$fail -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "TESTS FAILED"; exit 1; fi

clean:
	rm -f $(EXAMPLES) rkllm_bench libork_npu.a libork_npu.so src/*.o src/soc/*.o

.PHONY: all lib install test clean

attn_cost: tools/attn_cost.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm
