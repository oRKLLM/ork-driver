# ork-driver — userspace regcmd matmul library for the Rockchip NPU.
# Build on a Rockchip board (needs /dev/dri/cardN + the in-tree rknpu DRM driver).
CC      ?= cc
CFLAGS  ?= -O2 -Wall -Iinclude -Isrc
CORE    := src/npu.c src/soc.c src/soc/rk3588.c src/soc/rk3576.c
EXAMPLES := test_matmul layer decode model llama2

all: $(EXAMPLES)

$(EXAMPLES): %: examples/%.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) -lm

# static library for embedding in other projects (e.g. llama.cpp-rockchip)
libork_npu.a: $(CORE:.c=.o)
	$(AR) rcs $@ $^
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(EXAMPLES) libork_npu.a src/*.o src/soc/*.o

.PHONY: all clean
