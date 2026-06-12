# ork-driver — userspace regcmd matmul library for the Rockchip NPU.
# Build on a Rockchip board (needs /dev/dri/cardN + the in-tree rknpu DRM driver), or
# cross-compile for aarch64. No external dependencies (libc + the kernel DRM uABI only).
CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -Wall -Iinclude -Isrc
PREFIX  ?= /usr/local
CORE    := src/npu.c src/soc.c src/soc/rk3588.c src/soc/rk3576.c
EXAMPLES := test_matmul layer decode model llama2

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

# install the public header + both libs (override PREFIX=/path as needed)
install: libork_npu.a libork_npu.so
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m644 libork_npu.a libork_npu.so $(DESTDIR)$(PREFIX)/lib/
	install -m644 include/ork_npu.h $(DESTDIR)$(PREFIX)/include/

clean:
	rm -f $(EXAMPLES) libork_npu.a libork_npu.so src/*.o src/soc/*.o

.PHONY: all lib install clean
