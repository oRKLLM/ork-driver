# Modularization Plan — `src/npu.c` → `src/npu/` precision modules

**Status:** PLAN ONLY (no implementation). **Base:** `main` @ `f968b56` (v1.0.12, 2026-08-19).
**Branch to use:** `refactor/modularize-precision`.
**Layout:** scaffold `src/npu.c` beside a sibling `src/npu/` folder — the same `X.c` + `X/` idiom the
repo already uses for `src/soc.c` + `src/soc/rk3588.c`, applied fractally (`npu/i8.c` + `npu/i8/`).
**Scope of round 1:** split along the **precision axis** (`i4` / `i8` / `i16` / `f16`), keeping the
monolith as a *scaffold* that retains everything dtype-agnostic.

> **Re-derived 2026-08-19 against `origin/main`.** An earlier draft measured a checkout that had not
> fetched since 2026-07-20 and was 225 commits stale; every count below is from the current tree.

---

## 0. Why, and what "too big" measures out to

| file | lines | note |
|---|---:|---|
| `src/npu.c` | **15,313** | the monolith — 629 top-level symbols |
| `include/ork_npu.h` | 1,517 | public API |
| `src/orkd.c` | 1,003 | handler-organized — round 4, low priority |
| `src/orkd_client.c` | 757 | RPC transport — round 4, low priority |
| `src/regcmd_fold_refs.h` | 465 | captured fold templates |
| `src/ork_regs.h` | 299 | named-register layer (see §1) |
| everything else in `src/` | < 370 each | fine as-is |

`npu.c` is ~85 % of the core's source mass. Largest functions (the old 594-line `mcworker` is gone —
the doorbell path replaced it, which lowers the round-2 floor considerably):

```
420  run_chain_i8_impl        329  ork_dyn_begin_colsplit   273  ork_dyn_begin_mc
230  run_multicore            175  ork_npu_replay_softmax_f16  174  run
173  ork_csub_worker          160  ork_ssm_scan_f32         159  ork_mm_run_chain_i4
159  ork_dyn_end              149  ork_dyn_begin            149  ork_npu_benchmark_chain
```

---

## 1. This extends an extraction track that is already underway

`main` already carries the first pieces of exactly this direction — the plan should continue them, not
compete with them:

- **`src/ork_regs.h`** (299 lines) — "the naming layer for the register-remap refactor (task #40)".
  regcmd synth now sets registers **by name** via `setrn()`, width-validated. Every precision module
  will depend on this; `npu/internal.h` re-exports it rather than redefining anything.
- **`src/ork_op_manifest.h`** — the lib↔orkd op-parity contract, already a standalone unit.
- **`include/ork_slice.h`** + the SLICE-AND-DICE section, **`include/ork_spine.h`** +
  `src/spine_kernels.h`, **`src/regcmd_fold_refs.h`** — new subsystems that arrived *already*
  factored into their own headers.

The precedent is set. What's missing is the `.c` side.

---

## 2. Layout feasibility — verified, not assumed

Smoke-tested end-to-end (build, link, run) with a throwaway `src/npu.c` + `src/npu/i8.c` +
`src/npu/i8/pack.c` tree:

- **Already the repo's idiom** — `CORE` lists `src/soc.c src/soc/rk3588.c src/soc/rk3576.c` today.
- **`%.o: %.c` needs no change** — `src/npu/i8.c` → `src/npu/i8.o` compiles and links as-is.
- **`-Isrc` makes includes uniform at every depth** — `#include "npu/internal.h"` resolves identically
  from `src/npu.c`, `src/npu/i8.c`, and `src/npu/i8/pack.c`. No `../` anywhere; use that one spelling.
- **`.gitignore`'s `*.o` is depth-independent.**
- **Fractal** — round 2 is the same rule one level down, no later reorganization.
- **`src/regcmd_*.h` and `src/ork_regs.h` stay at `src/`.** Six files outside `src/` reach them through
  `-Isrc` (`tools/vec_fuzz.c`, `tools/stride_test.c`, `tools/i4_multim_fuzz.c`, `examples/test_norm.c`,
  `examples/test_layouts.c`, `examples/test_registers.c`). Relocating them is a separate change.
- **Two Makefile edits:** add the files to `CORE`, and change `clean:` to `rm -f $(COBJ)` (it currently
  hardcodes `src/*.o src/soc/*.o`).

---

## 3. Hard constraints

1. **C11, libc-only, one language** (AGENTS §2). `CORE`, `clean:` and `check_registry.sh` are the only
   build edits.
2. **Board-only verification.** `npu.c` does not compile on macOS (`sys/prctl.h`, DRM uABI). Use
   `tools/util/board` (the SSH wrapper) and/or `tools/util/sync_daemon.sh` (2 s bidirectional rsync,
   Mac stays source-of-truth). Full build ≈20 s, `make test` ≈33 s — cheap enough to gate every commit.
3. **Bit-exactness is the acceptance test.** Fixed-seed inputs + static goldens ⇒ `make test` passing
   *is* the proof NPU output is unchanged (AGENTS §3).
4. **`tests/sbc_attest.txt` covers `$(CORE)`** → the hash changes; a board `make test` + committed
   attest is mandatory or `sbc-attest.yml` / `version-bump.yml` fail.
5. **`tools/check_registry.sh` hardcodes `src/npu.c`** at lines 22, 41, 73 and runs from `make all` →
   generalize it first or the build breaks the moment code leaves `npu.c`.
6. **Cross-repo lockstep.** The fork's `ggml-ork` CMake compiles the vendored `src/npu.c` by name; the
   new `src/npu/*.c` must be added there in the same submodule bump.
7. **Never discard experimental code** (AGENTS §2). If a lift blocks, park it and consult.
8. **Public API byte-stable.** `include/ork_npu.h` untouched in round 1.

---

## 4. The private ABI: `src/npu/internal.h`

The real work is naming the surface that currently hides behind `static`. Measured usage in `npu.c`:

```
604 bsync   517 setr   322 bdestroy   307 bcreate   192 ork_now_us   134 rknpu_submit_ioctl
 96 synth_i8   71 ork_npu_enter   62 synth   58 budget   51 validate_regcmd   47 act
 41 dom_activate   31 pgup   29 synth_i4   28 setrn   27 bimport   26 ork_dom   22 mc_ensure
 19 submit1   17 set_i8_out8   17 set_i16_out   16 pin_big_core   15 set_i8_silu   13 int8_ks
```

Contents, in order:

1. **Types** — `struct buf`, `struct ork_pw`, `struct ork_dom_scratch`, `struct ork_npu`,
   `struct ork_w`, `struct ork_w_sliced`, `struct fold_scratch`, `ORK_MAXCORE`, `ORK_MAXDOM`,
   `typedef ork_f16 f16`. ⚠ Move the `ork_kv_resident` `#ifndef ORK_KV_RESIDENT_T` guard **verbatim** —
   the CMake/ggml-ork build pulls `ork_npu.h` into the same TU and would otherwise double-typedef.
2. **Re-exports** — `#include "ork_regs.h"`, `"ork_op_manifest.h"`, `"ork_slice.h"`, `"ork_spine.h"`.
3. **Dtype predicates** — `ORK_I8_LIVE`, `ORK_INT_DT`, `ORK_KW_DT`.
4. **Env-knob accessors** as `static inline` (one-line cached `getenv`; per-TU duplication is harmless).
5. **`static inline` hot helpers** — `setr`, `setrn`, `pgup`, `ork_dom`, `ork_now_us`, `bsync`,
   `bsync_off`, `act`. These are the inner-loop ones; header-inlining preserves today's inlining (§8.1).
6. **`extern` protos** for the rest of the cross-module surface.

**Naming rule.** Already-`ork_`-prefixed names keep them. Generic names that would pollute
`libork_npu.so` get an `orki_` prefix: `act`, `budget`, `pack`, `run`, `synth*`, `submit1[_db]`,
`mc_ensure`, `bcreate`, `bdestroy`, `bimport[_f]`, `bscratch`, `validate_regcmd`, `dma_find`,
`pin_big_core`, `pin_little_core`, `int8_ks`, `fused_mtile`, `f16_mtile`, `apply_ork_geom`,
`dom_activate`, `dom_reserve`, `rknpu_submit_ioctl`, `set_i8_out8`, `set_i16_out`, `set_f16_out`,
`set_i8_silu`, `set_i8_ewmul`, `imp_reg`, `imp_unreg`. Do the rename as its own commit, before any
move, so the move diffs stay pure.

**Cross-boundary edges are pre-catalogued** — `npu.c:4655–4660` is a forward-decl block naming exactly
the statics that jump between the future `npu/i4.c` and the scaffold (`run_i4_bchain_db`,
`ork_dyn_begin_mc_i4[_grouped]`, `i4_submit_tmo_ms`, `ork_dyn_grouped_end`, `ork_dyn_end`).

---

## 5. Round 1 target layout

> **Rule:** if a symbol's name, its regcmd template, or its dtype argument is precision-tagged, it moves
> to that precision's module. Otherwise it stays in the scaffold.

```
src/
  npu.c              <- scaffold (dtype-agnostic core)
  npu/
    internal.h
    sdp.c  f16.c  i16.c  i4.c  ssm.c
    i8.c  i8/        <- i8 is a FOLDER from day one (see below)
      regcmd.c  pack.c  fold.c  run.c  chain.c  dyn.c  probe.c
  regcmd_*.h  ork_regs.h  ork_op_manifest.h   <- unchanged
  soc.c  soc/                                  <- unchanged; the precedent
```

**Why `i8` is a folder immediately.** On the current tree the int8 surface totals **~5,500 lines** —
the `fold` subsystem alone (`synth_i8_mfold`, `ork_npu_mfold_chain{,_cap,_multi,_v}`,
`ork_npu_fold_{batch,run_i8,op_i8,run_w,batch_w}`, the `fold_*` helpers, `ork_mm_load_fold_i8` /
`ork_w_attach_fold_i8` / `ork_w_dump_fold_i8_cpu`) is ~750 lines, and the dynamic API is ~1,900. A
single `npu/i8.c` would be a third of the monolith and defeat the purpose. Splitting it now costs
nothing extra — the folder is where round 2 would have put it anyway.

| file | contents | est. lines |
|---|---|---:|
| `src/npu/internal.h` | private ABI (§4) | ~280 |
| `src/npu.c` *(scaffold)* | env knobs; IOVA guard; import registry (`imp_reg`/`imp_unreg`, `bimport_f`, `bscratch`, `reimport_inplace`, `ork_ctx_fd_reap`); live-buffer registry + SIGTERM teardown; `bcreate`/`bdestroy`/`bsync`/warena; dma-heap + `ork_dma_*`; domains (`dom_activate`, `dom_prime`, `dom_reserve`, `ork_dom_flush_if_dirty`, `ork_dom_reanchor`, `ork_npu_set_ndomains`, pack-domain, alloc/free); `dump_submit`/`trace_submit`/`rknpu_submit_ioctl`/`submit1[_db]`; device lifecycle (`ork_npu_init[_orkd]`/`free`/`soft_reset`/`recover`/`reap_stuck`/`force_fault`/`dump_state`/`ork_kmsg`, SRAM, governor warn, version); pool + `pin_big/little_core` + `ork_parallel_for` + `mc_ensure` + `run_multicore`; **mode-transition layer** (`XSPEC`, `ork_npu_enter`, xprof); generic `pack()`/`run()` + `ork_mm_run`/`ork_mm_run_i8`; slice **dispatch** (`ork_mm_pack_sliced`/`run_sliced`/`free_sliced`, `slice_acc_worker`, `slice_rescue_or_refuse`); `ork_w_free`/`ork_mm_free`/`ork_w_*`; stage + stream-pool; async submit; `ork_submit_seq` + dispatch shims; `ork_bmm_*` dispatch; norm/softmax/fwht; CPU pack helpers; profiling globals | **~3,900** |
| `src/npu/sdp.c` | shared SDP/LUT substrate: `silu_f`/`gelu_f`/`rsqrt_f`/`exp_f`, `chain_build_lut_fn`, `silu_build_curve[_biased]`, `silu_calibrate_idx`/`_idx16`, `build_act_lut16`, `ork_mm_silu_build_lut`, `ork_mm_chain_build_exp_lut`, `sdp_canon`, `ork_npu_sdp_stamp` | ~300 |
| `src/npu/i8/regcmd.c` | `synth_i8`, `synth_i8_mfold`, `set_i8_out8`, `set_i8_silu[32]`, `set_i8_ewmul`, `splice_ew_lane`, `set_mul_geom`, `apply_ork_geom`, i8 fuzz hooks, `int8_ks`, `fused_mtile`, `ork_npu_synth_i8_dump` | ~500 |
| `src/npu/i8/pack.c` | `pack_i8[_f32/_dequant/_import]`, `load_i8[_import/_flags]`, `mm_import_i8`, `adopt_imported_i8`, `repack_i8*`, `w_dump_i8*`, resident KV (`ork_kv_*`), int8→fp16 JIT inflate, `slice_pack_i8` | ~750 |
| `src/npu/i8/fold.c` | `ork_npu_mfold_chain{,_cap,_multi,_v}`, `ork_fbc_thread`, `ork_fold_submit_all`, `ork_npu_fold_batch[_w]`, `fold_ref_for`/`fold_pidx`/`fold_setv`/`fold_build_tile`/`fold_nc16`/`fold_woff`/`fold_c4`, `fold_A_ensure`/`fold_fill_A`/`fold_scr_get`/`fold_run_one`/`fold_scratch_free*`, `ork_npu_fold_run_i8`/`fold_op_i8`/`fold_run_w`, `ork_mm_load_fold_i8`, `ork_w_attach_fold_i8`, `ork_w_dump_fold_i8_cpu` | ~750 |
| `src/npu/i8/run.c` | `ork_mm_run_i8_silu[32]`, `_ewmul`, `_out8`, `_out16`, i8 activations (`act_lut_i8[_biased]`, `silu/gelu/rsqrt/exp_i8`), `ewmul_i8`, `add_i8`, `mul_perchan_i8`, `row_max_i8`, `slice_run_i8`, `run_stream_i8[_sk]`, `ork_bmm_i8[_strided]` | ~700 |
| `src/npu/i8/chain.c` | `run_chain_i8_impl` + all `ork_mm_run_chain_i8*` wrappers (incl. `_ffn_exp_biased`), `chain_fullk_mcap_i8`, `chains_rr[_biased]`, `ork_npu_chain_progs`, `chain_selftest`, orkd chain wrappers (`ork_mm_ffn_orkd`, `ork_mm_attn_orkd`, `ork_mm_attn_rr_orkd`, `layer_mm`, `ork_mm_layer_i8`) | ~1,000 |
| `src/npu/i8/dyn.c` | the dynamic steered API: `ork_dyn_begin[_mc/_colsplit/_seq_i8[_mc]]`, `ork_csub_worker`, `ork_dyn_{progress,append,halt,end,dump,steps,remaining,spin_probe}`, `mc_recover_resubmit`, `mtile_cap`, term handler, submit queue (`ork_dyn_queue_*`), precompiled cache (`ork_pc_*`) | ~1,900 |
| `src/npu/i8/probe.c` | `probe_mtile_i8`, `probe_single_i8`, `probe_i8_out8`, `probe_i8_mm`, `probe_i8_ewmul*`, `probe_i8_mul`, `probe_add_i8`, `probe_bs_scale`, `probe_i8_silu*`, `probe_silu_std`, `probe_chain_i8`, `benchmark_chain`, `probe_seq_hetero`, `probe_sdp_chain_fwd`, `probe_batch`, doorbell/overlap prof, `ork_npu_replay_i8[_amap/_sweep]` | ~900 |
| `src/npu/f16.c` | `synth`, `set_f16_out[_fp16in]`, f16 fuzz, `f16_mtile`, fp16 `ork_mm_pack`; `ork_mm_run_f16_silu`, `set_f16_silu`, `ork_mm_build_f16_{lut,silu_lut,rsqrt_lut}`, `pack_f16_fused_act`/`run_f16_fused_act`/`run_f16_act`, `ork_mm_run_f16_f16out`; f16 probes (`probe_f16_mm[_f16out]`, `_stridedA`, `probe_slice_f16`, `f16_gap_probe`, `f16_percore_probe` + `ork_pcfd_thread`, `ork_f16_colsplit`); `mm_perchan_f16[_diag/_fused]`, `mul_perchan_f16[_contig]`, `ewmul_f16`, `add_f16`, `chain_mm_perchan_f16`; f16 streams (`run_stream_f16[_chain]`); `ork_bmm_fp16[_fused/_stream/_strided]`; RE replays (`replay_full_f16`, `replay_softmax_f16`, `replay_reshape_f16`, `reshape_probe_f16`, `probe_silu_std_f16`); `ork_mm_f16_scratch`, `inflate_i8_to_f16`, `repack_f16` | ~1,900 |
| `src/npu/i16.c` | `synth_i16`, `set_i16_out`; `act_lut_i16`, `silu/gelu/rsqrt/exp_i16`; `chain_mm_silu_i16`, `chain_gatesilu_i16`, `chain_mm_perchan_i16`; `ewmul_i16`, `add_i16`, `mul_perchan_i16`, `requant_perchan_i32`; `probe_silu_std_i16`, `replay_lut_i16`; `ssd_probe_mixchain` | ~950 |
| `src/npu/i4.c` | `synth_i4`, i4 fuzz, `tile_i4_*`; int4 quantization (`quant_chan_i4`, NF4 codebook + `quant_chan_nf4`, `expand_chan_i4_*`, `inflate_chan_nf4_*`, imatrix `wq_err_chan`/`wq_best_absmax`); packs/loads (`pack_i4[_grouped/_to_i8]`, `pack_i4a8[_im]`, `load_i4[_arena/_import]`, `load_i4a8[_import]`, `w_dump_i4a8`, `pack_i4a8_cpu_blob`, slice-inflate diagnostics, `slice_pack_i4`); runs (`ork_mm_run_i4`, `run_i4_mc_db`, `run_i4_grouped`, `slice_run_i4`); the bchain-doorbell path (`bch_db_cells[_off]`, `bch_db_worker`, `run_i4_bchain_db`, `i4_submit_tmo_ms`); MoE experts (`bch_mw_worker`, `run_i4_experts_bchain_db`, `ork_mm_run_i4_experts`); `ork_mm_run_chain_i4`, `ork_dyn_i4_probe`, `ork_dyn_begin_mc_i4[_grouped]`, `ork_dyn_grouped_end`; `run_stream_i4`; `probe_i4[_mm]`; `ork_bmm_i4[_strided]` | ~2,100 |
| `src/npu/ssm.c` | `ork_ssm_scan_f32` + pool (`ssm_pool_ensure/free`, `ssm_stg_i8`), little-core helper (`ssm_helper_*`, `ssm_marshal_gi`), `ork_ssm_prof_dump`, `ork_softplus`, SSM knobs, `ork_ssd_fused_scan_bench`, `ssd_probe_rawmm/fusedmm_f16` | ~330 |

**Result:** largest file **~3,900** (scaffold), down from 15,313; every precision module ≤ ~2,100 and
every `i8/` unit ≤ ~1,900. A precision-specific question loads one file, not the monolith.

---

## 6. Commit sequence (each commit builds *and* passes `make test` on the board)

| # | commit | risk | what it proves |
|---|---|---|---|
| 0 | `MODULARIZE_WIP.md` + branch off `main` @ `f968b56` | none | AGENTS §2 recovery-doc requirement |
| A | generalize `tools/check_registry.sh` to `src/*.c src/*/*.c src/*/*/*.c`; `clean:` → `rm -f $(COBJ)` | low | the build gate and `make clean` survive the split |
| B | in-place rename of generic internals to `orki_*` (still one TU) | low, compiler-checked | keeps renames out of the move diffs |
| C | create `src/npu/internal.h`; move types/macros/inline helpers + the `ork_regs.h`/`ork_slice.h`/`ork_spine.h` re-exports into it | low | header compiles; still one TU |
| D | lift `src/npu/sdp.c` | low | first real TU split; validates the recipe and the folder layout |
| E | lift `src/npu/i16.c` | low | most self-contained precision |
| F | lift `src/npu/ssm.c` | low | isolated |
| G | lift `src/npu/f16.c` | medium | RE replays + fused-act LUTs |
| H | lift `src/npu/i4.c` | medium | the `npu.c:4655–4660` fwd-decl block is its boundary list |
| I1 | lift `src/npu/i8/regcmd.c` + `pack.c` | medium | i8 substrate first |
| I2 | lift `src/npu/i8/fold.c` | medium | newest subsystem, cleanly bounded |
| I3 | lift `src/npu/i8/run.c` + `probe.c` | medium | |
| I4 | lift `src/npu/i8/chain.c` + `dyn.c` | **highest** | deepest entanglement; last, when the ABI has settled |
| J | docs: AGENTS §4 layout tree, `README.md`, `OPS_REGISTRY.md` paths, `tools/re/README.md` | none | AGENTS §2 doc-review rule |
| K | refresh + commit `tests/sbc_attest.txt`; bump the fork's `ggml-ork` CMake file list in lockstep | none | CI green, fork builds |

Per-move recipe (identical every time — no judgment calls):

```
1. extract the recorded line ranges verbatim into the target file
2. delete those ranges from src/npu.c
3. prepend  #include "npu/internal.h"  + only the regcmd_*.h that file actually needs
4. add the file to CORE in the Makefile
5. build on the board; fix ONLY "implicit declaration"/"unknown type" errors, and fix them ONLY by
   adding an extern to src/npu/internal.h — never by editing logic
6. run the verification ladder (§7)
```

---

## 7. Verification ladder (every commit D–I4)

| step | command | pass criterion |
|---|---|---|
| a | `board -c 'make -j'` | clean, **zero new `-Wall` warnings** |
| b | symbol diff | `nm -g --defined-only` over `$(COBJ)`, sorted, **identical** to the pre-split baseline (modulo the commit-B renames) |
| c | `board -c 'make test'` | all examples exit 0 — static goldens ⇒ **bit-exact NPU output** |
| d | `make test` under `ORK_SSM_KEEPWARM=0` and under `ORK_MIXED_NOTHRASH=1` | mode-transition profiles differ only at non-default knobs (AGENTS §4) |
| e | `make mode_probe && sudo env ORK_MM_TIMEOUT=2500 timeout 300 ./mode_probe` | op→op reset/wedge matrix unchanged |
| f | `ORK_DEBUG_RESET=1` on `make test` | `ACT_RESET` count and call sites unchanged |
| g | *(final)* `sudo ./mc_prof` + `ork_bench` + `make bench-llama` | per-core µs/submit and end-to-end tok/s within baseline — see §8.1 |
| h | *(final)* `make test` writes `tests/sbc_attest.txt` | commit it |

Capture the (b)/(e)/(f)/(g) baselines **before commit A** and store them in `MODULARIZE_WIP.md`.

---

## 8. Risks, ranked

1. **Loss of cross-TU inlining → measurable slowdown.** Everything is `static` in one TU today, so GCC
   inlines `setr`/`setrn`/`bsync`/`ork_now_us`/`pgup` freely into hot paths. *Mitigation:* those live
   `static inline` in `npu/internal.h` (§4.5); measure with `mc_prof` (per-core µs/submit is the
   sensitive metric) + `ork_bench` at step (g). If a regression appears, header-inline the offending
   helper, or add `-flto`. **Do not accept a perf regression to get a nicer file tree.**
2. **Symbol collisions / `.so` pollution** from de-static-ing `run`, `pack`, `act`, `budget`, `synth`.
   *Mitigation:* the `orki_` rule (§4), as its own reviewable commit.
3. **`ork_kv_resident` double-typedef** under the CMake/ggml-ork build → move the guard verbatim; verify
   by building the fork.
4. **CI attest failure** — prefer **one push at the end** of the branch over per-commit pushes.
5. **`check_registry.sh` breaks the build** the instant a `REGCMD_*` or probe leaves `npu.c` →
   commit A first. (Failing loudly and locally is a feature.)
6. **Fork drift** — the `ggml-ork` CMake names `src/npu.c`; commit K bumps both, deletion-aware squash.
7. **Unlanded work lives in the FORK's submodule, not this repo — triaged 2026-08-19.** The tree that
   matters is `~/Dev/llama.cpp/ggml/src/ggml-ork/ork-driver` (fork on `feat/moe-decode-chain`
   @ `58f52c75f`). That submodule is on `main` @ `f968b56` with **zero tracked modifications** — but
   **15 untracked files**, all dated 2026-08-18, that exist in no commit anywhere:
   - `src/ork_gptq.c` (129 lines) + `examples/test_gptq.c` — a dependency-free GPTQ int4 quantizer
     (Frantar et al. 2022), gated `ORK_GPTQ`, UNVALIDATED pending an AutoGPTQ cross-check (task #56).
     **`origin/main`'s `include/ork_npu.h:376` already declares `ork_gptq_i4()`** — main ships a public
     declaration whose only implementation is unpushed. A byte-identical copy also sits on the board, so
     it is two-copy, not one — but neither is in git.
   - 9 more untracked probes: `test_bimport_dom`, `test_i4_2import`, `test_i4_domains`, `test_moe_par`,
     `test_moe_prog`, `test_moe_smoke`, `test_npubw`, `test_sram_bw`, `test_sram_npubw`, `test_sram_pipe`.
     These add `examples/` sources, so they interact with the Makefile `EXAMPLES` list the split edits.
   - `COLSPLIT_MULTIPREC_WIP.md` (512 lines) — **stale**: it plans task #45 (port colsplit to fp16/int4,
     then remove `mcworker`), which has since **landed**. In `main`, `ORK_F16_COLSPLIT` defaults to **1**,
     colsplit is the only multicore path, and every blocking `mcworker` / `i4_mcworker` / INCR / CBATCH
     path is removed (#45/#52) — the 39 remaining `mcworker` hits are comments. AGENTS §2 says don't
     commit stale scratch docs: fold it into the wiki or delete.
   - `MOE_LAYER_BLOCK_WIP.md` (862 lines) — **active** (#54, MoE per-layer expert residency across the 8
     IOMMU domains). This will touch the same `npu.c` regions the split moves; sequence it, don't race it.
   **Board copy** (`~/llama.cpp/…/ork-driver`, detached at `3c8e652` v1.0.8): its ~930-line `git diff` is
   an artifact of the stale HEAD — `src/npu.c` there is byte-identical to `origin/main`, as are 8 of 10
   modified files and both diff probes. Two things still need doing there: **delete the stale shadowing
   headers** `src/ork_npu.h` (v1.0.8) and `src/ork_native_cpu.h` (129 lines adrift) plus the duplicate
   `src/ork_slice.h`/`src/ork_spine.h` — `src/npu.c:40` is `#include "ork_npu.h"` and GCC searches the
   including file's own directory first, so **the board has been compiling against the stale header**
   (missing `ork_npu_active_domain()` / `ork_dma_alloc_flags()`) — and **re-point the submodule HEAD at
   `main`**. The verification ladder (§7) assumes a board build reflects the tree; today it does not.
   For the duration, sync **all of `src/`** via `tools/util/sync_daemon.sh`, never a single file.
8. **A lift blocks mid-way** → AGENTS §2: do not revert. Park it, record it in `MODULARIZE_WIP.md`,
   consult.

*(Branch topology is no longer a risk: `feat/op-domain-id` is fully merged into `origin/main`, and
`feat/orkd-guard-settle-stash` is the preservation stash created by the deliberate revert `1edb68e` —
all 60 of its commits have upstream equivalents. Nothing pending to land.)*

---

## 9. Rounds 2–4 (sketch)

- **Round 2 — finish the layer axis.** Apply the same rule one level down where a module is still
  large: `src/npu/i4.c` + `src/npu/i4/{pack,run,bchain,experts,chain}.c`; `src/npu/f16.c` +
  `src/npu/f16/{regcmd,run,perchan,replay,stream}.c`; and the scaffold's own bulk to
  `src/npu/core/{device,buf,domain,submit,sched,mode,prof}.c`, leaving `src/npu.c` a thin entry point.
  Keeping layer files under `core/` stops the folder mixing precision axes with layer axes. Target: no
  file over ~1.2 k lines. `run_chain_i8_impl` (420) and `ork_dyn_begin_colsplit` (329) also want
  *internal* decomposition — behavior-risky, so separate and individually benched.
- **Round 3 — public header.** `include/ork_npu.h` (1,517 lines) → umbrella including
  `ork_npu/{types,i8,i4,f16,chain,probe}.h`. `ork_npu.h` stays the single public include (AGENTS §5).
  Optional companion: move `src/regcmd_*.h` into `src/npu/regcmd/` and fix the six `tools/`+`examples/`
  includes that reach them via `-Isrc`.
- **Round 4 — daemon.** `orkd.c` (1,003) → `orkd.c` + `orkd/{handlers,ring,domain,layer}.c`;
  `orkd_client.c` (757) → `orkd_client.c` + `orkd_client/{conn,ops}.c`. Lower priority: both are
  already handler-organized.

---

## 10. Definition of done for round 1

- [ ] `src/npu.c` ≤ ~4 k lines; no file under `src/npu/` over ~2.1 k.
- [ ] `make test` green on RK3588, all goldens **unchanged** (no regen).
- [ ] `make test` green under `ORK_SSM_KEEPWARM=0` and `ORK_MIXED_NOTHRASH=1`.
- [ ] `mode_probe` matrix and `ORK_DEBUG_RESET` counts identical to baseline.
- [ ] `mc_prof` / `ork_bench` / `make bench-llama` within baseline.
- [ ] `nm` symbol set identical modulo the commit-B renames.
- [ ] `make check-registry` and `make check-attest` pass; `tests/sbc_attest.txt` committed.
- [ ] Fork builds against the bumped submodule.
- [x] **Board reconciled (risk 7) — DONE 2026-08-19.** Stale shadowing headers + a stale Aug-8
      `src/ork_gptq.o` deleted, worktree stashed, submodule HEAD fast-forwarded to `main` @ `f968b56`,
      clean rebuild + `make test` ALL PASS. `ork_gptq_i4` stubbed and landed (`61ccb48`); the real
      implementation and the 9 probes stay parked on `origin/wip/moe-saga-2026-08-18`.
- [ ] Use `tools/util/sync_daemon.sh` for the duration (sync all of `src/`, never a single file).
- [ ] `AGENTS.md` §4 layout, `README.md`, `OPS_REGISTRY.md`, `tools/re/README.md` updated.
- [ ] `MODULARIZE_WIP.md` deleted (or folded into the wiki).

## Round 2 — reuniting stranded doc-comments (2026-08-20)

Round 1 moved code out of `npu.c` and left its **doc-comments behind**. Nothing was duplicated at the
destination: of the stranded blocks, 100% had no counterpart in the module that received the code. So
`npu.c` documented functions it no longer contained, and 36 functions across 17 modules had no doc at all.

Moved: 36 blocks / 205 lines, each one where the comment's FIRST LINE names the symbol it documents —
verified by a comment-stripping fingerprint (`gcc -fpreprocessed -dD -E` over every source, hashed before
and after) so the change is provably comment-only, zero behaviour delta.

**Deliberately left in `npu.c`: 70 blocks / 378 lines.** These are the ambiguous ones — a block whose first
line names no symbol (a section banner like `===== SUBMIT QUEUE =====`), or one that cites several
modules' symbols without naming its own subject. Automated placement by "first symbol mentioned" was tried
and scored **31/88 correct**, which is far too low for RE narrative that AGENTS treats as load-bearing, so
it was reverted rather than shipped. Placing the remaining 378 lines needs a human read of each block; it
is worth doing, but not worth guessing.

Two tooling bugs surfaced here and are fixed:
- the symbol extractor was greedy (`s/^.*[ *]\(name\)(.*/\1/`), so a one-liner that CALLS another function
  recorded the callee as defined locally — 32 phantom definitions, which is what made the first placement
  pass look plausible while being wrong;
- knob accessors (`int ork_f16_colsplit(void){ ...getenv... }`) counted as implementations, which briefly
  turned a correct dagger in the capability matrix into a false claim of native fp16 multicore.

## Round 3 — splitting the public header (2026-08-20)

`include/ork_npu.h` was 1519 lines and 56% comment. It is included by **445 files in this repo plus the
fork's `ggml-ork.cpp`**, so it stays exactly where it is and keeps its name: it is now a 49-line umbrella
holding the include guard, the base typedefs and the version macros, then including nine parts from
`include/ork/`. No consumer changes.

The split is **contiguous** — every part is a verbatim line range of the original, included in the original
order. Two invariants were checked mechanically, and they are the reason this round needed no `make test`:
- concatenating the parts reproduces the original body **byte for byte** (1484 lines both sides);
- the declaration set is **identical** (295 both sides).
Headers are not in `ATTEST_SRCS` (it hashes `.c` only), so a header-only change cannot alter the attest;
correctness here is "does it still compile", which is a board build with no NPU execution.

| part | lines | holds |
|---|--:|---|
| `ork/context.h`  | 134 | lifecycle, SoC introspection, core budget, IOMMU domains |
| `ork/dma.h`      |  46 | zero-copy buffers, dma-buf import |
| `ork/weights.h`  | 238 | pack / load / dump / free, resident KV, streaming pool |
| `ork/run.h`      | 127 | matmul entrypoints, fused matmul+activation, sliced rescue |
| `ork/ops.h`      | 451 | SDP ops and norms — **interleaved with their RE probes** |
| `ork/dynamic.h`  |  86 | MoE/chained matmuls, NONBLOCK doorbell, queue, precompiled chains |
| `ork/seq.h`      | 265 | op vocabulary, the op→op chaining lookup, `ork_submit_seq` |
| `ork/chain.h`    | 139 | static chains, round-robin dispatch, streams, async |
| `ork/bmm.h`      | 111 | batched GEMM (attention), floor-decomp, mode hooks, SSM |

**What the split exposed, and did not fix.** `ork/ops.h` is 451 lines because the production SDP surface
(ewmul/add/silu/gelu/exp/rsqrt, per-channel multiply, RMSNorm, RoPE, softmax) and the probe/replay/fuzz
surface alternate every 20–60 lines through that whole region — they were written together while the ops
were being reverse-engineered. A contiguous split cannot separate them, and separating them means moving
declarations one at a time. That is deliberately NOT done here: round 2 measured exactly that kind of
heuristic placement at 31/88 correct. Splitting `ops.h` into `sdp.h` + `probe.h` is the round-4 candidate,
and it wants a human read, not a rule.

Worth noting for whoever does it: roughly a third of the "public" header is RE probe surface that no
production consumer calls. Separating it would make the actual supported API legible for the first time.

## Round 4 — separating the supported API from the RE surface (2026-08-20)

Round 3 left `ork/ops.h` at 451 lines because production SDP ops and their RE probes alternate every
20–60 lines. This splits it into `ork/sdp.h` (239) and `ork/probe.h` (225).

**Classified by evidence, not by name.** The rule is *who calls it*: a declaration is production if it has
a caller in `src/` (excluding its own defining file), in `examples/` (which are the test suite), or in the
fork's `ggml-ork`. Otherwise it is reached only from `tools/` and is RE surface. Naming would have been the
wrong instrument — `ork_npu_mul_perchan_f16_contig` sounds production and has no caller; `ork_npu_chain_progs`
sounds like a probe and has six library callers.

Result: **46 production / 53 RE — 53% of what shipped in the public header has no production consumer.**
That is the headline. `ork_npu.h` was presenting the reverse-engineering apparatus and the supported API as
one undifferentiated surface, and no reader could tell which was which.

Because this REORDERS declarations (unlike round 3's contiguous cut), three things were checked:
- every content line of `ops.h` appears exactly once across the two new headers (396 = 396, sorted compare);
- the whole-header declaration set is unchanged (296 = 296);
- a board check, completed after the fact (the board was tied up by another session's suite when
  round 4 was committed, so the commit message says "board build still pending" — it no longer is):
  all 41 library translation units pass `gcc -std=c11 -fsyntax-only` on the RK3588 with zero output,
  and a C++17 TU including <ork_npu.h> is clean, which is the path ggml-ork.cpp actually uses.

`ork_chain_prog`, the one type declared in `ops.h`, goes to `sdp.h`, which the umbrella includes first.

### Dead declarations — found, NOT removed

Eight exported declarations have no reference anywhere in the repo, the examples, the tools, or the fork:

| symbol | defined in |
|---|---|
| `ork_mm_chain_build_exp_lut` | `src/npu.c` |
| `ork_npu_benchmark_chain` | `src/npu/i8/probe.c` |
| `ork_npu_chain_selftest` | `src/npu/i8/probe.c` |
| `ork_npu_mfold_chain_cap` | `src/npu/i8/fold.c` |
| `ork_npu_mfold_chain_multi` | `src/npu/i8/fold.c` |
| `ork_npu_probe_chain_i8` | `src/npu/i8/chain.c` |
| `ork_npu_replay_i8_amap` | `src/npu/i8/probe.c` |
| `ork_npu_reshape_probe_f16` | `src/npu/f16/replay.c` |

These are NOT deleted. Several are `#39`/`#54` RE apparatus whose probe driver may live outside this repo or
have been folded into a tool since, and AGENTS is explicit that experimental code is not discarded on an
agent's own judgement. They are listed here so the decision can be made deliberately, per symbol, by someone
who knows whether the experiment is finished.

## Round 5 — finishing the stranded docs, using git history as the anchor (2026-08-20)

Round 2 left 69 doc blocks in `npu.c` because rule-based placement scored 31/88 and I would not ship
that. This round places 40 of them (215 lines) on **exact provenance instead of inference**.

The insight is that the evidence was in the repository all along. Round 1 moved functions out of `npu.c`
and left their docs behind — which means in the **pre-split tree, each stranded doc sat immediately above
the function it documents**. So: find the block's verbatim text in the 15,185-line pre-split `npu.c`, read
the definition on the next line, and look up where that function lives now. No naming heuristic, no vote.

The difference is not marginal. Where the earlier heuristics said:

| block | heuristic said | history proves |
|---|---|---|
| `set_f16_silu — graft the SiLU LUT output stage…` | `i8/regcmd.c` | **`f16/run.c`** |
| `SINGLE-SUBMIT fp16 matmul, per-channel scale fused` | `i8/regcmd.c` | **`f16/perchan.c`** |
| `Run-SCRATCH allocator … bimport` | `core/sched.c` | **`core/buf.c`** |
| `fp16 multicore doorbell colsplit` | `i8/colsplit.c` | **`f16/stream.c`** |

Adjacency is required, not just "the text appears somewhere": a block whose next pre-split line is another
comment belonged to a multi-block run and is NOT placed by this rule.

`npu.c` 3172 → 2953. Provably comment-only by the stripped-source fingerprint — and this time the stripper
was checked to emit real content (17,552 lines), after the round-2 version turned out to be hashing nothing
but filenames.

Three blocks the round-2 detector had called stranded are **not** stranded: their subject
(`orki_budget`, `ork_w_dump`, `ork_stage_fill`) is still in `npu.c`. They stay, and the count of genuinely
stranded blocks was 66, not 69.

Two more were not deletions but round 1's `orki_` prefixing (`set_f16_out_fp16in`, `splice_ew_lane`), and
one subject became a `static inline` in `internal.h` (`ork_softmax_npu_enabled`) while the op it gates
(`ork_npu_softmax_f16`, undocumented) lives in `norm.c` — that is where its doc went.

### Still in `npu.c`: 26 blocks / ~130 lines

- **17 blocks** whose text is absent from the pre-split image, i.e. written or edited after it. A later
  pre-image would anchor them; the same history method applies, it just needs the right tree.
- **8 blocks** that were part of a multi-block run even before the split — section banners like
  `===== SUBMIT QUEUE =====` that head a group rather than document one function.
- **1 block**, the `effective w4a8` banner, whose pre-split neighbour is semantically unrelated
  (`ork_xs32`, a PRNG) — proof that adjacency alone is not sufficient for banners, which is why the
  banners are excluded rather than placed.

## Round 6 — generalized history anchoring, then splitting `i8/probe.c` (2026-08-20)

**Part 1: the doc residue.** Round 5 anchored blocks against ONE pre-split image, which only worked for
docs written before it. Generalizing: walk all 511 historical `npu.c` versions newest-first and, per block,
stop at the first tree where that block's verbatim text is immediately followed by a definition. Different
blocks resolve against different commits — 6, 9, 11, 62 commits back — which is exactly why a single
pre-image missed them.

That placed 8 more blocks (43 lines), and identified 4 more that must **stay** because their subject
(`orki_budget`, `ork_w_dump`, `ork_stage_fill`, `ork_w_free`) is still defined in `npu.c`. Running total of
false "stranded" classifications from round 2's detector: 7.

For section banners (which head a group rather than document one function) adjacency is the wrong test —
the `effective w4a8` banner's pre-split neighbour is `ork_xs32`, a PRNG. Instead: find the banner's section
in the pre-split tree and see which module received its functions. Only 2 banners had unambiguous evidence
(≥2 definitions, 100% agreement) and moved — DYNAMIC STEERED SUBMISSION → `i8/dyn.c` (2/2), SUBMIT QUEUE →
`i8/queue.c` (9/9). The rest scored 25–57% and stayed. `npu.c` 2952 → 2892.

**Part 2: `i8/probe.c` 1,270 → five files.** It was the largest module after the scaffold and pure RE
surface. Split as CONTIGUOUS line ranges by probe family:

| file | body lines | holds |
|---|--:|---|
| `i8/probe.c` | 131 | fuzz-override hooks + matmul probes (mtile, single, out8, mm) |
| `i8/probe_sdp.c` | 332 | ewmul (three template variants), mul, add, standalone SiLU |
| `i8/probe_replay.c` | 216 | verbatim regcmd replay, A-variant sweep, A-layout mapper, fused-SiLU cfg |
| `i8/probe_prof.c` | 167 | submit-floor + overlap profiling, batch / b-scale acceptance |
| `i8/probe_chain.c` | 395 | chain benchmark, `ork_npu_chain_progs`, hetero / SDP-fwd / self-test |

Safe to split because `probe.c` has **no file-scope statics**; the one file-scope definition
(`orki_i8_fovr[]`, externed via `internal.h`) and the one macro (`CPU_ROUTER`, lines 775–799) each fall
entirely inside a single group.

Verified by a tree-wide **code-multiset** invariant — strip comments from every source, drop filenames,
sort, compare. Result: **zero removals**, and all 104 additions are replicated `#include`/`#define`
preamble (4 files × 26 lines). No code line was added or lost by the move. Plus `make clean && make` on the
board and an exported-symbol diff over `libork_npu.so`.

`CORE` in the Makefile gained the four new TUs. Forgetting that would not fail the build — the objects
simply would never be compiled and the symbols would go missing at link, so it is checked, not assumed.

### Follow-up this exposed — since fixed
`ork_npu_chain_progs` had **six library callers** (round 4 classified it production) yet lived in
`probe_chain.c`. The contiguous split preserved that misplacement rather than fixing it. Now moved to
`i8/chain.c`, where the heterogeneous chain paths that depend on it live.

Moving it turned up a second defect in the same place: the comment sitting above it was **not its doc**. It
described a self-test filling `*t0_cnt/*t1_cnt` — i.e. `ork_npu_chain_selftest`, which was itself
undocumented 200 lines further down. `chain_progs`' real contract was in `include/ork/sdp.h` all along. So
the move both relocated the function and returned the doc to the function it describes.

Verified by the code-multiset invariant being **exactly neutral** (17,656 lines both sides — a pure
relocation adds and loses nothing, unlike the probe split which legitimately added replicated preamble),
literal-aware brace balance 0 in both files, and a board build + `make test`.

## Round 7 — the prerequisite, not the split (2026-08-20)

Round 7 was going to split `orkd.c` (1,003 lines). It didn't, and the reason is worth recording because it
is the first time in seven rounds that measurement said *don't*.

**Why not.** Splitting `orkd.c` means de-staticing daemon internals, because the file is densely connected:
every one of the 19 `handle_*` functions calls `readn`/`writen`/`send_msg`. Measured cost at five candidate
cut points:

| cut | de-statics |
|---|--:|
| before the handlers (topic cut) | **22** |
| before `ring_service` | 40 |
| handlers + `dispatch_one` together | 40 |
| before `daemonize` | 36 |
| before `orkd_warmup` | 37 |

The intuition that moving `dispatch_one` in with the handlers would let all 19 stay `static` is **wrong** —
it just moves the cost to `main`/`daemonize`, which then need the helpers, `wk_*`, `dom_*` and the globals.
There is no cheap seam. 22 is the floor, of which ~6 (the small I/O wrappers) could stay internal as
`static inline` in a private header.

22 de-statics is a defensible price for splitting a 1,003-line file. Doing it with **no behavioural test**
is not: `make test` never started the daemon, so the only gate was "it compiles". Every other round had a
behavioural gate.

**So round 7 built the gate instead.** `make test` now ends with `orkd_probe mm`, which auto-spawns `orkd`,
packs a weight and runs a matmul THROUGH the daemon, and compares against an EXACT integer CPU reference
(A is all-ones, so `C[m,n] = sum_k B[k,n]`) — exit 3 on mismatch. Validated standalone first: 128/128
correct. It runs LAST because the daemon takes ownership of the NPU, and the suite SIGTERMs any surviving
`orkd` afterwards (never `-9`; an abrupt kill mid-submit wedges the IOMMU and costs a power-cycle).

Measured daemon lifecycle: `orkd` idle-reaps on its own ~6 s after the last client disconnects, cleanly and
with no IOMMU error. That made the Makefile reaper look like belt-and-braces — but in the real suite it
FIRES, every run: the suite reaches the reap step within the 6 s linger, so without it `make test` would
exit while a daemon still owned the NPU. On a shared board that is not cosmetic. Keep it.

**Round 8 can now split `orkd.c`** with the same safety as rounds 1-6: cut before the handlers, 22
de-statics with `orkd_` prefixes (AGENTS requires a prefix once a symbol crosses a TU), ~6 of them as
`static inline` in a private `src/orkd_internal.h`.

## Round 8 — splitting `orkd.c`, now that the daemon has a test (2026-08-20)

`orkd.c` 1,003 → **`orkd.c` 469 + `orkd_handlers.c` 561 + `orkd_internal.h` 51**. The cut puts the 19
per-opcode request handlers in their own TU; the dispatch loop, socket accept, ring service and `main()`
stay in `orkd.c`.

**The cost is 28 de-statics, not the 22 I quoted in round 7.** That earlier figure came from a looser
`statics()` regex, and re-measuring properly gave 31 — which then came *down* to 28 for two reasons worth
recording, because both are the same mistake in different clothes:

- `g_dom_inuse` appeared to cross the boundary. It doesn't: the only mention inside the handler block is in
  a **comment**. Handlers reach the pool through `dom_alloc_explicit`/`dom_release`.
- `ring_service` likewise — comment-only.

Text matching over comments inflated the count by 3. Stripping comments before measuring is the fix, and it
is the same lesson that produced round 2's 31/88 misplacement and check 7's false alarm on
`ORK_F16_FORCE_WEDGE`.

**No mutable global crosses the split.** `g_ring_c` (the daemon-side A-ring scratch) would have, so the cut
is placed 4 lines earlier, leaving it with `ring_service` — the only code that touches it. A shared mutable
global across a new TU boundary is a worse smell than a shared function, and here it was avoidable for free.

I said in round 7 that ~6 small I/O helpers would become `static inline` in the private header. They did
not. For per-message syscall wrappers the inlining is worth nothing, and moving function bodies into a
header costs discoverability; with the `orkd_` prefix the namespace concern is already handled. All 28 are
plain external declarations — one mechanism, not two.

Verified by a **rename-aware code-multiset** invariant: apply the identical 27 renames and de-statics to the
baseline `orkd.c`, strip comments, normalise whitespace, sort, compare. **Zero lines lost**; the 20
additions are all the replicated `#include` block plus `orkd_internal.h`.

Renames touched **code positions only** — comment spans were computed and excluded — because `drain` also
appears as an English verb ("drain n bytes into the void"). The one genuine symbol reference living inside a
comment (`dom_alloc_explicit` in the domain-pool note) was patched by hand.

### The interface was bigger than the function list
My cut analysis counted only static FUNCTIONS and VARIABLES. The first build failed because the handlers
also need **3 macros** (`ORKD_MAX_WEIGHTS`, `ORKD_MAX_BYTES`, `ORKD_NDOM`) and the **full definitions of 4
structs** (`cweight`, `ckv`, `client`, `work`) — they touch `cl->fd`, the weight table and the KV table, so
a forward declaration is not enough. Those moved into `orkd_internal.h` as well (moved, not duplicated).

Splitting a translation unit shares **types and macros**, not just symbols. A de-static count is a lower
bound on the interface, never the whole of it.

Also caught on the way: the generated header initially contained four *definitions* rather than
declarations, because the prototype extractor only stripped a trailing `{` and four of the helpers are
one-liners with their body on the same line. The stray `{` opened a scope inside the header, which is why
gcc reported "invalid storage class for function" against `spine_kernels.h` and "defined but not used" for
non-static functions — both symptoms of the file nesting inside a function. The extractor now truncates at
the first brace.

### The Makefile trap this round
`orkd:`'s recipe was `$(CC) ... -o $@ $< $(COBJ)`. `$<` is only the FIRST prerequisite, so adding
`orkd_handlers.c` to the dependency list would NOT have compiled it — the link fails on all 19 handlers.
The recipe now names both TUs explicitly. Prerequisite lists and compile lines are separate things; adding
to one is not adding to the other.

## Round 9 — `orkd_client.c`, and the case where `static inline` is right (2026-08-20)

`orkd_client.c` 757 → **`orkd_client.c` 219 + `orkd_client_ops.c` 548 + `orkd_client_internal.h` 29**. The
cut separates the connection/spawn/atexit machinery, the A-ring fast path and the domain calls from the
per-opcode RPC wrappers.

**This split adds ZERO external symbols**, which is the interesting part. Measuring the interface at five
candidate cuts (comments stripped, and counting macros and types as well as statics — both lessons from
round 8) showed `orkd_client.c` is far more loosely coupled than `orkd.c`:

| cut | interface size |
|---|--:|
| before `cdrain` | 5 |
| **before `orkd_pack_i8`** | **3** (all forward, zero backward) |
| before `orkd_sdp_call` | 3 |
| before `orkd_run_chain_i8` | 3 |
| before `orkd_dmabuf_probe` | 3 |

The three are `wn`, `rn`, `cdrain` — 1-to-5-line loops around a single `read()`/`write()`. Those became
`static inline` in a private header, so each TU keeps its own internal-linkage copy and nothing new is
exported.

**This is the option I rejected in round 8, and rejecting it there was right.** The rule that separates the
two cases is size, not principle: a 5-line syscall wrapper belongs in a header, a 40-line request handler
does not. Round 8's 19 handlers had to become extern; round 9's 3 wrappers had to not. Same question,
opposite answers, and the deciding factor is whether a reader would expect to find the body there.

One line was deliberately dropped rather than moved: `static void cdrain(int fd, size_t n);`, a forward
declaration that existed only because `orkd_ring_setup` used `cdrain` above its definition. With `cdrain`
now defined in a header included at the top, it is obsolete. The multiset invariant flagged it as the single
"lost" line, which is exactly what that check is for — it forced the removal to be justified rather than
unnoticed.

### The typedef blind spot
The build still failed once. `orkd_client_ops.c` dereferences `c->fd`, so it needs the full
`struct orkd_conn` — but my interface analysis looked for the literal string `struct orkd_conn` in the ops
half, and that half only ever writes the TYPEDEF ALIAS (`orkd_conn *c`). The public `orkd_client.h` keeps
the type opaque (`typedef struct orkd_conn orkd_conn;`) with the definition in the .c, which is good design
and exactly what hid it.

**A typedef'd struct crosses a TU boundary under a name the `struct` keyword never appears in.** Round 8
taught that the interface includes types and macros, not just symbols; round 9 adds that detecting a type's
use means searching for its alias too. The definition now lives in the private header, so both halves see
it while callers outside still cannot.

### Nine recipes, one variable
`src/orkd_client.c` was named in **nine** Makefile places — one `COBJ` entry, four prerequisite lists and
four compile lines. Adding a second TU would have meant nine edits, and missing any one fails at link (or,
worse, silently omits code). They now all go through `ORKD_CLIENT_SRC` / `ORKD_CLIENT_HDR`, so the next file
added to this split is a one-line change. This is round 8's `$<` trap generalized: a build system that
repeats a file list will eventually disagree with itself.

## Round 10 — `i8/dyn.c`, the free split (2026-08-20)

`i8/dyn.c` 1,000 → **`dyn.c` 491 + `dyn_seq.c` 165 + `dyn_ctl.c` 405**. It was the largest non-scaffold
file left; the `regcmd_*.h` above it in a size listing are captured data tables, not code.

**The interface cost is zero.** `dyn.c` had **no file-scope statics at all** — every one of its 20
definitions is an extern `ork_dyn_*`/`orki_*`, and it defines no macros, structs or typedefs beyond
`_GNU_SOURCE`. Measuring six candidate cuts returned 0 forward and 0 backward at every one. That is the
opposite extreme from round 8's `orkd.c` (28 symbols, no cheap seam), and it is worth recording that the
range exists: two files of comparable size can differ completely in how expensive they are to split. The
cost is a property of the call graph and the use of internal linkage, not of the line count.

Cut thematically, since nothing constrained where:

| file | lines | holds |
|---|--:|---|
| `i8/dyn.c` | 491 | the doorbell BEGIN paths — `ork_dyn_begin`, `ork_dyn_begin_mc`, and the shared drain/completion helpers |
| `i8/dyn_seq.c` | 165 | the heterogeneous single-core chain — `ork_dyn_begin_seq_i8{,_mc}`, `ork_dyn_seq_end` |
| `i8/dyn_ctl.c` | 405 | the control surface — spin probe, anomaly dump, step accounting, append, mid-flight halt, multi-core resubmit recovery |

Verified by the tree-wide code-multiset invariant: **zero removals**, and all 50 additions are replicated
`#include`/`#define` preamble (2 new files × 25 lines). `CORE` gained both new TUs — that is checked, not
assumed, because omitting it does not fail the build; the objects simply never compile.

## Round 11 — widening the daemon gate to match what rounds 8-10 moved (2026-08-20)

Round 7 gave `orkd` its first behavioural test, and rounds 8-9 then restructured the daemon and its client
substantially. But `orkd_probe mm` only exercises **pack → run → free**. Everything else rounds 8-9 touched
— the A-ring transport, the heterogeneous seq path, the domain calls — was being validated by "it links".

Three sibling probes already existed as board tools and already self-validate with a nonzero exit; they were
simply never wired in:

| probe | covers what `orkd_probe mm` does not |
|---|---|
| `orkd_ring_probe` | `orkd_ring_setup/submit/collect` + `handle_ring_setup` + `ring_service` + `g_ring_c` — bit-exact socket-vs-ring, sync **and** pipelined |
| `orkd_seq_probe` | `handle_seq` + `orkd_submit_seq` — six attention SDP ops checked against a CPU reference, via Path B |
| `orkd_dom_api` | `handle_dom_req`/`handle_dom_rel` + `orkd_domain_alloc/free` — bit-exact per domain |

That is close to exactly the code rounds 8, 9 and 10 relocated, which is the point: the gate should cover
what the refactor moved, not just what was easy to test first.

**The four daemon tests are now a grouped block rather than entries in the generic loop.** `orkd_seq_probe`
needs `ORK_USE_ORKD=1` to take Path B, and the loop has one shared environment — so it gets its own line
with `$(SUDO) env VAR=..`, never `VAR=.. sudo`, which strips it (the trap already documented in AGENTS §3).
Grouping also keeps the "daemon owns the NPU" tests together, immediately before the SIGTERM reaper.

Each was validated STANDALONE before being trusted in the suite. A probe that has never been a gate test is
not known-good just because it exists — wiring in a flaky one would poison every future run, and the cost of
finding out later is far higher than the cost of checking now.


## Round 12 — the NPU guard, and what a bad contention check cost (2026-08-20)

Round 11's board validation **collided with a parallel session's `make test`**. Result: three
`RKNPU: switch iommu domain time out` faults with `rknpu_gem_object_create error`, and — the part that
actually hurt the other run — an orphaned `orkd` left holding an IOMMU domain, which blocked *their* domain
switches until it was SIGTERMed. It exited in 1 s and the fault count froze; damage was bounded, but caused.

**The cause was my detection, not my timing.** I polled `pgrep -x make`. Their suite runs its recipe as
`/bin/sh -c fail=0; for t in ...`, so `make` never matched and the board read as free.

The parallel session had already written a better check as `/tmp/npu_guard.sh`, and its own comment names
the exact bug: *"name-based pgrep misses renamed binaries."* It asks who holds the **render node** via
`/proc/*/fd` → `renderD12[89]`, which sees the device rather than a process name, plus NPU utilisation and
whether a fault storm is in progress (a wedge in flight looks idle by utilisation alone).

Landed at `tools/util/npu_guard.sh`, attributed in the header, detection logic unchanged — `/tmp` does not
survive a reboot, and two sessions gating on two different mechanisms is how this happened.

**Added on top: an flock.** Checking and then launching is a race; the window between "guard says free" and
"my submit starts" is precisely where the incident landed. With `--` the lock is held for the lifetime of
the command. Verified all four paths, not assumed:

| case | result |
|---|---|
| free board, check only | `GUARD OK`, exit 0 |
| `-- <cmd>` on a free board | runs, lock released |
| second session while held | `GUARD BLOCKED`, **exit 1**, command did not run |
| `ORK_NPU_LOCK_WAIT=15` | queued, then ran |

The blocked case initially reported `exit=0` — that was `tail`'s status through a pipe, not the guard's.
Re-tested without the pipe: exit 1. A guard that prints BLOCKED and exits 0 is worse than no guard, and
that one is easy to accept by eye.

### Also in round 12: the docs had gone stale under the refactor
AGENTS §4's layout map still described the pre-round-6 tree — no `dyn_seq`/`dyn_ctl`, none of the four
`probe_*` files, no `include/ork/` (rounds 3-4), and none of the four orkd files (rounds 8-9). A map that
describes a tree that no longer exists is worse than no map, because it is the first thing a new agent
reads. Fixed, along with which half of the public header is supported API versus RE surface.
