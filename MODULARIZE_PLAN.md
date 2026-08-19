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
