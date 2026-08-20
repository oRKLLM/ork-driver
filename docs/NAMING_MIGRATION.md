# Naming migration — dtype-first (2026-08-20)

Every precision-tagged symbol moved the dtype to the **front** of the name, so `ork_i8_*` reads as that
datapath's namespace and mirrors the module tree (`src/npu/i8/run.c`). C has no namespaces; the prefix is
the namespace.

## The rule

```mustache
ork_{{dtype}}_{{family}}_{{verb}}{{#mechanism}}_{{mechanism}}{{/mechanism}}{{#modifiers}}_{{.}}{{/modifiers}}
```

`dtype` = `i8`|`i4`|`i4a8`|`nf4`|`f16`|`i16` · `family` = `mm`|`npu`|`bmm`|`dyn`|`w` ·
`mechanism` = `chain`|`stream`|`fold`|`slice` (before the dtype) ·
`modifier` = `silu`|`out8`|`grouped`|`import`|… (after it)

Dtype-**agnostic** surfaces keep no tag and did not move: `ork_dma_*`, `ork_npu_init`, `ork_pc_*`,
`ork_dyn_begin`, `ork_w_dump`, `ork_stage_fill`. `ork_dyn_begin_mc` also stays untagged — one
implementation genuinely serving i8 *and* f16.

## Conversions

A function converting between precisions takes the **source** dtype as its prefix and names the
destination explicitly, so the name reads in data-flow order:

```
ork_mm_pack_i4_to_i8   ->  ork_i4_mm_pack_to_i8
orki_tile_direct_i4_i8 ->  orki_i4_tile_direct_to_i8
```

## Are my calls broken?

**Yes — this is a clean break.** There are no compatibility shims. Every renamed symbol changed name with
no alias left behind, which is why the commit is marked `refactor!`.

That was deliberate: a shim layer would have added ~51 deprecated symbols to a public header that a prior
audit already found to be 53% surface with no production caller. Paying down that surface was the point of
the exercise; growing it to soften a rename would have undone it.

Fixing a consumer is mechanical — the full table below is the complete list, and every rename is a pure
token substitution with no signature or behaviour change:

```sh
# apply the table to a tree, longest names first so prefixes do not shadow
sed -i 's/\bork_mm_run_chain_i8_ffn\b/ork_i8_mm_run_chain_ffn/g;
        s/\bork_mm_run_i8\b/ork_i8_mm_run/g' your_file.c
```

Nothing else moved: signatures, argument order, return values and semantics are unchanged. If it compiled
before and the name resolves now, it behaves identically — verified by a code-multiset invariant over the
whole tree and by `make matrix` reproducing the capability table unchanged.

## Full map (193 renames)

`(internal)` = not part of the public API; external consumers will not have called it.

| old | new | |
|---|---|---|
| `ork_bmm_i4` | `ork_i4_bmm` | (internal) |
| `ork_bmm_i4_strided` | `ork_i4_bmm_strided` | (internal) |
| `ork_bmm_i8` | `ork_i8_bmm` | (internal) |
| `ork_bmm_i8_strided` | `ork_i8_bmm_strided` | (internal) |
| `ork_dyn_begin_mc_i4` | `ork_i4_dyn_begin_mc` | (internal) |
| `ork_dyn_begin_mc_i4_grouped` | `ork_i4_dyn_begin_mc_grouped` | (internal) |
| `ork_dyn_begin_seq_i8` | `ork_i8_dyn_begin_seq` | (internal) |
| `ork_dyn_begin_seq_i8_mc` | `ork_i8_dyn_begin_seq_mc` | (internal) |
| `ork_dyn_i4_probe` | `ork_i4_dyn_probe` | (internal) |
| `ork_gptq_i4` | `ork_i4_gptq` | (internal) |
| `ork_mm_adopt_imported_i8` | `ork_i8_mm_adopt_imported` | (internal) |
| `ork_mm_build_f16_lut` | `ork_f16_mm_build_lut` | (internal) |
| `ork_mm_build_f16_rsqrt_lut` | `ork_f16_mm_build_rsqrt_lut` | (internal) |
| `ork_mm_build_f16_silu_lut` | `ork_f16_mm_build_silu_lut` |  |
| `ork_mm_f16_scratch` | `ork_f16_mm_scratch` |  |
| `ork_mm_import_i8` | `ork_i8_mm_import` |  |
| `ork_mm_inflate_i8_to_f16` | `ork_i8_mm_inflate_to_f16` |  |
| `ork_mm_layer_i8` | `ork_i8_mm_layer` | (internal) |
| `ork_mm_load_fold_i8` | `ork_i8_mm_load_fold` | (internal) |
| `ork_mm_load_i4` | `ork_i4_mm_load` |  |
| `ork_mm_load_i4_arena` | `ork_i4_mm_load_arena` | (internal) |
| `ork_mm_load_i4_import` | `ork_i4_mm_load_import` |  |
| `ork_mm_load_i4a8` | `ork_i4a8_mm_load` |  |
| `ork_mm_load_i4a8_import` | `ork_i4a8_mm_load_import` |  |
| `ork_mm_load_i8` | `ork_i8_mm_load` |  |
| `ork_mm_load_i8_flags` | `ork_i8_mm_load_flags` | (internal) |
| `ork_mm_load_i8_import` | `ork_i8_mm_load_import` |  |
| `ork_mm_pack` | `ork_f16_mm_pack` |  |
| `ork_mm_pack_f16_fused_act` | `ork_f16_mm_pack_fused_act` | (internal) |
| `ork_mm_pack_i4` | `ork_i4_mm_pack` |  |
| `ork_mm_pack_i4_grouped` | `ork_i4_mm_pack_grouped` |  |
| `ork_mm_pack_i4_to_i8` | `ork_i4_mm_pack_to_i8` | (internal) |
| `ork_mm_pack_i4a8` | `ork_i4a8_mm_pack` |  |
| `ork_mm_pack_i4a8_im` | `ork_i4a8_mm_pack_im` |  |
| `ork_mm_pack_i8` | `ork_i8_mm_pack` |  |
| `ork_mm_pack_i8_dequant` | `ork_i8_mm_pack_dequant` |  |
| `ork_mm_pack_i8_f32` | `ork_i8_mm_pack_f32` | (internal) |
| `ork_mm_pack_i8_import` | `ork_i8_mm_pack_import` |  |
| `ork_mm_repack_f16` | `ork_f16_mm_repack` |  |
| `ork_mm_repack_i8` | `ork_i8_mm_repack` |  |
| `ork_mm_repack_i8_dequant` | `ork_i8_mm_repack_dequant` | (internal) |
| `ork_mm_repack_i8_f32` | `ork_i8_mm_repack_f32` | (internal) |
| `ork_mm_run` | `ork_f16_mm_run` |  |
| `ork_mm_run_chain_i4` | `ork_i4_mm_run_chain` |  |
| `ork_mm_run_chain_i4_async` | `ork_i4_mm_run_chain_async` | (internal) |
| `ork_mm_run_chain_i8` | `ork_i8_mm_run_chain` |  |
| `ork_mm_run_chain_i8_async` | `ork_i8_mm_run_chain_async` | (internal) |
| `ork_mm_run_chain_i8_ffn` | `ork_i8_mm_run_chain_ffn` | (internal) |
| `ork_mm_run_chain_i8_ffn_exp` | `ork_i8_mm_run_chain_ffn_exp` | (internal) |
| `ork_mm_run_chain_i8_ffn_exp_biased` | `ork_i8_mm_run_chain_ffn_exp_biased` | (internal) |
| `ork_mm_run_chain_i8_gsilu` | `ork_i8_mm_run_chain_gsilu` | (internal) |
| `ork_mm_run_chain_i8_sdpsilu` | `ork_i8_mm_run_chain_sdpsilu` | (internal) |
| `ork_mm_run_f16_act` | `ork_f16_mm_run_act` | (internal) |
| `ork_mm_run_f16_f16out` | `ork_f16_mm_run_f16out` | (internal) |
| `ork_mm_run_f16_fused_act` | `ork_f16_mm_run_fused_act` | (internal) |
| `ork_mm_run_f16_silu` | `ork_f16_mm_run_silu` |  |
| `ork_mm_run_i4` | `ork_i4_mm_run` |  |
| `ork_mm_run_i4_async` | `ork_i4_mm_run_async` | (internal) |
| `ork_mm_run_i4_experts` | `ork_i4_mm_run_experts` | (internal) |
| `ork_mm_run_i4_grouped` | `ork_i4_mm_run_grouped` |  |
| `ork_mm_run_i8` | `ork_i8_mm_run` |  |
| `ork_mm_run_i8_async` | `ork_i8_mm_run_async` | (internal) |
| `ork_mm_run_i8_ewmul` | `ork_i8_mm_run_ewmul` | (internal) |
| `ork_mm_run_i8_out16` | `ork_i8_mm_run_out16` |  |
| `ork_mm_run_i8_out8` | `ork_i8_mm_run_out8` |  |
| `ork_mm_run_i8_silu` | `ork_i8_mm_run_silu` |  |
| `ork_mm_run_i8_silu32` | `ork_i8_mm_run_silu32` | (internal) |
| `ork_mm_run_stream_f16` | `ork_f16_mm_run_stream` |  |
| `ork_mm_run_stream_f16_chain` | `ork_f16_mm_run_stream_chain` |  |
| `ork_mm_run_stream_i4` | `ork_i4_mm_run_stream` | (internal) |
| `ork_mm_run_stream_i4_async` | `ork_i4_mm_run_stream_async` | (internal) |
| `ork_mm_run_stream_i8` | `ork_i8_mm_run_stream` |  |
| `ork_mm_run_stream_i8_async` | `ork_i8_mm_run_stream_async` | (internal) |
| `ork_mm_run_stream_i8_sk` | `ork_i8_mm_run_stream_sk` | (internal) |
| `ork_npu_add_f16` | `ork_f16_npu_add` |  |
| `ork_npu_add_i16` | `ork_i16_npu_add` | (internal) |
| `ork_npu_add_i8` | `ork_i8_npu_add` | (internal) |
| `ork_npu_chain_gatesilu_i16` | `ork_i16_npu_chain_gatesilu` | (internal) |
| `ork_npu_chain_mm_perchan_f16` | `ork_f16_npu_chain_mm_perchan` | (internal) |
| `ork_npu_chain_mm_perchan_i16` | `ork_i16_npu_chain_mm_perchan` | (internal) |
| `ork_npu_chain_mm_silu_i16` | `ork_i16_npu_chain_mm_silu` | (internal) |
| `ork_npu_ewmul_f16` | `ork_f16_npu_ewmul` |  |
| `ork_npu_ewmul_i16` | `ork_i16_npu_ewmul` | (internal) |
| `ork_npu_ewmul_i8` | `ork_i8_npu_ewmul` |  |
| `ork_npu_exp_i16` | `ork_i16_npu_exp` |  |
| `ork_npu_exp_i8` | `ork_i8_npu_exp` | (internal) |
| `ork_npu_exp_i8_biased` | `ork_i8_npu_exp_biased` | (internal) |
| `ork_npu_f16_gap_probe` | `ork_f16_npu_gap_probe` | (internal) |
| `ork_npu_f16_percore_probe` | `ork_f16_npu_percore_probe` | (internal) |
| `ork_npu_fold_op_i8` | `ork_i8_npu_fold_op` | (internal) |
| `ork_npu_fold_run_i8` | `ork_i8_npu_fold_run` | (internal) |
| `ork_npu_gelu_i16` | `ork_i16_npu_gelu` | (internal) |
| `ork_npu_gelu_i8` | `ork_i8_npu_gelu` |  |
| `ork_npu_l2norm_f16` | `ork_f16_npu_l2norm` | (internal) |
| `ork_npu_mm_perchan_f16` | `ork_f16_npu_mm_perchan` | (internal) |
| `ork_npu_mm_perchan_f16_diag` | `ork_f16_npu_mm_perchan_diag` | (internal) |
| `ork_npu_mm_perchan_f16_fused` | `ork_f16_npu_mm_perchan_fused` | (internal) |
| `ork_npu_mul_perchan_f16` | `ork_f16_npu_mul_perchan` |  |
| `ork_npu_mul_perchan_f16_contig` | `ork_f16_npu_mul_perchan_contig` | (internal) |
| `ork_npu_mul_perchan_i16` | `ork_i16_npu_mul_perchan` | (internal) |
| `ork_npu_mul_perchan_i8` | `ork_i8_npu_mul_perchan` | (internal) |
| `ork_npu_probe_add_i8` | `ork_i8_npu_probe_add` | (internal) |
| `ork_npu_probe_f16_mm` | `ork_f16_npu_probe_mm` | (internal) |
| `ork_npu_probe_f16_mm_f16out` | `ork_f16_npu_probe_mm_f16out` | (internal) |
| `ork_npu_probe_f16_stridedA` | `ork_f16_npu_probe_stridedA` | (internal) |
| `ork_npu_probe_i16_out` | `ork_i16_npu_probe_out` | (internal) |
| `ork_npu_probe_i4` | `ork_i4_npu_probe` | (internal) |
| `ork_npu_probe_i4_mm` | `ork_i4_npu_probe_mm` | (internal) |
| `ork_npu_probe_i8_ewmul` | `ork_i8_npu_probe_ewmul` | (internal) |
| `ork_npu_probe_i8_ewmul_lin` | `ork_i8_npu_probe_ewmul_lin` | (internal) |
| `ork_npu_probe_i8_ewmul_tmpl` | `ork_i8_npu_probe_ewmul_tmpl` | (internal) |
| `ork_npu_probe_i8_mm` | `ork_i8_npu_probe_mm` | (internal) |
| `ork_npu_probe_i8_mul` | `ork_i8_npu_probe_mul` | (internal) |
| `ork_npu_probe_i8_out8` | `ork_i8_npu_probe_out8` | (internal) |
| `ork_npu_probe_i8_silu` | `ork_i8_npu_probe_silu` | (internal) |
| `ork_npu_probe_i8_silu_cfg` | `ork_i8_npu_probe_silu_cfg` | (internal) |
| `ork_npu_probe_mtile_i8` | `ork_i8_npu_probe_mtile` | (internal) |
| `ork_npu_probe_silu_std` | `ork_i8_npu_probe_silu_std` | (internal) |
| `ork_npu_probe_silu_std_f16` | `ork_f16_npu_probe_silu_std` | (internal) |
| `ork_npu_probe_silu_std_i16` | `ork_i16_npu_probe_silu_std` | (internal) |
| `ork_npu_probe_single_i8` | `ork_i8_npu_probe_single` | (internal) |
| `ork_npu_probe_slice_f16` | `ork_f16_npu_probe_slice` | (internal) |
| `ork_npu_replay_full_f16` | `ork_f16_npu_replay_full` | (internal) |
| `ork_npu_replay_i8` | `ork_i8_npu_replay` | (internal) |
| `ork_npu_replay_i8_sweep` | `ork_i8_npu_replay_sweep` | (internal) |
| `ork_npu_replay_lut_i16` | `ork_i16_npu_replay_lut` | (internal) |
| `ork_npu_replay_reshape_f16` | `ork_f16_npu_replay_reshape` | (internal) |
| `ork_npu_replay_softmax_f16` | `ork_f16_npu_replay_softmax` | (internal) |
| `ork_npu_rmsnorm_f16` | `ork_f16_npu_rmsnorm` |  |
| `ork_npu_rope_neox_f16` | `ork_f16_npu_rope_neox` |  |
| `ork_npu_row_max_i8` | `ork_i8_npu_row_max` |  |
| `ork_npu_rsqrt_i16` | `ork_i16_npu_rsqrt` | (internal) |
| `ork_npu_rsqrt_i8` | `ork_i8_npu_rsqrt` | (internal) |
| `ork_npu_silu_i16` | `ork_i16_npu_silu` |  |
| `ork_npu_silu_i8` | `ork_i8_npu_silu` |  |
| `ork_npu_softmax_f16` | `ork_f16_npu_softmax` |  |
| `ork_npu_synth_i8_dump` | `ork_i8_npu_synth_dump` | (internal) |
| `ork_pack_i4a8_cpu_blob` | `ork_i4a8_pack_cpu_blob` |  |
| `ork_slice_direct_i4a8_kind` | `ork_i4a8_slice_direct_kind` | (internal) |
| `ork_slice_direct_inflate_i8` | `ork_i8_slice_direct_inflate` | (internal) |
| `ork_slice_inflate_i4a8` | `ork_i4a8_slice_inflate` | (internal) |
| `ork_slice_inflate_i4a8_kind` | `ork_i4a8_slice_inflate_kind` | (internal) |
| `ork_slice_tile_i8` | `ork_i8_slice_tile` | (internal) |
| `ork_ssd_probe_fusedmm_f16` | `ork_f16_ssd_probe_fusedmm` | (internal) |
| `ork_ssd_probe_rawmm_f16` | `ork_f16_ssd_probe_rawmm` | (internal) |
| `ork_stage_fill_i8` | `ork_i8_stage_fill` | (internal) |
| `ork_stream_pool_add_i4a8` | `ork_i4a8_stream_pool_add` | (internal) |
| `ork_stream_pool_add_i8` | `ork_i8_stream_pool_add` |  |
| `ork_stream_pool_add_i8_raw` | `ork_i8_stream_pool_add_raw` | (internal) |
| `ork_w_attach_fold_i8` | `ork_i8_w_attach_fold` | (internal) |
| `ork_w_dump_bf_i8_cpu` | `ork_i8_w_dump_bf_cpu` |  |
| `ork_w_dump_fold_i8_cpu` | `ork_i8_w_dump_fold_cpu` | (internal) |
| `ork_w_dump_i4a8` | `ork_i4a8_w_dump` |  |
| `ork_w_dump_i8_cpu` | `ork_i8_w_dump_cpu` |  |
| `ork_w_dump_i8_cpu_st` | `ork_i8_w_dump_cpu_st` |  |
| `orki_act_lut_i16` | `orki_i16_act_lut` | (internal) |
| `orki_act_lut_i8` | `orki_i8_act_lut` | (internal) |
| `orki_bmm_gather_f16` | `orki_f16_bmm_gather` | (internal) |
| `orki_bmm_gather_i8` | `orki_i8_bmm_gather` | (internal) |
| `orki_chain_fullk_mcap_i8` | `orki_i8_chain_fullk_mcap` | (internal) |
| `orki_expand_chan_i4_f32` | `orki_i4_expand_chan_f32` | (internal) |
| `orki_expand_chan_i4_i8` | `orki_i4_expand_chan_to_i8` | (internal) |
| `orki_inflate_chan_nf4_f32` | `orki_nf4_inflate_chan_f32` | (internal) |
| `orki_inflate_chan_nf4_i8` | `orki_nf4_inflate_chan_to_i8` | (internal) |
| `orki_quant_chan_i4` | `orki_i4_quant_chan` | (internal) |
| `orki_quant_chan_nf4` | `orki_nf4_quant_chan` | (internal) |
| `orki_run_chain_i8_impl` | `orki_i8_run_chain_impl` | (internal) |
| `orki_run_i4_bchain_db` | `orki_i4_run_bchain_db` | (internal) |
| `orki_run_i4_experts_bchain_db` | `orki_i4_run_experts_bchain_db` | (internal) |
| `orki_run_i4_mc_db` | `orki_i4_run_mc_db` | (internal) |
| `orki_set_f16_out` | `orki_f16_set_out` | (internal) |
| `orki_set_f16_out_fp16in` | `orki_f16_set_out_fp16in` | (internal) |
| `orki_set_i16_out` | `orki_i16_set_out` | (internal) |
| `orki_set_i8_ewmul` | `orki_i8_set_ewmul` | (internal) |
| `orki_set_i8_out8` | `orki_i8_set_out8` | (internal) |
| `orki_set_i8_silu` | `orki_i8_set_silu` | (internal) |
| `orki_set_i8_silu32` | `orki_i8_set_silu32` | (internal) |
| `orki_slice_pack_i4` | `orki_i4_slice_pack` | (internal) |
| `orki_slice_pack_i8` | `orki_i8_slice_pack` | (internal) |
| `orki_slice_run_i4` | `orki_i4_slice_run` | (internal) |
| `orki_slice_run_i8` | `orki_i8_slice_run` | (internal) |
| `orki_synth` | `orki_f16_synth` | (internal) |
| `orki_synth_i4` | `orki_i4_synth` | (internal) |
| `orki_synth_i8` | `orki_i8_synth` | (internal) |
| `orki_synth_i8_mfold` | `orki_i8_synth_mfold` | (internal) |
| `orki_tile_direct_i4_i8` | `orki_i4_tile_direct_to_i8` | (internal) |
| `orki_tile_f32_i8` | `orki_i8_tile_f32` | (internal) |
| `orki_tile_i4_A` | `orki_i4_tile_A` | (internal) |
| `orki_tile_i4_Aslice` | `orki_i4_tile_Aslice` | (internal) |
| `orki_tile_i4_B` | `orki_i4_tile_B` | (internal) |
| `orki_tile_i8_range` | `orki_i8_tile_range` | (internal) |
| `orki_tile_i8_to_import_tiles` | `orki_i8_tile_to_import_tiles` | (internal) |
| `orki_tile_i8_to_tiles` | `orki_i8_tile_to_tiles` | (internal) |

## Why

The twelve modularization rounds re-shaped the tree along precision lines (`src/npu/<dtype>/`). The
names now match that structure. A convention already existed — dtype-*last* — and was followed by 150 of
152 symbols, but it was undocumented and unenforced, which is how its exceptions appeared. `check 11` in
`tools/check_registry.sh` now enforces the new rule, with exemptions catalogued in
`tools/naming_exempt.txt`.

Verified as a pure rename: the code-multiset invariant (baseline with the map applied vs the result) is
exactly neutral, and `make matrix` reproduces the capability table unchanged — a rename alters no
capability.
