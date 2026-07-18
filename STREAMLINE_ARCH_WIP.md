# Streamline Architecture — Full Submit Consolidation Strategy

## North star (the desired architecture)
**Every NPU op path converges on ONE submit spine. No path issues its own blocking submit.**

```
   run_i8 / run_multicore / run_chain / run_stream / bmm / i4 / SDP / PPU / scans
                                   |
                                   v
   [ ork_npu_enter ]  mode/reset/warm policy      (EXISTS — 51 sites, the reset-half already consolidated)
                                   |
                                   v
   [ SRAM-resident STATIC TABLE ]  compiled-once regcmd, resident, DRAM-failover
        { deduped real ops · shared spin · warm/establishment · POST/recovery }
                                   |
                                   v
   [ ork_submit_seq -> ork_dyn_begin_mc ]  NONBLOCK doorbell submit + host-bounded poll   (EXISTS — the wrapper)
                                   |
                                   v
   [ ork_npu_recover ]  on a doorbell timeout: dump -> soft-reset -> NONBLOCK probe -> continue / fault
```

The wrapper (`ork_submit_seq` over the `ork_dyn_begin_mc` doorbell) and the mode layer (`ork_npu_enter`)
ALREADY EXIST. This is **not** a new-API project — it is **migrating the ~107 hand-rolled
`rknpu_submit_ioctl` call sites onto the existing spine**, tweaking the wrapper where a path needs coverage
it doesn't have yet. The reset-half was consolidated by `ork_npu_enter`; the **submit-half was never
consolidated** — that is the whole of this work.

### The submit interface (endpoint mental model)
`ork_submit(ordered [ {regcmd-ref, data-refs}, … ])` → one NONBLOCK doorbell submit. A submit is an ORDERED
list of **regcmd references** (resident programs in the SRAM table — a block, or a single op) each paired with
**data references** (A/B/C pointers). The call's whole job: point the task descriptors at the referenced
regcmds in order, attach the data pointers, fire ONE NONBLOCK submit, poll doorbells. No per-submit synth (the
regcmd is resident/compiled-once), no copy (data by reference). `ork_seq_op={w,M,A,C}` already IS this shape
(w=regcmd-ref, A/C=data-refs). Two design choices this forces: (a) data-by-reference = JIT-patch the data
address into the resident regcmd per submit (truest "by reference") vs `ork_pc` fixed-address content-refresh;
(b) ordering = the descriptor array's regcmd_addr list (Model A) vs the baked 0x0010 chain (Model B) — the A/B probe.

### Interface vs mechanism (decoupled — the key layering)
The reference INTERFACE (caller passes `{regcmd-ref, data-ptr}`, moves no bytes) is STABLE and SEPARATE from
the data-movement MECHANISM. Given a reference, the mechanism moves data whichever way THAT case needs —
zero-copy in-place (C, and A where valid), copy-to-scratch (A at M=1 until ZC-A is fixed), JIT-patch the
address into the resident regcmd, per-core scatter, K-split accumulate — all private, per-case, invisible to
the caller. Consequence: **ZC-A is NOT a gate for the interface** — it's an internal bandwidth optimization
behind the stable contract (fix it later → every migrated path drops the internal A-copy transparently, no
interface change, no re-migration). So migrations route onto the reference interface NOW (mechanism copies-A
-for-M=1 under the hood, as `ork_dyn_begin` already does); data-movement (ZC-A, JIT-patch, SRAM residence) is
optimized independently. **G0 — resolve zero-copy A at M=1** is therefore a drop-in bandwidth win, not a blocker.

### Migration philosophy — NO legacy fallback (paradigm shift)
Migrate aggressively onto the new stack; do NOT hedge with a runtime legacy fallback. **Git is the recovery**
(revert), not a runtime path. A fallback would MASK the very misses we need to see. Every path moved onto the
spine that misses auto-fires the stuck-descriptor core-dump (`ork_dyn_end` → `ork_dyn_dump`) — the more we put
on the stack, the faster the doorbell dispatch fix surfaces. Failures are meant to be visible + diagnosed, not
silently caught.

### Why NONBLOCK everywhere (the load-bearing reason)
A **blocking** submit on a wedged NPU enters the kernel's `continue-wait` and re-waits PAST its own timeout in
an **uninterruptible D-state** (measured 61s→122s, unkillable, `timeout` can't kill it). So blocking submits
cannot be bounded from userspace. **NONBLOCK is the only hang-immune path** — the ioctl returns immediately,
the host polls the doorbell with its own timeout, and a wedge becomes a clean bounded fault → `recover()`,
never an uninterruptible hang. "Blocking is correct" was wrong; blocking is legacy (predates the doorbell).

## Three wins that stack (the point of the spine)
- **BUILD-half** — compile the regcmd ONCE (recipe, `ork_pc_*`): kills per-token `synth`+`validate` (~44% of a submit).
- **DISPATCH-half** — NONBLOCK doorbell (`ork_dyn_begin_mc`/`ork_submit_seq`): no kernel wait, multi-core, doorbell rendezvous.
- **RESIDENCE** — the static table lives in on-chip SRAM (otherwise-unused; DRAM failover): frees DRAM/IOVA.
Plus **persistence** (big `task_number` reserve → one submit spans many steps), **self-heal** (`recover`), and
**hang-immunity** (NONBLOCK).

## The SRAM-resident static table (compiled once, zero per-token/per-recovery synth)
One pool holds EVERY fixed regcmd the runtime needs:
| entry class | count | notes |
|---|---|---|
| deduped real-op programs | ~100–250 | one per resident weight; the actual work |
| shared spin no-op | 1 | self-loop; the persistent-chain runway |
| warm / establishment ops | few (per dtype) | cold mode-establishment, pre-compiled not re-synth'd |
| POST / recovery probe ops | few | resident so `recover()` is submit-only, no synth |
Footprint at the 13k budget: descriptor array (13107×40B ≈ 512KB) + table (≤~220KB) + spin/warm/POST (<10KB)
≈ 610–740KB, fits the 956KB SRAM with DRAM failover. **The descriptor array dominates, not the programs.**
Key decoupling: `task_number` (≤13107 descriptors, the spin/step budget) is INDEPENDENT of distinct programs
(regcmd bytes). Per-op tracking arrays index only the ~100–250 real ops, NOT the 13k steps (no heap-grow needed).

## What EXISTS now (committed this session on streamline-arch)
- `ba77747` SRAM-resident precompiled table (`ork_pc_*` auto-SRAM) + `ork_dma_alloc_sram` — validated bit-exact, latency-neutral in isolation.
- `a878169` `ork_dyn_queue_idle` — linger idle-exit halt (null-terminate on drain).
- `1d679d1` persistent spin tail (un-gated, reserve>P; dedicated no-op; safe teardown) + self-heal/dump:
  `ork_npu_dump_state`, `ork_dyn_dump` (chain-aware stuck-descriptor + [prev,STUCK,next] linkage window),
  `ork_npu_soft_reset`, `ork_npu_recover` (NONBLOCK, hang-proof), `ork_npu_force_fault`, `ork_npu_dma_rw`.
- Pre-existing spine pieces: `ork_submit_seq` (op-sequence router→doorbell), `ork_dyn_begin_mc` (NONBLOCK
  multi-core doorbell), `ork_pc_*` (compile-once recipe), `ork_npu_chain_progs` (PC-chain assembler),
  `ork_npu_enter` (mode/reset/warm, 51 sites).

## Phased plan (ADDITIVE — each path migrated bit-exact, make test green each step)
- **P0 — audit.** Classify the ~107 `rknpu_submit_ioctl` sites: CORE run paths (migration targets) vs RE/probe
  scaffolding (leave). Same for the residual hand-rolled `act(RESET)` (mostly probes, not core).
- **P1 — generalize the wrapper.** Extend `ork_submit_seq`/`ork_dyn_begin_mc` to cover what core paths need but
  it lacks: multi-core partition (have), K-split/N-tile tiling, fp16 M>1, int4, and the SDP/PPU op kinds. This
  is "tweak the existing wrapper," not a new one.
- **P2 — recipe = compile-once into the SRAM table.** Fold `ork_pc`'s compile-once into the wrapper so a fixed
  chain (decode, scan, FFN) compiles once into the SRAM pool and re-runs with only an A-refresh. Add the spin /
  warm / POST entries to the same pool.
- **P3 STARTED — run_multicore M=1 int8 DECODE migrated onto the spine.** The blocker was a
  parallelization-model mismatch (run_multicore splits one matmul's N-columns SUB-nmax across cores; the
  doorbell partitioned whole ops). Resolved by adding `ork_dyn_begin_colsplit_m1` (sub-nmax N-column tiling
  across cores on the NONBLOCK doorbell, matching `t0=i*NN/nc` bit-exact; scratch+copy-back = coherency-safe
  under multi-core). `run_multicore` now routes M=1 int8 decode (Sn==1, K<=4096 Bf, nc>1) through it — no
  legacy fallback. Validated: make test golden bit-exact (model/decode), decode latency neutral-to-faster
  (873 vs 884ms/500-iter, NONBLOCK dispatch). NOTE: production decode is CPU, so this affects the NPU-decode
  path (used when NPU decode is forced), not the default fast decode.
  - **M>1 prefill colsplit: WIRED — poll fix unblocked it, now at parity.** The M=256 2.8x regression was the
    `ork_dyn_end` busy-poll's per-row `civac` (768 words/iter for M=256x3 cores) contending with the large NPU
    writeback. Fix: `ork_dyn_end` now (a) caches per-entry done (stops re-civac'ing a finished core) and
    (b) after a 1ms tight window that fully covers dispatch-bound decode, sleeps 50us between polls — so a
    multi-ms prefill stops hammering the bus (adds only <=poll-granularity latency) while decode stays tight.
    Result (mc_prof K=2048 N=4096): M=1 470 vs 505 ms (FASTER), M=64 654 vs 648 (neutral), M=256 1820 vs 1802
    (PARITY, was 5147). So `run_multicore` now routes ALL int8 (Sn==1, K<=4096 Bf, nc>1) — decode AND prefill —
    onto the doorbell colsplit. make test golden bit-exact. Consolidation at no perf cost (decode faster).
    NEXT P3: end-to-end integration bench (ork_bench) to confirm; extend colsplit to wide-N / wide-K (combine
    with N-tile/K-split) so the big FFN projections route too; then run_stream / bmm.
  - **Wide-N colsplit (Sn>1): DONE for M=1 (decode) — bit-exact.** ork_dyn_begin_colsplit now handles Sn>1: a
    core's contiguous column range [c0,c1) can span N-slices, so it emits one program per overlapping slice
    (its column sub-range, weight Bf[ns] at the in-slice offset), writing into the core's [M,Ncore] scratch;
    since the range is contiguous in C the copy-back stays a single strided block. Validated bit-exact
    (cacheable output): K=3584 N=18944 (ffn_gate/up-like, Sn=3), N=16384/24576, nc=3. M>1 wide-N stays on
    mcworker for now (the per-row-last-col done across multi-slice strided writes is unverified). run_multicore
    routes wide-N M=1 decode onto it. make test golden bit-exact.
  - **ZC-OUT reconfirmed:** dma-OUTPUT (ork_dma_alloc as C) under MULTI-CORE colsplit is non-deterministically
    wrong (verified: 112/256/240 zeros across identical runs) — the documented ZC-OUT-multicore-unsafe path
    (per-core bsyncs not serialized vs the writeback). Default cacheable output is bit-exact; dma-output stays
    the off-by-default opt-in. Applies equally to the old mcworker path, so no new risk.
  - **Wide-K colsplit (K>4096, ffn_down): DONE for M=1 (decode) — bit-exact.** Each core does a per-core
    K-split over its column range: Sk partial [1,Ncore] programs (Bb[ks] at the column offset), chained;
    end() SUMS the Sk partials into C's columns [c0,c1) at row-stride N (oSk + ostride combined). Validated
    bit-exact: K=8192 (Sk=8) & K=18944 (Sk~19, ffn_down-like), N split across 3 cores. run_multicore routes
    wide-K M=1 decode onto it. make test golden bit-exact.
  - **P3 status: ALL int8 FFN/attention decode (M=1) + Sn==1 prefill (any M) are on the doorbell spine.**
    Routed: base (Sn==1, K<=4096, any M) + wide-N (Sn>1) M=1 + wide-K (K>4096) M=1. NOT yet: M>1 wide-N /
    wide-K prefill (multi-slice/K-split done + strided writes unverified at M>1), and Sn>1 && K>4096 combined.
    NEXT: end-to-end ork_bench (decode tok/s on the fork); then M>1 wide-N/K prefill; then run_stream / bmm.
- **Graceful SIGTERM/SIGINT in the doorbell (631744d): DONE + validated.** The async poll (ork_dyn_end) spun
  uninterruptibly — a kill -TERM during an NPU submit-wait was IGNORED, orphaning the process (forced a reboot;
  kill -9 mid-submit risks an IOMMU wedge). Now a chained handler sets a flag; the poll breaks + DRAINS
  (bsync + writeback + free) before the signal takes effect; never swallows it (idle => original disposition
  immediately; in-poll => re-raise after drain). Validated on the board: kill -TERM of a doorbell workload
  AND of the real ork_bench both exit cleanly in ~0.6-1.6s (was: uninterruptible), no orphan, NPU healthy.
  make test byte-identical.
- **End-to-end ork_bench: attempted, ENVIRONMENTALLY BLOCKED (not a code issue).** Rebuilt libggml-ork against
  streamline-arch (confirmed `[ork] ork-driver 0.7.6` loads, not the old 0.6.46). But the run produced garbage
  (`response: !!!`) because the NPU weight import ENOMEM'd (`PRIME_FD_TO_HANDLE: Cannot allocate memory` x60):
  the orkllm systemd service auto-starts on boot and holds the SINGLE-STREAM NPU domain, so ork_bench can't
  import. The auto-dump FIRED on the resulting doorbell miss (`S=3 progress=-1 STUCK`) — downstream of the
  missing weights, and proof the diagnostic infra works. To get a valid number: stop orkllm (systemctl was
  flaky over the non-login ssh; the service also auto-restarts) so the NPU is free, then re-run. Deferred —
  the code is validated at make-test-golden + mc_prof (parity/faster) + SIGTERM-functional levels.
  BOARD-OPS NOTE: a graceful `sudo reboot` did NOT bring sshd back (pinged, ssh refused ~12 min) => recovered
  via HA "Rock 5B Plug" power-cycle (came back clean, no SPI corruption since it still pinged). Fork submodule
  WIP was stashed for the rebuild and RESTORED after (checkout 22b7480 + stash pop + rebuild).
- **P3 — migrate core run paths onto the spine, one at a time, byte-identical:** `run_chain_i8` →
  `run_stream_f16_chain` → `run_multicore` → `bmm` → `run_i4`. Each: build `ork_seq_op[]`/recipe, submit via the
  spine, drop the hand-rolled `rknpu_submit_ioctl`. `make test` bit-exact + attest refresh per path.
- **P4 — persistence + self-heal on the spine:** the 13k `task_number` reserve (shared self-loop spin tail);
  `recover()` multi-core (loop all cores — current probe is core-0-only); `linger` idle-exit.
- **P5 — pipeline / overlap** (from the original plan): overlap CPU prep of chain N+1 with NPU drain of N.

## Open questions / gating (from this session)
- **A/B selection (gates dedup of real ops):** does the HW sequence by `task[i].regcmd_addr` (A → the deduped
  table is a cheap descriptor-playlist) or the baked `0x0010` chain (B → pointer composition)? The spin tail
  sidesteps it (self-loop works both ways), but regime-B dedup needs the answer. Small decisive board probe.
- **`task_number=13107` acceptance:** untested at scale (largest submitted ~64). Go/no-go board probe.
- **Cold-dispatch miss:** intermittent establishment miss on the cold first op (warm path stable; dump confirms
  descriptors are well-formed → a dispatch failure, not a regcmd bug). `recover()` rescuing a *live* miss unproven.
- **Dump depth:** the queryable signals are limited — DMA byte-counters unpopulated, `hw_elapse`=0 for NONBLOCK,
  `int_status` not populated on this path. The **doorbell is the only live NONBLOCK signal** (which `ork_dyn_dump`
  uses). NPU internal HW registers need a kernel debugfs/ioctl patch (optional, custom-kernel feasible).

## Validation + board-safety
- Every migration bit-exact / rel-L2 ≤ ~1e-3 vs the path it replaces; `make test` green + `sbc_attest.txt`
  refreshed before each lands.
- NONBLOCK + host-bounded poll everywhere → no uninterruptible hang; `recover()` on timeout.
- Board-safety: timeout every NPU cmd; SIGTERM not -9; force-recompile after rsync of a CORE .c; wedge → reboot
  / plug-cycle. **Do NOT run deliberate-fault experiments before `make test`** — they degrade the board and can
  wedge an unrelated test (observed this session: `test_bmm` D-state hung a degraded board; clean on reboot).

## State
- Branch `streamline-arch` (pushed to origin @ `a89d45c`). Board clone `~/ork-driver` on the SBC.
- Committed: SRAM table (`ba77747`), linger (`a878169`), spin tail + self-heal/dump (`1d679d1`),
  **P1a** (`a89d45c`): single-slice run_chain_i8 routed onto the doorbell spine, no fallback, auto-dump-on-miss.
- **P0 audit DONE**: ~35 submit sites in ~20 core fns are migration targets; ~60 in ~40 probe/RE fns leave; ~8 are the spine itself.
- **G1 N-tiling: DONE** (M=1 direct + M>1 scratch/scatter, bit-exact, committed 0fbf4b1). See below.
- **G2 K-split (K>4096): FIRST INCREMENT DONE — wide-K DECODE (M=1) bit-exact.** `ork_dyn_begin_mc` now
  accepts int8 K>4096 for Sn==1 && M==1: emits Sk per-K-slice partial programs (A_ks = contiguous offset
  `A+k0` since M=1, no gather; Bb[ks] K-slice weight), chained; `end()` host-SUMS the Sk `[1,N]` partials
  into C (marked by `oSk[gi]`); `done_i` full-surface polls the partial scratch. Validated bit-exact:
  K=8192 (Sk=8) and K=18944 (Sk~19, ffn_down-like), cacheable + dma output. All predicates key on `K>4096`
  (NOT `Sk>1` — K<=4096 has Sk>1 too but uses the full-K Bf path; keying on Sk>1 broke byte-identical).
  make test ALL PASS byte-identical.
  - **G2 M>1 K-split (wide-K PREFILL): DONE — bit-exact.** M 1..64 now handled. Findings that simplified it:
    (a) NO M-tile chunking needed — for KS=1024 (default) every K-slice has Kp in {1024,512} (since K%512==0),
    whose mg_max*64 cap is 320/704 >= 64 >= M, so the whole M-tile fits ONE program per K-slice (a defensive
    per-slice cap check rejects a pathological ORK_KTILE); (b) added the per-K-slice A-GATHER (contiguous
    [M,Kp] tile; for M==1 it degenerates to a plain copy); (c) NO CC-budget batching needed at M<=64 — the
    partials (<=~17MB for ffn_down M=64 N=3584) fit mcc after a grow, and maf is grown to hold the M*K gather
    (all Sk tiles resident in one submit). Validated bit-exact: K=8192/18944, M=1/8/64. make test byte-ident.
  - **G2 NEXT: combined Sn>1 && K>4096** (wide-N AND wide-K in one op) — currently guarded to Sn==1. Needs the
    (ns,ks) 2D program grid with per-(ns) accumulation. Rare shape (ffn_down is Sn==1); low priority.
- **Wide-M PREFILL (M>64): DONE — bit-exact.** `ork_dyn_begin_mc` now M-tiles M>64 (plain: int8, Sn==1,
  K<=4096 with full-K Bf) into `mtile_cap(K)`-row chained programs (shared `mtile_cap` helper = mg_max*64;
  A rows are contiguous so no gather; each tile writes disjoint rows of the [M,N] scratch; end() straight-copies
  to a cacheable C). Validated bit-exact: K=4096 M=128 (2 tiles) / M=256 (4 tiles), K=2048 M=256 (2 tiles).
  This completes the int8 matmul shape space on the doorbell (N-tile + K-split + M any) => `run_multicore`'s
  int8 prefill/decode can now migrate onto the spine (P3). dma-OUTPUT at M>1 stays the ZC-OUT caveat
  (copy-back to a non-cacheable ork_dma_alloc dst is flaky — off by default, orthogonal). Combining M>64 with
  N-tiling / K-split (2D/3D grid) is the remaining tiling follow-up; those shapes stay M<=64 for now.
- **Then** G3/G4 fp16-M>1 / int4-M>1, G5 SDP/PPU. Each: extend wrapper → migrate a path → make test
  byte-identical + attest.
  - **KERNEL CONSTRAINT (found starting P1b):** a PC-CHAIN cannot span >1 N-slice — distinct `Bb[ns]`/`Bf[ns]`
    buffers make the CDMA walker reject the cross-buffer next-pointer (errno 110 "cdma address wild"; the same
    reason chain-K-split is `Sn==1`-only, per `mcworker`). So G1 is NOT "chain Sn programs in one submit".
    Viable designs: (a) distribute N-slices across cores (`Sn<=nc`, each core a single-slice chain);
    (b) **per-N-slice separate NONBLOCK submits** (each references ONE buffer = kernel-safe; the doorbell fires
    `Sn` submits and polls all) — most general, recommended; (c) re-lay-out the weight contiguously (bigger).
    NEXT concrete step: implement (b) — teach `ork_dyn_begin_mc` to accept `Sn>1` by emitting one single-slice
    submit per N-slice (per core), each writing its C column-slice; the per-op doorbell = the last slice's last col.
  - **STRIDED OUTPUT IS PART OF THE OP INTERFACE (design refinement).** An N-sliced op writes disjoint COLUMN
    slices of C (row-stride = full N), NOT a contiguous block. So the op reference/data-structure must carry the
    output-layout descriptor — `{slice-weight-ref Bf[ns], C-column-offset n0=ns*NMAX, row-stride N}` — and the
    mechanism honors it rather than re-deriving. Implementation lever: `synth_i8(...,stride)` already writes a
    strided output (`s = stride>0 ? stride : N`), so NO host-scatter is needed (unlike `mcworker`, which scatters
    from contiguous scratch). Each N-slice program = one strided-output sub-op. The doorbell stays per-OP: the
    last slice (ns=Sn-1) covers column N-1, so it writes the per-row completion sentinel last — unchanged.
  - **CHAIN-vs-SEPARATE-SUBMIT is resolved empirically (per no-fallback philosophy):** first attempt chains the
    Sn slice-programs in one submit and lets the board report (bit-exact => simple G1; errno-110 "cdma address
    wild" => auto-dump fires => pivot to per-slice separate submits). Do NOT assume; migrate and observe.
    RESULT (board-validated): chaining N-slice weight buffers WORKS (no errno-110, no doorbell miss) — the
    kernel-CDMA fear was UNFOUNDED for the plain (non-K-split) N-slice chain. Two output findings:
      * **M=1 wide-N: LANDED, bit-exact** (ork_dyn_ntile_test: Sn=2 & Sn=3, single- & multi-core). Fix was
        coherency: strided column-slice writes leave the row interior uncovered by int8's last-col seed, so
        dirty CPU lines race the NPU writes -> non-deterministic zeros. `seed_all` now folds Sn>1 into the
        fp16 full-surface clean branch. This covers the wide-N DECODE case (lm_head).
      * **M>1 wide-N: capability boundary (rejected), scatter needed.** The strided M-ROW output stride
        follows the compute width Nc (not full N), so for M>1 the rows collide (~12% correct, mostly
        wrong-nonzero). This is exactly why `mcworker` writes contiguous [M,Nc] scratch + host-scatters
        instead of a strided M-row write. Guarded off (`Sn>1 && M>1 => return NULL`) until scatter lands.
    RESOLVED & UNGATED (2026-07-17): M>1 wide-N is bit-exact on the doorbell. The scatter was CORRECT all
    along — the zeros were NOT a scatter/multi-slice bug. Isolation study (ork_dyn_ntile_test cmode: dma vs
    malloc C) proved it: the DIRECT (zero-copy) output path is coherency-unreliable for M>1 (M=8 direct drops
    thousands of words to 0, non-deterministic, at Sn==1 too — the ZC-OUT class); the SCRATCH path (NPU ->
    cacheable mcc -> bsync FROM_DEVICE -> CPU copy/scatter to the caller's C) is the reliable completion
    barrier and is bit-exact. Fix: route ALL M>1 through scratch (never direct output for M>1); the earlier
    test only failed because its C was always an ork_dma_alloc (the flaky direct/ZC-OUT destination). Now:
    M=1 wide-N uses direct zero-copy (bit-exact); M>1 (single-slice AND wide-N scatter) uses scratch copy-back
    to the caller's cacheable C (bit-exact, single- & multi-core). Gate removed. Output zero-copy to a
    resident dma buffer at M>1 remains the separate ZC-OUT opt-in (off by default, coherency-unsafe) — NOT a
    doorbell/N-tiling issue. make test ALL PASS byte-identical.

    [historical] The mcworker-style scatter is wired:
      * build: Sn>1 && M>1 op writes each slice as a CONTIGUOUS [M,Nc] block (stride=0, offset M*n0) to
        in-domain scratch (forced non-direct); ork_dyn_end scatters each block to C columns [n0,n0+Nc) at
        row-stride N; done_i polls the block/full surface. Placement is CORRECT — wrong-nonzero = 0 across
        all cases (the M-row collision is fixed).
      * REMAINING BUG: non-deterministic ZEROS (~0.5-15% of outputs, varying run-to-run for identical
        configs) that SURVIVE a full-surface done-poll (every scratch word polled non-sentinel before
        scatter). got=0 (not the sentinel), wrong-nonzero=0. Only on the chained MULTI-slice M>1 path.
        Ruled out: placement (wrong=0), done-timing (full-surface poll unchanged it), the seed (covers all
        M*N words). Leading hypotheses: (1) async NPU write-drain to DRAM not complete at doorbell-detect
        for the larger chained block output; (2) mcc scratch cacheability differs from ork_dma_alloc (M=1
        direct C works), so the clean/invalidate coherency dance behaves differently on scratch.
      * NEXT DIAGNOSTICS (cheap, decisive): (a) run the failing shape with a BLOCKING submit (drop the 0x2
        NONBLOCK flag) — bit-exact => confirms async-drain; (b) compare vs ork_dyn_test's M>1 Sn==1 scratch
        path (does single-slice M>1 via ork_dyn_begin_mc pass? isolates block-layout/multi-slice vs M>1);
        (c) check mcc bcreate flags/cacheability vs ork_dma_alloc. UNGATE (remove the Sn>1 && M>1 guard)
        the moment it lands bit-exact. Scatter scaffolding is preserved (dormant behind the guard).

- **CONSOLIDATION REQUIREMENT — preserve legacy chain-format coverage in the examples.** As paths migrate onto
  the doorbell spine, an example MUST keep exercising `run_chain_i8`'s ORIGINAL hand-rolled chain format (the
  40-byte task + in-regcmd 0x0010/0x0014 descriptor image) as a pinned regression/reference — do NOT let the
  migration turn it into dead/untested code. P1a already shifted `run_chain_i8`'s routable (M=1, DMA-C) case
  onto the doorbell, so the legacy format is no longer hit by that case; add/keep a test that builds and
  validates the legacy chain image directly (independent of the production route). This is coverage, NOT a
  production fallback (production has none — git is the recovery).
- Wiki (`Exp-2026-07-16` doorbell page) is deliberately deferred until the architecture reaches desired state,
  then corrected in one pass (currently mis-files the reserve/spin wrap as a "negative result").
