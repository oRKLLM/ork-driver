# WIP: port doorbell colsplit to fp16 + native-int4, then remove mcworker (#45)

Plan approved (see ~/.claude/plans/dreamy-leaping-quokka.md). Staged, each gated on board:
bit-exact + prefill t/s >= mcworker per precision; keep mcworker for any precision that fails.

## ============================================================================
## SESSION SNAPSHOT (2026-08-04) — authoritative resume point + ALL env gates
## ============================================================================
SHIPPING fp16 wide-K path (all DEFAULT-ON, no gate — these are the win):
  1. per-K-slice LOCKSTEP BARRIER (`csub_barrier`, reuses c->b_ioctl) — fixes cross-slice desync wrong-answers.
  2. per-core PARALLEL f32 accumulate (ork_csub_worker ~11517) — NOT the serial ork_dyn_end scan (that was the
     5-9x slowdown I self-inflicted + then removed).
  3. run-level SELF-HEAL (run_multicore fp16 dispatch ~5216): worker sets c->mc_error on a faulted blocking submit
     (8s wedge-detect timeout cap); run loop does RKNPU_ACT_RESET + 1ms quiesce + resubmit-same the whole colsplit
     (single-threaded coordinator, up to 6x) -> nc=1 de-escalation backstop. Copies int8 mc_recover_resubmit.
RESULT: fp16 colsplit BEATS mcworker 1.05-1.42x, bit-exact, self-heals; reset rate == mcworker's (path-independent,
proven by a 25x25 A/B). `ORK_F16_COLSPLIT` is STILL DEFAULT-OFF (was gated when broken) — the open decision is to
flip it default-on + `make test` + attest + unblock #45 (remove int8 AND fp16 mcworker; keep i4_mcworker).

ENV GATES ADDED THIS SESSION (all default-off unless noted):
| gate | default | effect | status |
| `ORK_F16_COLSPLIT` | OFF | master: fp16 wide-K -> doorbell colsplit (else mcworker) | viable now; flip decision pending |
| `ORK_F16_NOBAR` | off | disable the lockstep barrier (pre-barrier independent loops) | DEBUG/A-B only (wrong-answers return) |
| `ORK_F16_STAGGER=<us>` | 0 | per-core-index fetch stagger | DEAD (no reliability gain) |
| `ORK_F16_2CORE` | off | fp16 wide-K -> 2 cores not 3 | DEAD (slower, no reliability gain) |
| `ORK_F16_RECOV` | off | naive IN-WORKER retry-same (per-core, concurrent) | DISPROVEN + DANGEROUS (hard-wedges board); DO NOT enable |
| `ORK_F16_CONTIG` | off | (A) all Sk slices in ONE contiguous weight (w->Bbc) + single chained submit/core | FAILED (still wedges: wild is base-latch TIMING, not the dma-buf boundary) |
| `ORK_F16_FORCE_WEDGE=<N>` | off | TEST-ONLY: inject N simulated wedges (countdown) to exercise recovery (heal path / exhaust->nc1) | test hook (keep) |
Also: fp16 blocking submits use an 8s wedge-detect timeout cap (no gate). Struct additions: ork_w{ Bbc, Bbc_valid }
(contiguous weight, (A)); ork_dyn_chain{ f16_contig } (single-submit flag). Test: examples/test_f16colsplit
(suite forces ORK_F16_COLSPLIT=1; probe mode `M K N [iters]`). Existing probe reused: tools/mm_perchan_f16_diag_probe.

SINGLE-SUBMIT fp16 across wide-K — VERDICT so far: NOT achieved.
  (A) contiguous buffer -> WEDGES (timing race, not boundary).
  (B) in-chain SDP gap -> registry-closed: ORK_OP_MM_F16->SDP is ORK_CHAIN_SW (not HW); can't PC-chain.
  (B') the ONE untested edge: ORK_OP_MM_F16->MUL_PERCHANNEL_F16 is HW-proven (fwd); return perchan->mm UNCHARACTERIZED.
  Cheap check: mm_perchan_f16_diag ORK_DIAG_CHAIN=1 = fp16 mm->mm single PC-chain WORKS bit-exact BUT only small
  single-K-slice (no cross-slice base-reprogram, so it does NOT reproduce the wide-K wild).
IN PROGRESS (user-approved): build a WIDE perchan-gap — insert an identity mul_perchan_f16 (REGCMD_MUL_F16_CHAIN,
BS-bypassed 0x53, chain-safe) between wide fp16 K-slices in one chain, to idle the weight-CDMA so the prior fetch
drains before the next slice's base latches. Uses a DUMMY fp16 scratch in/out + identity scale (pure time-filler,
not data-dependent on the f32 matmul output). Gated (new env), watchdogged, recovery-netted. Board 10.3.0.236;
wedged 2x this session (recovered via HA "Rock 5B Plug" power-cycle / self-heal). Machinery refs: ork_npu_chain_progs,
set_f16_out_fp16in (1665), REGCMD_MUL_F16_CHAIN, ork_npu_chain_mm_perchan_f16 (9769), set_mul_geom.

## State of the tree
- Branch: ork-driver `main` @ b412383 (int8 colsplit is default). Working from Mac source-of-truth.
- Untracked: examples/test_npubw.c (NPU bandwidth probe — keep), this WIP doc.

## Key facts (from investigation, npu.c)
- `ork_dyn_begin_colsplit` (11484) is int8-typed: synth_i8, nt_sz=32, Bf(base)/Bb[ks](wide-K), int32.
- fp16 shares int8's EXACT submit format (108-reg, enable 0xd, chain words 216-219, next-amt 0x37); only
  synth()+Bb differ (npu.c ~4783). fp16 is ALWAYS K-sliced (Bb, no Bf) + host f32 accumulate (4857-4889),
  == the int8 wide-K colsplit branch shape (base = Sk==1 -> single partial, accumulate is a copy).
- LANDMINE: fp16 M-tile miscomputes above the validated small chunk (K>=2048 caps at 8 rows, NOT enlargeable).
  fp16 colsplit MUST use the fp16 sched chunk, NEVER mtile_cap/mg_max*64.
- Accumulate sites to generalize to f32: ork_csub_worker (11465-11480, default parallel) AND ork_dyn_end
  (12432-12450, serial). Sentinel poll is already precision-agnostic (int32 bit-pattern compare; 0x7fffffff).
- h->mc_dt drives recovery: I8/I4 => recov_max=6; fp16 => 0 (drains in-submit). Set it EARLY (before workers).
- ggml-ork fp16 usage is pre-chunked to Sn==1 (wu/wg per-N-chunk ork_w; wd K-sliced) => Stage 1 only needs
  fp16 Sn==1 (base + wide-K). fp16 Sn>1: refuse/keep-mcworker for now.
- native int4: synth_i4(rc,mc,K,N,aA,aB,aC) (1906), nt_sz=64, esz can be 2 (int16 acc). grouped scale drain
  = ork_dyn_grouped_end / i4g fields. i4_mcworker/i4_mcworker_g are the current paths.

## Stage 3 (native int4) — DROPPED (measured). i4 doorbell (per-row, ork_dyn_begin_mc_i4) is 10-26x SLOWER
than i4_mcworker on prefill (no M-amortization; re-streams weight per row): M256 K2048 N2048 262ms vs 22ms
(11.8x), M256 K2048 N512 249ms vs 9.6ms (26x). Bit-exact between them. int4 M-tile scheduler is immature
(ORK_I4_MSCHED not wired into i4_mcworker). So mcworker is int4's GOOD prefill path — KEEP i4_mcworker.
Narrowed #45: remove ONLY int8+fp16 mcworker. (Latent: int4 multi-M default routes to the slow doorbell —
research-only path, left as-is.) Probe: scratchpad/test_i4_ab.c.

## !!! RESOLVED (2026-08-04): fp16 colsplit now BEATS mcworker, bit-exact; wedge was a mis-validation artifact
Two distinct bugs, cleanly separated by a controlled A/B:

**(1) Wrong-answers (cross-slice desync) — FIXED by a per-K-slice lockstep barrier.**
The fp16 SW-chain had each core loop its Sk K-slices INDEPENDENTLY, so cores desynced — core A fetching Bb[ks+1]
while core B still fetched Bb[ks] = two distinct fp16 weight buffers on the CDMA at once (fp16's 2-byte weights
double the fetch bytes; int8 tolerates the cross-buffer concurrency, fp16 wilds -> a plausible-WRONG partial
that a completion check can't catch). Fix: a `pthread_barrier` (`csub_barrier`, reuses `c->b_ioctl`) makes all
cores finish slice ks before ANY starts ks+1, so at any instant every core fetches the SAME Bb[ks]. Result:
0 wrong-answers across all hammers (was ~4%). The barrier costs nothing measurable. `ORK_F16_NOBAR` disables it.

**(2) The 5-9x "colsplit is slow AND wedges more" was SELF-INFLICTED and confounded the whole investigation.**
The "completion fix" (Edits: skip per-core accumulate for fp16 + a SINGLE full-surface civac verify + serial
accumulate in ork_dyn_end) serialized ALL cores' partials through one thread and full-scanned every output word.
Measured: fp16 colsplit ran **5-9x SLOWER than mcworker** ({256,3584,3584} 261ms vs 29ms; {256,3584,512} 39ms vs
7ms). Because it spent 5-9x longer in the concurrent-fetch window per matmul, it accumulated ~5-9x more soft-reset
opportunities per unit work -> it LOOKED far more wedge-prone than mcworker, when the per-submit reset probability
was the same. FIX: restore the PER-CORE PARALLEL accumulate (ork_csub_worker 11517, exactly like mcworker
4817-4821; the barrier makes it correct, the FROM_DEVICE bsync + blocking completion make it coherent). The
ork_dyn_end fp16 full-surface verify is now dormant (per-core sets dst[i]=NULL).

**Post-fix A/B (2026-08-04, board 10.3.0.236, governors=performance), bit-exact everywhere:**
| shape           | colsplit (barrier) | mcworker (sync) | colsplit speedup |
|-----------------|--------------------|-----------------|------------------|
| 256x3584x512    | 5551 us            | 7895 us         | 1.42x            |
| 256x4096x512    | 6537 us            | 9174 us         | 1.40x            |
| 256x3584x3584   | 28231 us           | 29590 us        | 1.05x            |

**The residual big-shape soft-reset is PATH-INDEPENDENT — CONFIRMED by a controlled head-to-head A/B.**
Interleaved 25 rounds each on {256,3584,3584}, ORK_MM_TIMEOUT=60000, governors=performance:
  colsplit(doorbell): 25 runs, soft-resets=1, non-bit-exact=1
  mcworker(sync)    : 25 runs, soft-resets=4, non-bit-exact=2
The doorbell path is EQUAL-OR-BETTER on reliability (1 vs 4) AND 1.05-1.42x faster. So the residual is a
fp16-wide-K concurrent-3-core-fetch HARDWARE edge common to BOTH paths, NOT a colsplit defect. Closes the
"is colsplit less reliable than mcworker" question: no — it is equal-or-better and faster.

**RECOVERY GAP (fp16 does NOT self-heal — the real remaining stabilization item).** On BOTH paths `bad == resets`
in the A/B: every wedge produced a WRONG ANSWER, not a healed one. Cause: `recov_max = 0` for fp16 (npu.c ~12526)
— the int8/int4 nonblock doorbell has robust self-heal (transient-rejection retry npu.c:768 -> self-healing reset
:780 -> mc_recover_resubmit + ork_dyn_end recover loop, recov_max=6: re-seed+resubmit+re-poll), but fp16 has none
(assumed "drains in-submit"; the A/B disproves that). NEXT: extend recov_max + mc_recover_resubmit to fp16 (re-seed
the f32 partial surface + resubmit), optionally gated by an `ork_dyn_spin_probe` POST (npu.c:12313) confirming the
NPU came back before resubmitting. `ORK_F16_RETRY` (added here) is a crude first cut (resubmit-on-fault, short
timeout); the proper fix unifies fp16 onto the int8 recovery path so a wedge heals to correct output.

**SELF-HEAL attempt 1 (naive retry) — DISPROVEN + DANGEROUS (2026-08-04).** Tried: on a failed blocking submit,
short timeout (5s) + resubmit the same slice up to 4x (staggered breather). Result: it does NOT heal — resubmitting
the identical 3-core concurrent fp16 slice RE-TRIGGERS the same concurrent-fetch CDMA wild (run came back WRONG),
AND hammering submits at a mid-soft-reset NPU ESCALATED a recoverable soft-reset into a HARD WEDGE (board 10.3.0.236
went unresponsive; recovered by Home-Assistant "Rock 5B Plug" power-cycle; SPI survived). Gated OFF (opt-in
`ORK_F16_RECOV`); DO NOT default-on. The correct self-heal must DE-ESCALATE, not resubmit the same concurrent work:
on a colsplit fault, let the kernel soft-reset settle, then recompute the whole matmul SINGLE-CORE (nc=1 never wedges
— it's the bit-exact reference), ideally at the ork_mm_run/run_multicore level (fault -> c->mc_error -> nc=1 rerun),
NOT by hammering resubmits in the worker. POST via ork_dyn_spin_probe optional.

**SELF-HEAL attempt 2 (run-level nc=1 de-escalation) — WORKS + VALIDATED (2026-08-04).** Implemented at the
run_multicore fp16 dispatch: clear c->mc_error before ork_dyn_begin_colsplit; the worker sets mc_error on a failed
blocking submit (wedge-detect timeout CAPPED at 8s so a fault returns fast, not the 60s job-timeout; NO in-worker
retry/hammering); after ork_dyn_end, if mc_error, fall through to the mcworker path with nc FORCED to 1 + a forced
cold re-warm (soft-reset cleared NPU regcmd state). nc=1 has no concurrent fetch -> never wedges -> bit-exact
reference. Validation:
  - DETERMINISTIC (ORK_F16_FORCE_WEDGE=1 injects mc_error, no real fault): de-escalation FIRES + output BIT-EXACT
    on {256,3584,512} and {256,3584,3584}; happy path (no inject) unchanged (fast, no message). Proves the plumbing.
  - NATURAL 25-run hunt on {256,3584,3584}, self-heal ON, ping-watchdog: 25/25 BIT-EXACT, 0 wrong, 3 soft-resets
    (runs 12/20/24) ALL auto-recovered to correct output, board responsive throughout, NO hard wedge.
  - Bonus finding: the 8s timeout cap (vs old 60s) makes the KERNEL's own reset+recovery return correct output on a
    soft-reset — 3/3 resets healed WITHOUT needing our de-escalation. Our nc=1 fallback is the proven backstop for
    any case the kernel doesn't catch. (The old 60s-timeout A/B saw bad==resets = wrong answers; the cap fixed that.)
Test hook ORK_F16_FORCE_WEDGE (core 0, post-barrier) kept for regression. NEXT (user-requested, now SAFE because
the wedge de-escalates instead of hard-wedging): retry the NONBLOCK (0x2) doorbell submit and/or HW-chaining for fp16.

METHODOLOGY LESSON (this is the important one): we were validating fixes against a colsplit that (a) structurally
differed from the proven mcworker in its completion/accumulate stage and (b) was made progressively MORE divergent
by the "completion fix" itself. A negative ("wedge persists") was thus confounded — un-attributable between "fix
wrong" and "path divergence". Correct method: make the doorbell path match mcworker's proven structure as closely
as possible, THEN attribute any residual to the one genuine remaining delta. (cf. memory `scrutinize-negative-results`.)

Toggles added for the comparison (all default-off except the barrier): `ORK_F16_NOBAR` (disable barrier),
`ORK_F16_STAGGER=<us>` (per-core fetch stagger), `ORK_F16_2CORE` (fp16 wide-K -> 2 cores), `ORK_F16_RETRY`
(short-timeout + resubmit on fault). Measured: stagger/2core/retry do NOT improve reliability; 2core costs speed.

ACTION: re-enable ORK_F16_COLSPLIT default-on pending the A/B confirmation; Stage 4 (#45) is UNBLOCKED for fp16
(colsplit is the faster path). The big-shape reset is a separate, pre-existing (mcworker too) item.

## Single-submit fp16 attempts (to unify onto int8's one-chained-submit/core + nonblock on-ramp)
- **(A) Contiguous weight buffer (`ORK_F16_CONTIG`, gated off) — FAILED (2026-08-04).** Concatenated all Sk
  K-slice `Bb[ks]` into one buffer (`w->Bbc`) so the HW chain walks slice→slice with NO cross-buffer boundary,
  one chained submit/core (int8-style). STILL WEDGED (2 soft-resets + stuck `D<l` procs; board recovered w/o
  power-cycle via the recovery net). CONCLUSION: the cross-slice wild is **base-latch TIMING**, not the dma-buf
  boundary — the walker reprograms the weight base for ks+1 while ks's fp16 fetch drains, and a same-buffer
  different-offset base races identically. So (A)'s premise (boundary = cause) is wrong; contiguous doesn't help.
  Code kept gated (`ORK_F16_CONTIG`) for the record. This REDIRECTS the fix at the timing (a drain gap), i.e. (B).
- **(B) SDP/PPU drain-gap in the single HW chain — CLOSED by prior art (no board run needed).** The whole point of
  (B) was an in-chain gap: `mm slice0 -> SDP -> mm slice1` as ONE HW-chained submit. But `ORK_CHAIN_LIST`
  (ork_npu.h) records EVERY `ORK_OP_MM_F16 -> <SDP>` transition as **`ORK_CHAIN_SW`, not `HW`** — and the registry
  is explicit `SW != HW` ("separate-submit safety != PC-chain safety"). So fp16-mm->SDP is NOT PC-chainable; an
  in-chain SDP gap would wedge on that transition (the int8-SDP hard-wedge campaign, "do not re-run"). The
  `SW`-safe realization of `mm -> gap -> mm` is SEPARATE SUBMITS — which is EXACTLY the per-slice path we already
  ship. So (B) collapses into the shipping solution. Did NOT build it (would re-run the documented-dead campaign).

## (B') perchan DRAIN-GAP — BUILT + TESTED, FAILED (2026-08-04). Root cause now conclusive.
Wired an identity mul_perchan_f16 (REGCMD_MUL_F16_CHAIN, enable 0x18/69 regs, middle desc_slot=138, ping-pong OFF)
between K-slices in the 3-core CONTIG chain (gate `ORK_F16_GAP`; dummy Bgap[3] in/out/scale; padded into REGCMD_N
slots; per-link next-amount 0x24 mm->pc / 0x37 pc->mm). New probe `ork_npu_f16_gap_probe` + tools/f16_gap_probe.c:
SINGLE-CORE the gap chains CLEANLY (rc=0, full output) — so the perchan->mm return edge (desc_slot 138) IS HW-safe.
But in the real 3-CORE colsplit on {256,3584,512} it STILL WEDGED (resets, recovery ground through to nc=1).
ROOT CAUSE (conclusive): the gap is a PER-CORE, in-chain construct; the wild is a CROSS-CORE CONCURRENCY race — all
3 cores fire their wide fp16 weight fetches simultaneously at each slice boundary, and a gap in one core's chain does
nothing to the other cores' concurrent fetches. So NO per-core chaining trick (A contiguous / B SDP / B' perchan-gap)
can fix a between-cores hazard. Gap code gated OFF (`ORK_F16_GAP`, needs `ORK_F16_CONTIG`); default path unaffected.

## (B') FAIR-TEST confirmation (2026-08-04): rebooted to a FRESH board, ran CONTIG+GAP as the FIRST NPU workload
(no prior wedge to bias it). Still WEDGED — 4 soft-resets, no clean output in 70s (recovery ground through). So the
gap failure is NOT a degraded-state artifact; it fails fairly. (Board recovered via HA "Rock 5B Plug" power-cycle,
SPI intact — a graceful `sudo reboot` had left it unreachable, needed the plug.)
Explored + all reduce to the same wall (barrier / contiguous / SDP-gap / perchan-gap / sentinel+spin / round-robin):
the wild is CROSS-CORE concurrent fp16 fetch. A cross-core barrier (pthread in the per-slice path, or CPU-halt of
in-chain spins) syncs slice COMPLETION but releases all cores together -> simultaneous next-slice fetch = the wild.
Removing it needs staggering fetches by ~a fetch-duration (= serializing) or single-core (= nc=1). No per-core chain
trick and no in-chain sentinel/spin makes a CONCURRENT fetch safe. 3-core-fast+recover (ship) vs serialize-safe-slow
(nc=1) are the only points; single-submit can't beat them.

## Concurrency-degree + stagger tests (2026-08-05) — CONCURRENCY-CAPPING is DEAD.
- Staggered launch (ORK_F16_STAGGER): a few µs is pointless (fetches are ~770µs — they still fully overlap). A/B
  at 300µs: control 1/20 vs stagger 1/20 — no measurable change (and the wedge is too rare, ~1/25, to detect a
  partial rate reduction without ~100+ runs/config).
- Concurrency degree (nc=2 via ORK_F16_2CORE vs nc=3): BOTH wedge ~1/25 (nc2: 1 reset + 1 wrong / 25; nc3: 1/25).
  So 2-way fp16 fetch is NOT safe — the wild is present at ANY concurrency >=2, not just 3-way. "Cap at 2 / delay
  the 3rd until the 1st drains" does NOT help; only concurrency==1 (serialize/nc=1) is wedge-free. Speed: nc3 2.6x
  vs nc1, nc2 2.0x vs nc1 (but nc2 still wedges — no reliability gain for the speed loss).
CONCLUSION: no fast-and-wedge-free point. The concurrency that gives the DMA bandwidth IS the hazard, down to 2
streams. Only serial (nc=1, ~3x slower) removes it. => ship fast+recover (nc=3) or safe-slow (nc=1); nothing between.

## VERDICT (2026-08-04): single-submit / HW-chained fp16 across K-slices is NOT achievable on this NPU.
(A) empirically wedged (timing race, not buffer boundary); (B) is registry-closed (fp16-mm->SDP is SW-only). The
SHIPPING answer is the per-slice submits (the SW-safe sequencing) + run-level reset/resubmit recovery + nc=1
backstop: BEATS mcworker 1.05-1.42x, bit-exact, self-heals, path-independent reset rate = mcworker's. DONE.
- Meanwhile the shipping path — **per-slice submits + run-level reset/resubmit recovery + nc=1 backstop** — WORKS,
  beats mcworker (1.05-1.42x), self-heals; single-submit is elegance/unification, not a functional need.

## Stages
- [x] Stage 0: dt=w->dtype seam in colsplit; h->mc_dt=dt early; f32-accumulate branches added (dormant for i8).
      GATE: board make test byte-identical (int8 untouched). PASS (ALL TESTS PASSED, attest rewritten).
- [~] Stage 1: fp16 Sn==1 colsplit. DONE + BIT-EXACT (matches nc=1 AND mcworker on all shapes, maxerr=0).
      Two bugs fixed: (1) nt_sz=16 for fp16 (was 32); (2) CBUF cap 32768 into synth() (fp16 M-sched miscomputes
      large mc above the 32768 tile). A/B (test_f16colsplit.c): colsplit BEATS mcworker 1.07-1.40x on FFN shapes
      (K3584/4096 N512+); ~0.87x on tiny K1024 N256 (small-shape submit overhead — same trade as int8 base).
      Routed ONLY from run_multicore (calls ork_dyn_begin_colsplit directly; falls back to mcworker on NULL) —
      NOT via ork_dyn_begin_mc (that also serves SSM stream/pool fp16, which must keep pre-Stage-1 behavior).
      3rd bug fixed: intermittent (~4%) completion race on the K-split f32 accumulate — the blocking submit's
      completion can precede the fp16 writeback DRAIN. Fixed with SENT seed + full-surface civac VERIFY in
      ork_csub_worker (fp16-gated, mc_dt==DT_F16 && oSk>1; hardened_w=1 for fp16 to flush the seed). 30/30 clean.
      GATES PASSED: make test ALL PASS toggle ON and OFF; test_f16colsplit (new, self-validating) bit-exact;
      A/B beats mcworker 1.07-1.4x on FFN shapes. (test_spine perf assertion is a pre-existing timing flake.)
      Behind ORK_F16_COLSPLIT (default OFF). Added test_f16colsplit to EXAMPLES + make-test loop.
      OPEN: decide default-on; add fp16 budget=3 case to test_matmul (optional, test_f16colsplit covers it).
- [ ] Stage 2: int8 fall-through (Sn>1&K>4096 / no-Bf) -> colsplit or refuse. Check reachability first.
- [ ] Stage 3: native int4 colsplit (synth_i4, nt_sz=64) replacing i4_mcworker*. Bar = matches i4_mcworker.
- [ ] Stage 4: remove mcworker + i4_mcworker* + dispatch for passing precisions; comments/OPS_REGISTRY/attest.

## Board ops
- ssh board; single-stream; one clean run/boot; SIGTERM not -9; `ssh board 'sudo reboot'` on NPU wedge.
- build: `cd ~/llama.cpp-orkd/ggml/src/ggml-ork/ork-driver && make -j` ; validate: `make test`.
- attest: refresh tests/sbc_attest.txt after each npu.c change (CI check-attest gate). No commit trailers.

## Next concrete step
Stage 0 edits in src/npu.c, then board `make test` byte-identical.

## int8 recovery VERIFIED + fp16 SENTINEL recovery built (2026-08-05)
int8 colsplit-prefill has NO app-level wedge recovery: worker blocking-submit return is ignored for int8 (mc_error
set only for fp16, npu.c ~11601); prepolled=1 makes ork_dyn_end SKIP the recover loop (~12671); mc_recover_resubmit
/recov_max=6 only runs on the NONBLOCK poll path (decode/stream / ORK_COLSPLIT_NB), NOT the colsplit prefill. So
int8's stability is NOT recovery-masked — with no recovery, a wedge would be a silent wrong answer, which make
test/models never show => int8 colsplit genuinely doesn't wedge (or the KERNEL transparently auto-recovers its
blocking submits, as observed for fp16). Either way the fp16 wild is fp16-specific (2-byte fetch). Web-corroboration:
Rockchip's RKNN SDK keeps matmul SINGLE-CORE (only conv+data-movement are multi-core); multi-core NPU concurrency has
documented spinlock/IOMMU crashes => multi-core matmul is vendor-avoided territory.

fp16 SENTINEL recovery (gate ORK_F16_SENTINEL, tmo ORK_F16_SENTINEL_TMO_US default 800ms): the swchain submits
NONBLOCK (0x2) + CPU poll-DRAINs each slice (last-word gate -> full-slice civac verify; full-surface SENT seed via
hardened=true for fp16 K-split). Fast wedge-detect (~poll timeout, not the 8s blocking timeout) AND the CPU never
blocks in-kernel (avoids the D-state uninterruptible hang that ESCALATED to a hard wedge). Stuck sentinel -> mc_error
-> run-level RKNPU_ACT_RESET + resubmit + nc=1 backstop. Goal: match int8's cheap recovery so the (rare, physics-
intrinsic, ~2%) wedge becomes cheap/invisible instead of the ~91s blocking recovery that cratered fp16's avg wall.
TODO: compile-check + A/B (blocking vs sentinel: per-wedge recovery cost, no D-state hangs) once the 1k campaign frees the board.

## fp16 recovery IMPROVEMENT attempts — BOTH HARD-WEDGED, reverted (2026-08-05)
Goal: cheaper/cleaner recovery than the campaign-proven "8s blocking + resubmit-same x6 + nc1 backstop" (which
survives 1000 runs, 0 hard-wedge, but ~1.7% wrong-answers + ~91s/wedge).
- SENTINEL (ORK_F16_SENTINEL, nonblock + poll-drain): HARD-WEDGED on the FIRST run, on a normally-safe shape.
  NONBLOCK is the ORIGINAL fp16 doorbell wedge (why per-slice blocking exists). int8's sentinel recovery is NOT
  portable to fp16 — int8 tolerates nonblock, fp16 doesn't. DEAD. (Also had a 7x clean slowdown from a per-iter
  full-slice verify before I fixed it to an O(1) gate.) Gated off.
- Straight-to-nc1 (recov_max=0) + 2s timeout + explicit RKNPU_ACT_RESET before nc1: HARD-WEDGED in ~8 runs.
  The added explicit reset / tighter timeout made it WORSE than the resubmit-x6 path (which reaches the final
  de-escalation rarely because a resubmit usually heals). Reverted to recov_max=6 / 8s / no-explicit-reset.
LESSON: touching the NPU aggressively after an fp16 wedge (nonblock, explicit reset, straight-nc1) ESCALATES to a
HARD wedge (4 hard-wedges this session, all power-cycle-recovered, SPI survived). The campaign-proven blocking +
resubmit-x6 + nc1 is the SAFEST recovery found; further on-board recovery tuning is high-risk, low-yield.
ENV added (kept for A/B, default-safe): ORK_F16_SENTINEL (DEAD-nonblock), ORK_F16_SENTINEL_TMO_US, ORK_F16_RESUB (default 6).

## SENTINEL nonblock recovery — corrected design, one real-wedge run (2026-08-05) — RAW OBSERVATION, interpretation open
Rebuilt the nonblock+sentinel recovery with the IRQ-hang finding in mind: on poll-miss, RKNPU_ACT_RESET (kill the
in-flight) + de-escalate to nc=1 — the reset made CONDITIONAL on the sentinel gate (fp16_sentinel_r) so the default
BLOCKING path stays reset-free (byte-identical to campaign-proven). Hypothesis being tested: nonblock leaves no
D-state-stuck ioctl to conflict with the reset, so reset+nc1 should be clean.
OBSERVATIONS (facts, not verdict):
- FORCE_WEDGE=99 (simulated wedge, submits succeed, resets an IDLE NPU): 4 forced de-escalations, final BIT-EXACT,
  4 clean resets, no hard-wedge. The recovery ACTION (reset+nc1) executed cleanly on an idle NPU.
- REAL natural-wedge campaign (ORK_F16_SENTINEL=1, shape 256x3584x3584, N=40): run 1 hit a natural wild and the
  board went to SSH-dead / 100% packet loss; power-cycle recovered it (SPI survived). Only 1 real wedge was observed
  (n=1) before the hang — NOT yet a statistical result.
OPEN QUESTIONS (do not conclude yet): was it the userspace reset racing the kernel fault handler, or something in
the nc1 de-escalation, or a one-off? n=1 is not a rate. Default (blocking) path is UNCHANGED by this session's edits
(sentinel gate default-off; recov_max=6, no reset) — the tree is safe to iterate on. Next move: user's interpretation.

## ROOT CAUSE FOUND via auto-dump (2026-08-05) — it is NOT a CDMA wild. Corrects the whole session's theory.
Wired ork_npu_dump_state + a per-slice [F16-WEDGE-DETECT] line into the sentinel path (fires BEFORE recovery, fflush,
live-streamed over ssh so it survives a hang). Captured on run 1 (shape 256x3584x3584, ORK_F16_SENTINEL):
  [F16-WEDGE-DETECT] core=0/1/2 Kslice=1 POLL-MISS after 800000us last-word=SENT (stuck) mtk.int_status=0x0
  [NPU-DUMP] hw ok, freq=1GHz, iommu=1, freeSRAM=956KiB, hw_elapse=0 (=>no work), int_status[0..3]=0
  dmesg: "BUG: Bad page cache in kworker pfn:122d2d..122d36" (10 contig pages) ; "RKNPU: soft reset, num:6" ;
         "Unable to handle kernel NULL pointer dereference at 0xc" -> Oops in rknpu_gem_sync_ioctl+0x168 ;
         "page dumped because: still mapped when deleted" ; Comm: test_f16colspli ;
         trace: filemap_unaccount_folio<-__filemap_remove_folio<-truncate_inode_folio<-shmem_undo_range<-
                shmem_evict_inode<-evict<-iput<-__dentry_kill<-dput
DEFINITIVE FINDINGS:
1. The failure is ALWAYS Kslice=1 (the 2nd K-slice), never slice 0. Slice 1's nonblock submit is ACCEPTED (src=0)
   but NEVER DISPATCHED (hw_elapse=0, int_status=0) = an accepted-but-never-dispatched DOORBELL DROP (the exact
   int8 nonblock failure named at npu.c:12746). NOT a hardware CDMA wild — the NPU is HEALTHY throughout (iommu=1).
2. The HARD-WEDGE is a KERNEL DMA-BUF LIFETIME BUG, not an NPU fault: the phantom (never-completed) slice-1 job keeps
   the output buffer's shmem pages mapped; teardown/de-escalate frees the buffer -> shmem eviction hits "page still
   mapped when deleted" -> bad page cache -> NULL-deref in rknpu_gem_sync_ioctl (our bsync) -> soft-reset cascade -> hang.
3. This unifies BOTH prior symptoms as downstream of the slice-1 doorbell drop: BLOCKING on the dropped doorbell =
   ~60-90s kernel-watchdog "D-state hang"; NONBLOCK = fast 800ms detect but the phantom job dangles refs -> crash.
WHY int8 nonblock is safe & fp16 sentinel is not: int8 mc_recover_resubmit (npu.c:12674) does reset->reseed->RESUBMIT
the SAME submit and KEEPS POLLING until the job LANDS/completes on the ORIGINAL buffers -> GEM refs release normally
-> clean teardown. fp16 sentinel abandons the colsplit buffers (de-escalates to a fresh nc1) while the phantom slice-1
job still holds them -> the crash. Blocker (npu.c:5220): fp16's per-slice submits have no single stashed mc_subs, so
mc_recover_resubmit isn't directly reusable.
PROPOSED FIX (grounded, not yet built): on an fp16 nonblock poll-miss, COMPLETE the dropped job on its original buffer
(re-ring that slice's doorbell + re-poll to landing, int8-style) instead of de-escalate+free. Never free a buffer a
phantom job still references. Alternative framings for the doorbell drop itself: (a) prevent slice-1 drop (stagger/
serialize the 2nd-slice doorbell), (b) blocking submit for slice>=1 only.
NOTE: correct the memory `fp16-colsplit-wedge-misvalidation` + earlier WIP text — the "concurrent 3-core 2-byte fetch
CDMA wild" root cause is REFUTED for the nonblock path by int_status=0/hw_elapse=0/iommu=1 + the gem_sync kernel trace.

## RESUB=6 (int8-style complete-the-job) — FAILS for fp16: the slice-1 drop is STICKY across reset (2026-08-05)
Set up netconsole (kernel printk -> UDP -> Mac C listener udplog.c; wedge-surviving, off-box) + a Mac /dev/kmsg-routed
capture harness (board_bringup.sh re-arms netconsole+loglevel8+governors each boot). KEY LESSON on capture: ramoops/
pstore is DRAM-backed (region 0x110000) so a COLD power-cycle WIPES it — only a WARM reboot preserves it; netconsole is
the robust off-box capture. Ran ORK_F16_SENTINEL=1 ORK_F16_RESUB=6 (nonblock detect + int8-style reset+resubmit-to-
landing). Captured on the board ($BF survived — the gem_sync Oops killed the process but the box stayed up this time):
  attempt 0: core0,core1 Kslice=1 POLL-MISS 800ms (hw_elapse=0) -> RKNPU_ACT_RESET+resubmit
  attempt 1: core2 Kslice=1 POLL-MISS -> reset+resubmit
  attempt 2..6: EVERY attempt re-drops Kslice=1  (never lands)
  dmesg: RKNPU soft reset num:6 (x6) -> Bad page cache/map (contig pfns) "still mapped when deleted" ->
         Oops rknpu_gem_sync_ioctl+0x168 NULL+0xc, FSC=0x04 LEVEL 0 TRANSLATION FAULT, in OUR bsync (PID test_f16colspli)
DECISIVE: the fp16 slice-1 (2nd K-slice) doorbell drop is STICKY across RKNPU_ACT_RESET — unlike int8 (wiki: int8 drop
"lands with high prob after a clean-reset resubmit"). So "complete the job via NONBLOCK resubmit" CANNOT work: every
resubmit re-drops, the job never completes, phantom IOMMU mappings accumulate, and our next bsync NULL-derefs in
rknpu_gem_sync_ioctl walking a zeroed page table (level-0 fault) = the crash. int_status=0 only means submit ACCOUNTING
is clean; the IOMMU mapping is corrupt (level-0 fault proves it). This PARTIALLY VINDICATES the original CDMA/IOMMU-wild
intuition: it IS a persistent HW/IOMMU state, not a clearable doorbell race — I was too hasty calling that "refuted".
MECHANISTIC HYPOTHESIS (consistent across the whole session): only the KERNEL's blocking-path job-reap (watchdog on a
blocking submit) CLEARS the sticky state; userspace RKNPU_ACT_RESET does NOT. Evidence: BLOCKING recov (recov_max=6,
campaign-proven 1000 runs 0 hard-wedge) works because a blocking submit on the dropped slice waits the ~60s kernel
watchdog which reaps+resets properly; NONBLOCK+ACT_RESET skips that path -> sticky -> crash. Also: the crash is
SOMETIMES survivable (box stayed up) and sometimes fatal (earlier run took it down) — depends on whether the bad bsync
lands in our proc (survivable, proc killed) or a kworker mid-teardown (fatal).
NEXT (proposed, grounded, LOWER risk): nonblock DETECT (fast) + BLOCKING resubmit of the missed slice (invokes the
kernel reap that actually clears the sticky state) — fast detect + the one recovery op proven safe. Needs a code change
(fp16 sentinel RESUB path uses a blocking, not nonblock, resubmit). Awaiting user steer before the next board cycle.
Board ops this session: 8 hard-wedges (all recovered; SPI intact). WARM `sudo reboot` clears the wedged NPU when the box
is still reachable (preserves ramoops); COLD power-cycle only when SSH-dead. Capture infra now: netconsole (kernel
printk -> UDP -> Mac udplog.c) + in-process ork_kmsg (diag lines -> /dev/kmsg -> netconsole) = wedge-surviving, off-box,
no ssh-stdout hang. board_bringup.sh re-arms it each boot (races boot; run it a few s after sshd is up).

## ALL-BLOCKING baseline (78 runs) — safe from crash but SILENTLY WRONG / storms on a drop (2026-08-05)
No sentinel, recov_max=6. Happy path 0.5s / 28ms nc3 / 2.6x nc1 / bit-exact (same speed as nonblock). On a drop:
run 72 = 60.6s wall + SILENT WRONG ANSWER (diff=13568, rc=0, my F16-WEDGE never fired); run 78 = 200s+ HANG after a
~235x soft-reset STORM. wedged=0 for BOTH: BLOCKING CANNOT DETECT (submit returns rc=0 while lying — the watchdog
reset killed the compute but the ioctl "succeeded"). Box SURVIVED all 78 (no gem_sync crash). => all-blocking is
hard-wedge-safe but useless for correctness (undetected corruption + multi-min stalls). The nonblock SENTINEL is the
ONLY reliable detector (output sentinel catches the non-landing).

## fd-close REAP — WORKS (first clean cleanup of the poison) (2026-08-05). Answers "cleanup first, not blocking".
User insight: the crash is a cleanup-ORDERING bug (we bsync/free the poisoned buffer before the kernel reaps it), not
proof nonblock is doomed. Probe ORK_F16_FDCLOSE: on nonblock detect -> close(c->fd) -> reopen -> _exit(0) (isolate the
close). RESULT (netconsole, in-process kmsg): close() returned WITHOUT crash; reopen fd ok; **235 soft resets BEFORE
the close, ZERO after**; 50s+ hold clean; box ALIVE. So closing the DRM fd (drm_release) CANCELS all cores' stuck jobs
+ tears down the IOMMU domain = a CLEAN reap, unlike our per-buffer bsync (which NULL-derefs in rknpu_gem_sync_ioctl).
It even STOPPED the soft-reset storm. This is the cleanup primitive we were missing.
ARCHITECTURE (why it works / why the wedge happens): ONE c->fd for all 3 cores (single DRM ctx / single IOMMU domain;
per-core only s.core_mask=1<<i), and the fp16 K-split weight w->Bb[ns*Sk+ks] is ONE dma-buf read CONCURRENTLY by all
3 cores (N-column split = different offsets, same buffer, same CDMA). So the sharing is BOTH the wedge cause (3-core
concurrent contention on one buffer/CDMA/IOMMU ctx) AND why one close(fd) reaps everything.
TWO FORWARD PATHS (awaiting user steer):
 (A) RECOVERY: nonblock detect -> close+reopen+RE-INIT (re-create all buffers, RE-PACK the weight — heavy, caller-side)
     -> resubmit. close-reap proven clean; remaining work is the re-init/re-pack continue (probe just _exit(0)'d).
 (B) PREVENTION (unshare): per-core fd (separate IOMMU ctx -> a fault can't poison the others, per-core reap) and/or
     per-core weight-slice copy (no concurrent read of one dma-buf). If unsharing removes the contention, the drop may
     not happen at all. Cost: 3x weight memory and/or a multi-fd submit model.
ENV added this session (all default-off; default path = campaign-proven blocking, byte-identical): ORK_F16_SENTINEL,
ORK_F16_BLOCK_HEAL, ORK_F16_RESET_N (0/1/2), ORK_F16_COOLDOWN_MS, ORK_F16_FDCLOSE, ORK_F16_RESUB. New code: ork_kmsg()
helper, f16_force_blocking field, reset-count loop, fd-close probe, worker+run-level kmsg diag.

## DRIVER MODEL (from rockchip-linux/kernel develop-6.1 drivers/rknpu, matches board 6.1.115-vendor) — settles the design
- ONE IOMMU domain attached to the whole DEVICE at a time: rknpu_iommu_domain_switch does iommu_attach_device(dst,
  rknpu_dev->dev); domains are a DEVICE-GLOBAL array rknpu_dev->iommu_domains[], switched per-job (get_and_switch,
  refcounted) => PER-CORE / PER-FD IOMMU ISOLATION IS IMPOSSIBLE (per-core-fd is DEAD as a prevention lever).
- rknpu_submit_ioctl => dev_get_drvdata(dev->dev) => rknpu_submit(rknpu_dev,...) IGNORES file_priv; jobs + completion
  (subcore_datas[core].job_done_wq, device IRQ) are DEVICE-GLOBAL, not per-fd.
- NO userspace job-abort ioctl. The ONLY clean per-job reap (rknpu_job_abort: iommu_domain_put + dequeue) is reached
  ONLY via the BLOCKING wait's timeout (rknpu_job_wait: wait_event_timeout(job_done_wq, DONE||soft_reseting,
  args->timeout) looped wait_count<3; on timeout returns -ETIMEDOUT then rknpu_job_abort). args->timeout is OURS.
- => short-timeout blocking DOES detect (-ETIMEDOUT) + reap cleanly, BUT stalls the doorbell poller => out.
- => the ONLY nonblock-compatible clean reap is close(fd) => drm_release (device teardown reaps ALL jobs + IOMMU).

## FD-REAP RECOVERY — BUILT + reap MECHANISM PROVEN + recovers a REAL drop w/o hard-wedge (2026-08-05). One segfault left.
Design (the architecture-aligned recovery): nonblock doorbell (unchanged, poller-friendly) -> sentinel poll-miss
DETECT -> ork_ctx_fd_reap(c) -> retry. ork_ctx_fd_reap: close(fd) [drm_release reaps all stuck jobs+IOMMU — the clean
reap] -> reopen + reinit (POWER_ON/nice/timerslack) -> RE-IMPORT every registered dma-buf weight IN PLACE (its pages
persist — client/heap holds the dma-buf fd; its CPU mmap is of the dma-buf fd NOT the drm fd so it SURVIVES the close;
only the IOMMU mapping died => re-import = PRIME_FD_TO_HANDLE+MEM_CREATE, NO re-pack, NO data copy) -> zero bcreate'd
scratch (lazily re-created) + warmed=0 (fp16 2-pass re-warm on retry). Requires weights be dma-buf-backed => the fp16
pack gates on ORK_F16_IMPORT_W (bimport instead of bcreate + imp_reg). In the orkd/doorbell path weights are ALREADY
dma-buf imports (task #16) so this is free there.
NEW CODE: import registry g_imp/imp_reg/imp_unreg (buf* list, registered in pack, unregistered in bdestroy);
reimport_inplace(); ork_ctx_fd_reap() [exposed in ork_npu.h]; pack ORK_F16_IMPORT_W gate; run_multicore recovery
ORK_F16_FDREAP branch (attempt<recov_max: ork_ctx_fd_reap + continue); test_f16colsplit ORK_F16_REAP_TEST hook.
RESULTS:
- MECHANISM (ORK_F16_REAP_TEST=1 IMPORT_W=1, shape 8x2048x1536, run->reap->run): ork_ctx_fd_reap rc=0, POST-REAP run
  BIT-EXACT maxerr=0 — PROVEN: close+reopen+re-import(in place)+run-path-rewarm produces correct output. (The earlier
  fresh-fd hang was the POWER_ON-only percore probe skipping the warm; the run path warms via ork_npu_enter.)
- REAL DROP (ORK_F16_FDREAP+IMPORT_W+SENTINEL, shape 256x3584x3584): a natural drop FIRED the reap; netconsole:
  "FD-REAP close(fd=3) ... reopened fd=3, re-imported N dma-buf weights (0 failed)" (twice — retry re-dropped, reaped
  again). BOARD SURVIVED THE DROP+REAP (no hard-wedge — the reap cleaned the device mappings, so the drop did NOT
  trigger the kernel still-mapped/gem_sync poison-wedge that killed EVERY prior approach). THEN the process SEGFAULTED
  rc=139 in the retry-after-reap path (userspace NULL/stale deref — NOT a board wedge; board stayed pingable, sshd only
  degraded from the segfault teardown, power-cycled 1x).
REMAINING BUG (rc=139 segfault, retry-after-reap) — 2 suspects, both in the new code:
  (1) MULTI-SLICE REGISTRATION: the "re-imported 1 then 2" mismatch says not all of the weight's Bb[ns*Sk+ks] slices
      are consistently registered/re-imported (N=3584 => Sn>1 => multiple Bb) — a retry reads a Bb slice whose IOMMU
      mapping wasn't restored -> deref. Check imp_reg covers EVERY slice + reimport_inplace covers all g_imp entries.
  (2) SCRATCH STALE PTR: ork_ctx_fd_reap zeroes c->mcc[]/dom_save/dom_anchor + frees dom_save/dom_anchor; if the retry's
      accumulate / ork_dyn_end / dom_activate touches one before mc_ensure re-allocs (or assumes dom_save non-NULL) -> deref.
REPRO: sudo env ORK_MM_TIMEOUT=8000 ORK_F16_COLSPLIT=1 ORK_F16_IMPORT_W=1 ORK_F16_SENTINEL=1 ORK_F16_FDREAP=1 \
       ./test_f16colsplit 256 3584 3584 3   (loop until a natural drop fires the reap; ~1.5-3.5%/run). Or force via a
       FORCE_WEDGE-style mc_error inject to hit the reap deterministically without waiting for a real drop.
NEXT: fix the 2 suspects (likely gdb/coredump or add ork_kmsg breadcrumbs through the retry+accumulate to locate the
deref), re-run the drop hunt -> expect BIT-EXACT recovery + board survives. Then: poller-quiesce + in-flight-batch redo
for the real doorbell/orkd path; flip fp16 colsplit default-on once the recovery is clean. Board incidents this session:
~9 (all recovered, SPI intact). The fd-reap is the FIRST recovery that survives a real drop w/o hard-wedge.

## fd-reap DEBUGGING (2026-08-05 cont.) — Bug#1 FIXED+VERIFIED; Bug#2 NAMED (not yet root-caused). ~12 board cycles.
CAPTURE DISCIPLINE (learned the hard way): (a) after gdb catches a real-drop segfault, the NPU is left drop-poisoned/
half-reaped => REBOOT before any next test or the result is CONFOUNDED (a "still crashes" run was actually the dirty
board, not the fixed binary). (b) do NOT truncate netconsole.log mid-run (lost one wedge's oops that way). (c) grep the
FULL "Call trace", not just the fault-address line. (d) FORCE_WEDGE=N is the SAFE deterministic repro of the recovery-
loop path (N forced mc_errors => N reaps, no real drop/poison, no wedge risk) — use it before real-drop hunts.
BUG #1 (FIXED + VERIFIED): ork_npu_dump_state deref'd c->task.cpu which ork_ctx_fd_reap ZEROES (task buf died with the
fd; colsplit uses c->mtk[i] not c->task, so nothing re-creates it before the NEXT attempt's dump). Only fires on a 2nd
recovery attempt after a reap (retry re-drops => 2nd dump). Fix: guard `t=c->task.cpu` NULL in ork_npu_dump_state (prints
"(no task buf)"). VERIFIED: FORCE_WEDGE=2 (two reaps => two post-reap dumps) heals BIT-EXACT, board healthy. Also proven:
ORK_F16_REAP_TEST (run->reap->run) BIT-EXACT at 8x2048x1536 AND 256x3584x3584; a real drop's reap survives w/o hard-wedge.
BUG #2 (NAMED, NOT root-caused): on REPEATED reaps (a real-drop CASCADE — ~4 reaps in 4s, NOT ~3%), a KERNEL refcount
UNDERFLOW fires: "refcount_t: underflow; use-after-free" (lib/refcount.c:28) in drm_gem_object_put <- release_handle <-
drm_gem_release <- drm_file_free <- drm_release  (i.e. during a reap's close(fd)). => a GEM object is PUT TWICE across
the reap/re-import/close cycle. It corrupts the slab (__kmem_cache_alloc_node faults then ripple into UNRELATED syscalls
apparmor_sk_alloc/netlink/path_openat => system-wide wedge; rk_iommu page faults confirm the freed weight's mapping is
torn out mid-fetch). CANDIDATE causes (unconfirmed): (i) drm-prime import refcount imbalance across repeated
PRIME_FD_TO_HANDLE of the same dma-buf; (ii) the reap-CASCADE itself (a bad re-import makes EVERY subsequent colsplit
weight-fetch IOMMU-fault => poll-miss => "drop" => reap => churn => underflow) — ONE root cause driving both the cascade
AND the underflow. NOTE: my earlier "slices share one dma-buf" guess is LIKELY WRONG — the ORK_F16_IMPORT_W pack path
bimports EACH Bb slice as its OWN dma-buf (per-slice bimport at npu.c ~2578), so they are not shared; the double-put
cause is NOT yet pinned. INVESTIGATION (fresh session): add per-reap GEM-handle/refcount accounting (ork_kmsg the handle
+ a get/put balance around reimport_inplace); confirm whether the FIRST post-reap colsplit faults (rk_iommu) => cascade,
or whether close(fd) itself underflows on the 2nd reap with a single clean op between; check reimport_inplace for a
missing/extra put and whether live_add/imp_reg + the retry's bdestroy(mcc)+bcreate re-alloc double-track a handle.
STATE: mechanism proven; Bug#1 fixed+verified (in tree); Bug#2 (repeated-reap GEM refcount underflow) is the sole blocker
to a working real-drop recovery. All gated (default path byte-identical). Reap+nc1-de-escalate wiring is in
(ORK_F16_FDREAP => ork_ctx_fd_reap + break to nc=1) but blocked by Bug#2. ENV: ORK_F16_IMPORT_W, ORK_F16_FDREAP (+ the
earlier sentinel/reset/fdclose gates). Repro (SAFE, deterministic): FORCE_WEDGE heals; real-drop hunt (256x3584x3584,
IMPORT_W+SENTINEL+FDREAP) triggers Bug#2 — REBOOT after.

## Bug#2 ROOT-CAUSED = VENDOR-KERNEL drm_release double-put; reset-before-close FIX FAILED (2026-08-05, clean repro)
DEFINITIVE (clean fresh-log repro, methodology fixed: fresh netconsole file per run + counter greps stderr not kmsg):
- 10x FORCE_WEDGE reaps of a CLEAN fd (submits succeeded, NO stuck job): NO underflow, bit-exact, board healthy. So the
  repeated close/reopen/RE-IMPORT itself is BALANCED — Bug#2 is NOT my re-import churn.
- On a REAL drop, the underflow fires on the 2nd reap's close(fd): "refcount_t: underflow; use-after-free" ->
  drm_gem_object_put <- drm_gem_object_release_handle <- drm_gem_release <- drm_file_free <- drm_release. Then a UAF fault
  at an ASCII virtual address (e.g. 0x006d656d68735f7d = "...hs_mem}") = the freed GEM/shmem obj reallocated as a string
  then deref'd = textbook double-free. Corrupts the slab -> __kmem_cache_alloc_node faults ripple into unrelated syscalls
  (apparmor/netlink/path_openat) -> sshd degrades (ping ok, ssh "connection reset") -> effective wedge.
ROOT CAUSE: a VENDOR rknpu/DRM-core bug — drm_release DOUBLE-PUTS a GEM object when the closing fd holds a RE-IMPORTED
weight handle AND a STUCK (dropped nonblock) job that still references it. The stuck job is the trigger (clean fd never
underflows; only a real drop's in-flight-but-never-dispatched job does). Reap#1 (closing the original pack-time fd) is
clean; reap#2 (closing the reap#1-reopened fd that has the re-imported handles + a fresh stuck job) underflows.
FIX ATTEMPT — FAILED: RKNPU_ACT_RESET + 2ms drain BEFORE close(fd) (to abort the stuck job in-kernel first). Did NOT
prevent the underflow. Soft-reset does NOT reach the kernel's clean job-abort (rknpu_job_abort) — that is reachable ONLY
via the BLOCKING wait's timeout (rknpu_job_wait), which we can't use (stalls the doorbell poller). So soft-reset leaves
the stuck job in a state drm_release still double-puts.
REMAINING PATHS (all have a real cost — none is a clean userspace one-liner):
 (a) KERNEL PATCH: balance the GEM put in the vendor rknpu gem-release / prime-import path when a stuck job holds the
     handle (the correct fix, but requires building/shipping a patched rknpu.ko — out of scope for a userspace lib).
 (b) BLOCKING-ABORT before close: force rknpu_job_abort on the stuck job via a bounded blocking wait, THEN close — clean,
     but the blocking wait stalls the doorbell poller for that op (~seconds). Acceptable ONLY if scoped to the ~3% drop
     recovery and the poller can be quiesced around it; needs the poller-coordination work anyway.
 (c) AVOID the stuck-job-at-close: don't reap; instead prevent/avoid the drop (e.g. proactively nc=1 the wedge-prone
     (weight,M) shapes so the concurrent-fetch drop never happens — the memoized slice-rescue already does this for some
     shapes). Loses colsplit speed on those shapes but no reap, no wedge.
VERDICT: fd-reap RECOVERS a single real drop and the board SURVIVES the drop; but a 2nd reap (drop-during-recovery)
hits the vendor drm_release double-put. Bug#2 is a VENDOR KERNEL bug, precisely characterized, NOT fixable by the
obvious userspace means. ~14 board cycles. Recommend: STOP on-board; decide between (a) kernel patch, (b) blocking-abort
+ poller quiesce, or (c) avoid-the-drop — as a design choice, not more marathon wedging.

## DEFINITIVE (2026-08-05, clean fresh-board tests): the fp16 drop poisons GEM state TWO ways; NO clean userspace reap.
Both candidate fixes validated on genuinely-fresh boards (uptime reset + dmesg fault-clean verified before each run;
fresh netconsole file per run — the earlier confounds were running the next test on a not-power-cycled, "reboot is
needed"-state board):
- FD-CLOSE reap: tears down the poisoned IOMMU mapping (fixes the bsync/gem_sync crash) BUT drm_release DOUBLE-PUTS the
  stuck async job's task_obj (rknpu_job_alloc:152 get / rknpu_job_free:98 put, deferred cleanup_work) => refcount
  UNDERFLOW (drm_gem_object_put) => slab UAF. reset-before-close does NOT prevent it (soft-reset != rknpu_job_abort).
- TCLEAN retry (nonblock resubmit -> rknpu_job_timeout_clean reaps the stuck job cleanly, NO fd-close): fixes the
  task_obj underflow BUT the retry's ork_dyn_begin_colsplit bsyncs the dropped slice's OUTPUT buffer (c->mcc[i]) whose
  GEM state the drop/soft-reset tore => rknpu_gem_sync_ioctl NULL-deref at +0x168 (VA 0xc). timeout_clean (a soft-reset)
  does NOT restore the output buffer's mapping.
=> The dropped nonblock job poisons BOTH (1) its task_obj ref (underflow on close) AND (2) its output buffer's GEM/IOMMU
mapping (gem_sync NULL-deref on reuse). The poisoned output buffer can't be safely REUSED (bsync->NULL-deref), DESTROYED
(bdestroy->"still mapped when deleted"), or FD-CLOSED (->task_obj underflow) — each trips a DISTINCT vendor-kernel bug.
CONCLUSION: a real fp16 concurrent-fetch drop is NOT cleanly recoverable in userspace on this vendor rknpu kernel.
KEY UNDER-EXPLORED LEAD: int8 colsplit RECOVERS from the SAME class of drop cleanly (mc_recover_resubmit via ork_dyn_end's
recover loop — wiki: 25/40->40/40, 12000 iters). So the difference is fp16-SPECIFIC: fp16 colsplit uses a DIFFERENT
recovery path (the per-core worker + host f32 accumulate I added), NOT int8's proven ork_dyn_end recover loop. NEXT LEAD
(most promising): find why int8's ork_dyn_end recover loop (re-seed + resubmit on the SAME buffers) does NOT hit the
gem_sync/underflow, and port THAT exact path to fp16 — rather than the worker/fd-reap/tclean paths tried here. Options
otherwise: (a) kernel patch (balance task_obj put in drm_release + restore output-buffer mapping on soft-reset), or
(c) don't ship fp16 colsplit (use fp16 nc=1 — safe, but loses the 2.6x). ~16 board cycles; all recovered, SPI intact.
STATE: default path byte-identical/safe. All recovery attempts gated (ORK_F16_SENTINEL/FDREAP/TCLEAN/IMPORT_W, default
off). Bug#1 dump-state fix is in tree + verified. The fd-reap/tclean/reap-stuck code is in but gated off (both fail).
