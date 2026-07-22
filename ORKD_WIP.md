# ORKD / SDP-doorbell — Remaining Work (WIP)

Scratch tracking doc (per AGENTS.md). Delete or fold into the wiki once these land. Branch: **`feat/orkd`**.

## Status snapshot (2026-07-19)

**Done + committed on `feat/orkd` (9 commits local, NOT yet pushed to origin):**
- int8 SDP on the NONBLOCK doorbell — mechanism→engine→multi-core→scheduler (Stages 1–4):
  `6b794f0` (64-B chain align) · `0c823d2` (S1) · `b4fcb1e` (S2 `ork_dyn_begin_seq_i8`) · `2bab9d2` (S3 multi-core groups) · `b80d2ac` (S4 `ork_submit_seq` grouping).
- Multi-consumer orkd validated: `1e5ecd8` (two connections, plain matmul, 6+30 iters) · `38e505c` (two connections, grouped SEQ, 6+20 iters). Both bit-exact, no wedge.
- `3ff4690` — Path-B `group` wire-forwarding (grouped chains route through the daemon). `make test` ALL PASS + attest `f4819a63` + `check-attest` OK.
- `5c0eeee` — Makefile dedup (`i16out_probe`).

Board: RK3588 `10.3.0.236`, healthy/idle. Test tools: `test_orkd_2conn{,_seq}`, `sdp_chain_probe`, `test_orkd_transparent` (routed via `ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd`).

---

## 2026-07-20 — ★ the "multi-domain switch-timeout wedge" was a SELF-INFLICTED CONCURRENCY ARTIFACT (RESOLVED)
Forensics (task-file timestamps + a `load average: 38` snapshot on 07-19, two `make test`s hung 1.5–2.3h):
the wedge clustered in a chaotic session with orphaned/stacked DIRECT-NPU processes. The NPU is single-stream;
my direct-NPU stress harnesses (`dom_switch/pack/race/chain_stress`, all `ork_npu_init()` direct) BYPASS orkd's
serialization, so running them concurrently/orphaned drove concurrent submits → IOMMU wedge (persistent →
poisoned later procs → looked frequent). PROOF it's not a per-domain bug: everything THROUGH orkd (serialized)
is clean — `2conn_seq 8×125k`=1M clean, and **3 concurrent `dom_chain_stress` via orkd = 60k chains, 0 wedges**
(the exact load that hard-wedges DIRECT). ~3.5M direct ops run strictly SEQUENTIALLY today = 0 wedges.
⇒ orkd is fine; its serialization IS the fix. The `db_flying_dom` guard + retirement settle (built for this
phantom) were REVERTED (`9cc2416`), preserved on branch `feat/orkd-guard-settle-stash`. KEPT: `21441d0`
(real domain-propagation fix), features, 1µs timer-slack. LESSON: multi-process NPU work goes THROUGH orkd;
never exercise it with concurrent direct-NPU harnesses. Optional follow-on: advisory flock on the device so a
2nd direct (non-orkd) process fails fast instead of wedging.

## 2026-07-20 — client domains + ring idle-backoff + full SDP routing (commit `3ee7e61`)

Three features landed on `feat/orkd` (local, not pushed):
- **(1) Client-managed IOMMU domains.** SDK `ork_npu_domain_alloc/free` + `ork_npu_set_pack_domain` route to
  orkd (`ORKD_DOM_REQ`/`ORKD_DOM_REL`, coordinated pool); `orkd_pack.domain` carries the client-chosen domain;
  daemon validates it's owned + releases all on drop. `tools/orkd_dom_api.c` (direct + routed). **RPC/SDK layer
  PROVEN** (domains requested + returned; pack carries the id). **End-to-end pack+run bit-exact = NOT yet
  re-validated** because the board's IOMMU switch is currently WEDGED again (see below).
- **(2) A-ring idle-backoff.** Spin only in a hot window (`ORKD_RING_SPIN_MS`, def 3) after last ring activity,
  then poll-timeout `ORKD_RING_BACKOFF_MS` (def 2). VALIDATED bit-exact (sync+pipelined), 95 µs/op preserved.
- **(3) Full SDP-family routing.** `ORKD_SDP` extended to int16 silu/gelu/rsqrt/exp/ewmul/add + i8 exp/rsqrt;
  Path-B routes added; `ORKD_SUITE` gained test_ewmul_i16/test_silu/test_gelu/test_add. **A-SUITE PASSED.**
- `make test` ALL PASS byte-identical at dom0 (Path-B routes fire only when `c->daemon` set), attest refreshed.

**★ RESOLVED — Task 1 end-to-end VALIDATED + drain-before-switch GUARD landed (2026-07-20).**
- **Task 1 end-to-end:** post-reboot, `orkd_dom_api` PASSES bit-exact DIRECT and ROUTED (ORKD_DOM_REQ → pack into
  the client-chosen domain → run → ORKD_DOM_REL). Multi-domain is robust: **~67 000 clean domain switches this
  session** (50k forced blocking `run_i8` alternation + ~16k doorbell SEQ across 8 domains) with ZERO wedges — at
  the ~1/2400 estimate that's ~20 expected, so the *drained* switch path is definitively safe.
- **Root cause (refined + confirmed with the user):** the wedge is NOT the switch path itself — it's the known
  ~1/2000–4000 **multi-core doorbell-MISS** (a nonblock round whose output sentinel never lands = an undrained
  submit) coinciding with a **cross-domain switch**: the kernel's domain-switch idle-wait races the phantom
  submit and times out (reboot-persistent). The blocking path never misses (50k clean); the SEQ path misses ~1/2400
  but its recovery normally clears it before a switch — the wedge is the rare miss-AT-a-domain-boundary coincidence.
- **GUARD (commit pending): drain-before-switch, at `dom_activate` + the doorbell drain paths.** Ctx flag
  `db_flying_dom`: `ork_dyn_end` (mc) and `ork_dyn_seq_end` (SEQ) set it to the round's domain ONLY if the round
  stayed STUCK (recovery exhausted / sentinel never landed), else clear to -1. `dom_activate`, when actually
  switching domains, if a stuck round is flagged, QUIESCES first — ACT_RESET + 1ms settle (mirrors
  `mc_recover_resubmit`) + `dom_cool` (invalidate now-stale warm flags, active + parked) + clear the flag — so the
  kernel switch can't race the phantom. **No-op in normal operation** (clean rounds clear the flag), so it can't
  regress the 67k-clean path. Validated: 20k forced switches with `ORK_DEBUG_RESET` → guard fires 0×; SEQ path
  8×1000 per-client-domains bit-exact.
- **★ CORRECTED by a 1M run: the wedge is a CLEAN-ROUND retirement race, and the fix is an every-switch SETTLE
  (not the stuck-gated guard).** A 1M forced-switch run REPRODUCED the wedge — it hit at ~44–67k switches, and the
  stuck-gated guard fired 0× / didn't prevent it. So the wedge is NOT a stuck round: it's the retirement
  micro-gap — the output sentinel lands (our completion signal) before the kernel RETIRES the task (no userspace
  retirement signal; int_status/dma_rw read 0-always), and the kernel's switch-idle-wait rarely (~1/50k switches,
  measured — NOT the 1/2400 doorbell-miss, a distinct + rarer event) races the un-retired tail → timeout,
  reboot-persistent. **FIX (commit pending): `dom_settle_us()` — a short nanosleep before EVERY cross-domain
  switch in `dom_activate` (default 50µs, `ORK_DOM_SETTLE_US`, 0=off) so the tail retires before the switch.**
  Only fires on a real switch (no cost in single-domain use / `make test`). VALIDATED: **1,000,000 clean switches
  with the settle, zero wedges** (vs wedging at ~44–67k without it — ~15–20× past the failure point); 500k also
  clean. The stuck-gated `db_flying_dom` guard (`4bcd8e8`) is KEPT (no-op on clean paths, still covers the
  distinct stuck-round case) but is NOT what prevents this wedge — the settle is. Not absolute proof (the event is
  stochastic), but 1M-clean-vs-44k-wedge is compelling.

Remaining orkd follow-ons (unchanged): full-suite RPC surface for stream/int4/bmm/ssm; `ork_w_free` daemon-free;
push `feat/orkd`; ggml-ork decode-client wiring.

---

## A. orkd — deeper features (task #8, in progress; core proven, these make it a real runtime)

- [x] **≥3-client deadlock FIXED (`53e46f7`) — the A-sched correctness keystone.** The poll loop serviced clients against a `pfd` array that didn't match the post-`accept()` `nc`: a client accepted this tick sat at an index poll() never filled, so its `revents` was uninitialized stack garbage; a spurious `POLLIN` made the single-threaded daemon `recvmsg`-block on a client that hadn't spoken → whole-loop deadlock (more accepts → more likely; N=4 hung at connect). Fix: snapshot `npoll=nc` before accept, service only `i<npoll`, no `i--` on drop-compaction. **Board-validated N=2/3/4/6 grouped + N=4×30 grouped + N=4×20 ungrouped, all bit-exact, 0 aborts.** orkd multi-consumer now proven well past 2 clients. (This is the bug A-hard #15 deferred here.)
- [x] **Scheduler features — DONE (`12ba587`), except per-client-domains is plumbed-but-not-yet-viable:**
  - [x] **Domain-aware priority queue** — `wk_pick` order = PRIORITY → DOMAIN AFFINITY (prefer the active domain among equal-prio, amortizing the scratch-swap) → FIFO. No-op == proven order when single-domain.
  - [x] **Priority band** — `ork_npu_set_priority()` (public) → per-conn prio → `orkd_run.flags` → `wk_pick`. Default 0. Preemption effect only observable under concurrent queued RUNs (future decode client), not the synchronous tests.
  - [~] **Per-client IOMMU domains — NOT blocked; multi-domain WORKS on the -sram kernel (CORRECTED 2026-07-20).** The plumbing (opt-in `ORKD_PER_CLIENT_DOMAINS`, per-client domain stamped on packs/work, released on drop) + the whole multi-domain code path (per-weight domain threading, `dom_activate` scratch swap, doorbell ops carrying `iommu_domain_id`) are CORRECT and the kernel supports them. **Earlier "kernel broke multi-domain" was WRONG — it was a runtime wedge I induced.** On a clean boot `domain_correct` PASSES (bit-exact 2-domain int8, clean `switch iommu domain 0↔1`, zero timeouts). The `switch iommu domain time out, id:1` failures were a RUNTIME wedge: dmesg showed ~17h clean uptime then timeouts starting only when my testing first hit a non-0 domain — a domain switch attempted while the NPU was busy (undrained `ork_dyn_*` NONBLOCK submit) times out and jams the switch mechanism; a reboot clears it. NO stock-kernel restore needed. **DOMAIN-PROPAGATION FIX LANDED + VALIDATED (2026-07-20):** the standalone SDP/activation leaf helpers (`ork_npu_ewmul_i8/f16/i16`, `add_i8`/`probe_add_i8`, `add_f16`/`ppu_scratch3`, `add_i16`, `probe_silu_std` = all i8 activations via `act_lut_i8`, + `probe_silu_std_f16`'s 2nd submit) hardcoded `iommu_domain_id=-1` (global default 0) for scratch `bcreate` AND `rknpu_submit_ioctl`, while reusing `c->regcmd`/`c->task` that `dom_activate` swaps to `c->dom_active` — so under a non-0 client domain (chained after a matmul that activated dom 1) the SDP submit read its program from dom-1 IOVA under a dom-0 mapping → errno-110 timeout (misreported as `op=ork_dyn_colsplit` since the SDP path doesn't update `g_last_op`). Fix = thread `int dom=c->dom_active` into every scratch+submit in those helpers (the `#35 FIX` idiom already in `probe_silu_std_f16/_i16`); `ppu_scratch3` also got a realloc-on-domain-change guard. Byte-identical at dom 0 (21 `-1`→`dom` + 7 decls). **`test_orkd_2conn_seq 4` w/ `ORKD_PER_CLIENT_DOMAINS=1` PASSES bit-exact — 4 clients each on its own domain; grouped `[mm→ewmul→mm]` now rides one HW `seq-chain` (the domain bug had been forcing SW-fallback).** No iommu-switch-timeout, no wedge. Confirms the queue's domain tracking (`struct work.domain`, `wk_pick` affinity, `g_active_dom`) was already right; this closed the one place the tracked domain wasn't propagated. **Remaining:** the user-requested API surface — orkd op-submit RPC carrying a client-chosen domain id + SDK domain-management APIs (request/release). Pending before commit: full `make test` byte-identical gate + `sbc_attest.txt` refresh (npu.c is an ATTEST_SRC). See [[npu-sram-enabled]].
  - [x] **Fairness quantum** — the contention-adaptive row-slice quantum (v1) is retained and is the right lever; a dedicated dispatch thread was deliberately NOT added (the quantum already bounds socket-I/O latency to one quantum; the NPU is single-stream, so a thread adds concurrency risk with no gain).
  - [x] **≥3-client deadlock FIXED (`53e46f7`)** — see below.
- [x] **Ring is now a SEAMLESS precision-agnostic transport + async API — DONE (`4328f03`).** `ork_mm_run`/`run_i8`/`run_i4` transparently use the ring when attached (any precision, op fits a 64K slot) and fall back to the socket otherwise — no ring-specific calls; gated by `ORK_ORKD_RING` (daemon busy-polls while attached). One precision-agnostic `orkd_ring_submit(...,dtype,...)`/`collect` (daemon's ring_service already dispatched by dtype). Async `ork_mm_submit`/`ork_mm_collect` (public, precision-agnostic, ≤ORKD_RING_SLOTS in flight). Validated: test_orkd_transparent ORK_ORKD_RING=1 bit-exact ALL precisions (small→ring, big→socket fallback), make test PASS (attest f92cfc00), multi-consumer regression PASS. **★HONEST: the async PIPELINE gives ~0 single-client latency gain (probe 101 vs 101 µs/op) — the single-threaded daemon runs ops serially (per-op = daemon NPU+memcpy bound; ~101µs is already ~NPU-bound, 3-core M=1 direct ~93µs), and a lone spin-waiting client has nothing to overlap by queuing ahead. The async API's real value is CPU‖NPU overlap in a REAL decode loop (client does norm/quant/sampling during the NPU op) — not measurable in a spin-only probe — + substrate for a daemon-side doorbell pipeline. The SYNC ring is the real win (~1.6× vs socket).** ★DESIGN: ring = small, individually-issued ops (decode); big ops auto-fall-back (don't fit 64K + transport negligible vs ms compute); CHAINS/SEQ stay on the socket (they batch S ops into 1 round-trip → per-op transport already ~0). Ring and chaining are complementary transport-amortizers.
- [x] **Decode-latency path: shared-memory ring — DONE (`380b03b`).** SPSC shm ring (`src/orkd_ring.h`; 8 slots × 64K) removes the per-op socket round-trip: client writes a request slot + busy-polls; daemon busy-polls the ring (between socket polls) + writes the result in place. Establishment stays on the socket (ORKD_RING_SETUP + SCM_RIGHTS fd). `orkd_ring_setup()` + `orkd_run_i8_ring()` (client); `handle_ring_setup()` + `ring_service()` (daemon). **Board: bit-exact + ~1.9× at M=1 (183→96 µs/op, ~87 µs saved/op, consistent ~80–87 µs across shapes) — recovers the reducible-host overhead ([[submit-floor-decomposed]]).** Multi-consumer regression PASS. Transport-only (npu.c untouched, no attest). FOLLOW-ONS: idle-backoff (the daemon busy-polls a core while a ring is attached); ring paths for fp16/int4/SDP/chain (only int8 wired); wire the ring into the transparent ork_mm_run route (currently an explicit client API + probe); precompiled-cache refs so the daemon skips regcmd synth too (the other half of "per-op cost ≈ 0").
- [x] **Whole-suite conversion (the "first client" milestone) — v1 DONE (`952f1cd`): `make test-orkd`.** Runs the ROUTABLE subset through ONE persistent orkd (each example UNCHANGED, self-validating vs its golden/CPU ref; only the NPU-access path flips to the orkd RPC via `ORK_USE_ORKD=1`), asserts the daemon PID is stable across the run (proves one daemon serviced every test), tears it down after. Board-validated ALL PASS: test_activations · quant (int8 mm golden) · layer/decode/model 1/model 12 (fp16 mm; model 12 = 84 resident weights) · test_ewmul_i8/f16 (SDP), bit-exact/tol-match, pid stable, clean SIGTERM, 0 aborts. **Bumped `ORKD_MAX_WEIGHTS` 64→512** (a model client holds one resident weight per matmul per layer; `ork_w_free` has no ctx so weights reclaim only on socket-EOF → the table must exceed a model's full resident set). **REMAINING for full-suite**: the internal-entrypoint tests stay direct until their entrypoints join the RPC surface — `run_stream_*` (test_matmul/test_stream_interleave/test_affinity), int4 chain/stream/grouped/i4a8 (test_chain_i4/i4/perplexity_i4), bmm+ssm (test_bmm*/test_ssd_chunk_npu/test_mode_transition), int16 SDP + gelu/exp/rsqrt (test_ewmul_i16/test_silu/test_add/test_gelu). Also: `ork_w_free` can't daemon-free (no ctx) — a proper fix (thread ctx or a client-side registry) would let long-running clients recycle weights instead of relying on disconnect-reclaim.
- [x] **Optional hardening — ALL DONE (2026-07-19):**
  - [x] >2 consumers — **the N≥3 hang is FIXED** (`53e46f7`, the poll/`revents` deadlock; see the A-sched keystone above). Board-proven to N=6.
  - [x] **Genuine TWO-PROCESS proof (`orkd_2proc`)** — `fork()`+`exec()`s N *separate* client binaries (default `./test_orkd`) concurrently against one daemon (fresh exec'd image per child → no inherited fork/NPU state, sidesteps the old wedge). Board-validated: 2 AND 3 concurrent separate processes, all bit-exact (socket + zc-A + zc-A+C), every child exit 0, daemon PID stable, clean teardown, 0 aborts. Confirms the earlier `test_orkd_multi` wedge was the fork-then-use-in-child bug, NOT the daemon.
  - [x] **Grouped-SEQ routing CONFIRMED** — pre-started orkd `ORKD_FOREGROUND=1 ORK_SEQ_DEBUG=1`, ran the 2-consumer grouped seq: daemon log shows `6× [seq] grouped run [0,3) ng=1 -> seq-chain`, `0× SW-fallback` — every grouped SEQ routed AS GROUPED on the doorbell (the evidence the empty-log runs lacked).

## B. SDP-doorbell follow-ups (optional extensions) — RECHECKED 2026-07-19

- [x] **More SEQ op-kinds — DONE.** `SEQ_CLASS[]` implements ADD_I8/SILU_I8/GELU_I8/EWMUL_I8/EWMUL_F16/ADD_F16 as SW-break SDP ops (validated by `seqfull`), plus **`ORK_OP_SILU_I16` HW-CHAINED** (`2c2f9b9`).
  - **fp16 SiLU — NOT VIABLE on this NPU (resolved, not a gap).** Confirmed on-board: fused fp16 silu (`ork_mm_run_f16_silu`) gives garbage PPL (9072 vs int8 9.15) and is gated OFF; the standalone fp16 SDP silu (single-LUT recipe AND the vendor 2-stage replay) under-writes elements. So the `SILU_F16` SEQ target was a dead end; the fp16-silu dead-path was backed out.
  - **int16 SiLU HW-CHAINED (`2c2f9b9`) — the accurate, working answer.** Ported the FFN static-graph chain's LUT-in-chain into the generic doorbell seq: `seq_op_ok` accepts `SILU_I16`; `begin_seq_i8_mc` forces single-core + runs a ping-pong-OFF LUT-load prologue (resident curve, one scale/chain); `seq_build_op` emits the `REGCMD_SILU_STD_I16` compute task (atom-8 EWCUBEH, dslot 138) reading the resident LUT; a silu terminal rides the **B2 witness**. Board-validated `seq-grp-silu16 [i8→silu_i16→i8]` DIRECT + ROUTED, max|err|=75 (RKNN-class), make test ALL PASS, 0 wedges. `build_act_lut16` factored out of `act_lut_i16`.
  - [x] **int8 SiLU HW-chained too (`a3b1bd2`).** The clean parallel to int16: `seq_op_ok` accepts `SILU_I8` (N%16); `seq_build_op` emits `REGCMD_SILU_STD` in the atom-16 `ORK_SEQCUBE` cube (int8 de-marshal reused); the silu detection + LUT prologue generalized to track PRECISION (one resident LUT/chain → all silu share kind+scale), curve via `silu_calibrate_idx`+`silu_build_curve` (int8) or `build_act_lut16` (int16). Board: `seq-grp-silu8 [i8→silu_i8→i8]` DIRECT + ROUTED, max|err|=2 (tol 2), make test ALL PASS, 0 wedges. **⇒ BOTH int8 and int16 SiLU now HW-chain in the generic doorbell seq.**
- [x] **Terminal-SDP chains — DONE (`1427d4f`), witness-matmul path.** A grouped seq ending in an SDP op now HW-chains: `ork_submit_seq` splices a tiny lazy-per-ctx int8 witness matmul (K=512,N=16, zeros) as the terminal of any group ending in a non-matmul, so the SDP rides the chain (writes its forward descriptor to the witness) and the witness's int32 sentinel gates completion (output discarded). Yields the proven `mm→…→SDP→mm` shape — NO new hardware surface (the fp16-SDP-inf-poison alternative was rejected: it needs fp16 in the int8-only seq builder + an untested int8→fp16-SDP mixed chain). SW-break fallback on pack-fail/over-cap. Board-validated: `seq-grp-sdpterm [i8→ewmul_i8]` bit-exact DIRECT and ROUTED-through-orkd; `make test` ALL PASS (augmentation gated on a group actually ending in SDP → no regression), attest refreshed, 0 aborts.
- [x] **`chain_progs` retirement — DONE (`caa13d5`).** Migrated `ork_bmm_fp16_fused` (the last PRODUCTION consumer) off `ork_npu_chain_progs` onto `ork_mm_run_stream_f16_chain` (the doorbell fp16 PC-chain). Board-validated: `make test` ALL PASS incl. `TEST_BMM_FUSED` (bit-exact) + `test_ssd_chunk_npu` (SSD-scan critical path) at default AND `ORK_SSM_KEEPWARM=0`/`ORK_MIXED_NOTHRASH=1`, 0 ACT_RESET regression, 0 aborts, attest refreshed. `chain_progs` stays for the 8 tools/ RE probes (they exercise heterogeneous mid-chain SDP/LUT ops the fp16 stream primitives can't express — full symbol removal isn't warranted).
- [x] **`mode_probe` reconfirm — DONE (2026-07-20).** Ran after this session's npu.c edits (scheduler + ring wiring): all 24 op→op transitions `none=ok(0,0)`, **0 wedged submits**, no `-22`/abort. Transition matrix unchanged — the scheduler/ring changes (gated) didn't perturb it.

### Section B disposition (2026-07-20): B4 done; B1/B2/B3 deferred with cause
- **B1 SILU_F16** — deferred: no standalone `ork_npu_silu_f16` SDP primitive exists (only fused `ork_mm_run_f16_silu` + a probe), so it's a NEW fp16-SiLU RE task, not LUT plumbing; no consumer (int8/int16 SiLU cover production). Low value / not-quick.
- **B2 terminal-SDP** — deferred/not-yet-viable: the fraught int8-SDP-doorbell-chaining wall (SDP output has no free poison sentinel; 3 prior board experiments hung). SW-break already runs SDP correctly. Needs the SDP-aware-chain breakthrough first.
- **B3 chain_progs retirement** — deferred: `bmm_fp16_fused` chains via `chain_progs` on the SSD-scan critical path; migrating it is a risky refactor with cleanup-only value (no perf/functional gain).

## C. Cleanup / backlog

- [x] **`$(CORE)`→`$(COBJ)` tool mislink sweep — DONE (`ab9c760`, + `mc_prof` in `738500b`).** Flipped all ~100 tool/test targets from `$(CORE)` to `$(COBJ)`; preserved the 2 non-tool uses (COBJ def + ATTEST_SRCS). Board-verified the previously-broken tools now link (softmax_replay, chain_gatesilu_probe, perchan_bench, mm_perchan_f16_diag_probe, chain_probe, …) + the `-fopenmp`/`-march=native`/pthread ones + a broad sampling. Makefile-only, no attest change.
- [x] **Task #4 — int4 batch/grouped on the spine — DONE (A1+A2+B all on the doorbell; commits c772d62/a932117/12f52ef). LOW VALUE, behind the W4A4-viability wall — architectural completeness.**
  - DONE (on the doorbell): int4 decode M=1 (`begin_mc_i4`, per-row); int4 batch M≥2 **single-slice** (`Sk==1 && Sn==1`) — `run_i4_mc_db` decomposes M rows into M single-row doorbell tasks, DEFAULT-ON (`npu.c:3126`, `ORK_I4_NODB=1` reverts).
  - [x] **A1 — N-tile `Sn>1` (wide-N) — DONE + validated (bit-exact, pending make-test attest+commit).** `begin_mc_i4` now emits `Sn` chained column-slice programs per M=1 row-task (each writes its `[Nc]` int16 at its column offset in the row's `[N]` int16 output → drain widens the full row unchanged; still raw int32 accumulator, no per-group scale). Gate `ork_mm_run_i4` relaxed to `Sk==1` (drop `Sn==1`); seq int4 already permitted Sn>1. Validated `tools/i4_sn_probe`: Sn=1/2/3 (N up to 24576) bit-exact vs CPU AND vs `ORK_I4_NODB=1` blocking ref, 0 wedges. (Falls back to `run_i4_mc` if the chain exceeds the per-core buffers.)
  - [x] **A2 — K-split `Sk>1` (wide-K, K>ORK_I4_KS=10752) — DONE + validated (pending make-test attest+commit).** CORRECTION to the earlier framing: the PLAIN `ork_mm_run_i4` (int32 C) K-split is **int-accumulate**, NOT float-grouped — `i4_mcworker` int16→int32 widen-adds the Sk partials (the float+per-group-scale path is `i4_mcworker_g`/`ork_mm_run_i4_grouped` = item B, separate). So A2 = int16-partial-sum K-split, like int8's `begin_mc` K-split via `oSk`. `begin_mc_i4` now emits Sn*Sk programs/row (per-K-slice A re-tile via `tile_i4_Aslice(k0,Kp)`, weight `Bb[ns*Sk+ks]`, Sk blocks of [N] int16); `ork_dyn_end`'s `oSk>1` sum made esz-aware (int16 for int4, int8 path byte-identical). Sk<=16. Validated `i4_sn_probe`: Sk=2/3 (K=21504/32256) + combined Sk=2×Sn=2, bit-exact vs CPU (small values) AND vs blocking ref, 0 wedges. CAVEAT: each 10752-wide K-slice int16 partial overflows for adversarial data — same limit as `run_i4_mc`, so A2 is parity-with-the-blocking-path, not a new capability.
  - [x] **B — Grouped int4 `ork_mm_run_i4_grouped` — DONE + validated (pending make-test attest+commit).** The genuinely FLOAT-grouped path. Built ISOLATED: dedicated `ork_dyn_begin_mc_i4_grouped` (row-decomposed across cores, K-slice = gsize G, Sn*Sk programs/row, Sk int16 partial blocks) + dedicated `ork_dyn_grouped_end` (FLOAT scale-accumulate `C[m][n]=Σ_g aS[m*Sk+g]·bS[g*N+n]·partial_g[n]`). Shared `ork_dyn_end` UNTOUCHED (new `i4g*` chain fields read only by the grouped drain). `ork_mm_run_i4_grouped` routes by default (`ORK_I4G_NODB=1` reverts; NULL/over-large → blocking `i4_mcworker_g`). Sk<=256. Validated `examples/i4` gtest (in make test): M=1/4/8, Sk=16/32, G=64/128 all dequant maxerr=0.0000 EXACT via doorbell AND blocking control, 0 wedges.
  - **⇒ TASK #4 FULLY DONE** (decode + batch + A1 N-tile + A2 K-split + B grouped, all on the doorbell). Still architectural completeness behind the W4A4-not-viable wall (no consumer).
  - Experimental batch variants (`ORK_I4_INCR` / `ORK_I4_BCHAIN` / `ORK_I4_CBATCH`) are opt-in research alternatives, not blockers.
  - CAVEAT: W4A4 int4 is **not-yet-viable on real models** ([[npu-int4-datapath-symmetric]], [[int4-hadamard]]) — no int4 config both fast & coherent today. So the remaining migration is architectural completeness, not a perf/quality lever; the common int4 path already rides the spine.
- [ ] **Push `feat/orkd` to origin** — 9 commits local only.
