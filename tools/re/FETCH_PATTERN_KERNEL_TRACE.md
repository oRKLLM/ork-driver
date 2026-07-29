# Fetch-Pattern RE — Kernel DMA-Address Tracing Plan (the single remaining forward route)

**Status:** PARKED / forward-plan. All userspace probe avenues are exhausted (see "Why" below). This is the
only route left to recover the NPU's actual CDMA fetch pattern. It is a deliberate kernel-side project on a
clean board — **not** a userspace probe. Resume from this document.

## Goal

Recover the RK3588 NPU's **actual CDMA fetch address sequence** for a *single* matmul submit — the ordered list
of DRAM/IOVA addresses the feature-DMA reads — so we can:
1. **Crack the mfold input layout** (`(m,k) → IOVA` map) that userspace probing cannot, and
2. **Generalize fetch-pattern optimizations** (contiguity, weight-reuse, tiling order) from ground-truth HW behavior.

## Why kernel tracing is the ONLY route left (userspace is exhausted)

Recorded across the 2026-07-28/29 sessions (see memory `mfold-0x1080-wedge`):

| userspace approach | blocker |
|---|---|
| on-board mfold layout sweep (any input pack) | mfold synth wedges after **~1 submit/boot** (state accumulation) |
| value probe (K-window mask / one-hot) | same synth → wedge on any multi-submit sweep |
| offline `cdma_calib` enumerator | **cannot discriminate** — every bijection is a valid matmul in sim |
| timing probe (fetch order via A-writes) | **no host-writable in-place A path**: blocking+thread → SIGBUS (bsync race); doorbell+zero-copy-A → SIGBUS at M>1; doorbell+host-A → memcpy-snapshot (writes invisible) |

**The key property that makes kernel tracing viable:** it reads the pattern **passively from the hardware in
ONE submit**. That sidesteps *both* walls — no repeated mfold submits (stays inside the 1-submit/boot safe
envelope), and no userspace buffer writes (no bsync/zero-copy SIGBUS).

## Approach — IOMMU/SMMU page-fault tracing (primary)

The only component that *sees the addresses* is the IOMMU/SMMU (the NPU does the DMA; the driver only programs
the regcmd; the CPU never sees the fetches). So instrument the translation path:

1. **Protect the A (feature) buffer's IOVA pages** — map them fault-on-access (present=0 / read-protected) for
   the NPU's stream ID, for one flagged submit.
2. **Log in the fault handler** — on each translation fault: record `(faulting IOVA, monotonic timestamp,
   stream id)`, then **fix up** (map the page, resume the DMA) — a "single-step DMA trace". The faulting-IOVA
   sequence *is* the fetch order at **page (4 KiB) granularity**.
3. **Decode** `IOVA → logical (m,k)` using the buffer base + known dims → the layout + order.

Page granularity (4 KiB) is coarser than atom-level but resolves **surface/tile order**, which is exactly the
layout-family ambiguity `cdma_calib` bounded but couldn't disambiguate. For finer resolution, shrink the traced
region (protect only a few pages spanning the boundary of interest).

### Calibrate, then apply
- **First run it on the NORMAL int8 matmul** (known layout = NC1HWC2 C2=16, `nc_off`) to validate the
  IOVA→(m,k) decoder against ground truth. Normal matmul is stable → safe to iterate.
- **Then one mfold submit** (`ORK_MFOLD=1` synth) — a *single* traced submit is within the 1-submit/boot envelope.

## Concrete steps

1. **Locate the driver + fault path.** rknpu DRM driver (`drivers/rknpu`) or mainline `drivers/accel/rocket`;
   find the IOMMU fault handler (rknpu prints `PC_INTERRUPT_RAW_STATUS_DMA_READ_ERROR` / "cdma address wild"
   today — that path already receives a faulting address; we make it **log+resume** instead of treating it as fatal).
2. **Add a debug trace mode** (module param / debugfs): for a buffer tagged by IOVA range, install fault-on-access
   PTEs, and in the handler append `(iova, ktime, sid)` to a ring buffer exposed via debugfs; fix up + resume.
3. **Userspace driver hook:** one env/ioctl to tag the A buffer's IOVA range for tracing before a submit
   (ork-driver side: expose the A `dma`/`obj` + size).
4. **Run** one known-A/W matmul (normal, then mfold), read the debugfs trace, decode.
5. **Analyze** with the existing offline harness (`cdma_calib` already models IOVA offsets → feed the traced
   sequence in and match the (m,k) map).

## Risks / prerequisites (the real cost)

- **Fault-handler resume is the main engineering risk.** rknpu currently treats a DMA read fault as fatal
   (wedge). Making it log-and-resume without wedging the NPU is the crux — may need the SMMU stall-and-resume
   model (arm-smmu). If the HW can't resume a faulted CDMA read, fall back to **PMU/event-queue** tracing
   (SMMU translation events for the NPU stream, no fault injection) — lower resolution, no resume needed.
- **Kernel build + module reload access** on the RK3588 board (heavier ops than userspace; reboot-on-wedge).
- **Timing perturbation** — faulting changes timing, but we want *order*, not absolute latency, so this is fine.
- **Board discipline** — kernel changes can hard-wedge (SPI-reflash risk per AGENTS); test on the validated
   `10.3.0.236` with the recovery ladder ready.

## Lighter first probe (before full fault-tracing)

Check what the **existing** rknpu fault path already prints: a deliberately-OOB single access makes the CDMA
fault and dmesg logs the faulting IOVA ("cdma address wild"). That yields *one* address, not the sequence — but
it's a zero-kernel-change way to confirm the IOVA→address decode and the fault-handler's logging format before
investing in the resume path.

## Success criteria

Recover the `(m,k) → IOVA` fetch map (+ order) for one submit; validate it reproduces the known C2=16 normal-matmul
layout; then apply to the mfold synth to settle the input layout that userspace probing could not.

## Artifacts in hand (resume context)

- `tools/re/cdma_calib.c` — offline CDMA byte-address model + enumerator (silicon-anchored on the standard
  layout; bounds the mfold family to ~22 OOB-safe candidates; can consume a traced sequence to match the map).
- `tools/re/timing_probe.c` / `timing_probe2.c` — the two SIGBUS'd timing attempts (kept as negative record).
- `tools/re/validate_mfold*.c` — single-submit bit-exact validator (known A/W, dumps raw Cout) + input-pack sweep.
- Board WIP (contiguous-synth experiment) preserved in git stash on the board tree.
- Full reconciliation + all four userspace dead-ends: memory `mfold-0x1080-wedge`.

## Strategic note

ork is already **~90–110% of rkllm on 7B prefill without the fold**; this pursues the ~220→352 GMAC/s kernel gap.
Reopen only if that kernel-throughput gap becomes worth a multi-day kernel-tracing effort.
