# Chain assembler — SUPERSEDED (historical)

> **⚠️ SUPERSEDED.** The `chain_progs` + `set_i16_out` PC-chain approach described in the historical section
> below was **abandoned**. It never worked end-to-end (the data-connected matmul→silu chain hung: task0 didn't
> execute; the standalone `set_i16_out` matmul wedges the NPU with a blocking submit). **Do not build on it,
> do not run its probes** (`chain_gatesilu_probe`, `chain_probe`, `i16out_probe`, `chain_selftest`) — they
> wedge the NPU.
>
> **The working chain assembler is the NONBLOCK DOORBELL SEQ** (`ork_submit_seq` / `ork_dyn_begin_seq_i8[_mc]`):
> it chains matmul + SDP (silu / ewmul / add) as one self-healing doorbell submit, int8 AND int16, multi-core
> grouped, and is already routed through orkd (`ORKD_SEQ` → `handle_seq` → `ork_submit_seq`). See commits
> `ace4bc3` (reusable seq engine), `21718e2` (multi-core grouped), `7400024` (grouped chains → doorbell),
> `87b6c09`/`2c77ef2` (int16+int8 SiLU HW-chained in the seq), `f97db3f` (fused chain through orkd).
>
> **ggml-ork** has a full `ORK_FFN_CHAIN` SwiGLU handler (keeps int8 intermediates on-NPU) but it's disabled
> under orkd (fd-local ops). **The open work (task #20)** = route the FFN SwiGLU chain through orkd's
> `ORKD_SEQ` / `ork_submit_seq` so the whole FFN inner is one coalesced on-NPU chain under the daemon.

---

## Historical reference — the decoded vendor multi-task chain mechanism (still valid, still useful)

The one durably useful result from this line of work: the RK3588 multi-task chain mechanism was decoded from
the captured 9-task softmax graph (`regcmd_softmax_f16.h`) and **replays bit-exact on-board** (single submit,
`task_number=9`, ping-pong OFF, rc=0). Mechanism:

**Pure ADDRESS ALIASING through resident scratch buffers.** A task writes its output to a scratch IOVA
(0x4020 for SDP/DPU out; CNA output for the 0xd matmul); a downstream task names that SAME IOVA as its input
(0x5018/0x5038 for SDP, 0x1070 for CNA feature). There is NO forwarding datapath and NO chain-output register
— the PC descriptor (0x0010/0x0014) only sequences WHICH regcmd runs next; data flows through shared addresses
in a persistent image with NO act(RESET) between tasks. It's a DAG (a task can feed two consumers), not a
linear pipe. `matmul(0xd) → silu(0x18)` is proven by t2→t3 of the replay.

Working recipe: ONE contiguous IOVA image (tasks + data + weights), tasks at 64-byte-aligned slots, the
0x0014 next-amount recomputed as `(next_regcfg_amount + 3) / 2`, `flags=0x1` (ping-pong OFF), `task_number=N`,
`subcore_task[0]={0,N}`, one submit. (The doorbell seq applies the same aliasing idea via `ork_dyn_begin_seq_i8`.)

Also decoded (may be useful for the coalesced FFN chain): the residual can be FUSED into the matmul's SDP
output stage (int32 accumulator + residual add BEFORE the int16/fp16 cast — the NVDLA SDP element-wise/bias
sub-unit, operand regs in the 0x40xx block), removing the CPU residual seam so a whole block ends with a
fused-residual output.
