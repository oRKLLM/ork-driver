# SBC Performance Wiki: NVMe Storage & PCIe Bandwidth Optimization

This document records the reverse-engineering analysis, diagnostic methodology, and kernel-level performance tuning applied to the **Radxa ROCK 5B (RK3588)** storage interface. It outlines how to maximize file reading and model weight loading throughput for large language models (like GGUF files) to eliminate I/O-bound startup and execution delays.

---

## 1. Hardware & Physical Link Introspection

The Radxa ROCK 5B features an M.2 M-key connector supporting PCIe Gen 3 x4 lanes routed directly from the RK3588 SoC.

To verify that your physical hardware link is fully negotiated without degradation (which can be caused by sub-par NVMe adapters, poor contact, or thermal limitations), you can query the Linux kernel `/sys` filesystem directly:

```bash
# Query negotiated PCIe link speed
cat /sys/devices/platform/fe150000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/current_link_speed

# Query negotiated PCIe link width
cat /sys/devices/platform/fe150000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/current_link_width
```

### Physical Limits & Profiles:
* **Target Speed:** `8.0 GT/s PCIe` (Gen 3)
* **Target Width:** `4` (x4 lanes)
* **Maximum Theoretical Bus Bandwidth:**
  $$\text{Throughput} = 4 \text{ lanes} \times 8.0 \text{ GT/s} \times \frac{128\text{b}}{130\text{b}} \approx 3.938 \text{ GB/s}$$
* **Real-world maximum sequential read (with protocol overhead):** **~3.2 to 3.4 GB/s**

---

## 2. The OS Kernel Bottleneck: Sequential Read-Ahead

When `llama.cpp` maps a model weight file using virtual memory (`mmap`), the weights are not read into physical memory at startup. Instead, page faults are triggered on-demand as threads read the mapped memory addresses during model initialization or first-token prefill.

By default, Linux distributions on Rockchip SBCs (like Radxa Debian or Armbian) set the block device sequential read-ahead buffer to a tiny default value:

* **Default Read-Ahead:** `/sys/block/nvme0n1/queue/read_ahead_kb` = **`128` (128 KB)**

### The Throttling Mechanism:
1. **Queue Depth Serialization (QD=1):** With a read-ahead size of only 128KB, the OS kernel block layer requests only tiny sequential slices from storage per page fault. This prevents the NVMe controller from scheduling parallel operations across its flash NAND channels, effectively pinning NVMe operations to a serial Queue Depth of 1.
2. **Synchronous Page-Fault Overhead:** Handling page-table traps sequentially on a single Cortex-A76 core saturates the CPU thread handling the load, making CPU processing speeds (not NVMe hardware) the ultimate bottleneck.
3. **The Symptom:** When loading weights, the NVMe SSD peaks at ~2 GB/s only for a brief moment and averages much lower (often 400–800 MB/s), failing to saturate the PCIe Gen 3 x4 link.

---

## 3. High-Leverage Kernel Tuning (The Solution)

To bypass the page-fault bottleneck and leverage the NVMe controller's internal channel parallelism, we must scale up the read-ahead size. This prompts the Linux kernel to issue large, pipelined, asynchronous read operations ahead of the CPU page-fault trap, keeping NVMe queue depths saturated and hiding I/O latency.

### The Optimization Command:
Run the following command on the target board with root privileges to set the NVMe block device read-ahead size to **4 MB (4096 KB)**:

```bash
echo 4096 | sudo tee /sys/block/nvme0n1/queue/read_ahead_kb
```

### Performance Impact:
* **Read-ahead size:** Increased from 128 KB to 4096 KB (32× increase).
* **I/O Pipelining:** Enables deep queue depths (QD=16 to QD=32) at the kernel driver layer.
* **Loading Speedup:** Sequential reads of massive GGUF weight files will now sustain **3.2+ GB/s**, matching the absolute physical limit of the PCIe Gen 3 x4 bus and significantly accelerating cold-start and prefetch phases.

---

## 4. Complementary OS Tuning Checklist

To ensure no other system governors choke storage throughput, confirm the following parameters:

1. **CPU Scaling Governor:** Ensure your performance cores are pinned to maximum clock speed during load. This avoids CPU-bound deserialization bottlenecks:
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
   ```
2. **I/O Scheduler:** Verify that the NVMe I/O scheduler is configured to `none` (or `kyber`) to avoid unnecessary CPU-bound sorting overhead on high-speed NVMe block devices:
   ```bash
   cat /sys/block/nvme0n1/queue/scheduler
   # Should output: [none] mq-deadline kyber
   ```
