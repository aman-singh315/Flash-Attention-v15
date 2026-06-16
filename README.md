# FlashAttention CUDA — from scratch

Written from scratch in CUDA C++. No libraries, no Triton, no PyTorch. Just a kernel that implements the full Flash Attention algorithm with warp-level primitives, built iteratively across 15 versions.

---

## Numbers

```
Sequence Length:      8192 tokens
Head Dimension:       64
Precision:            float32

Time:                 15.40 ms
GFLOPS:               1115.69
Memory Bandwidth:     52.30 GB/s
Arithmetic Intensity: 21.33 FLOP/byte
```

---

## What's in the kernel

| Technique | Details |
|---|---|
| **Warp-per-row** | Each warp handles one query row. Each lane owns 2 dimensions, mapping D=64 perfectly across 32 threads |
| **Shared memory tiling** | K and V staged in `[32×64]` tiles (8 KB each, 16 KB total). Every value read from HBM exactly once per tile |
| **`float4` vectorized loads** | 128-bit coalesced transactions for cooperative K/V loading — 4× fewer load instructions |
| **Register-cached Q** | `qi0`, `qi1` loaded once into registers and reused across all tiles and all j-iterations |
| **4× unrolled inner loop** | Four rows of K/V pulled into registers per iteration, exposing instruction-level parallelism and hiding shared memory latency |
| **Packed `uint64` warp shuffle** | Two floats packed into a `uint64_t` via a union and shuffled together — halves `__shfl_down_sync` count for dot-product reduction |
| **Online softmax** | Running `(m, l)` updated per tile. Accumulator rescaled by `exp(m_old - m_new)` before adding new terms. N×N matrix never written to memory |
| **`exp2f` + LOG2E** | `exp2f(x * log2e)` instead of `expf(x)` — maps to a single `EX2` hardware instruction |
| **`__launch_bounds__(256, 3)`** | Bounds hint lets `nvcc` make tighter register allocation decisions, improving occupancy |
| **Tail loop** | Remainder rows that don't fill a group of 4 handled separately, keeping the hot path branch-free |

---

## The algorithm

Naive attention writes the full N×N score matrix to HBM. At N=8192 that's 256 MB of intermediate storage, read and written every forward pass.

Flash Attention tiles K and V so they stay in shared memory and computes softmax incrementally using a running max and sum. Each tile, the output accumulator is corrected by a rescaling factor before new terms are added. The result is numerically identical to standard attention, with O(N) memory instead of O(N²).

---

## How it got here

The kernel didn't start here. Each version introduced one change:

- **v3** — shared memory tiling. Stopped reloading K/V from HBM on every query row
- **v8** — online softmax. Earlier versions had a bug where rescaling was applied after the tile instead of per-row
- **v10** — 4× unrolling. Biggest single jump in GFLOPS
- **v12** — packed uint64 shuffle. Fewer instructions, cleaner code
- **v14** — switched to `exp2f`. Small but consistent gain
- **v15** — `__launch_bounds__`, tail loop cleanup

Plenty of versions in between were dead ends — wider tiles, different thread layouts, async copies. Some made things worse.

---

## What's next

- `cp.async` for double-buffered tile loading
- FP16 / BF16 precision
- Tensor Core path (WMMA or PTX)
- FlashAttention-2 style work partitioning across warps
- Persistent thread blocks for long sequences

---

## Build

```bash
nvcc -O3 --use_fast_math -arch=sm_86 flash_15.cu -o flash_attn
./flash_attn
```

Replace `sm_86` with your GPU's compute capability. Requires CUDA 11+, Volta or newer.

---

## References

- Dao et al., *FlashAttention*, NeurIPS 2022  
- Dao et al., *FlashAttention-2*, ICLR 2024  
- NVIDIA CUDA C++ Programming Guide
