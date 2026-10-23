# HIP VGPR spill gate

The gfx1151 gate bounds known compiler resource debt per kernel. A passing gate means that the pinned toolchain did not exceed those bounds. It does not mean that every kernel is spill-free or that a spilling kernel has acceptable runtime performance.

## Metric definitions and architecture

`VGPRs` / `.vgpr_count` is the allocated register count per lane. `VGPRs Spill` / `.vgpr_spill_count` counts 32-bit components of static spill-save operations. It can count repeated stores and reused scratch slots. `ScratchSize [bytes/lane]` / `.private_segment_fixed_size` is a separate byte count. Adding allocated VGPRs to spill saves does not give simultaneous register demand.

For the worst Q2_K routed-compact kernel in the recorded Clang 24 build, final assembly contains 256 allocated VGPRs, 1,036 spill-save components and 3,392 scratch bytes per lane. Its 460 `scratch_store_b32`, 48 `scratch_store_b64` and 120 `scratch_store_b128` instructions independently account for exactly 1,036 stored dwords. The next Q2_K case has 256 VGPRs, 766 spill saves and 2,732 scratch bytes. These are real spills.

gfx1151 can address 256 VGPRs per lane, or 32 KiB per wave32. It has 1,536 physical wave32 VGPRs per SIMD, allocated in groups of 24, shared among resident waves. The physical pool is not a per-wave allowance. The original gate targeted gfx908/CDNA; commit `7d4dd51a76705f9bc42e14ea0920c2c122480ed1` changed the target to gfx1151 without recalibrating the `VGPRs + Spill > 256` heuristic or its ignore list.

Primary sources for the pinned Clang 24 revision:

- [Spill-save counter increment](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Target/AMDGPU/SIRegisterInfo.cpp#L2749).
- [Addressable registers, physical pool and allocation granularity](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1373).
- [gfx1151 physical-register feature](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Target/AMDGPU/AMDGPU.td#L2194).

## Same-source toolchain comparison

Both images compiled source `05dc1d444842cf99ef061bc0a75a3f3bfeb4b629` in separate builds on 2026-10-23, with Release, HIP, gfx1151 and `GGML_HIP_EXPORT_METRICS=On`. Both complete `ggml-hip` builds passed. No GPU was attached. The images also contain different HIP/rocPRIM headers, so this comparison measures the toolchains as shipped rather than isolating every difference to LLVM alone.

| Metric | ROCm 7.2.1 / Clang 22 | ROCm 10.1 / Clang 24 |
| --- | ---: | ---: |
| Unique kernel symbols in final assembly | 8,599 | 9,143 |
| Symbols with VGPR spill saves | 364 | 257 |
| Historical checker, unignored reports | 339 | 238 |
| Worst Clang 24 Q2_K case: VGPRs / spill saves / scratch bytes | 172 / 0 / 0 | 256 / 1036 / 3392 |
| Second Q2_K case | 147 / 0 / 0 | 256 / 766 / 2732 |

All 8,599 Clang 22 symbols also occur in Clang 24. Among them, 97 spill-save counts increase, including 25 newly spilling symbols; 285 decrease, including 132 that stop spilling. Seven current spilling symbols have equal spill-save counts. The additional 544 Clang 24 symbols do not spill. This is a mixture of existing debt, improvements and toolchain regressions. It is not evidence that the source changes introduced all 238 reports. Two independent Clang 24 builds agree on every kernel's VGPR, spill-save and scratch count.

The location-based remark parser also disagreed with final assembly for 12 Clang 22 symbols and one Clang 24 symbol in the new parallel builds. The gate therefore reads final `*.lto.s` metadata directly, without source-location association or a YAML dependency.

## Baseline policy

`gfx1151-vgpr-baseline.json` records the exact compiler identity, image digest, source revision and minimum kernel coverage for each translation unit. Each of the 257 spilling symbols has its own VGPR, spill-save and scratch-byte ceiling, its Clang 22 comparison in that order, and a reason. The 97 increased counts, including the two severe Q2_K regressions, are explicit compiler-migration debt. Runtime cost and kernel tuning remain unqualified; the baseline does not assert that the regressions are harmless.

New spilling symbols fail, including small spills that the old sum threshold missed. Existing symbols fail if any recorded ceiling increases; a smaller value in one metric cannot offset an increase in another. Values above the architectural 256-VGPR limit fail. Wrong compilers or architectures, incomplete metadata, missing baseline symbols and reduced translation-unit coverage fail. Repeated symbols are checked using the maximum of each metric across emitted copies. No wildcard or unbounded ignore remains.

To change the baseline, compile the proposed source with the pinned image and flags, compare final metadata against the recorded source and toolchain, and explain each changed ceiling. Deleting a kernel or changing the compiler also requires a reviewed baseline update. Do not generate or enlarge the baseline automatically during CI. Prefer reducing ceilings after measured improvements. Kernel changes require the repository's hardware correctness and performance flow.

Run the CPU regression fixtures with `python3 scripts/hip/test_vgpr_check.py`. The workflow supplies the HIP assembly directory, baseline and exact compiler version to `gcn-cdna-vgpr-check.py`; the raw metrics log remains available for diagnosis.
