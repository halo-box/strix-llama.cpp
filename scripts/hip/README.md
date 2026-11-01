# HIP VGPR and SGPR spill gate

The gfx1151 gate requires zero VGPR-spilling and zero SGPR-spilling kernels with the pinned ROCm 10.1 / AMD Clang 24 compiler. It reads final `*.lto.s` code-object metadata, checks every emitted kernel and takes the maximum of each resource metric across repeated names. No kernel in the tracked baseline has a VGPR spill exception; ten flash-attention kernels have bounded SGPR spill budgets (see below).

## Metric definitions and architecture

`VGPRs` / `.vgpr_count` is the allocated register count per lane. `VGPRs Spill` / `.vgpr_spill_count` counts 32-bit components of static spill-save operations, including repeated stores and reused scratch slots. `SGPRs Spill` / `.sgpr_spill_count` counts SGPR spill saves; on gfx1151 they go to VGPR lanes while a lane is free, so they need not add private bytes. `ScratchSize [bytes/lane]` / `.private_segment_fixed_size` is a separate byte count. Adding allocated VGPRs to spill saves does not give simultaneous register demand.

gfx1151 can address 256 VGPRs per lane, or 32 KiB per wave32. It has 1,536 physical wave32 VGPRs per SIMD, allocated in groups of 24, shared among resident waves. The physical pool is not a per-wave allowance. Any emitted kernel above 256 allocated VGPRs fails independently of the spill check.

Primary sources for the pinned Clang 24 revision:

- [Spill-save counter increment](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Target/AMDGPU/SIRegisterInfo.cpp#L2749).
- [Addressable registers, physical pool and allocation granularity](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1373).
- [gfx1151 physical-register feature](https://github.com/ROCm/llvm-project/blob/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/lib/Target/AMDGPU/AMDGPU.td#L2194).

## Zero-spill contract and coverage

`gfx1151-vgpr-baseline.json` records the schema, architecture, exact compiler identity, image digest, source revision, the minimum total unique kernel count and minimum emitted kernel count for each translation unit. The current inventory has 6,274 unique names, 6,614 TU/name copies and 154 nonempty translation units. `minimum_kernel_count` requires at least 6,274 unique names, independently of the per-TU counts. The coverage map requires at least those 6,614 emitted copies in their recorded TUs; an empty inventory or a missing or reduced TU fails. Coverage counts must be positive integers and the baseline coverage map must be nonempty. An empty or absent `kernels` map means that any positive `.vgpr_spill_count` or `.sgpr_spill_count` fails, including a single spill save.

The compiler identity must match exactly. Every input assembly file must target gfx1151, every kernel metadata record must include all resource fields and wave32, and duplicate TU inventories or malformed metadata fail. The gate checks the exact emitted inventory, including stubs and copies outside runtime-selected paths. It does not filter names by dispatch reachability. Missing input, a parser that matches nothing, and incomplete builds cannot pass using an empty exception map.

Private-segment bytes are no longer governed by an allowlist. Five Q1_0 MMQ bodies retain 16 private bytes per lane with zero VGPR spill saves. This is a known observation, not an exception or a scratch-byte ceiling. A zero-spill result does not require zero private-segment bytes, and it does not establish numerical correctness, occupancy or runtime performance. Kernel changes still require the repository's hardware correctness and performance flow.

The checker retains support for explicit legacy budgets so its existing regression fixtures and historical baselines can still be read; a budget without `sgpr_spills` allows no SGPR spill. The tracked gfx1151 baseline has no VGPR budget and ten SGPR budgets. Three (reason `sgpr_lane_spill`) cover flash_attn_ext_f16 D64 4x8 and softcap D128 16x2 / 32x2, whose spill-free variants changed the output bytes or used scratch. Seven (reason `sgpr_lane_spill_slower`) cover flash_attn_ext_f16 D64 16x2, D128 32x2, softcap D128 4x8 / 8x4 and softcap D256 4x8 / 8x4, and flash_attn_mask_to_KV_max<64>, whose spill-free variants measured slower.

## Regenerate the coverage map

Build the complete `ggml-hip` target from the proposed source with the pinned image, Release, HIP, `GPU_TARGETS=gfx1151` and `GGML_HIP_EXPORT_METRICS=On`. Use a clean build directory so stale assembly cannot supply coverage for a missing TU. Pass the directory containing all emitted `*.lto.s` files to the checker. Depending on the generator, those files can be in the build root or `ggml/src/ggml-hip`.

After reviewing the inventory and confirming zero spills, regenerate the coverage map, total minimum and source revision:

```sh
python3 - BUILD_ASSEMBLY_DIRECTORY <<'PYTHON'
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

spec = importlib.util.spec_from_file_location('vgpr_check', 'scripts/hip/gcn-cdna-vgpr-check.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
path = Path('scripts/hip/gfx1151-vgpr-baseline.json')
baseline = json.loads(path.read_text())
kernels, units = checker.read_assembly(Path(sys.argv[1]), baseline['architecture'])
assert kernels and units and not any(k['spill_stores'] for k in kernels.values())
assert all(not k['sgpr_spills'] or name in baseline.get('kernels', {}) for name, k in kernels.items())
assert all(k['vgprs'] <= 256 for k in kernels.values())
baseline['translation_units'] = units
baseline['minimum_kernel_count'] = len(kernels)
baseline['source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
path.write_text(json.dumps(baseline, indent=2) + '\n')
PYTHON
```

Compare the new TU/name inventory against the recorded source before accepting reduced coverage. Keep the exact compiler and image identity unless a toolchain migration has been separately reviewed. Never regenerate coverage automatically in CI.

Run the CPU regression fixtures with `python3 scripts/hip/test_vgpr_check.py`. The existing workflow passes the assembly directory, `--baseline` and exact `--compiler-version` to `gcn-cdna-vgpr-check.py`; its job shape is unchanged and its raw metrics log remains available for diagnosis.

## Historical parser failure

The original remark parser required a `remark:` prefix that Clang 24 omitted, so it could pass with zero parsed symbols. Source-location association also confused template instantiations. The gate now reads final assembly metadata directly and requires nonempty per-TU coverage. Historical compiler-migration budgets have been retired after the spill-elimination campaign; they are not evidence of an allowance in the current contract.
