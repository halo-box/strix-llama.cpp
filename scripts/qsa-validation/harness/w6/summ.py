#!/usr/bin/env python3
# W6: writes summaries/w6_tests.md and summaries/w6_time.md from the logs
import os,re
L=os.path.expanduser('~/dev/wt_rewrite_logs/W6/'); O=os.path.expanduser('~/dev/wt_rewrite/scripts/qsa-validation/summaries/')
r=lambda f: open(L+f,errors='replace').read()
t=['# W6 tests (HEAD b1cb6764c; logs strix-002:~/dev/wt_rewrite_logs/W6)','',
'Every run one op, -j 1, under flock. `--fail-unsupported` on every Vulkan run of the two ops: a "not supported" outside the list (group 4; K/V bigger than the binding limit) is a FAIL. "Ratio" = NMSE / band of the census class (§14.2 (c): G1 row of the split class; no row measured on a Vulkan tier).','',
'Devices: RADV = Radeon 8060S, Mesa 26.0.3; lavapipe = llvmpipe Mesa 25.3.6, LLVM 21.1.8, 256 bits (Fedora 43 packages extracted to wt_lab/w6/lvp43; the system lavapipe crashes, see below).','',
'## Op runs (`runs1.out`, `attn_report.txt`)','','```',r('runs1.out').strip(),'',r('attn_report.txt').strip(),'```','',
'Lavapipe "not supported" outside group 4: K/V views above its 128 MiB maxStorageBufferRange, 8 cases: f16 n_kv 270336 (2), 1048576, 2097156; q8_0 n_kv 270336, 1048576, 2097156; f32 n_kv 1048576 (`sa_expected_unsupported`, by the device limit).','',
'## Validation, standalone, switches (`runs2.out`)','','```',r('runs2.out').strip(),'```','']
for f in sorted(x for x in os.listdir(L) if x.startswith('val_')):
    txt=r(f); m=re.search(r'\d+/\d+ tests passed',txt)
    t.append(f'- `{f}`: {m.group(0) if m else "?"}, VUID / UNASSIGNED messages: {len(re.findall("VUID-|UNASSIGNED-",txt))}')
t+=['','## test-sparse-attn-vk','','```']
for f in ('vkt_radv.log','vkt_lvp.log'):
    t+=[f]+[l[:200] for l in r(f).splitlines() if not l.startswith('ggml_vulkan')]
t+=['```','','## Regressions (`final.out`)','','```',r('final.out').strip(),'```','',
'hip SPARSE_ATTN rc 1: the 20 rot != 0 cases, W10 on HIP, reported "not supported" under --fail-unsupported, 0 compare errors. hip QSA_DECODE_MASKLESS rc 1: the one case W2 already lists (q8_0 GQA 4 interleaved).','',
'## Mutants (`mutants.out`, `mut_*_syncval.log`, `mut_short_scratch_gpuav.log`)','','```',r('mutants.out').strip(),'```','',
'sync_attend, sync_plan: not caught, neither by the stress test nor by synchronization validation (submit-time validation on: 0 SYNC-HAZARD). short_scratch: caught by the bit-exact compare; GPU-AV reported nothing on it (0 VUID), so the "0 out-of-range" of the validation runs has no positive control.','']
open(O+'w6_tests.md','w').write('\n'.join(t)+'\n')
h=['# W6 AT-B2 / AT-B3: op time on Vulkan (HEAD b1cb6764c, RADV, `harness/w6/time_campaign.sh`)','',
'3 interleaved processes per arm, 5 reps after 3 warm-ups, R = 2 attend nodes. Rows: captured text and random (C11 shape: c3_capture ub 2048 at pos0 32768, n_kv 34816, 11 x 2048 q + 1 x 512 q, all 12 layers; d4096, d131072: C18 rows, n_q 4096, layers 0, 5, 11). Q / K / V: SYN (uniform +-1.7): the graph uses the op from W7 on, so the model data cannot reach it yet. Statistic: per process the mean over layers of the per-layer median; median over processes. Logger = GGML_VK_PERF_LOGGER op time, wall = ggml_backend_graph_compute. "sel" = the old selected-key FLASH_ATTN_EXT in the same build, its pre-pass inside.','',
'```',r('time_an.txt').strip(),'```','']
open(O+'w6_time.md','w').write('\n'.join(h)+'\n')
