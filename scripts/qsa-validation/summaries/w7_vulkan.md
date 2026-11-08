# W7 Vulkan (AT-B5, Q1-vk / AT-B6, census), build-vk at 6e07acd3d, W0-Vulkan = ~/dev/wt_lab/w0lab12/build-vk

Source: strix-002:~/dev/wt_lab/w7/chainV.out, vqr.out, vk/logs/. Commands: scripts/qsa-validation/harness/w7/vk.sh VB VC V5 VQ V1M; the Q1-vk rerun script ~/dev/wt_lab/w7/vqr.sh.

## Census on Vulkan0 (vk.sh VC: llama-perplexity c8192 x 1 chunk, b 4096, ub 4096 / 512, LLAMA_QSA_TRACE=1)

```
PHASE VC start 05:33:05
VC f16_f16 u4096 rc=0 Final estimate: PPL = 2.0936 +/- 0.05706 masked=0 cpu=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 24;sa_attend/vk/cm/direct/f16/f16 24;]
VC f16_f16 u512 rc=0 Final estimate: PPL = 2.0943 +/- 0.05696 masked=0 cpu=0 guard=[passes=19 inspected=396 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 144;sa_attend/vk/cm/direct/f16/f16 144;]
VC q8_0_q8_0 u4096 rc=0 Final estimate: PPL = 2.1006 +/- 0.05740 masked=0 cpu=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 24;sa_attend/vk/cm/direct/q8_0/q8_0 24;]
VC q8_0_q8_0 u512 rc=0 Final estimate: PPL = 2.1145 +/- 0.05806 masked=0 cpu=0 guard=[passes=19 inspected=396 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 144;sa_attend/vk/cm/direct/q8_0/q8_0 144;]
VC q4_0_q4_0 u4096 rc=0 Final estimate: PPL = 2.1295 +/- 0.05891 masked=0 cpu=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 24;sa_attend/vk/cm/direct/q4_0/q4_0 24;]
VC q4_0_q4_0 u512 rc=0 Final estimate: PPL = 2.1383 +/- 0.05946 masked=0 cpu=0 guard=[passes=19 inspected=396 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 144;sa_attend/vk/cm/direct/q4_0/q4_0 144;]
VC bf16_bf16 u4096 rc=0 Final estimate: PPL = 2.1021 +/- 0.05768 masked=0 cpu=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 24;sa_attend/vk/cm/direct/bf16/bf16 24;]
VC bf16_bf16 u512 rc=0 Final estimate: PPL = 2.0972 +/- 0.05734 masked=0 cpu=0 guard=[passes=19 inspected=444 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 192;sa_attend/vk/cm/direct/bf16/bf16 192;]
VC q8_0_f16 u4096 rc=0 Final estimate: PPL = 2.0942 +/- 0.05706 masked=0 cpu=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 24;sa_attend/vk/cm/direct/q8_0/f16 24;]
VC q8_0_f16 u512 rc=0 Final estimate: PPL = 2.0980 +/- 0.05715 masked=0 cpu=0 guard=[passes=19 inspected=396 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 144;sa_attend/vk/cm/direct/q8_0/f16 144;]
VC f16_q8_0 u4096 rc=0 Final estimate: PPL = 2.1032 +/- 0.05765 masked=0 cpu=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 24;sa_attend/vk/cm/direct/f16/q8_0 24;]
VC f16_q8_0 u512 rc=0 Final estimate: PPL = 2.0909 +/- 0.05684 masked=0 cpu=0 guard=[passes=19 inspected=396 inspected_min=12 layers=12 cpu_launch=0] sa=[sa_plan/G1 144;sa_attend/vk/cm/direct/f16/q8_0 144;]
PHASE VC done 05:50:19
```

## AT-B5 (Vulkan0 sched_reserve, llama-server main + MTP, VK_LOG_MEMORY=1)

| cell | T | new Vulkan0 main / MTP | W0 Vulkan0 main / MTP | new T - new f16 (main / MTP) | M1 (<= f16 + 1 MiB) | new f16 - W0 f16 (main / MTP) |
|---|---|---|---|---|---|---|
| c32768 ub4096 | f16 | 1494.02 / 1121.72 | 1494.02 / 1121.72 | +0.00 / +0.00 | PASS | +0.00 / +0.00 |
| c32768 ub4096 | q8_0 | 1494.29 / 1121.98 | 2635.39 / 1657.84 | +0.27 / +0.26 | PASS |  |
| c32768 ub4096 | q4_0 | 1494.29 / 1121.98 | 2635.39 / 1657.84 | +0.27 / +0.26 | PASS |  |
| c32768 ub4096 | bf16 | 1494.02 / 1121.72 | 2515.48 / 1593.58 | +0.00 / +0.00 | PASS |  |
| c32768 ub512 | f16 | 189.93 / 184.43 | 217.93 / 221.43 | +0.00 / +0.00 | PASS | -28.00 / -37.00 |
| c32768 ub512 | q8_0 | 190.19 / 184.70 | 385.98 / 313.79 | +0.26 / +0.27 | PASS |  |
| c32768 ub512 | q4_0 | 190.19 / 184.70 | 385.98 / 313.79 | +0.26 / +0.27 | PASS |  |
| c32768 ub512 | bf16 | 189.93 / 184.43 | 377.72 / 304.53 | +0.00 / +0.00 | PASS |  |
| c131072 ub4096 | f16 | 1503.77 / 1147.22 | 1759.77 / 1243.22 | +0.00 / +0.00 | PASS | -256.00 / -96.00 |
| c131072 ub4096 | q8_0 | 1504.04 / 1147.48 | 6796.63 / 5082.97 | +0.27 / +0.26 | PASS |  |
| c131072 ub4096 | q4_0 | 1504.04 / 1147.48 | 6796.63 / 5082.97 | +0.27 / +0.26 | PASS |  |
| c131072 ub4096 | bf16 | 1503.77 / 1147.22 | 6796.37 / 5002.70 | +0.00 / +0.00 | PASS |  |
| c131072 ub512 | f16 | 607.80 / 629.28 | 631.80 / 617.28 | +0.00 / +0.00 | PASS | -24.00 / +12.00 |
| c131072 ub512 | q8_0 | 608.07 / 629.54 | 1179.11 / 1018.92 | +0.27 / +0.26 | PASS |  |
| c131072 ub512 | q4_0 | 608.07 / 629.54 | 1179.11 / 1018.92 | +0.27 / +0.26 | PASS |  |
| c131072 ub512 | bf16 | 607.80 / 629.28 | 1178.84 / 1005.65 | +0.00 / +0.00 | PASS |  |

AT-B5: FAIL: f16<=W0 MTP c131072 u512

## Q1-vk / AT-B6 (new-Vulkan(T) - W0-Vulkan(T), pinned on the W0-Vulkan f16 reference vref8t0, c8192 x 8, ub4096; B_vk = 3.214e-4, summaries/w12_bands.md:311)

Three arms of the first pass (vq_w0_f16_f16, vq_new_f16_f16, vq_w0_q8_0_q8_0) ended in vk::DeviceLostError (rc 134) while the kernel logged amdgpu_gem_va_update_vm Couldn't update BO_VA (-12) (journalctl -k, 06:16:31, 06:17:57, 06:19:22), while `free -g` showed 0 GB free and 46 GB buff/cache and the HIP KLD references (62 GB) were still on disk; after those references were deleted, the remaining arms and the one budgeted rerun of the three (_r1, same reference files, hard-linked) ran with rc 0 and no further BO_VA error (that the page cache caused the loss is UNPROVEN).

| pair | new arm | W0 arm | n | mean diff | SE | band B_vk | verdict | rows_absent / moe_absent (new, W0) |
|---|---|---|---|---|---|---|---|---|
| f16_f16 | vq_new_f16_f16_r1 | vq_w0_f16_f16_r1 | 8 | +2.391e-04 | 2.62e-05 | 3.214e-04 | PASS | 0/0, 0/0 |
| q8_0_q8_0 | vq_new_q8_0_q8_0 | vq_w0_q8_0_q8_0_r1 | 8 | +7.544e-07 | 2.63e-06 | 3.214e-04 | PASS | 0/0, 0/0 |
| q4_0_q4_0 | vq_new_q4_0_q4_0 | vq_w0_q4_0_q4_0 | 8 | -9.980e-05 | 7.37e-05 | 3.214e-04 | PASS | 0/0, 0/0 |
| bf16_bf16 | vq_new_bf16_bf16 | vq_w0_bf16_bf16 | 8 | -5.144e-06 | 4.16e-06 | 3.214e-04 | PASS | 0/0, 0/0 |
| q8_0_f16 | vq_new_q8_0_f16 | vq_w0_q8_0_f16 | 8 | -2.075e-05 | 4.90e-06 | 3.214e-04 | PASS | 0/0, 0/0 |
| f16_q8_0 | vq_new_f16_q8_0 | vq_w0_f16_q8_0 | 8 | -1.321e-05 | 5.53e-06 | 3.214e-04 | PASS | 0/0, 0/0 |
