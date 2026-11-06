# W6 tests (HEAD b1cb6764c; logs strix-002:~/dev/wt_rewrite_logs/W6)

Every run one op, -j 1, under flock. `--fail-unsupported` on every Vulkan run of the two ops: a "not supported" outside the list (group 4; K/V bigger than the binding limit) is a FAIL. "Ratio" = NMSE / band of the census class (§14.2 (c): G1 row of the split class; no row measured on a Vulkan tier).

Devices: RADV = Radeon 8060S, Mesa 26.0.3; lavapipe = llvmpipe Mesa 25.3.6, LLVM 21.1.8, 256 bits (Fedora 43 packages extracted to wt_lab/w6/lvp43; the system lavapipe crashes, see below).

## Op runs (`runs1.out`, `attn_report.txt`)

```
build rc=0
EXIT radv SPARSE_ATTN_PLAN rc=0 secs=4
EXIT radv SPARSE_ATTN rc=0 secs=416
EXIT scalar SPARSE_ATTN_PLAN rc=0 secs=3
EXIT scalar SPARSE_ATTN rc=0 secs=324
EXIT portable_sg32 SPARSE_ATTN_PLAN rc=0 secs=4
EXIT portable_sg32 SPARSE_ATTN rc=0 secs=414
EXIT portable SPARSE_ATTN_PLAN rc=0 secs=3
EXIT portable SPARSE_ATTN rc=0 secs=414
EXIT vdirect SPARSE_ATTN_PLAN rc=0 secs=3
EXIT vdirect SPARSE_ATTN rc=0 secs=324
EXIT vstaged SPARSE_ATTN_PLAN rc=0 secs=3
EXIT vstaged SPARSE_ATTN rc=0 secs=417
EXIT vpack SPARSE_ATTN_PLAN rc=0 secs=3
EXIT vpack SPARSE_ATTN rc=0 secs=321
EXIT lvp SPARSE_ATTN_PLAN rc=0 secs=4
EXIT lvp SPARSE_ATTN rc=0 secs=427
RUNS1_DONE

== run_radv_SPARSE_ATTN_PLAN.log
  CENSUS COVERAGE: Vulkan0: 1/1 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
  92/92 tests passed
  not supported: 93 {'G4': 93}
  K/V pairs seen: 0  rot reports (2 per case, CPU excluded): {}
== run_radv_SPARSE_ATTN.log
  CENSUS COVERAGE: Vulkan0: 85/85 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (332 excluded by a forcing switch): OK
  711/711 tests passed
  not supported: 580 {'G4': 580}
  K/V pairs seen: 81  rot reports (2 per case, CPU excluded): {'64/64': 4, '256/256': 4, '64/0': 4, '0/256': 4, '128/64': 4}
  cm/direct: reports 725 max ratio 0.518  [type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=4096,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end nmse=8.759e-08]
== run_scalar_SPARSE_ATTN_PLAN.log
  CENSUS COVERAGE: Vulkan0: 1/1 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
  92/92 tests passed
  not supported: 93 {'G4': 93}
  K/V pairs seen: 0  rot reports (2 per case, CPU excluded): {}
== run_scalar_SPARSE_ATTN.log
  CENSUS COVERAGE: Vulkan0: 84/84 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (333 excluded by a forcing switch): OK
  711/711 tests passed
  not supported: 580 {'G4': 580}
  K/V pairs seen: 81  rot reports (2 per case, CPU excluded): {'64/64': 4, '256/256': 4, '64/0': 4, '0/256': 4, '128/64': 4}
  scalar/staged: reports 725 max ratio 0.613  [type_k=f16,type_v=f16,n_q=1,n_kv=16384,n_sel=17,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0,s nmse=3.075e-08]
== run_portable_sg32_SPARSE_ATTN_PLAN.log
  CENSUS COVERAGE: Vulkan0: 1/1 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
  92/92 tests passed
  not supported: 93 {'G4': 93}
  K/V pairs seen: 0  rot reports (2 per case, CPU excluded): {}
== run_portable_sg32_SPARSE_ATTN.log
  CENSUS COVERAGE: Vulkan0: 81/81 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (336 excluded by a forcing switch): OK
  711/711 tests passed
  not supported: 580 {'G4': 580}
  K/V pairs seen: 81  rot reports (2 per case, CPU excluded): {'64/64': 4, '256/256': 4, '64/0': 4, '0/256': 4, '128/64': 4}
  portable/staged: reports 725 max ratio 0.002  [type_k=q4_0,type_v=q4_0,n_q=1,n_kv=16384,n_sel=257,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end= nmse=4.496e-10]
== run_portable_SPARSE_ATTN_PLAN.log
  CENSUS COVERAGE: Vulkan0: 1/1 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
  92/92 tests passed
  not supported: 93 {'G4': 93}
  K/V pairs seen: 0  rot reports (2 per case, CPU excluded): {}
== run_portable_SPARSE_ATTN.log
  CENSUS COVERAGE: Vulkan0: 81/81 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (336 excluded by a forcing switch): OK
  711/711 tests passed
  not supported: 580 {'G4': 580}
  K/V pairs seen: 81  rot reports (2 per case, CPU excluded): {'64/64': 4, '256/256': 4, '64/0': 4, '0/256': 4, '128/64': 4}
  portable/staged: reports 725 max ratio 0.003  [type_k=q4_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=257,group=1,gqa=4,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=1,at_end= nmse=5.364e-10]
== run_vdirect_SPARSE_ATTN_PLAN.log
  CENSUS COVERAGE: Vulkan0: 1/1 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
  92/92 tests passed
  not supported: 93 {'G4': 93}
  K/V pairs seen: 0  rot reports (2 per case, CPU excluded): {}
== run_vdirect_SPARSE_ATTN.log
  CENSUS COVERAGE: Vulkan0: 85/85 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (332 excluded by a forcing switch): OK
  711/711 tests passed
  not supported: 580 {'G4': 580}
  K/V pairs seen: 81  rot reports (2 per case, CPU excluded): {'64/64': 4, '256/256': 4, '64/0': 4, '0/256': 4, '128/64': 4}
  cm/direct: reports 725 max ratio 0.500  [type_k=q8_0,type_v=q8_0,n_q=9,n_kv=2051,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_ nmse=7.905e-08]
== run_vstaged_SPARSE_ATTN_PLAN.log
  CENSUS COVERAGE: Vulkan0: 1/1 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
  92/92 tests passed
  not supported: 93 {'G4': 93}
  K/V pairs seen: 0  rot reports (2 per case, CPU excluded): {}
== run_vstaged_SPARSE_ATTN.log
  CENSUS COVERAGE: Vulkan0: 85/85 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (332 excluded by a forcing switch): OK
  711/711 tests passed
  not supported: 580 {'G4': 580}
  K/V pairs seen: 81  rot reports (2 per case, CPU excluded): {'64/64': 4, '256/256': 4, '64/0': 4, '0/256': 4, '128/64': 4}
  cm/staged: reports 725 max ratio 0.509  [type_k=q8_0,type_v=q8_0,n_q=9,n_kv=2051,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_ nmse=8.045e-08]
== run_vpack_SPARSE_ATTN_PLAN.log
  CENSUS COVERAGE: Vulkan0: 1/1 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
  92/92 tests passed
  not supported: 93 {'G4': 93}
  K/V pairs seen: 0  rot reports (2 per case, CPU excluded): {}
== run_vpack_SPARSE_ATTN.log
  CENSUS COVERAGE: Vulkan0: 93/93 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (324 excluded by a forcing switch): OK
  711/711 tests passed
  not supported: 580 {'G4': 580}
  K/V pairs seen: 81  rot reports (2 per case, CPU excluded): {'64/64': 4, '256/256': 4, '64/0': 4, '0/256': 4, '128/64': 4}
  cm/direct: reports 94 max ratio 0.506  [type_k=f16,type_v=f16,n_q=9,n_kv=2051,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_en nmse=8.002e-08]
  cm/pack: reports 157 max ratio 0.425  [type_k=q8_0,type_v=q8_0,n_q=16,n_kv=512,n_sel=2051,group=1,gqa=12,n_stream=1,rot_q=0,rot_o=0,rows=allvis,kv_layout=1,at_ nmse=6.626e-08]
  cm/staged: reports 474 max ratio 0.499  [type_k=q8_0,type_v=q8_0,n_q=1,n_kv=16384,n_sel=2051,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end nmse=8.541e-08]
== run_lvp_SPARSE_ATTN_PLAN.log
  CENSUS COVERAGE: Vulkan0: 1/1 compiled keys of {sa_plan/} launched (0 excluded by a forcing switch): OK
  92/92 tests passed
  not supported: 93 {'G4': 93}
  K/V pairs seen: 0  rot reports (2 per case, CPU excluded): {}
== run_lvp_SPARSE_ATTN.log
  CENSUS COVERAGE: Vulkan0: 81/81 compiled keys of {sa_attend/,sa_fixup/,sa_generic/,sa_merge,sa_pack/,sa_portable/} launched (336 excluded by a forcing switch): OK
  703/703 tests passed
  not supported: 588 {'G4': 580, 'large/other 270336 f16': 2, 'large/other 1048576 f16': 1, 'large/other 2097156 f16': 1, 'large/other 270336 q8_0': 1, 'large/other 1048576 q8_0': 1, 'large/
  K/V pairs seen: 81  rot reports (2 per case, CPU excluded): {'64/64': 4, '256/256': 4, '64/0': 4, '0/256': 4, '128/64': 4}
  portable/staged: reports 717 max ratio 0.003  [type_k=f16,type_v=q8_0,n_q=1,n_kv=16384,n_sel=257,group=1,gqa=8,n_stream=1,rot_q=0,rot_o=0,rows=mix,kv_layout=0,at_end=0 nmse=6.810e-10]
```

Lavapipe "not supported" outside group 4: K/V views above its 128 MiB maxStorageBufferRange, 8 cases: f16 n_kv 270336 (2), 1048576, 2097156; q8_0 n_kv 270336, 1048576, 2097156; f32 n_kv 1048576 (`sa_expected_unsupported`, by the device limit).

## Validation, standalone, switches (`runs2.out`)

```
vulkaninfo radv 0
vulkaninfo lvp 0
plan lavapipe system mesa rc=139
val plan radv rc=0
val plan lvp rc=0
val attn radv [] rc=0
val attn radv [scalar] rc=0
val attn radv [portable] rc=0
val attn radv [vstaged] rc=0
val attn radv [vpack] rc=0
val attn lvp rc=0
vkt radv rc=0
vkt lvp rc=0
maxq attn [] rc=0
maxq attn [scalar] rc=0
maxq attn [portable] rc=0
maxq plan rc=0
mask15 plan rc=0
mask7 attn rc=0
RUNS2_DONE
```

- `val_attn_lvp.log`: 8/8 tests passed, VUID / UNASSIGNED messages: 0
- `val_attn_radv_.log`: 8/8 tests passed, VUID / UNASSIGNED messages: 0
- `val_attn_radv_portable.log`: 8/8 tests passed, VUID / UNASSIGNED messages: 0
- `val_attn_radv_scalar.log`: 8/8 tests passed, VUID / UNASSIGNED messages: 0
- `val_attn_radv_vpack.log`: 8/8 tests passed, VUID / UNASSIGNED messages: 0
- `val_attn_radv_vstaged.log`: 8/8 tests passed, VUID / UNASSIGNED messages: 0
- `val_plan_lvp.log`: 92/92 tests passed, VUID / UNASSIGNED messages: 0
- `val_plan_radv.log`: 92/92 tests passed, VUID / UNASSIGNED messages: 0

## test-sparse-attn-vk

```
vkt_radv.log
device: Radeon 8060S Graphics (RADV STRIX_HALO)
limits: the -c 1048576 reserve
  binding limit 1073741824, buffer type maximum 1073741824, alignment 4
  f16   ub 512  : K/V view 1073741824 B; plan dst 6306048 alloc 10508544; attend dst 12582912 alloc 15703680; supports 1 1
  f16   ub 4096 : K/V view 1073741824 B; plan dst 50446592 alloc 84066560; attend dst 100663296 alloc 103784064; supports 1 1
  f16   ub 8192 : K/V view 1073741824 B; plan dst 100892928 alloc 168132864; attend dst 201326592 alloc 204447360; supports 1 1
  bf16  ub 512  : K/V view 1073741824 B; plan dst 6306048 alloc 10508544; attend dst 12582912 alloc 15703680; supports 1 1
  bf16  ub 4096 : K/V view 1073741824 B; plan dst 50446592 alloc 84066560; attend dst 100663296 alloc 103784064; supports 1 1
  bf16  ub 8192 : K/V view 1073741824 B; plan dst 100892928 alloc 168132864; attend dst 201326592 alloc 204447360; supports 1 1
  q8_0  ub 512  : K/V view 570425344 B; plan dst 6306048 alloc 10508544; attend dst 12582912 alloc 15703680; supports 1 1
  q8_0  ub 4096 : K/V view 570425344 B; plan dst 50446592 alloc 84066560; attend dst 100663296 alloc 103784064; supports 1 1
  q8_0  ub 8192 : K/V view 570425344 B; plan dst 100892928 alloc 168132864; attend dst 201326592 alloc 204447360; supports 1 1
  q4_0  ub 512  : K/V view 301989888 B; plan dst 6306048 alloc 10508544; attend dst 12582912 alloc 15703680; supports 1 1
  q4_0  ub 4096 : K/V view 301989888 B; plan dst 50446592 alloc 84066560; attend dst 100663296 alloc 103784064; supports 1 1
  q4_0  ub 8192 : K/V view 301989888 B; plan dst 100892928 alloc 168132864; attend dst 201326592 alloc 204447360; supports 1 1
capability: no pipeline, no support
  default: plan 1 attend 1
  mask 15 (no pipeline): plan 0 attend 0
rot: H_q and H_o against fwht.comp
  tier default  rot 64  n_q 3  f16 : H_q bitwise, H_o bitwise
  tier default  rot 64  n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier default  rot 64  n_q 65 f16 : H_q bitwise, H_o bitwise
  tier default  rot 64  n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier default  rot 128 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier default  rot 128 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier default  rot 128 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier default  rot 128 n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier default  rot 256 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier default  rot 256 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier default  rot 256 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier default  rot 256 n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier scalar   rot 64  n_q 3  f16 : H_q bitwise, H_o bitwise
  tier scalar   rot 64  n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier scalar   rot 64  n_q 65 f16 : H_q bitwise, H_o bitwise
  tier scalar   rot 64  n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier scalar   rot 128 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier scalar   rot 128 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier scalar   rot 128 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier scalar   rot 128 n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier scalar   rot 256 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier scalar   rot 256 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier scalar   rot 256 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier scalar   rot 256 n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier portable rot 64  n_q 3  f16 : H_q bitwise, H_o bitwise
  tier portable rot 64  n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier portable rot 64  n_q 65 f16 : H_q bitwise, H_o bitwise
  tier portable rot 64  n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier portable rot 128 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier portable rot 128 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier portable rot 128 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier portable rot 128 n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier portable rot 256 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier portable rot 256 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier portable rot 256 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier portable rot 256 n_q 65 q8_0: H_q bitwise, H_o bitwise
stress: the workspace over a tensor the preceding dispatch reads
  attend stress: dst 0x183a65c + 73728, alloc 1708416, workspace 1634688; T 0x184ca54 + 8388608: workspace over T 1, dst clear of T 1, dst clear of the unsynced writes 1
  attend ref  : dst 0x204ca58 + 73728, alloc 1708416, workspace 1634688; T 0x184ca54 + 8388608: workspace over T 0, dst clear of T 1, dst clear of the unsynced writes 1
  attend workspace: layout 1, unstable iterations 0 of 19, attend bitwise to the reference 1, reader of T bitwise 1
  plan   stress: dst 0x182262c + 37376, alloc 62000, workspace 24624; T 0x182bc24 + 8388608: workspace over T 1, dst clear of T 1, dst clear of the unsynced writes 1
  plan   ref  : dst 0x202bc28 + 37376, alloc 62000, workspace 24624; T 0x182bc24 + 8388608: workspace over T 0, dst clear of T 1, dst clear of the unsynced writes 1
  plan scratch: layout 1, unstable iterations 0 of 19, attend bitwise to the reference 1, reader of T bitwise 1
OK: 0 failed checks
vkt_lvp.log
device: llvmpipe (LLVM 21.1.8, 256 bits)
limits: the -c 1048576 reserve
  binding limit 134217712, buffer type maximum 1073741824, alignment 16
  f16   ub 512  : K/V view 1073741824 B; plan dst 6306048 alloc 10508544; attend dst 12582912 alloc 12582912; supports 1 0
  f16   ub 4096 : K/V view 1073741824 B; plan dst 50446592 alloc 84066560; attend dst 100663296 alloc 100663296; supports 1 0
  f16   ub 8192 : K/V view 1073741824 B; plan dst 100892928 alloc 168132864; attend dst 201326592 alloc 201326592; supports 1 0
  bf16  ub 512  : K/V view 1073741824 B; plan dst 6306048 alloc 10508544; attend dst 12582912 alloc 12582912; supports 1 0
  bf16  ub 4096 : K/V view 1073741824 B; plan dst 50446592 alloc 84066560; attend dst 100663296 alloc 100663296; supports 1 0
  bf16  ub 8192 : K/V view 1073741824 B; plan dst 100892928 alloc 168132864; attend dst 201326592 alloc 201326592; supports 1 0
  q8_0  ub 512  : K/V view 570425344 B; plan dst 6306048 alloc 10508544; attend dst 12582912 alloc 12582912; supports 1 0
  q8_0  ub 4096 : K/V view 570425344 B; plan dst 50446592 alloc 84066560; attend dst 100663296 alloc 100663296; supports 1 0
  q8_0  ub 8192 : K/V view 570425344 B; plan dst 100892928 alloc 168132864; attend dst 201326592 alloc 201326592; supports 1 0
  q4_0  ub 512  : K/V view 301989888 B; plan dst 6306048 alloc 10508544; attend dst 12582912 alloc 12582912; supports 1 0
  q4_0  ub 4096 : K/V view 301989888 B; plan dst 50446592 alloc 84066560; attend dst 100663296 alloc 100663296; supports 1 0
  q4_0  ub 8192 : K/V view 301989888 B; plan dst 100892928 alloc 168132864; attend dst 201326592 alloc 201326592; supports 1 0
capability: no pipeline, no support
  default: plan 1 attend 1
  mask 15 (no pipeline): plan 0 attend 0
rot: H_q and H_o against fwht.comp
  tier default  rot 64  n_q 3  f16 : H_q bitwise, H_o bitwise
  tier default  rot 64  n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier default  rot 64  n_q 65 f16 : H_q bitwise, H_o bitwise
  tier default  rot 64  n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier default  rot 128 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier default  rot 128 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier default  rot 128 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier default  rot 128 n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier default  rot 256 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier default  rot 256 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier default  rot 256 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier default  rot 256 n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier scalar: not on this device
  tier portable rot 64  n_q 3  f16 : H_q bitwise, H_o bitwise
  tier portable rot 64  n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier portable rot 64  n_q 65 f16 : H_q bitwise, H_o bitwise
  tier portable rot 64  n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier portable rot 128 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier portable rot 128 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier portable rot 128 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier portable rot 128 n_q 65 q8_0: H_q bitwise, H_o bitwise
  tier portable rot 256 n_q 3  f16 : H_q bitwise, H_o bitwise
  tier portable rot 256 n_q 3  q8_0: H_q bitwise, H_o bitwise
  tier portable rot 256 n_q 65 f16 : H_q bitwise, H_o bitwise
  tier portable rot 256 n_q 65 q8_0: H_q bitwise, H_o bitwise
stress: the workspace over a tensor the preceding dispatch reads
  attend workspace: this tier has none, skipped
  plan   stress: dst 0x1822650 + 37376, alloc 62000, workspace 24624; T 0x182bc30 + 8388608: workspace over T 1, dst clear of T 1, dst clear of the unsynced writes 1
  plan   ref  : dst 0x202bc40 + 37376, alloc 62000, workspace 24624; T 0x182bc30 + 8388608: workspace over T 0, dst clear of T 1, dst clear of the unsynced writes 1
  plan scratch: layout 1, unstable iterations 0 of 19, attend bitwise to the reference 1, reader of T bitwise 1
OK: 0 failed checks
```

## Regressions (`final.out`)

```
hip build rc=0
vk build all rc=0
hip SPARSE_ATTN_PLAN rc=0   185/185 tests passed notsup=0
hip SPARSE_ATTN rc=1   1271/1291 tests passed notsup=20
hip QSA_PREFILL_MASKLESS rc=0   26/26 tests passed notsup=0
hip QSA_DECODE_MASKLESS rc=1   52/53 tests passed notsup=1
hip QSA_PREFILL rc=0   26/26 tests passed notsup=0
hip QSA_DECODE rc=0   92/92 tests passed notsup=0
vk QSA_PREFILL_MASKLESS rc=0   14/14 tests passed notsup=12
vk QSA_DECODE_MASKLESS rc=0   34/34 tests passed notsup=19
vk QSA_PREFILL rc=0   26/26 tests passed notsup=0
vk QSA_DECODE rc=0   92/92 tests passed notsup=0
cpu SPARSE_ATTN_PLAN rc=0   185/185 tests passed
cpu SPARSE_ATTN rc=0   1291/1291 tests passed
cpu-only build rc=0
cpu-only test-sparse-attn-vk rc=0 no Vulkan backend: skipped
ctest vk rc=0 main    =   4.97 sec*proc (2 tests)  Total Test time (real) =   4.98 sec 
FINAL_DONE
```

hip SPARSE_ATTN rc 1: the 20 rot != 0 cases, W10 on HIP, reported "not supported" under --fail-unsupported, 0 compare errors. hip QSA_DECODE_MASKLESS rc 1: the one case W2 already lists (q8_0 GQA 4 interleaved).

## Mutants (`mutants.out`, `mut_*_syncval.log`, `mut_short_scratch_gpuav.log`)

```
MUT sync_attend: build 0 check rc 0 FAIL-lines 0 validation hits 0; ['OK: 0 failed checks']
MUT sync_plan: build 0 check rc 0 FAIL-lines 0 validation hits 0; ['OK: 0 failed checks']
MUT short_scratch: build 0 check rc 1 FAIL-lines 3 validation hits 0; ['  0/1 tests passed']
MUT no_sort: build 0 check rc 1 FAIL-lines 67 validation hits 0; ['  27/92 tests passed']
MUT v_select: build 0 check rc 1 FAIL-lines 11 validation hits 0; ['  0/9 tests passed']
MUT score_select: build 0 check rc 1 FAIL-lines 11 validation hits 0; ['  0/9 tests passed']
MUT rot_q_butterfly: build 0 check rc 1 FAIL-lines 9 validation hits 0; ['FAILED: 8 failed checks']
MUT q8_v_scale: build 0 check rc 1 FAIL-lines 4 validation hits 0; ['  0/2 tests passed']
restore build 0
MUT_DONE
ALL_DONE
```

sync_attend, sync_plan: not caught, neither by the stress test nor by synchronization validation (submit-time validation on: 0 SYNC-HAZARD). short_scratch: caught by the bit-exact compare; GPU-AV reported nothing on it (0 VUID), so the "0 out-of-range" of the validation runs has no positive control.

