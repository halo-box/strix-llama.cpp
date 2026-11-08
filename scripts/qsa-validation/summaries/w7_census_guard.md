# W7 census, MTP census, guard, op tests on the final build dcaa19f16

Source: strix-002:~/dev/wt_lab/w7/chainF1.out; logs in ~/dev/wt_lab/w7/{optests,guard,census,mtp}/. Commands: scripts/qsa-validation/harness/w7/optests.sh, guard.sh, census.sh, mtp_census.sh

```
OPTEST CPU SPARSE_ATTN_PLAN rc=0   185/185 tests passed
OPTEST ROCm0 SPARSE_ATTN_PLAN rc=0   185/185 tests passed
OPTEST CPU SPARSE_ATTN rc=0   1291/1291 tests passed
OPTEST ROCm0 SPARSE_ATTN rc=1   1271/1291 tests passed
OPTEST ROCm0 QSA_PREFILL_MASKLESS rc=0   26/26 tests passed
OPTEST ROCm0 QSA_DECODE_MASKLESS rc=0   52/52 tests passed
OPTEST ROCm0 QSA_PREFILL rc=0   26/26 tests passed
OPTEST ROCm0 QSA_DECODE rc=0   92/92 tests passed
OPTEST Vulkan0 SPARSE_ATTN_PLAN rc=0   92/92 tests passed
OPTEST Vulkan0 SPARSE_ATTN rc=0   711/711 tests passed
OPTESTS_DONE
PHASE VO start 21:41:05
VO p1 rc=0
VO p2 rc=0
VO p3 rc=0
PHASE VO done 21:59:53
VK_DONE
GUARD pass_q8 rc=0 | PASS | 
GUARD pass_f16 rc=0 | PASS | 
GUARD pass_q4 rc=0 | PASS | 
GUARD pass_bf16 rc=0 | PASS | 
GUARD fake rc=0 | PASS | llama_init_from_model: failed to initialize the context: QSA layer 3 (cb): sparse attention K=q8_0 V=q8_0 is not supported on device FAKE0 (backend FAKE): no sparse-attention kernel in this backend; run QSA on the CPU with -ngl 0 or --no-kv-offload, or use a H
GUARD reject rc=0 | PASS | llama_init_from_model: failed to initialize the context: QSA layer 3 (cb): sparse attention K=q8_0 V=q8_0 is not supported on device ROCm0 (backend ROCm): rejected by the test hook GGML_CUDA_SA_REJECT; run QSA on the CPU with -ngl 0 or --no-kv-offload, or use 
GUARD reject_decode rc=0 | PASS | process_ubatch: QSA layer 3 (av): sparse attention K=q8_0 V=q8_0 is not supported on device ROCm0 (backend ROCm): rejected by the test hook GGML_CUDA_SA_REJECT; run QSA on the CPU with -ngl 0 or --no-kv-offload, or use a HIP (RDNA3.5) or Vulkan build
GUARD allvis rc=0 | RESULT PASS (failures 0) | 
GUARD split_tensor rc=1 1.11.235.867 E llama_init_from_model: failed to initialize the context: QSA sparse attention is not supported with --split-mode tensor: the meta backend has no SPARSE_ATTN handler; use --split-mode la
GUARD nctx_cap rc=1 0.55.563.617 E llama_init_from_model: failed to initialize the context: QSA sparse attention: n_ctx_seq = 2097664 is above the tested maximum of 2097156 cells
GUARD_DONE
RUN ppl_f16_f16_u4096.log rc=0 Final estimate: PPL = 2.1082 +/- 0.05778
CENSUS ppl_f16_f16_u4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_f16_f16_u512.log rc=0 Final estimate: PPL = 2.0978 +/- 0.05732
CENSUS ppl_f16_f16_u512.log expect_rc=0 recipe=Q1-B recipe_rc=0 masked_lines=0 guard=[passes=19 inspected=396 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_f16_f16_u16.log rc=0 Final estimate: PPL = 2.0875 +/- 0.05687
CENSUS ppl_f16_f16_u16.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=37 inspected=768 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN bench_f16_f16_p4096_d4096.log rc=0
CENSUS bench_f16_f16_p4096_d4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=4 inspected=96 inspected_min=24 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q8_0_q8_0_u4096.log rc=0 Final estimate: PPL = 2.1139 +/- 0.05827
CENSUS ppl_q8_0_q8_0_u4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q8_0_q8_0_u512.log rc=0 Final estimate: PPL = 2.1139 +/- 0.05827
CENSUS ppl_q8_0_q8_0_u512.log expect_rc=0 recipe=Q1-B recipe_rc=0 masked_lines=0 guard=[passes=19 inspected=444 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q8_0_q8_0_u16.log rc=0 Final estimate: PPL = 2.0874 +/- 0.05703
CENSUS ppl_q8_0_q8_0_u16.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=37 inspected=768 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN bench_q8_0_q8_0_p4096_d4096.log rc=0
CENSUS bench_q8_0_q8_0_p4096_d4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=4 inspected=96 inspected_min=24 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q4_0_q4_0_u4096.log rc=0 Final estimate: PPL = 2.1271 +/- 0.05896
CENSUS ppl_q4_0_q4_0_u4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q4_0_q4_0_u512.log rc=0 Final estimate: PPL = 2.1271 +/- 0.05896
CENSUS ppl_q4_0_q4_0_u512.log expect_rc=0 recipe=Q1-B recipe_rc=0 masked_lines=0 guard=[passes=19 inspected=444 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q4_0_q4_0_u16.log rc=0 Final estimate: PPL = 2.0881 +/- 0.05695
CENSUS ppl_q4_0_q4_0_u16.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=37 inspected=768 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN bench_q4_0_q4_0_p4096_d4096.log rc=0
CENSUS bench_q4_0_q4_0_p4096_d4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=4 inspected=96 inspected_min=24 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_bf16_bf16_u4096.log rc=0 Final estimate: PPL = 2.1047 +/- 0.05737
CENSUS ppl_bf16_bf16_u4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_bf16_bf16_u512.log rc=0 Final estimate: PPL = 2.1047 +/- 0.05737
CENSUS ppl_bf16_bf16_u512.log expect_rc=0 recipe=Q1-B recipe_rc=0 masked_lines=0 guard=[passes=19 inspected=444 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_bf16_bf16_u16.log rc=0 Final estimate: PPL = 2.0822 +/- 0.05638
CENSUS ppl_bf16_bf16_u16.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=37 inspected=768 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN bench_bf16_bf16_p4096_d4096.log rc=0
CENSUS bench_bf16_bf16_p4096_d4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=4 inspected=96 inspected_min=24 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q8_0_f16_u4096.log rc=0 Final estimate: PPL = 2.1204 +/- 0.05891
CENSUS ppl_q8_0_f16_u4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q8_0_f16_u512.log rc=0 Final estimate: PPL = 2.1204 +/- 0.05891
CENSUS ppl_q8_0_f16_u512.log expect_rc=0 recipe=Q1-B recipe_rc=0 masked_lines=0 guard=[passes=19 inspected=444 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_q8_0_f16_u16.log rc=0 Final estimate: PPL = 2.0955 +/- 0.05727
CENSUS ppl_q8_0_f16_u16.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=37 inspected=768 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN bench_q8_0_f16_p4096_d4096.log rc=0
CENSUS bench_q8_0_f16_p4096_d4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=4 inspected=96 inspected_min=24 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_f16_q8_0_u4096.log rc=0 Final estimate: PPL = 2.1152 +/- 0.05822
CENSUS ppl_f16_q8_0_u4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=5 inspected=108 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_f16_q8_0_u512.log rc=0 Final estimate: PPL = 2.1152 +/- 0.05822
CENSUS ppl_f16_q8_0_u512.log expect_rc=0 recipe=Q1-B recipe_rc=0 masked_lines=0 guard=[passes=19 inspected=444 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN ppl_f16_q8_0_u16.log rc=0 Final estimate: PPL = 2.0986 +/- 0.05762
CENSUS ppl_f16_q8_0_u16.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=37 inspected=768 inspected_min=12 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
RUN bench_f16_q8_0_p4096_d4096.log rc=0
CENSUS bench_f16_q8_0_p4096_d4096.log expect_rc=0 recipe= recipe_rc=0 masked_lines=0 guard=[passes=4 inspected=96 inspected_min=24 layers=12 cpu_launch=0] EXPECT PASS 0 diffs
CENSUS_DONE rc_all=0
MTP c_default expect_rc=0 EXPECT PASS masked=0 cpu=0 guard=[ctx=mtp passes=1019 inspected=811 inspected_min=0 layers=1 cpu_launch=0;ctx=main passes=33 inspected=684 inspected_min=12 layers=12 cpu_launch=0;] res=5
  mtp       1 producer=cb n_kv=28672 count=13 route=sparse
  mtp       1 producer=cb n_kv=28928 count=236 route=sparse
  mtp       1 producer=cb n_kv=29184 count=57 route=sparse
  mtp       1 producer=cb n_kv=5632 count=266 route=sparse
  mtp       1 producer=cb n_kv=5888 count=7 route=sparse
  mtp       1 producer=cb n_kv=6912 count=229 route=sparse
  mtp       1 producer=cb n_kv=7168 count=83 route=sparse
  mtp       1 producer=dense n_kv=1024 count=24 route=dense
  mtp       1 producer=dense n_kv=1280 count=479 route=dense
  mtp       1 producer=dense n_kv=1536 count=100 route=dense
  mtp       1 producer=dense n_kv=256 count=1 route=dense
MTP c_q8 expect_rc=0 EXPECT PASS masked=0 cpu=0 guard=[ctx=mtp passes=1019 inspected=810 inspected_min=0 layers=1 cpu_launch=0;ctx=main passes=33 inspected=684 inspected_min=12 layers=12 cpu_launch=0;] res=5
  mtp       1 producer=cb n_kv=28672 count=13 route=sparse
  mtp       1 producer=cb n_kv=28928 count=233 route=sparse
  mtp       1 producer=cb n_kv=29184 count=54 route=sparse
  mtp       1 producer=cb n_kv=5632 count=266 route=sparse
  mtp       1 producer=cb n_kv=5888 count=7 route=sparse
  mtp       1 producer=cb n_kv=6912 count=232 route=sparse
  mtp       1 producer=cb n_kv=7168 count=83 route=sparse
  mtp       1 producer=dense n_kv=1024 count=24 route=dense
  mtp       1 producer=dense n_kv=1280 count=479 route=dense
  mtp       1 producer=dense n_kv=1536 count=103 route=dense
  mtp       1 producer=dense n_kv=256 count=1 route=dense
MTP c_q4 expect_rc=0 EXPECT PASS masked=0 cpu=0 guard=[ctx=mtp passes=1019 inspected=811 inspected_min=0 layers=1 cpu_launch=0;ctx=main passes=33 inspected=684 inspected_min=12 layers=12 cpu_launch=0;] res=5
  mtp       1 producer=cb n_kv=28672 count=13 route=sparse
  mtp       1 producer=cb n_kv=28928 count=236 route=sparse
  mtp       1 producer=cb n_kv=29184 count=51 route=sparse
  mtp       1 producer=cb n_kv=5632 count=266 route=sparse
  mtp       1 producer=cb n_kv=5888 count=7 route=sparse
  mtp       1 producer=cb n_kv=6912 count=236 route=sparse
  mtp       1 producer=cb n_kv=7168 count=82 route=sparse
  mtp       1 producer=dense n_kv=1024 count=21 route=dense
  mtp       1 producer=dense n_kv=1280 count=479 route=dense
  mtp       1 producer=dense n_kv=1536 count=103 route=dense
  mtp       1 producer=dense n_kv=256 count=1 route=dense
MTP_DONE
```
