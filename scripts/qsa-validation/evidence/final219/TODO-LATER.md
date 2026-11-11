# Known issues to fix later (owner decision 2026-11-08)

The owner deferred these W7 findings: they do not block W8 and later items or Prove, and they are not fixed in this PR unless a later step fixes them for free. Prove still measures them and reports pass/fail. The PR description must list every item below under "Known issues / follow-ups", with its numbers.

1. Q1 direct form fails for q4_0 and bf16 (KLD(new || PR219), stage 2: q4_0 2.082e-3 vs band 1.441e-3; bf16 1.250e-4 vs 1.032e-4). Against the f16 reference both arms sit at the same distance (phase C: q4_0 new 4.999e-3 vs PR219 4.995e-3; bf16 new 1.07e-4 vs PR219 1.00e-4), and the Q1 difference form passes.
2. AT-D2 (-ngl 0, CPU only): new is 0.69-0.71x PR219 at pp512 and 0.89x at tg32. Cause: the CPU SPARSE_ATTN op (ggml-cpu/sparse-attn.cpp:217-270) is scalar and converts K/V row by row. Follow-up: vectorise it.
3. AT-B1 (Vulkan): q8_0 worst 0.921x f16, q4_0 worst 0.916x (criterion >= 0.95); bf16 and both mixed pairs pass; f16 equals PR219. V pack would pass but costs 68-264 MiB (against M1); V direct is kept.
4. AT-B5 (Vulkan memory): one MTP f16 cell is 12 MiB above PR219.
5. AT-D1: PR219 has 0 kind-4 regions where the plan expects 12; the criterion as written is not met (criterion wording).
6. M1: iq4_nl MTP is 32.29 MiB over f16 (HIP GET_ROWS limit, ggml-cuda.cu:9203-9209); W11's mapping is expected to remove it; Prove re-checks.
7. -nkvo: copy route; four d0 cells unresolved and bf16 not measured on the copy route (moved to Prove).
8. Q4 passes with thin f16-KV margins (6e-6 and 2e-6).
9. Q7 (MTP draft acceptance vs PR219, paired, greedy): stage 1 (14 prompts) and stage 2 (X8..X21, chain11) both done. Orchestrator's rough pooled figures (~34 pairs per type): mean new-PR219 default -0.13 pp, q8_0 -0.53, q4_0 +0.16, bf16 +0.05; per-prompt SD ~5 pp, so the 95% lower bound (-1.6 to -2.5 pp) stays below the -1 pp gate. Deferred by the owner as a known issue (no measurable mean drop; the gate is tighter than the per-prompt noise). Prove reports the exact figures in "Known issues (deferred)"; no prove-fix work on it.
