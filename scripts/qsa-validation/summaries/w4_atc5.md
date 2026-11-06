# W4 AT-C5 at op level (w4_reserve, GGML_SCHED_DEBUG_REALLOC=1)

Per run: reserve at (ub, n_ctx) with the HIP policy G (final AT-A rule, HEAD 330ece83f); every n_q in 1..ub x n_kv in {2052, n_ctx/4, n_ctx/2, n_ctx} x G in {policy, forced g4, forced g1}: alloc through the scheduler (plan and attend pinned to ROCm0, as llama pins them, PLAN-v3 §10); every 8th and the named n_q (1, 65, 66, 127, 128, 129, ub) computed. Results strix-002:~/dev/wt_lab/w4/reserve/res/ (the same grid with the pre-AT-A default policy: res_defaultpolicy/, 54/54 PASS; the unpinned first run res_v1/: finding F-W4-1).

```
EXIT f16_d4096_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT f16_d4096_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT f16_d4096_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT f16_d4096_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT f16_d4096_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT f16_d4096_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
EXIT f16_d32768_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT f16_d32768_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT f16_d32768_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT f16_d32768_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT f16_d32768_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT f16_d32768_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
EXIT f16_d131072_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT f16_d131072_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT f16_d131072_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT f16_d131072_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT f16_d131072_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT f16_d131072_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
EXIT q8_0_d4096_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT q8_0_d4096_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT q8_0_d4096_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT q8_0_d4096_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT q8_0_d4096_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT q8_0_d4096_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
EXIT q8_0_d32768_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT q8_0_d32768_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT q8_0_d32768_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT q8_0_d32768_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT q8_0_d32768_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT q8_0_d32768_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
EXIT q8_0_d131072_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT q8_0_d131072_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT q8_0_d131072_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT q8_0_d131072_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT q8_0_d131072_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT q8_0_d131072_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
EXIT q4_0_d4096_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT q4_0_d4096_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT q4_0_d4096_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT q4_0_d4096_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT q4_0_d4096_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT q4_0_d4096_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
EXIT q4_0_d32768_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT q4_0_d32768_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT q4_0_d32768_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT q4_0_d32768_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT q4_0_d32768_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT q4_0_d32768_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
EXIT q4_0_d131072_ub64 0 attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK PASS 
EXIT q4_0_d131072_ub65 0 attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK PASS 
EXIT q4_0_d131072_ub66 0 attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK PASS 
EXIT q4_0_d131072_ub127 0 attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK PASS 
EXIT q4_0_d131072_ub128 0 attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK PASS 
EXIT q4_0_d131072_ub129 0 attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK PASS 
DONE

RESERVE type f16 ub 127 n_ctx 135424 G_policy 4 gpu 19339008 cpu 4163072 | plan alloc 2618096 >= formula 2618096: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type f16 ub 127 n_ctx 135424 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2618096 <= 2618096: OK | PASS
RESERVE type f16 ub 128 n_ctx 135424 G_policy 4 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type f16 ub 128 n_ctx 135424 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type f16 ub 129 n_ctx 135424 G_policy 4 gpu 19668352 cpu 4228640 | plan alloc 2683944 >= formula 2683944: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type f16 ub 129 n_ctx 135424 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2683944 <= 2683944: OK | PASS
RESERVE type f16 ub 64 n_ctx 135424 G_policy 1 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type f16 ub 64 n_ctx 135424 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type f16 ub 65 n_ctx 135424 G_policy 4 gpu 13149056 cpu 2130720 | plan alloc 1371048 >= formula 1371048: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type f16 ub 65 n_ctx 135424 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1371048 <= 1371048: OK | PASS
RESERVE type f16 ub 66 n_ctx 135424 G_policy 4 gpu 13214592 cpu 2163488 | plan alloc 1379256 >= formula 1379256: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type f16 ub 66 n_ctx 135424 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1379256 <= 1379256: OK | PASS
RESERVE type f16 ub 127 n_ctx 37120 G_policy 4 gpu 19339008 cpu 4163072 | plan alloc 2618096 >= formula 2618096: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type f16 ub 127 n_ctx 37120 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2618096 <= 2618096: OK | PASS
RESERVE type f16 ub 128 n_ctx 37120 G_policy 4 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type f16 ub 128 n_ctx 37120 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type f16 ub 129 n_ctx 37120 G_policy 4 gpu 19668352 cpu 4228640 | plan alloc 2683944 >= formula 2683944: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type f16 ub 129 n_ctx 37120 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2683944 <= 2683944: OK | PASS
RESERVE type f16 ub 64 n_ctx 37120 G_policy 4 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type f16 ub 64 n_ctx 37120 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type f16 ub 65 n_ctx 37120 G_policy 4 gpu 13149056 cpu 2130720 | plan alloc 1371048 >= formula 1371048: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type f16 ub 65 n_ctx 37120 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1371048 <= 1371048: OK | PASS
RESERVE type f16 ub 66 n_ctx 37120 G_policy 4 gpu 13214592 cpu 2163488 | plan alloc 1379256 >= formula 1379256: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type f16 ub 66 n_ctx 37120 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1379256 <= 1379256: OK | PASS
RESERVE type f16 ub 127 n_ctx 8448 G_policy 4 gpu 19327744 cpu 4163072 | plan alloc 2606824 >= formula 2606824: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type f16 ub 127 n_ctx 8448 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2606824 <= 2606824: OK | PASS
RESERVE type f16 ub 128 n_ctx 8448 G_policy 4 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type f16 ub 128 n_ctx 8448 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type f16 ub 129 n_ctx 8448 G_policy 4 gpu 19632768 cpu 4228640 | plan alloc 2648344 >= formula 2648344: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type f16 ub 129 n_ctx 8448 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2648344 <= 2648344: OK | PASS
RESERVE type f16 ub 64 n_ctx 8448 G_policy 4 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type f16 ub 64 n_ctx 8448 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type f16 ub 65 n_ctx 8448 G_policy 4 gpu 13112832 cpu 2130720 | plan alloc 1334808 >= formula 1334808: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type f16 ub 65 n_ctx 8448 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1334808 <= 1334808: OK | PASS
RESERVE type f16 ub 66 n_ctx 8448 G_policy 4 gpu 13190656 cpu 2163488 | plan alloc 1355312 >= formula 1355312: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type f16 ub 66 n_ctx 8448 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1355312 <= 1355312: OK | PASS
RESERVE type q4_0 ub 127 n_ctx 135424 G_policy 1 gpu 19339008 cpu 4163072 | plan alloc 2618096 >= formula 2618096: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type q4_0 ub 127 n_ctx 135424 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2618096 <= 2618096: OK | PASS
RESERVE type q4_0 ub 128 n_ctx 135424 G_policy 1 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type q4_0 ub 128 n_ctx 135424 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type q4_0 ub 129 n_ctx 135424 G_policy 4 gpu 19668352 cpu 4228640 | plan alloc 2683944 >= formula 2683944: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type q4_0 ub 129 n_ctx 135424 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2683944 <= 2683944: OK | PASS
RESERVE type q4_0 ub 64 n_ctx 135424 G_policy 1 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type q4_0 ub 64 n_ctx 135424 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type q4_0 ub 65 n_ctx 135424 G_policy 1 gpu 13149056 cpu 2130720 | plan alloc 1371048 >= formula 1371048: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type q4_0 ub 65 n_ctx 135424 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1371048 <= 1371048: OK | PASS
RESERVE type q4_0 ub 66 n_ctx 135424 G_policy 1 gpu 13214592 cpu 2163488 | plan alloc 1379256 >= formula 1379256: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type q4_0 ub 66 n_ctx 135424 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1379256 <= 1379256: OK | PASS
RESERVE type q4_0 ub 127 n_ctx 37120 G_policy 4 gpu 19339008 cpu 4163072 | plan alloc 2618096 >= formula 2618096: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type q4_0 ub 127 n_ctx 37120 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2618096 <= 2618096: OK | PASS
RESERVE type q4_0 ub 128 n_ctx 37120 G_policy 4 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type q4_0 ub 128 n_ctx 37120 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type q4_0 ub 129 n_ctx 37120 G_policy 4 gpu 19668352 cpu 4228640 | plan alloc 2683944 >= formula 2683944: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type q4_0 ub 129 n_ctx 37120 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2683944 <= 2683944: OK | PASS
RESERVE type q4_0 ub 64 n_ctx 37120 G_policy 1 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type q4_0 ub 64 n_ctx 37120 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type q4_0 ub 65 n_ctx 37120 G_policy 4 gpu 13149056 cpu 2130720 | plan alloc 1371048 >= formula 1371048: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type q4_0 ub 65 n_ctx 37120 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1371048 <= 1371048: OK | PASS
RESERVE type q4_0 ub 66 n_ctx 37120 G_policy 4 gpu 13214592 cpu 2163488 | plan alloc 1379256 >= formula 1379256: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type q4_0 ub 66 n_ctx 37120 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1379256 <= 1379256: OK | PASS
RESERVE type q4_0 ub 127 n_ctx 8448 G_policy 4 gpu 19327744 cpu 4163072 | plan alloc 2606824 >= formula 2606824: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type q4_0 ub 127 n_ctx 8448 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2606824 <= 2606824: OK | PASS
RESERVE type q4_0 ub 128 n_ctx 8448 G_policy 4 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type q4_0 ub 128 n_ctx 8448 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type q4_0 ub 129 n_ctx 8448 G_policy 4 gpu 19632768 cpu 4228640 | plan alloc 2648344 >= formula 2648344: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type q4_0 ub 129 n_ctx 8448 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2648344 <= 2648344: OK | PASS
RESERVE type q4_0 ub 64 n_ctx 8448 G_policy 4 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type q4_0 ub 64 n_ctx 8448 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type q4_0 ub 65 n_ctx 8448 G_policy 4 gpu 13112832 cpu 2130720 | plan alloc 1334808 >= formula 1334808: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type q4_0 ub 65 n_ctx 8448 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1334808 <= 1334808: OK | PASS
RESERVE type q4_0 ub 66 n_ctx 8448 G_policy 4 gpu 13190656 cpu 2163488 | plan alloc 1355312 >= formula 1355312: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type q4_0 ub 66 n_ctx 8448 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1355312 <= 1355312: OK | PASS
RESERVE type q8_0 ub 127 n_ctx 135424 G_policy 4 gpu 19339008 cpu 4163072 | plan alloc 2618096 >= formula 2618096: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type q8_0 ub 127 n_ctx 135424 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2618096 <= 2618096: OK | PASS
RESERVE type q8_0 ub 128 n_ctx 135424 G_policy 4 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type q8_0 ub 128 n_ctx 135424 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type q8_0 ub 129 n_ctx 135424 G_policy 4 gpu 19668352 cpu 4228640 | plan alloc 2683944 >= formula 2683944: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type q8_0 ub 129 n_ctx 135424 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2683944 <= 2683944: OK | PASS
RESERVE type q8_0 ub 64 n_ctx 135424 G_policy 1 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type q8_0 ub 64 n_ctx 135424 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type q8_0 ub 65 n_ctx 135424 G_policy 1 gpu 13149056 cpu 2130720 | plan alloc 1371048 >= formula 1371048: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type q8_0 ub 65 n_ctx 135424 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1371048 <= 1371048: OK | PASS
RESERVE type q8_0 ub 66 n_ctx 135424 G_policy 1 gpu 13214592 cpu 2163488 | plan alloc 1379256 >= formula 1379256: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type q8_0 ub 66 n_ctx 135424 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1379256 <= 1379256: OK | PASS
RESERVE type q8_0 ub 127 n_ctx 37120 G_policy 4 gpu 19339008 cpu 4163072 | plan alloc 2618096 >= formula 2618096: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type q8_0 ub 127 n_ctx 37120 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2618096 <= 2618096: OK | PASS
RESERVE type q8_0 ub 128 n_ctx 37120 G_policy 4 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type q8_0 ub 128 n_ctx 37120 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type q8_0 ub 129 n_ctx 37120 G_policy 4 gpu 19668352 cpu 4228640 | plan alloc 2683944 >= formula 2683944: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type q8_0 ub 129 n_ctx 37120 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2683944 <= 2683944: OK | PASS
RESERVE type q8_0 ub 64 n_ctx 37120 G_policy 4 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type q8_0 ub 64 n_ctx 37120 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type q8_0 ub 65 n_ctx 37120 G_policy 4 gpu 13149056 cpu 2130720 | plan alloc 1371048 >= formula 1371048: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type q8_0 ub 65 n_ctx 37120 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1371048 <= 1371048: OK | PASS
RESERVE type q8_0 ub 66 n_ctx 37120 G_policy 4 gpu 13214592 cpu 2163488 | plan alloc 1379256 >= formula 1379256: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type q8_0 ub 66 n_ctx 37120 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1379256 <= 1379256: OK | PASS
RESERVE type q8_0 ub 127 n_ctx 8448 G_policy 4 gpu 19327744 cpu 4163072 | plan alloc 2606824 >= formula 2606824: OK | attend alloc 12557760 >= nbytes + G1 partial max 12557760 (n_q' 127): OK
RUNTIME type q8_0 ub 127 n_ctx 8448 shapes 1524 computed 228 nonfinite 0 reallocs 0 | max runtime attend alloc 12557760 <= reserved 12557760, plan 2606824 <= 2606824: OK | PASS
RESERVE type q8_0 ub 128 n_ctx 8448 G_policy 4 gpu 19479808 cpu 4195840 | plan alloc 2627328 >= formula 2627328: OK | attend alloc 12656640 >= nbytes + G1 partial max 12656640 (n_q' 128): OK
RUNTIME type q8_0 ub 128 n_ctx 8448 shapes 1536 computed 240 nonfinite 0 reallocs 0 | max runtime attend alloc 12656640 <= reserved 12656640, plan 2627328 <= 2627328: OK | PASS
RESERVE type q8_0 ub 129 n_ctx 8448 G_policy 4 gpu 19632768 cpu 4228640 | plan alloc 2648344 >= formula 2648344: OK | attend alloc 12755520 >= nbytes + G1 partial max 12755520 (n_q' 129): OK
RUNTIME type q8_0 ub 129 n_ctx 8448 shapes 1548 computed 252 nonfinite 0 reallocs 0 | max runtime attend alloc 12755520 <= reserved 12755520, plan 2648344 <= 2648344: OK | PASS
RESERVE type q8_0 ub 64 n_ctx 8448 G_policy 4 gpu 12910336 cpu 2097920 | plan alloc 1313792 >= formula 1313792: OK | attend alloc 9498624 >= nbytes + G1 partial max 9498624 (n_q' 64): OK
RUNTIME type q8_0 ub 64 n_ctx 8448 shapes 768 computed 108 nonfinite 0 reallocs 0 | max runtime attend alloc 9498624 <= reserved 9498624, plan 1313792 <= 1313792: OK | PASS
RESERVE type q8_0 ub 65 n_ctx 8448 G_policy 4 gpu 13112832 cpu 2130720 | plan alloc 1334808 >= formula 1334808: OK | attend alloc 9647040 >= nbytes + G1 partial max 9647040 (n_q' 65): OK
RUNTIME type q8_0 ub 65 n_ctx 8448 shapes 780 computed 120 nonfinite 0 reallocs 0 | max runtime attend alloc 9647040 <= reserved 9647040, plan 1334808 <= 1334808: OK | PASS
RESERVE type q8_0 ub 66 n_ctx 8448 G_policy 4 gpu 13190656 cpu 2163488 | plan alloc 1355312 >= formula 1355312: OK | attend alloc 9671616 >= nbytes + G1 partial max 9671616 (n_q' 65): OK
RUNTIME type q8_0 ub 66 n_ctx 8448 shapes 792 computed 132 nonfinite 0 reallocs 0 | max runtime attend alloc 9671616 <= reserved 9671616, plan 1355312 <= 1355312: OK | PASS
```

## Policy procs (w4_procs)

```
GROUP shortcut 0 V f16   n_kv     2304: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv     8192: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv    16384: 1:1 2:1 4:1 8:1 16:1 32:4 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv    32768: 1:1 2:1 4:1 8:1 16:1 32:4 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv    49152: 1:1 2:1 4:1 8:1 16:1 32:4 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv    65536: 1:1 2:1 4:1 8:1 16:1 32:4 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv    98304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv   135168: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv   262144: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V f16   n_kv  1048576: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv     2304: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv     8192: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv    16384: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv    32768: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv    49152: 1:1 2:1 4:1 8:1 16:1 32:4 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv    65536: 1:1 2:1 4:1 8:1 16:1 32:4 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv    98304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv   135168: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv   262144: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V bf16  n_kv  1048576: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv     2304: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv     8192: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv    16384: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv    32768: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv    49152: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv    65536: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv    98304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv   135168: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv   262144: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q8_0  n_kv  1048576: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv     2304: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv     8192: 1:1 2:1 4:1 8:1 16:1 32:1 64:4 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv    16384: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv    32768: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv    49152: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv    65536: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:4 127:4 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv    98304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:1 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv   135168: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:1 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv   262144: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:1 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 0 V q4_0  n_kv  1048576: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:1 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv     2304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv     8192: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv    16384: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv    32768: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv    49152: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv    65536: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv    98304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv   135168: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv   262144: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V f16   n_kv  1048576: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv     2304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv     8192: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv    16384: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv    32768: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv    49152: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv    65536: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv    98304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv   135168: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv   262144: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V bf16  n_kv  1048576: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv     2304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv     8192: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv    16384: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv    32768: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv    49152: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv    65536: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv    98304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv   135168: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv   262144: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q8_0  n_kv  1048576: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv     2304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv     8192: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv    16384: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv    32768: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv    49152: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv    65536: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv    98304: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv   135168: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv   262144: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
GROUP shortcut 1 V q4_0  n_kv  1048576: 1:1 2:1 4:1 8:1 16:1 32:1 64:1 96:1 127:1 128:4 255:4 256:4 511:4 512:4 1024:4 4096:4
DENSE K f16   V f16  : 1 1 1 1 1 1 1
DENSE K f16   V bf16 : 0 0 0 0 0 0 0
DENSE K f16   V q8_0 : 0 0 0 0 0 0 0
DENSE K f16   V q4_0 : 0 0 0 0 0 0 0
DENSE K bf16  V f16  : 0 0 0 0 0 0 0
DENSE K bf16  V bf16 : 0 0 0 0 0 0 0
DENSE K bf16  V q8_0 : 0 0 0 0 0 0 0
DENSE K bf16  V q4_0 : 0 0 0 0 0 0 0
DENSE K q8_0  V f16  : 0 0 0 0 0 0 0
DENSE K q8_0  V bf16 : 0 0 0 0 0 0 0
DENSE K q8_0  V q8_0 : 0 0 0 0 0 0 0
DENSE K q8_0  V q4_0 : 0 0 0 0 0 0 0
DENSE K q4_0  V f16  : 0 0 0 0 0 0 0
DENSE K q4_0  V bf16 : 0 0 0 0 0 0 0
DENSE K q4_0  V q8_0 : 0 0 0 0 0 0 0
DENSE K q4_0  V q4_0 : 0 0 0 0 0 0 0
FORCE checks wrong 0 -> OK
```
