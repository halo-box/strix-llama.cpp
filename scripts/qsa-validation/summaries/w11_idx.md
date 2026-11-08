# W11: indexer cache mapping (PLAN-v3 §9, D14), no session format change

Build: HIP `build/` at the W11 code commit. Logs on strix-002: `~/dev/wt_rewrite_logs/W11/`, `~/dev/wt_lab/w11/`.

## Unit test (tests/test-qsa-idx-type.cpp, CPU, no model file)
- mapping: qwen4exp auto q4_0, q4_1, q5_0, q5_1, iq4_nl -> q8_0; f32, f16, bf16, q8_0 unchanged; explicit override wins;
  GLM5_NEXT keeps -ctk with or without an override (9 x 10 cases).
- create_memory: qwen4exp main / MTP q4_0 -> indexer tensor q8_0; f16, q8_0, bf16 unchanged; GLM5_NEXT main / MTP q4_0 -> q4_0.
- session state: an old qwen4exp q4_0 state (q4_0 indexer) is rejected by the existing per-layer check
  (`state_read_data: mismatched key type (8 != 2, layer 0)`) with the wrapped message
  `failed to restore the sparse-attention indexer cache (first QSA layer 0): the session was saved with indexer type q4_0,
  this context uses q8_0; re-save the session, or load it with --cache-type-k-idx q4_0`; with that override it loads;
  qwen4exp f16 / q8_0 and GLM5_NEXT q4_0 / f16 states load; an attention-section mismatch keeps `failed to restore kv cache`.

## Real model, session file saved by W0-lab12 (PR219 arm), loaded by W11 (harness/w11/session.sh)
| -ctk | W11 load |
|---|---|
| q4_0 | rejected: `mismatched key type (8 != 2, layer 3)` then `error loading session file: failed to restore the sparse-attention indexer cache (first QSA layer 3): the session was saved with indexer type q4_0, this context uses q8_0; re-save the session, or load it with --cache-type-k-idx q4_0` |
| q4_0 + `--cache-type-k-idx q4_0` | loaded (48 tokens) |
| f16 | loaded (48 tokens) |
| q8_0 | loaded (48 tokens) |

## Real model smoke (harness/w11/smoke_kld.sh, c8192 b/ub4096, 1 chunk, KL base = pre-W11 build ~/dev/wt_lab/w10/base_bin)
| arm | Mean KLD | Max KLD | Same top p | PPL |
|---|---|---|---|---|
| q4_0 auto (indexer q8_0) | 0.028154 +- 0.001406 | 2.708 | 96.068 % | 2.1271 -> 2.124025 |
| q4_0 --cache-type-k-idx q4_0 | 0.000000 | 5.5e-5 | 100 % | 2.127075 (ratio 1.000044 +- 0.000044) |
| f16 auto | 0.000000 | 5.8e-5 | 100 % | 2.108191 |
| q8_0 auto | 0.000000 | 5.1e-5 | 100 % | 2.113919 |
The override / f16 / q8_0 rows equal W10's smoke against the same base digit for digit (IMPLEMENTATION.md W10).

## Census (W10 census.sh, graphs off, q4_0 auto): ub4096 23/24 gated, ub512 184/192 gated, EXPECT PASS (= W10).

## MTP server smoke (harness/w11/mtp_smoke.sh, -ctk/-ctkd q4_0, c32768, request S0)
- both contexts: `indexer K cache type = q8_0 (auto, -ctk q4_0)`; indexer KV lines `K (q8_0): 51.00 MiB` (main, 12 layers) and
  `K (q8_0): 4.25 MiB` (MTP); with the q4_0 override 27.00 / 2.25 MiB (x1.89, C9 d3). QSA lines `idx=q8_0`, cpu=0, request OK.
- census_check expect: every route key OK, but the gated share of the fusible G4 attends is 0/12 (floor 0.9), identically
  with `--cache-type-k-idx q4_0` (the pre-W11 indexer type): not caused by W11; reported for the owner / Prove.
