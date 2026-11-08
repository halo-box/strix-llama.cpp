# W7 perf: P1 and AT-D2 (-ngl 0)

Source: strix-002:~/dev/wt_lab/w7/perf/p1_*.json, d2b_*.json. Command: python3 scripts/qsa-validation/harness/w7/perf_an.py ~/dev/wt_lab/w7/perf P1 D2B. AT-D2 rule: new/W0 >= 1 - 2 x the W0 between-process spread (the printed threshold); the table verdict column uses theta 0 and is not the AT-D2 verdict.

### P1 text q8_0 ub512 pp4096 d0, new vs base 13a279bfa (theta 0.995)

| depth | ub | pp | tg | ref t/s (procs) | new t/s (procs) | ratio | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | 512 | 4096 | 0 | 1219.44 (4) | 1252.82 (3) | 1.0274 | [1.0227, 1.0325] | PASS |

### AT-D2 -ngl 0 f16, new vs W0-lab (threshold 1 - 2 x the W0 between-process spread, printed below) (theta 0.0)

| depth | ub | pp | tg | ref t/s (procs) | new t/s (procs) | ratio | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|
| 4096 | 512 | 0 | 32 | 10.77 (3) | 9.56 (3) | 0.8877 | [0.8826, 0.8925] | PASS |
| 4096 | 512 | 512 | 0 | 163.58 (3) | 116.65 (3) | 0.7131 | [0.6986, 0.7326] | PASS |
  W0 spread (4096, 512, 512, 0): 0.0408 -> threshold 0.9185
  W0 spread (4096, 512, 0, 32): 0.0042 -> threshold 0.9917

### AT-D2 -ngl 0 q8_0, new vs W0-lab (threshold 1 - 2 x the W0 between-process spread, printed below) (theta 0.0)

| depth | ub | pp | tg | ref t/s (procs) | new t/s (procs) | ratio | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|
| 4096 | 512 | 0 | 32 | 10.53 (3) | 9.39 (3) | 0.8918 | [0.8786, 0.9067] | PASS |
| 4096 | 512 | 512 | 0 | 164.33 (3) | 113.88 (3) | 0.6930 | [0.6818, 0.7064] | PASS |
  W0 spread (4096, 512, 512, 0): 0.0075 -> threshold 0.9850
  W0 spread (4096, 512, 0, 32): 0.0067 -> threshold 0.9867

## totals {'PASS': 5}; eviction samples dropped 1 in 1 processes; processes with evictions 9/19, mean evicted_ms 140; P6 (a) samples below 0.9 x cell median: 0 []
