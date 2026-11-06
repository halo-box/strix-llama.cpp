# W4 G1 kernel gates on the C18 decode fixtures

w4_dec time mode, C18 protocol; results strix-002:~/dev/wt_lab/w4/bands/c18/. Gate: new <= same-run C18 harness kernel x 1.014 on wiki d131k (PLAN-v3 §13 W4); rand d131k and f16 reported.

```
## wikiL_p131072 n_q 512 (ms, mean of r1/r2; r1/r2 spread in brackets)
GATE wikiL_p131072  q8_0  n_q 512 G1   new 3.3093 C18 3.7544 ratio 0.881 limit 3.8070 [spread 0.0%] PASS
GATE wikiL_p131072  q8_0  n_q 512 G1M  new 3.3441 C18 3.8333 ratio 0.872 limit 3.8869 [spread 0.0%] PASS
GATE wikiL_p131072  q4_0  n_q 512 G1   new 3.0181 C18 3.5958 ratio 0.839 limit 3.6461 [spread 0.1%] PASS
GATE wikiL_p131072  q4_0  n_q 512 G1M  new 3.0440 C18 3.6462 ratio 0.835 limit 3.6972 [spread 0.0%] PASS
GATE wikiL_p131072  bf16  n_q 512 G1   new 2.9693 C18 3.5549 ratio 0.835 limit 3.6046 [spread 0.2%] PASS
GATE wikiL_p131072  bf16  n_q 512 G1M  new 2.9984 C18 3.6105 ratio 0.830 limit 3.6610 [spread 0.1%] PASS
GATE wikiL_p131072  f16   n_q 512 G1   new 2.9500 C18 3.3362 ratio 0.884 limit 3.3830 [spread 0.1%] (reported)
GATE wikiL_p131072  f16   n_q 512 G1M  new 2.9810 C18 3.4107 ratio 0.874 limit 3.4584 [spread 0.0%] (reported)
PR    wikiL_p131072  f16   n_q 512 PR-dec 4.6584 new G1 2.9500 ratio 0.633
PR    wikiL_p131072  q8_0  n_q 512 PR-dec 6.4795 new G1 3.3093 ratio 0.511
## wikiL_p131072 n_q 1 (ms, mean of r1/r2; r1/r2 spread in brackets)
GATE wikiL_p131072  q8_0  n_q   1 G1   new 0.0183 C18 0.0199 ratio 0.920 limit 0.0202 [spread 0.0%] PASS
GATE wikiL_p131072  q8_0  n_q   1 G1M  new 0.0272 C18 0.0293 ratio 0.928 limit 0.0297 [spread 0.0%] PASS
GATE wikiL_p131072  q4_0  n_q   1 G1   new 0.0176 C18 0.0208 ratio 0.846 limit 0.0211 [spread 0.0%] PASS
GATE wikiL_p131072  q4_0  n_q   1 G1M  new 0.0267 C18 0.0290 ratio 0.921 limit 0.0294 [spread 0.4%] PASS
GATE wikiL_p131072  bf16  n_q   1 G1   new 0.0172 C18 0.0199 ratio 0.864 limit 0.0202 [spread 0.0%] PASS
GATE wikiL_p131072  bf16  n_q   1 G1M  new 0.0262 C18 0.0287 ratio 0.914 limit 0.0291 [spread 0.0%] PASS
GATE wikiL_p131072  f16   n_q   1 G1   new 0.0171 C18 0.0188 ratio 0.912 limit 0.0191 [spread 0.6%] (reported)
GATE wikiL_p131072  f16   n_q   1 G1M  new 0.0263 C18 0.0284 ratio 0.928 limit 0.0288 [spread 0.4%] (reported)
PR    wikiL_p131072  f16   n_q   1 PR-dec 0.0239 new G1 0.0171 ratio 0.719
PR    wikiL_p131072  q8_0  n_q   1 PR-dec 0.0316 new G1 0.0183 ratio 0.579
## randL_p131072 n_q 512 (ms, mean of r1/r2; r1/r2 spread in brackets)
GATE randL_p131072  q8_0  n_q 512 G1   new 3.7035 C18 4.0913 ratio 0.905 limit 4.1486 [spread 0.1%] (reported)
GATE randL_p131072  q8_0  n_q 512 G1M  new 3.7412 C18 4.1604 ratio 0.899 limit 4.2187 [spread 0.1%] (reported)
GATE randL_p131072  q4_0  n_q 512 G1   new 3.2290 C18 3.7915 ratio 0.852 limit 3.8446 [spread 0.0%] (reported)
GATE randL_p131072  q4_0  n_q 512 G1M  new 3.2820 C18 3.8616 ratio 0.850 limit 3.9156 [spread 0.1%] (reported)
GATE randL_p131072  bf16  n_q 512 G1   new 3.7321 C18 4.0711 ratio 0.917 limit 4.1281 [spread 0.0%] (reported)
GATE randL_p131072  bf16  n_q 512 G1M  new 3.7938 C18 4.1406 ratio 0.916 limit 4.1986 [spread 0.1%] (reported)
GATE randL_p131072  f16   n_q 512 G1   new 3.7344 C18 3.9394 ratio 0.948 limit 3.9946 [spread 0.1%] (reported)
GATE randL_p131072  f16   n_q 512 G1M  new 3.7892 C18 4.0051 ratio 0.946 limit 4.0612 [spread 0.2%] (reported)
PR    randL_p131072  f16   n_q 512 PR-dec 5.1994 new G1 3.7344 ratio 0.718
PR    randL_p131072  q8_0  n_q 512 PR-dec 7.5000 new G1 3.7035 ratio 0.494
## randL_p131072 n_q 1 (ms, mean of r1/r2; r1/r2 spread in brackets)
GATE randL_p131072  q8_0  n_q   1 G1   new 0.0184 C18 0.0200 ratio 0.922 limit 0.0202 [spread 0.0%] (reported)
GATE randL_p131072  q8_0  n_q   1 G1M  new 0.0272 C18 0.0294 ratio 0.927 limit 0.0298 [spread 0.4%] (reported)
GATE randL_p131072  q4_0  n_q   1 G1   new 0.0176 C18 0.0209 ratio 0.844 limit 0.0211 [spread 0.0%] (reported)
GATE randL_p131072  q4_0  n_q   1 G1M  new 0.0267 C18 0.0290 ratio 0.919 limit 0.0295 [spread 0.0%] (reported)
GATE randL_p131072  bf16  n_q   1 G1   new 0.0172 C18 0.0199 ratio 0.864 limit 0.0202 [spread 0.0%] (reported)
GATE randL_p131072  bf16  n_q   1 G1M  new 0.0263 C18 0.0287 ratio 0.915 limit 0.0291 [spread 0.4%] (reported)
GATE randL_p131072  f16   n_q   1 G1   new 0.0172 C18 0.0188 ratio 0.912 limit 0.0191 [spread 0.0%] (reported)
GATE randL_p131072  f16   n_q   1 G1M  new 0.0264 C18 0.0285 ratio 0.926 limit 0.0289 [spread 0.0%] (reported)
PR    randL_p131072  f16   n_q   1 PR-dec 0.0239 new G1 0.0172 ratio 0.720
PR    randL_p131072  q8_0  n_q   1 PR-dec 0.0316 new G1 0.0184 ratio 0.581
ALL_GATES PASS
```
