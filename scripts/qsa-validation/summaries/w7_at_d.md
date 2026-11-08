# W7 AT-D (HIP graph placement)

Source: strix-002:~/dev/wt_lab/w7/at_d/*.log; D1, D3, D2 at 0294c0274+789a85c12 (chain11.out), D4, D5 at dcaa19f16 (chainF1.out). Command: scripts/qsa-validation/harness/w7/at_d.sh D1 D3 D2 D4 D5

```
PHASE D1 start 09:04:36
d1 f16 new rc=0 kind4 per graph:    1022 0;     17 12;
d1 f16 w0 rc=0 kind4 per graph:    1035 0;
d1 q8_0 new rc=0 kind4 per graph:    1022 0;     17 12;
d1 q8_0 w0 rc=0 kind4 per graph:    1035 0;
u16 f16 OVL=15 rc=0
u16 f16 OVL=11 rc=0
u16 f16 hash: 0 differing lines of 20
u16 q8_0 OVL=15 rc=0
u16 q8_0 OVL=11 rc=0
u16 q8_0 hash: 0 differing lines of 20
PHASE D1 done 10:47:26
PHASE D3 start 10:47:26
d3_f16_new rc=0 Final estimate: PPL = 1.3832 +/- 0.05075 s/pass - ETA 0.70 minutes guard=[passes=7 inspected=84 inspected_min=12 layers=12 cpu_launch=0]
  placement       5 3 CPU
  placement       5 7 CPU
  placement       5 11 CPU
  placement       5 15 CPU
  placement       5 19 CPU
  placement       5 23 CPU
  placement       5 27 ROCm0
  placement       5 31 ROCm0
  placement       5 35 ROCm0
  placement       5 39 ROCm0
  placement       5 43 ROCm0
  placement       5 47 ROCm0
d3_q8_0_new rc=0 Final estimate: PPL = 1.3873 +/- 0.05087 s/pass - ETA 0.45 minutes guard=[passes=7 inspected=156 inspected_min=18 layers=12 cpu_launch=0]
  placement       5 3 CPU
  placement       5 7 CPU
  placement       5 11 CPU
  placement       5 15 CPU
  placement       5 19 CPU
  placement       5 23 CPU
  placement       5 27 ROCm0
  placement       5 31 ROCm0
  placement       5 35 ROCm0
  placement       5 39 ROCm0
  placement       5 43 ROCm0
  placement       5 47 ROCm0
PHASE D3 done 11:07:00
PHASE D2 start 11:07:00
d2_f16_0_w0 rc=0 Final estimate: PPL = 2.0851 +/- 0.03865 s/pass - ETA 3.93 minutes guard=[]
d2_f16_1_new rc=0 Final estimate: PPL = 2.0835 +/- 0.03866 s/pass - ETA 5.17 minutes guard=[passes=35 inspected=636 inspected_min=12 layers=12 cpu_launch=0]
d2_f16_2_w0 rc=0 Final estimate: PPL = 2.0851 +/- 0.03865 s/pass - ETA 12.08 minutes guard=[]
d2_f16_3_new rc=0 Final estimate: PPL = 2.0835 +/- 0.03866 s/pass - ETA 6.77 minutes guard=[passes=35 inspected=636 inspected_min=12 layers=12 cpu_launch=0]
d2_f16_4_w0 rc=0 Final estimate: PPL = 2.0851 +/- 0.03865 s/pass - ETA 4.00 minutes guard=[]
d2_f16_5_new rc=0 Final estimate: PPL = 2.0835 +/- 0.03866 s/pass - ETA 4.83 minutes guard=[passes=35 inspected=636 inspected_min=12 layers=12 cpu_launch=0]
d2_q8_0_0_w0 rc=0 Final estimate: PPL = 2.0866 +/- 0.03865 s/pass - ETA 5.52 minutes guard=[]
d2_q8_0_1_new rc=0 Final estimate: PPL = 2.0858 +/- 0.03869 s/pass - ETA 4.08 minutes guard=[passes=35 inspected=840 inspected_min=24 layers=12 cpu_launch=0]
d2_q8_0_2_w0 rc=0 Final estimate: PPL = 2.0866 +/- 0.03865 s/pass - ETA 4.83 minutes guard=[]
d2_q8_0_3_new rc=0 Final estimate: PPL = 2.0858 +/- 0.03869 s/pass - ETA 4.85 minutes guard=[passes=35 inspected=840 inspected_min=24 layers=12 cpu_launch=0]
d2_q8_0_4_w0 rc=0 Final estimate: PPL = 2.0866 +/- 0.03865 s/pass - ETA 4.25 minutes guard=[]
d2_q8_0_5_new rc=0 Final estimate: PPL = 2.0858 +/- 0.03869 s/pass - ETA 12.95 minutes guard=[passes=35 inspected=840 inspected_min=24 layers=12 cpu_launch=0]
PHASE D2 done 13:43:18
AT_D_DONE
CHAIN11_DONE
PHASE D4 start 23:23:54
d4_f16_u4096_new rc=0 Final estimate: PPL = 1.9477 +/- 0.03451 s/pass - ETA 0.78 minutes guard=[passes=7 inspected=156 inspected_min=12 layers=12 cpu_launch=0]
d4_f16_u4096_w0 rc=0 Final estimate: PPL = 1.9471 +/- 0.03445 s/pass - ETA 0.77 minutes guard=[]
d4 sched f16 u4096: cache_[kv] split inputs to ROCm0 lines: 772
d4_f16_u512_new rc=0 Final estimate: PPL = 1.9442 +/- 0.03428 s/pass - ETA 0.83 minutes guard=[passes=35 inspected=732 inspected_min=12 layers=12 cpu_launch=0]
d4_f16_u512_w0 rc=0 Final estimate: PPL = 1.9460 +/- 0.03439 s/pass - ETA 0.85 minutes guard=[]
d4 sched f16 u512: cache_[kv] split inputs to ROCm0 lines: 772
d4_f16_u16_new rc=0 Final estimate: PPL = 1.9387 +/- 0.03409 s/pass - ETA 3.42 minutes guard=[passes=69 inspected=1428 inspected_min=12 layers=12 cpu_launch=0]
d4_f16_u16_w0 rc=0 Final estimate: PPL = 1.9376 +/- 0.03415 s/pass - ETA 3.43 minutes guard=[]
d4 sched f16 u16: cache_[kv] split inputs to ROCm0 lines: 748
d4_q8_0_u4096_new rc=0 Final estimate: PPL = 1.9453 +/- 0.03438 s/pass - ETA 0.78 minutes guard=[passes=7 inspected=156 inspected_min=12 layers=12 cpu_launch=0]
d4_q8_0_u4096_w0 rc=0 Final estimate: PPL = 1.9602 +/- 0.03493 s/pass - ETA 0.77 minutes guard=[]
d4 sched q8_0 u4096: cache_[kv] split inputs to ROCm0 lines: 772
d4_q8_0_u512_new rc=0 Final estimate: PPL = 1.9453 +/- 0.03438 s/pass - ETA 0.95 minutes guard=[passes=35 inspected=828 inspected_min=12 layers=12 cpu_launch=0]
d4_q8_0_u512_w0 rc=0 Final estimate: PPL = 1.9508 +/- 0.03462 s/pass - ETA 0.98 minutes guard=[]
d4 sched q8_0 u512: cache_[kv] split inputs to ROCm0 lines: 772
d4_q8_0_u16_new rc=0 Final estimate: PPL = 1.9406 +/- 0.03434 s/pass - ETA 3.33 minutes guard=[passes=69 inspected=1428 inspected_min=12 layers=12 cpu_launch=0]
d4_q8_0_u16_w0 rc=0 Final estimate: PPL = 1.9447 +/- 0.03441 s/pass - ETA 3.80 minutes guard=[]
d4 sched q8_0 u16: cache_[kv] split inputs to ROCm0 lines: 748
PHASE D4 done 00:03:00
PHASE D5 start 00:03:00
d5_nkvo_auto_new rc=0 Final estimate: PPL = 1.9728 +/- 0.05134 s/pass - ETA 0.43 minutes guard=[passes=19 inspected=348 inspected_min=12 layers=12 cpu_launch=0]
d5_off_u512_new rc=0 Final estimate: PPL = 1.9565 +/- 0.03476 s/pass - ETA 0.38 minutes guard=[passes=35 inspected=732 inspected_min=12 layers=12 cpu_launch=0]
d5_off_nkvo_u512_new rc=0 Final estimate: PPL = 1.9475 +/- 0.03440 s/pass - ETA 0.92 minutes guard=[passes=35 inspected=732 inspected_min=12 layers=12 cpu_launch=0]
d5_off_u4096_new rc=0 Final estimate: PPL = 1.9485 +/- 0.03445 s/pass - ETA 0.23 minutes guard=[passes=7 inspected=156 inspected_min=12 layers=12 cpu_launch=0]
d5_off_nkvo_u4096_new rc=0 Final estimate: PPL = 1.9477 +/- 0.03451 s/pass - ETA 0.88 minutes guard=[passes=7 inspected=156 inspected_min=12 layers=12 cpu_launch=0]
d5_off_u512_w0 rc=0 Final estimate: PPL = 1.9573 +/- 0.03483 s/pass - ETA 0.45 minutes guard=[]
d5_off_nkvo_u512_w0 rc=0 Final estimate: PPL = 1.9505 +/- 0.03452 s/pass - ETA 1.03 minutes guard=[]
d5_off_u4096_w0 rc=0 Final estimate: PPL = 1.9574 +/- 0.03482 s/pass - ETA 0.47 minutes guard=[]
d5_off_nkvo_u4096_w0 rc=0 Final estimate: PPL = 1.9469 +/- 0.03431 s/pass - ETA 0.90 minutes guard=[]
LLAMA_QSA_TRACE il=3 producer=cb n_q=512 n_kv=2560 n_stream=1 k=f16 v=f16 count=2 route=host-kv-copy ctx=main backend=ROCm0 G=4 n_sel=2051 gqa=12
```
