# W6 AT-B2 / AT-B3: op time on Vulkan (HEAD b1cb6764c, RADV, `harness/w6/time_campaign.sh`)

3 interleaved processes per arm, 5 reps after 3 warm-ups, R = 2 attend nodes. Rows: captured text and random (C11 shape: c3_capture ub 2048 at pos0 32768, n_kv 34816, 11 x 2048 q + 1 x 512 q, all 12 layers; d4096, d131072: C18 rows, n_q 4096, layers 0, 5, 11). Q / K / V: SYN (uniform +-1.7): the graph uses the op from W7 on, so the model data cannot reach it yet. Statistic: per process the mean over layers of the per-layer median; median over processes. Logger = GGML_VK_PERF_LOGGER op time, wall = ggml_backend_graph_compute. "sel" = the old selected-key FLASH_ATTN_EXT in the same build, its pre-pass inside.

```
rows force arm procs | wall attend, plan, total (ms) | logger op attend, plan (ms)
c11rand      default  bf16       3 |    10.321    1.481    11.858 |    10.054    1.030
c11rand      default  f16        3 |     7.896    1.641     9.520 |     7.707    1.189
c11rand      default  f16:q8_0   3 |    12.754    1.596    14.347 |    12.479    1.188
c11rand      default  q4_0       3 |    15.104    1.609    16.700 |    14.621    1.187
c11rand      default  q8_0       3 |    16.086    1.570    17.728 |    15.911    1.181
c11rand      default  q8_0:f16   3 |     9.958    1.684    11.642 |     9.743    1.189
c11rand      default  sel        3 |     8.938    0.000     8.938 |     8.432      nan
c11rand      vdirect  bf16       3 |    10.400    1.421    11.820 |    10.051    1.020
c11rand      vdirect  f16        3 |     7.950    1.621     9.571 |     7.742    1.184
c11rand      vdirect  q4_0       3 |    15.066    1.620    16.646 |    14.587    1.189
c11rand      vdirect  q8_0       3 |    16.037    1.611    17.682 |    15.824    1.187
c11rand      vpack    bf16       3 |     9.565    1.370    10.936 |     9.161    1.011
c11rand      vpack    f16        3 |     7.971    1.582     9.553 |     7.699    1.181
c11rand      vpack    q4_0       3 |     9.705    1.538    11.233 |     9.404    1.181
c11rand      vpack    q8_0       3 |    10.046    1.584    11.582 |     9.757    1.176
c11rand      vstaged  bf16       3 |    19.179    1.430    20.637 |    18.926    1.026
c11rand      vstaged  f16        3 |    15.948    1.653    17.619 |    15.799    1.190
c11rand      vstaged  q4_0       3 |    16.846    1.598    18.491 |    16.149    1.181
c11rand      vstaged  q8_0       3 |    18.149    1.598    19.748 |    17.924    1.188
c11text      default  bf16       3 |     9.843    1.466    11.359 |     9.563    1.023
c11text      default  f16        3 |     7.278    1.627     8.925 |     7.070    1.187
c11text      default  f16:q8_0   3 |    12.394    1.629    14.056 |    12.199    1.193
c11text      default  q4_0       3 |    15.393    1.654    17.074 |    14.552    1.186
c11text      default  q8_0       3 |    16.012    1.578    17.713 |    15.840    1.186
c11text      default  q8_0:f16   3 |     9.799    1.626    11.425 |     9.577    1.187
c11text      default  sel        3 |     8.235    0.000     8.235 |     7.709      nan
c11text      vdirect  bf16       3 |     9.843    1.425    11.246 |     9.539    1.017
c11text      vdirect  f16        3 |     7.232    1.654     8.887 |     7.070    1.190
c11text      vdirect  q4_0       3 |    15.325    1.591    16.916 |    14.435    1.187
c11text      vdirect  q8_0       3 |    15.969    1.609    17.556 |    15.746    1.186
c11text      vpack    bf16       3 |     8.687    1.454    10.135 |     8.328    1.038
c11text      vpack    f16        3 |     7.245    1.637     8.885 |     7.055    1.184
c11text      vpack    q4_0       3 |     8.850    1.559    10.409 |     8.564    1.184
c11text      vpack    q8_0       3 |     9.109    1.666    10.791 |     8.909    1.189
c11text      vstaged  bf16       3 |    16.928    1.462    18.379 |    16.679    1.001
c11text      vstaged  f16        3 |    13.746    1.599    15.371 |    13.599    1.189
c11text      vstaged  q4_0       3 |    17.279    1.647    18.923 |    16.065    1.188
c11text      vstaged  q8_0       3 |    17.565    1.611    19.176 |    17.253    1.187
d131krand    default  bf16       3 |    25.557    2.455    28.006 |    25.302    1.949
d131krand    default  f16        3 |    23.322    2.397    25.729 |    23.137    1.950
d131krand    default  f16:q8_0   3 |    27.258    2.428    29.765 |    27.013    1.968
d131krand    default  q4_0       3 |    30.975    2.454    33.576 |    30.657    1.974
d131krand    default  q8_0       3 |    33.555    2.514    36.061 |    33.285    1.987
d131krand    default  q8_0:f16   3 |    21.773    2.599    24.341 |    21.589    1.985
d131krand    default  sel        3 |    24.557    0.000    24.557 |    24.141      nan
d131krand    vdirect  bf16       3 |    25.654    2.585    28.110 |    25.299    1.959
d131krand    vdirect  f16        3 |    23.283    2.492    25.888 |    23.085    1.969
d131krand    vdirect  q4_0       3 |    30.979    2.464    33.474 |    30.588    1.983
d131krand    vdirect  q8_0       3 |    33.530    2.471    36.010 |    33.225    1.962
d131krand    vpack    bf16       3 |    30.865    2.486    33.305 |    30.696    1.958
d131krand    vpack    f16        3 |    23.441    2.401    25.912 |    23.210    1.963
d131krand    vpack    q4_0       3 |    29.680    2.528    32.229 |    29.501    1.979
d131krand    vpack    q8_0       3 |    30.584    2.430    33.023 |    30.330    1.966
d131krand    vstaged  bf16       3 |    44.955    2.452    47.449 |    44.751    1.951
d131krand    vstaged  f16        3 |    38.751    2.387    41.099 |    38.491    1.952
d131krand    vstaged  q4_0       3 |    37.882    2.510    40.392 |    36.704    1.974
d131krand    vstaged  q8_0       3 |    41.950    2.439    44.347 |    41.701    1.970
d131ktext    default  bf16       3 |    21.246    2.371    23.738 |    20.973    1.972
d131ktext    default  f16        3 |    16.534    2.523    19.130 |    16.251    1.955
d131ktext    default  f16:q8_0   3 |    26.359    2.492    28.829 |    26.001    1.960
d131ktext    default  q4_0       3 |    32.483    2.436    34.919 |    30.792    1.967
d131ktext    default  q8_0       3 |    33.498    2.465    35.905 |    33.233    1.966
d131ktext    default  q8_0:f16   3 |    20.876    2.441    23.317 |    20.430    1.983
d131ktext    default  sel        3 |    18.198    0.000    18.198 |    17.658      nan
d131ktext    vdirect  bf16       3 |    21.062    2.511    23.596 |    20.812    1.969
d131ktext    vdirect  f16        3 |    16.429    2.481    18.869 |    16.133    1.958
d131ktext    vdirect  q4_0       3 |    31.938    2.535    34.425 |    30.584    1.970
d131ktext    vdirect  q8_0       3 |    33.338    2.339    35.669 |    33.072    1.988
d131ktext    vpack    bf16       3 |    23.467    2.687    26.096 |    23.228    1.962
d131ktext    vpack    f16        3 |    16.352    2.479    18.746 |    16.004    1.974
d131ktext    vpack    q4_0       3 |    22.345    2.578    24.917 |    22.138    1.962
d131ktext    vpack    q8_0       3 |    23.213    2.422    25.650 |    22.896    1.959
d131ktext    vstaged  bf16       3 |    38.561    2.420    40.995 |    38.323    1.949
d131ktext    vstaged  f16        3 |    32.112    2.486    34.599 |    31.930    1.951
d131ktext    vstaged  q4_0       3 |    37.385    2.439    39.817 |    35.689    1.976
d131ktext    vstaged  q8_0       3 |    38.688    2.456    41.106 |    38.188    1.979
d4krand      default  bf16       3 |    20.616    2.619    23.180 |    20.232    2.084
d4krand      default  f16        3 |    15.134    2.547    17.678 |    14.585    2.102
d4krand      default  f16:q8_0   3 |    26.127    2.570    28.665 |    25.704    2.095
d4krand      default  q4_0       3 |    32.120    2.512    34.631 |    30.831    2.093
d4krand      default  q8_0       3 |    34.063    2.499    36.599 |    33.633    2.087
d4krand      default  q8_0:f16   3 |    20.885    2.599    23.512 |    20.507    2.091
d4krand      default  sel        3 |    16.583    0.000    16.583 |    16.032      nan
d4krand      vdirect  bf16       3 |    20.474    2.543    23.045 |    20.214    2.095
d4krand      vdirect  f16        3 |    15.179    2.540    17.757 |    14.595    2.089
d4krand      vdirect  q4_0       3 |    32.424    2.625    35.048 |    30.903    2.086
d4krand      vdirect  q8_0       3 |    34.278    2.543    36.820 |    33.884    2.069
d4krand      vpack    bf16       3 |    14.283    2.503    16.768 |    13.938    2.081
d4krand      vpack    f16        3 |    15.253    2.516    17.754 |    14.619    2.089
d4krand      vpack    q4_0       3 |    14.563    2.540    17.103 |    14.200    2.077
d4krand      vpack    q8_0       3 |    14.567    2.518    17.066 |    14.148    2.083
d4krand      vstaged  bf16       3 |    33.899    2.520    36.419 |    33.720    2.105
d4krand      vstaged  f16        3 |    27.404    2.579    29.979 |    27.143    2.075
d4krand      vstaged  q4_0       3 |    35.333    2.615    37.939 |    33.535    2.078
d4krand      vstaged  q8_0       3 |    36.655    2.552    39.407 |    36.170    2.083
d4ktext      default  bf16       3 |    20.341    2.673    23.014 |    20.023    2.106
d4ktext      default  f16        3 |    15.040    2.503    17.557 |    14.490    2.069
d4ktext      default  f16:q8_0   3 |    25.870    2.583    28.392 |    25.515    2.075
d4ktext      default  q4_0       3 |    32.392    2.586    35.009 |    30.483    2.093
d4ktext      default  q8_0       3 |    33.790    2.503    36.313 |    33.429    2.072
d4ktext      default  q8_0:f16   3 |    20.570    2.540    23.121 |    20.274    2.091
d4ktext      default  sel        3 |    16.468    0.000    16.468 |    15.811      nan
d4ktext      vdirect  bf16       3 |    20.373    2.534    22.903 |    19.986    2.115
d4ktext      vdirect  f16        3 |    14.891    2.580    17.506 |    14.454    2.080
d4ktext      vdirect  q4_0       3 |    32.418    2.613    35.005 |    30.652    2.113
d4ktext      vdirect  q8_0       3 |    33.783    2.580    36.363 |    33.391    2.087
d4ktext      vpack    bf16       3 |    14.127    2.574    16.619 |    13.717    2.097
d4ktext      vpack    f16        3 |    14.930    2.623    17.495 |    14.393    2.074
d4ktext      vpack    q4_0       3 |    14.261    2.579    16.838 |    14.001    2.077
d4ktext      vpack    q8_0       3 |    14.293    2.603    16.913 |    13.966    2.078
d4ktext      vstaged  bf16       3 |    33.525    2.709    36.149 |    33.367    2.089
d4ktext      vstaged  f16        3 |    27.015    2.475    29.663 |    26.802    2.070
d4ktext      vstaged  q4_0       3 |    34.383    2.672    37.055 |    32.655    2.080
d4ktext      vstaged  q8_0       3 |    37.169    2.529    39.689 |    36.193    2.095

AT-B2: new f16 (default arm) / in-build sel_cm, per rows file: pass <= 1.10
c11rand      wall: attend/sel 0.883  plan+attend/sel 1.065  sel 8.938 ms | logger: attend/sel 0.914  plan+attend/sel 1.055  sel 8.432 ms
c11text      wall: attend/sel 0.884  plan+attend/sel 1.084  sel 8.235 ms | logger: attend/sel 0.917  plan+attend/sel 1.071  sel 7.709 ms
d131krand    wall: attend/sel 0.950  plan+attend/sel 1.048  sel 24.557 ms | logger: attend/sel 0.958  plan+attend/sel 1.039  sel 24.141 ms
d131ktext    wall: attend/sel 0.909  plan+attend/sel 1.051  sel 18.198 ms | logger: attend/sel 0.920  plan+attend/sel 1.031  sel 17.658 ms
d4krand      wall: attend/sel 0.913  plan+attend/sel 1.066  sel 16.583 ms | logger: attend/sel 0.910  plan+attend/sel 1.041  sel 16.032 ms
d4ktext      wall: attend/sel 0.913  plan+attend/sel 1.066  sel 16.468 ms | logger: attend/sel 0.916  plan+attend/sel 1.047  sel 15.811 ms

AT-B3: ratio to new f16 of the same build (default arm), attend by logger then wall; per V variant
c11rand      bf16       vdirect 1.304/1.317  vstaged 2.456/2.429  vpack 1.189/1.211  default 1.304/1.307
c11rand      f16        vdirect 1.005/1.007  vstaged 2.050/2.020  vpack 0.999/1.009  default 1.000/1.000
c11rand      f16:q8_0   default 1.619/1.615
c11rand      q4_0       vdirect 1.893/1.908  vstaged 2.095/2.133  vpack 1.220/1.229  default 1.897/1.913
c11rand      q8_0       vdirect 2.053/2.031  vstaged 2.326/2.298  vpack 1.266/1.272  default 2.064/2.037
c11rand      q8_0:f16   default 1.264/1.261
c11text      bf16       vdirect 1.349/1.352  vstaged 2.359/2.326  vpack 1.178/1.194  default 1.353/1.352
c11text      f16        vdirect 1.000/0.994  vstaged 1.924/1.889  vpack 0.998/0.995  default 1.000/1.000
c11text      f16:q8_0   default 1.726/1.703
c11text      q4_0       vdirect 2.042/2.106  vstaged 2.272/2.374  vpack 1.211/1.216  default 2.058/2.115
c11text      q8_0       vdirect 2.227/2.194  vstaged 2.440/2.413  vpack 1.260/1.252  default 2.241/2.200
c11text      q8_0:f16   default 1.355/1.346
d131krand    bf16       vdirect 1.093/1.100  vstaged 1.934/1.928  vpack 1.327/1.323  default 1.094/1.096
d131krand    f16        vdirect 0.998/0.998  vstaged 1.664/1.662  vpack 1.003/1.005  default 1.000/1.000
d131krand    f16:q8_0   default 1.167/1.169
d131krand    q4_0       vdirect 1.322/1.328  vstaged 1.586/1.624  vpack 1.275/1.273  default 1.325/1.328
d131krand    q8_0       vdirect 1.436/1.438  vstaged 1.802/1.799  vpack 1.311/1.311  default 1.439/1.439
d131krand    q8_0:f16   default 0.933/0.934
d131ktext    bf16       vdirect 1.281/1.274  vstaged 2.358/2.332  vpack 1.429/1.419  default 1.291/1.285
d131ktext    f16        vdirect 0.993/0.994  vstaged 1.965/1.942  vpack 0.985/0.989  default 1.000/1.000
d131ktext    f16:q8_0   default 1.600/1.594
d131ktext    q4_0       vdirect 1.882/1.932  vstaged 2.196/2.261  vpack 1.362/1.351  default 1.895/1.965
d131ktext    q8_0       vdirect 2.035/2.016  vstaged 2.350/2.340  vpack 1.409/1.404  default 2.045/2.026
d131ktext    q8_0:f16   default 1.257/1.263
d4krand      bf16       vdirect 1.386/1.353  vstaged 2.312/2.240  vpack 0.956/0.944  default 1.387/1.362
d4krand      f16        vdirect 1.001/1.003  vstaged 1.861/1.811  vpack 1.002/1.008  default 1.000/1.000
d4krand      f16:q8_0   default 1.762/1.726
d4krand      q4_0       vdirect 2.119/2.142  vstaged 2.299/2.335  vpack 0.974/0.962  default 2.114/2.122
d4krand      q8_0       vdirect 2.323/2.265  vstaged 2.480/2.422  vpack 0.970/0.963  default 2.306/2.251
d4krand      q8_0:f16   default 1.406/1.380
d4ktext      bf16       vdirect 1.379/1.355  vstaged 2.303/2.229  vpack 0.947/0.939  default 1.382/1.352
d4ktext      f16        vdirect 0.997/0.990  vstaged 1.850/1.796  vpack 0.993/0.993  default 1.000/1.000
d4ktext      f16:q8_0   default 1.761/1.720
d4ktext      q4_0       vdirect 2.115/2.155  vstaged 2.254/2.286  vpack 0.966/0.948  default 2.104/2.154
d4ktext      q8_0       vdirect 2.304/2.246  vstaged 2.498/2.471  vpack 0.964/0.950  default 2.307/2.247
d4ktext      q8_0:f16   default 1.399/1.368

allocations of the attend node (bytes), layer 0, per arm
c11rand      default  bf16       plan 42033408 attend 53452416
c11rand      default  f16        plan 42033408 attend 53452416
c11rand      default  f16:q8_0   plan 42033408 attend 53452416
c11rand      default  q4_0       plan 42033408 attend 53452416
c11rand      default  q8_0       plan 42033408 attend 53452416
c11rand      default  q8_0:f16   plan 42033408 attend 53452416
c11rand      default  sel        plan 0 attend 0
c11rand      vdirect  bf16       plan 42033408 attend 53452416
c11rand      vdirect  f16        plan 42033408 attend 53452416
c11rand      vdirect  q4_0       plan 42033408 attend 53452416
c11rand      vdirect  q8_0       plan 42033408 attend 53452416
c11rand      vpack    bf16       plan 42033408 attend 124755584
c11rand      vpack    f16        plan 42033408 attend 53452416
c11rand      vpack    q4_0       plan 42033408 attend 124755584
c11rand      vpack    q8_0       plan 42033408 attend 124755584
c11rand      vstaged  bf16       plan 42033408 attend 53452416
c11rand      vstaged  f16        plan 42033408 attend 53452416
c11rand      vstaged  q4_0       plan 42033408 attend 53452416
c11rand      vstaged  q8_0       plan 42033408 attend 53452416
c11text      default  bf16       plan 42033408 attend 53452416
c11text      default  f16        plan 42033408 attend 53452416
c11text      default  f16:q8_0   plan 42033408 attend 53452416
c11text      default  q4_0       plan 42033408 attend 53452416
c11text      default  q8_0       plan 42033408 attend 53452416
c11text      default  q8_0:f16   plan 42033408 attend 53452416
c11text      default  sel        plan 0 attend 0
c11text      vdirect  bf16       plan 42033408 attend 53452416
c11text      vdirect  f16        plan 42033408 attend 53452416
c11text      vdirect  q4_0       plan 42033408 attend 53452416
c11text      vdirect  q8_0       plan 42033408 attend 53452416
c11text      vpack    bf16       plan 42033408 attend 124755584
c11text      vpack    f16        plan 42033408 attend 53452416
c11text      vpack    q4_0       plan 42033408 attend 124755584
c11text      vpack    q8_0       plan 42033408 attend 124755584
c11text      vstaged  bf16       plan 42033408 attend 53452416
c11text      vstaged  f16        plan 42033408 attend 53452416
c11text      vstaged  q4_0       plan 42033408 attend 53452416
c11text      vstaged  q8_0       plan 42033408 attend 53452416
d131krand    default  bf16       plan 84066560 attend 103784064
d131krand    default  f16        plan 84066560 attend 103784064
d131krand    default  f16:q8_0   plan 84066560 attend 103784064
d131krand    default  q4_0       plan 84066560 attend 103784064
d131krand    default  q8_0       plan 84066560 attend 103784064
d131krand    default  q8_0:f16   plan 84066560 attend 103784064
d131krand    default  sel        plan 0 attend 0
d131krand    vdirect  bf16       plan 84066560 attend 103784064
d131krand    vdirect  f16        plan 84066560 attend 103784064
d131krand    vdirect  q4_0       plan 84066560 attend 103784064
d131krand    vdirect  q8_0       plan 84066560 attend 103784064
d131krand    vpack    bf16       plan 84066560 attend 380608128
d131krand    vpack    f16        plan 84066560 attend 103784064
d131krand    vpack    q4_0       plan 84066560 attend 380608128
d131krand    vpack    q8_0       plan 84066560 attend 380608128
d131krand    vstaged  bf16       plan 84066560 attend 103784064
d131krand    vstaged  f16        plan 84066560 attend 103784064
d131krand    vstaged  q4_0       plan 84066560 attend 103784064
d131krand    vstaged  q8_0       plan 84066560 attend 103784064
d131ktext    default  bf16       plan 84066560 attend 103784064
d131ktext    default  f16        plan 84066560 attend 103784064
d131ktext    default  f16:q8_0   plan 84066560 attend 103784064
d131ktext    default  q4_0       plan 84066560 attend 103784064
d131ktext    default  q8_0       plan 84066560 attend 103784064
d131ktext    default  q8_0:f16   plan 84066560 attend 103784064
d131ktext    default  sel        plan 0 attend 0
d131ktext    vdirect  bf16       plan 84066560 attend 103784064
d131ktext    vdirect  f16        plan 84066560 attend 103784064
d131ktext    vdirect  q4_0       plan 84066560 attend 103784064
d131ktext    vdirect  q8_0       plan 84066560 attend 103784064
d131ktext    vpack    bf16       plan 84066560 attend 380608128
d131ktext    vpack    f16        plan 84066560 attend 103784064
d131ktext    vpack    q4_0       plan 84066560 attend 380608128
d131ktext    vpack    q8_0       plan 84066560 attend 380608128
d131ktext    vstaged  bf16       plan 84066560 attend 103784064
d131ktext    vstaged  f16        plan 84066560 attend 103784064
d131ktext    vstaged  q4_0       plan 84066560 attend 103784064
d131ktext    vstaged  q8_0       plan 84066560 attend 103784064
d4krand      default  bf16       plan 83968256 attend 103784064
d4krand      default  f16        plan 83968256 attend 103784064
d4krand      default  f16:q8_0   plan 83968256 attend 103784064
d4krand      default  q4_0       plan 83968256 attend 103784064
d4krand      default  q8_0       plan 83968256 attend 103784064
d4krand      default  q8_0:f16   plan 83968256 attend 103784064
d4krand      default  sel        plan 0 attend 0
d4krand      vdirect  bf16       plan 83968256 attend 103784064
d4krand      vdirect  f16        plan 83968256 attend 103784064
d4krand      vdirect  q4_0       plan 83968256 attend 103784064
d4krand      vdirect  q8_0       plan 83968256 attend 103784064
d4krand      vpack    bf16       plan 83968256 attend 120561280
d4krand      vpack    f16        plan 83968256 attend 103784064
d4krand      vpack    q4_0       plan 83968256 attend 120561280
d4krand      vpack    q8_0       plan 83968256 attend 120561280
d4krand      vstaged  bf16       plan 83968256 attend 103784064
d4krand      vstaged  f16        plan 83968256 attend 103784064
d4krand      vstaged  q4_0       plan 83968256 attend 103784064
d4krand      vstaged  q8_0       plan 83968256 attend 103784064
d4ktext      default  bf16       plan 83968256 attend 103784064
d4ktext      default  f16        plan 83968256 attend 103784064
d4ktext      default  f16:q8_0   plan 83968256 attend 103784064
d4ktext      default  q4_0       plan 83968256 attend 103784064
d4ktext      default  q8_0       plan 83968256 attend 103784064
d4ktext      default  q8_0:f16   plan 83968256 attend 103784064
d4ktext      default  sel        plan 0 attend 0
d4ktext      vdirect  bf16       plan 83968256 attend 103784064
d4ktext      vdirect  f16        plan 83968256 attend 103784064
d4ktext      vdirect  q4_0       plan 83968256 attend 103784064
d4ktext      vdirect  q8_0       plan 83968256 attend 103784064
d4ktext      vpack    bf16       plan 83968256 attend 120561280
d4ktext      vpack    f16        plan 83968256 attend 103784064
d4ktext      vpack    q4_0       plan 83968256 attend 120561280
d4ktext      vpack    q8_0       plan 83968256 attend 120561280
d4ktext      vstaged  bf16       plan 83968256 attend 103784064
d4ktext      vstaged  f16        plan 83968256 attend 103784064
d4ktext      vstaged  q4_0       plan 83968256 attend 103784064
d4ktext      vstaged  q8_0       plan 83968256 attend 103784064
```

