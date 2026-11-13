# Two Strix Halo boxes as one: RPC over USB4

Two Strix Halo machines (Radeon 8060S, 128 GB each), joined by one USB4 cable, run a model neither can hold alone.
With the ggml RPC backend, Qwen3.8-Flash-Next at **Q8_0 (188 GB)** decodes at **21.8 tok/s** and prefills at
**750 tok/s** (pp4096). A model that fits on one box loses about 6% decode when split, and gains prefill on long prompts.

This note covers the setup, the measurements, and two small changes on this branch.

## Setup

**Hardware:** two Strix Halo boxes (Ryzen AI Max+ 395, Radeon 8060S, 128 GB unified memory), Ubuntu 26.04, kernel 7.0,
Vulkan (RADV). Both run the same commit of this repo, built with `-DGGML_VULKAN=ON -DGGML_RPC=ON`.

**Link:** USB4 host-to-host networking (`thunderbolt-net`), interface `thunderbolt0`, MTU 65520, static IPs
(10.0.0.1 and 10.0.0.2).

1. **Check the link rate.** Both ends should show `20.0 Gb/s` on two lanes (40 Gbps). A non-certified cable trains at
   `10.0 Gb/s` (20 Gbps) without any error:

   ```bash
   cat /sys/bus/thunderbolt/devices/1-2/rx_speed /sys/bus/thunderbolt/devices/1-2/rx_lanes
   ```

   We first had a 20 Gbps cable. A certified 40 Gbps USB4 cable fixed it.
2. **Open the firewall on the link only**, not on the LAN: `sudo ufw allow in on thunderbolt0`.
3. **Start the RPC server** on one box, bound to the link address:

   ```bash
   ggml-rpc-server -H 10.0.0.1 -p 50052 -d Vulkan0
   ```

   The RPC server has no authentication: never bind it to a LAN or public address.
4. **Run the model on the other box**, from fast local storage (NVMe). With `-lzm on-direct` the per-layer n-gram
   table is read from the model file during decode, so a slow disk slows decode.

   ```bash
   llama-server -m Qwen3.8-Flash-Next-Q8_0.gguf --rpc 10.0.0.1:50052 -dev RPC0,Vulkan0 -ts 52,48 \
     -ngl 999 -fa on -lzm on-direct -b 4096 -ub 2048
   ```

**Device order:** put the remote GPU first and the local GPU last (`-dev RPC0,Vulkan0`). The last device holds the
output head, so the logits (about 1 MB per token for a 248k vocabulary) stay local instead of crossing the link.

**Memory:** Q8_0 with the n-gram table on disk needs about 135 GB on the GPUs. A 52/48 split keeps each box well
below its limit (peak about 70 GB on the serving box).

## Link measurements

| | Value |
|---|---|
| Link rate | 40 Gbps (2 x 20 Gb/s), after the cable change |
| Round trip (ping) | 0.64-0.71 ms |
| TCP, one stream | 11.3 Gb/s (one direction), 8.7 Gb/s (other direction) |
| TCP, 2 or 4 streams | 11.3 Gb/s (no gain) |

TCP stops at about 11 Gb/s whatever the stream count, so the limit is the `thunderbolt-net` driver path (one queue,
no receive checksum offload), not the cable. That is ample for RPC: activations at a split are tens of KB per token.
Latency matters more than bandwidth for decode.

## Results

All with `llama-bench`, `-fa 1 -ngl 999 -lzm on-direct`, models on NVMe of the client box.

**Gyro-M (Qwen3.8-Flash-Next, 2-bit trellis, fits on one box): split overhead**

| | One box | Split 50/50 |
|---|---:|---:|
| tg128 | 27.6 | 25.6 (26.0 with this branch) |
| pp4096, ub 2048 | 661 | 710 |
| pp4096, ub 512 | 511 | 598 |
| pp512, ub 512 | 654 | 625 |

**Qwen3.8-Flash-Next Q8_0 (188 GB): does not fit one box**

| ubatch (batch 4096) | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| pp4096 | 505 | 602 | 688 | **750** |
| tg64 | 21.6 | 21.6 | 21.9 | 21.8 |

With the default batch of 2048, pp4096 is 712 tok/s. A batch of 4096 with ubatch 2048 lets the two boxes overlap two
micro-batches.

## Where the decode time goes

We traced the RPC commands of one decode step (`GGML_RPC_DEBUG=3` on the server). The split costs about 2.6 ms per token
before this branch:

- **Each cross-device read was a blocking round trip.** For each input of a split that lives on another backend, the
  scheduler fell back to a synchronous `ggml_backend_tensor_get`. Over RPC that is one round trip (about 0.65 ms) per
  input, and one boundary had three.
- **The same bytes were read twice.** A tensor and a reshaped view of it (same buffer, same data) were copied as two
  separate inputs.
- **A small weight crossed the link every token.** Qwen3.8-Flash-Next's hyper-connection norm weight of the first layer
  after the split (40 KB) is used by an op on the other side of the boundary, so it is copied each step. Not changed on
  this branch.

## Changes on this branch

1. **`ggml-backend.cpp`: batched cross-backend input reads.** When a split input needs a blocking read from a backend
   that has `get_tensor_async` (such as RPC), the scheduler now issues all those reads for the split, waits once per
   source backend, and then writes them. Inputs that alias the same bytes are read once. `GGML_SCHED_NO_BATCH_COPY=1`
   restores the old path.
2. **`ggml-rpc.cpp`: pipelined async reads.** Async `GET_TENSOR` requests go out immediately; their replies are read in
   order at the next barrier or before any command that needs its own reply (capped at 4 MB in flight). With change 1,
   the reads of one split cost one round trip instead of one each. Synchronous reads are unchanged. No protocol change:
   the server is untouched. `GGML_RPC_NO_PIPELINE=1` turns it off.

Measured on Gyro-M, tg64, 3 repetitions:

| Device order | Before | After |
|---|---:|---:|
| `RPC0/Vulkan0` | 25.73 | **26.03** (+1.2%) |
| `Vulkan0/RPC0` | 25.70 | 25.75 |

The gain is small: of the remaining 2.2 ms per token, about 1 ms is one round trip plus the one-way handoff, and the
rest is the extra Vulkan submit and synchronize on each side of the split.

## Ideas not tried yet

- **Lower link latency.** 0.65 ms is high for a direct link. Socket busy polling (`net.core.busy_read`,
  `net.core.busy_poll`) and CPU idle-state limits usually cut it substantially.
- **Speculative decoding (MTP draft) on the split model.** Several tokens per boundary crossing amortise the fixed cost.
- **Keep boundary ops with their weights,** so the per-token weight copy above disappears.
- **Longer prompts with a larger batch** (8192) to overlap more micro-batches in prefill.
