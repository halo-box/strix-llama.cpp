#!/usr/bin/env python3
# AT-A (d): a C3/C18 rows file cut to its first n_q queries per layer (record: char name[64]; int32 n_q, ns, nk,
# has_mask, pos0, kt; then n_q * ns int32), for the W2 plan harness (w2_plan) at that n_q
import sys, struct
src, dst, nq = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "rb") as f, open(dst, "wb") as o:
    while True:
        h = f.read(88)
        if len(h) < 88: break
        name = h[:64]; n_q, ns, nk, hm, pos0, kt = struct.unpack("<6i", h[64:])
        rows = f.read(4 * n_q * ns)
        k = min(nq, n_q)
        o.write(name + struct.pack("<6i", k, ns, nk, hm, pos0, kt)); o.write(rows[:4 * k * ns])
