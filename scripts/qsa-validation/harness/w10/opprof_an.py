#!/usr/bin/env python3
# OP_PROF summary for opprof.sh. The OPPROF records of one llama graph can be split over several flushes (one per backend
# graph compute), so the per-pass cost is taken from the whole log: every class's total ms divided by the number of
# 4096-token QSA passes (SPARSE_ATTN calls / 12 layers). d0: -c 4096, every pass is pp4096 at depth 0. d32k: -c 36864, the
# mean over the chunk's 9 ubatches, depth 0 .. 32768 (the rotation and gate costs do not depend on the depth; the attend
# grows with it).
import sys, os, collections
def cls(label):
    if label.startswith('SPARSE_ATTN_PLAN'): return 'plan'
    if label.startswith('SPARSE_ATTN'): return 'attend (+gate fused)' if '+' in label else 'attend'
    if label.startswith('MUL_MAT') and 'attn_inp_k_rot' in label: return 'hadamard k_rot (q + k)' 
    if label.startswith('MUL_MAT') and 'attn_inp_v_rot' in label: return 'hadamard v_rot (v + out)'
    if label.startswith('CONT gate_reshaped'): return 'gate cont+sigmoid+mul (unfused)'
    return None
out = sys.argv[1]
res = {}
for arm in ('w9', 'w10'):
    for d in ('d0', 'd32k'):
        p = os.path.join(out, f'{arm}_{d}.log')
        if not os.path.exists(p): continue
        tot, calls = collections.Counter(), 0
        for line in open(p, errors='replace'):
            if not line.startswith('OPPROF '): continue
            _, ms, n, label = line.rstrip('\n').split(' ', 3)
            c = cls(label)
            if c: tot[c] += float(ms)
            if label.startswith('SPARSE_ATTN ') or (label.startswith('SPARSE_ATTN') and not label.startswith('SPARSE_ATTN_PLAN')): calls += int(n)
        passes = calls / 12
        per = {k: v / passes for k, v in tot.items()} if passes else {}
        res[(arm, d)] = per
        print(f'{arm} {d}: {passes:g} passes; ' + '; '.join(f'{k} {v:.2f} ms' for k, v in sorted(per.items())))
for d in ('d0', 'd32k'):
    if ('w9', d) in res and ('w10', d) in res:
        ta = sum(v for k, v in res[('w9', d)].items() if k != 'plan'); tb = sum(v for k, v in res[('w10', d)].items() if k != 'plan')
        print(f'U6 {d}: rotation + gate + attention per pass (excl. plan; the k / v storage rotations included in both) w9 {ta:.2f} ms, w10 {tb:.2f} ms, diff {tb - ta:+.2f} ms ({"below" if tb < ta else "NOT below"} W9)')
