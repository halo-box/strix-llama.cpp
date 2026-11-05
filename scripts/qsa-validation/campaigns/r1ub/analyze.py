# r1ub analyze.py LOGDIR : per-arm per-chunk KLD (QKLD_CHUNK all=, %.9e) and paired differences
import sys, re, math, os
L = sys.argv[1]
T95 = {1: 6.3138, 3: 2.3534, 5: 2.0150, 7: 1.8946}

def load(tag):
    p = os.path.join(L, tag + '.log')
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p, errors='replace'):
        m = re.search(r'QKLD_CHUNK (\d+) all=(\S+)', line)
        if m:
            d[int(m.group(1))] = float(m.group(2))
    return d or None

def stats(x):
    n = len(x); mu = sum(x) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / (n - 1)) if n > 1 else float('nan')
    return mu, sd / math.sqrt(n), n

arms = sorted(f[:-4] for f in os.listdir(L) if f.endswith('.log') and not f.startswith('ref'))
print('## arms (KLD vs f16 ub4096 trim-0 reference, QSA+MoE pinned)')
for a in arms:
    d = load(a)
    if not d: continue
    v = [d[k] for k in sorted(d)]
    mu, se, n = stats(v)
    print(f"{a:22s} n={n} mean={mu:.6e} chunkSE={se:.3e} chunks: " + ' '.join(f"{x:.6e}" for x in v))

def pair(b, a, label=''):
    A, B = load(a), load(b)
    if not A or not B:
        print(f"{b} - {a}: missing"); return
    ks = sorted(set(A) & set(B))
    d = [B[k] - A[k] for k in ks]
    mu, se, n = stats(d)
    t = mu / se if se > 0 else float('nan')
    lcb = mu - T95.get(n - 1, float('nan')) * se if se == se else float('nan')
    print(f"{b:20s} - {a:20s} n={n} diff={mu:+.6e} SEp={se:.3e} t={t:+.2f} LCB95={lcb:+.3e} max|d|={max(abs(x) for x in d):.3e} pos={sum(x > 0 for x in d)}/{n} {label}\n      per-chunk: " + ' '.join(f"{x:+.3e}" for x in d))

print('\n## self-pairs (run-to-run)')
for U in ('u512', 'u64'):
    for T in ('f16', 'q8_0', 'q4_0'):
        pair(f'h_{T}_{U}_r2', f'h_{T}_{U}_r1')
for T in ('f16', 'q8_0', 'q4_0'):
    pair(f'h32_{T}_u256_r2', f'h32_{T}_u256_r1')
for T in ('f16', 'q8_0', 'q4_0'):
    pair(f'vk_{T}_u4096_r2', f'vk_{T}_u4096_r1')

print('\n## type vs f16 at the same ub (Q1-form cost of the type)')
for U in ('u4096', 'u512_r1', 'u64_r1'):
    for T in ('q8_0', 'q4_0'):
        pair(f'h_{T}_{U}', f'h_f16_{U}')
for U in ('u4096', 'u256_r1'):
    for T in ('q8_0', 'q4_0'):
        pair(f'h32_{T}_{U}', f'h32_f16_{U}')

print('\n## same type, ub change (benign reorder band of the type)')
for U in ('u512_r1', 'u64_r1'):
    for T in ('f16', 'q8_0', 'q4_0'):
        pair(f'h_{T}_{U}', f'h_{T}_u4096')
for T in ('f16', 'q8_0', 'q4_0'):
    pair(f'h32_{T}_u256_r1', f'h32_{T}_u4096')
for T in ('f16', 'q8_0', 'q4_0'):
    pair(f'h_{T}_u64_r1', f'h_{T}_u512_r1')

print('\n## Vulkan vs HIP, same type, same pins, ub4096 (Q1 form)')
for T in ('f16', 'q8_0', 'q4_0'):
    pair(f'vk_{T}_u4096_r1', f'h_{T}_u4096')
for T in ('q8_0', 'q4_0'):
    pair(f'vk_{T}_u4096_r1', f'vk_f16_u4096_r1')

print('\n## c32768 x 8 tier (QKL_TAIL=4096)')
for T in ('q8_0', 'q4_0'):
    pair(f'h32x8_{T}_u4096', f'h32x8_f16_u4096')
    pair(f'h32x8_{T}_u256', f'h32x8_f16_u256')
for T in ('f16', 'q8_0', 'q4_0'):
    pair(f'h32x8_{T}_u256', f'h32x8_{T}_u4096')

print('\n## Vulkan-internal (vs Vulkan f16 ub4096 trim-0 reference vref8t0)')
pair('v_q8_0_u4096', 'v_f16_u4096_self')
pair('v_f16_u2048', 'v_f16_u4096_self')
pair('v_q8_0_u4096', 'v_f16_u2048')
