# r1ub census_sum.py LOG... : sum C2COUNT exit launches per (kernel, nq class, nkv class, mask, ids, K, V) over nodes;
# list C1TRACE unique (kernel, nq) and the n_kv range per (kernel, nq)
import sys, re, collections
for p in sys.argv[1:]:
    tab = collections.Counter(); tr = collections.defaultdict(list)
    for line in open(p, errors='replace'):
        m = re.match(r'C2COUNT exit (\d+) ctx=\S+ (\S+) \S+ (\S+) (\S+) (mask=\d) (ids=\d) (K=\S+) (V=\S+)', line)
        if m:
            tab[(m.group(2),) + m.groups()[2:]] += int(m.group(1))
        m = re.match(r'C1TRACE (\S+) nq (\d+) nkv (\d+) ids (-?\d+) mask (\d)', line)
        if m:
            tr[(m.group(1), int(m.group(2)), m.group(4), m.group(5))].append(int(m.group(3)))
    print('==', p.split('/')[-1])
    for k, n in sorted(tab.items()):
        print(f"   {n:6d}  {' '.join(k)}")
    for k, v in sorted(tr.items()):
        print(f"   trace {k[0]} nq={k[1]} ids={k[2]} mask={k[3]}: {len(v)} distinct n_kv, {min(v)}..{max(v)}")
