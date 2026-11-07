#!/usr/bin/env python3
"""Graph-level reach checker for the QSA sparse-attention routes (PLAN-v3 §14.3).

Expected census rows are generated, not hand-written: the checker reads every executed ubatch shape from the
llama-side trace, asks the backend's shipped selection logic which census keys that shape must hit, and requires the
backend dispatch census to equal the sum, key by key.

Log contract (the lines the branch prints; W7 adds the llama side, W2-W6 the backend side):
  llama side, LLAMA_QSA_TRACE=1, printed at exit, one line per distinct (layer, shape), counted per executed ubatch:
    LLAMA_QSA_TRACE il=<layer> producer=<av|cb|te|dense|masked> n_q=<queries per stream> n_kv=<cells> n_stream=<s>
                    k=<type> v=<type> count=<executed ubatches> route=<sparse|dense|masked|host-kv-copy> ctx=<main|mtp>
                    backend=<name> G=<g> [n_sel=<n> gqa=<g>]
  backend side, read at exit through ggml_backend_debug_dispatch_census_count (§6.2 key strings):
    QSA_CENSUS <key> <count>
  selection dump (test-only proc ggml_backend_sparse_attn_select_dump, or any tool that implements it): it reads
  shape lines `<n_q> <n_kv> <n_stream> <k> <v> <shortcut 0|1> [<n_sel> <gqa>]` on stdin and prints, per shape, one line
    SA_SELECT <the shape line> -> <key>[*<launches>] [<key>[*<launches>] ...]
  (launches default to 1 per executed op; a dense shortcut shape prints no key).

Modes:
  expect  --log LOG (--select-cmd CMD | --select-file FILE)   census == sum over traced shapes of count x keys
  keyset  --log A --log B [--keys REGEX]                       same set of nonzero census keys (hooked-arm census)
  recipe  --log LOG --recipe NAME                              the zero / nonzero rows of the §14.3 table
  selftest                                                     runs the three modes on built-in synthetic logs
Exit status 0 = every check passed, 1 = a mismatch, 2 = bad input.
"""
import argparse, re, subprocess, sys
from collections import Counter, defaultdict

TRACE_RE = re.compile(r'LLAMA_QSA_TRACE il=(-?\d+) producer=(\w+) n_q=(\d+) n_kv=(\d+) n_stream=(\d+) k=(\w+) v=(\w+) count=(\d+)(.*)$')
CENSUS_RE = re.compile(r'^QSA_CENSUS (\S+) (\d+)\s*$')
SELECT_RE = re.compile(r'^SA_SELECT (\d+) (\d+) (\d+) (\w+) (\w+) ([01])((?: \d+)*) ->(.*)$')
# keys the shape cannot predict: the fused selection counts strips and layers, the cpu: rows are llama's own
EXPECT_SKIP = re.compile(r'^(qsa_expand$|cpu:)')
GATED_FLOOR = 0.9

# §14.3 table, default policy. Each recipe: regexes over census keys / trace producers that must be nonzero (any
# match with count > 0) and regexes that must be zero (every match has count 0). 'producer:' rows test the trace.
RECIPES = {
    'Q1-A':  {'nonzero': [r'^producer:cb$', r'^sa_attend/G4/[^/]+/[^/]+/packed'], 'zero': [r'^producer:av$', r'^sa_attend/G1/', r'^producer:masked$', r'^cpu:']},
    'Q1-B':  {'nonzero': [r'^sa_attend/G4/'], 'zero': [r'^producer:masked$', r'^cpu:']},
    'Q11a':  {'nonzero': [r'^sa_attend/G4/'], 'zero': [r'^producer:masked$', r'^cpu:']},
    'Q1-C':  {'nonzero': [r'^producer:cb$', r'^sa_attend/G1/'], 'zero': [r'^sa_attend/G4/', r'^producer:masked$', r'^cpu:']},
    'Q3-B':  {'nonzero': [r'^producer:cb$', r'^sa_attend/G4/[^/]+/[^/]+/direct'], 'zero': [r'^producer:masked$', r'^cpu:']},
    'Q11b':  {'nonzero': [r'^producer:(av|dense)$'], 'zero': [r'^producer:masked$', r'^cpu:']},
    'Q10':   {'nonzero': [r'^producer:te$'], 'zero': [r'^producer:masked$', r'^cpu:']},
    'Q7':    {'nonzero': [r'^producer:cb$'], 'zero': [r'^producer:masked$', r'^cpu:']},
    'P1':    {'nonzero': [r'^producer:(av|dense)$', r'^sa_attend/G4/[^/]+/[^/]+/packed'], 'zero': [r'^producer:masked$', r'^cpu:']},
    'P2':    {'nonzero': [r'^sa_attend/G4/[^/]+/[^/]+/direct'], 'zero': [r'^producer:masked$', r'^cpu:']},
    'Q12':   {'nonzero': [r'^cpu:sparse_attn$'], 'zero': [r'^sa_attend/', r'^sa_plan/']},
}

def parse_log(path):
    trace, census = [], Counter()
    for line in open(path, errors='replace'):
        m = TRACE_RE.search(line)
        if m:
            il, prod, nq, nkv, ns, k, v, cnt, rest = m.groups()
            extra = dict(re.findall(r'(\w+)=(\S+)', rest))
            trace.append(dict(il=int(il), producer=prod, n_q=int(nq), n_kv=int(nkv), n_stream=int(ns), k=k, v=v, count=int(cnt), **extra))
            continue
        m = CENSUS_RE.match(line.strip())
        if m: census[m.group(1)] += int(m.group(2))
    return trace, census

def shape_key(t):
    key = (t['n_q'], t['n_kv'], t['n_stream'], t['k'], t['v'], 1 if t['producer'] == 'av' else 0)
    if 'n_sel' in t: key += (int(t['n_sel']), int(t['gqa']))
    return key

# a shape reaches the backend census only when it runs a sparse op on the backend the census counts
def on_census(t):
    return t['producer'] not in ('dense', 'masked', 'cache') and t.get('backend', 'gpu') not in ('CPU',)

def parse_select(text):
    sel = {}
    for line in text.splitlines():
        m = SELECT_RE.match(line.strip())
        if not m: continue
        nq, nkv, ns, k, v, sc, more, rest = m.groups()
        keys = Counter()
        for tok in rest.split():
            key, _, mult = tok.partition('*')
            keys[key] += int(mult) if mult else 1
        sel[(int(nq), int(nkv), int(ns), k, v, int(sc)) + tuple(int(x) for x in more.split())] = keys
    return sel

def expected_census(trace, sel):
    exp, missing = Counter(), []
    for t in trace:
        if not on_census(t): continue   # dense shortcut, masked route, host-KV CPU route: no sparse key on the backend
        s = shape_key(t)
        if s not in sel: missing.append(s); continue
        for key, mult in sel[s].items(): exp[key] += mult * t['count']
    return exp, missing

def cmd_expect(log, select_text):
    trace, census = parse_log(log)
    if not trace: print(f'{log}: no LLAMA_QSA_TRACE lines'); return 2
    sel = parse_select(select_text)
    exp, missing = expected_census(trace, sel)
    rc = 0
    for s in sorted(set(missing)):
        print(f'MISSING_SELECT {s}'); rc = 1
    # The EPI of a G4 attend is not a function of its shape: the gate fusion's overlap check depends on where gallocr puts the MUL
    # output (ggml-cuda.cu, PLAN-v3 §2.2 item 4). So the attend and fixup keys are compared with gated and plain merged, and the gated
    # share of the fusible attends has its own floor, printed per run.
    def merged(c):
        out = Counter()
        for k, v in c.items():
            if not EXPECT_SKIP.search(k): out[k.replace('/gated', '/plain')] += v
        return out
    em, cm = merged(exp), merged(census)
    for key in sorted(set(em) | set(cm)):
        e, c = em.get(key, 0), cm.get(key, 0)
        ok = e == c
        print(f"{'OK  ' if ok else 'DIFF'} {key} expected={e} census={c}")
        rc |= 0 if ok else 1
    eg = sum(v for k, v in exp.items() if '/gated' in k and k.startswith('sa_attend/'))
    cg = sum(v for k, v in census.items() if '/gated' in k and k.startswith('sa_attend/'))
    if eg:
        share = cg / eg
        ok = share >= GATED_FLOOR
        print(f"{'OK  ' if ok else 'DIFF'} gated share of the fusible G4 attends: {cg}/{eg} = {share:.3f} (floor {GATED_FLOOR})")
        rc |= 0 if ok else 1
    print('EXPECT ' + ('PASS' if rc == 0 else 'FAIL'))
    return rc

def nonzero_keys(census, rx=None):
    r = re.compile(rx) if rx else None
    return {k for k, v in census.items() if v > 0 and (r is None or r.search(k))}

def cmd_keyset(logs, rx):
    sets = [nonzero_keys(parse_log(p)[1], rx) for p in logs]
    rc = 0
    for p, s in zip(logs[1:], sets[1:]):
        if s != sets[0]:
            rc = 1
            print(f'KEYSET DIFF {logs[0]} vs {p}: only first {sorted(sets[0] - s)} only second {sorted(s - sets[0])}')
    print(f"KEYSET {'PASS' if rc == 0 else 'FAIL'} ({len(sets[0])} keys{', filter ' + rx if rx else ''})")
    return rc

def recipe_counts(trace, census):
    c = Counter(census)
    for t in trace: c['producer:' + t['producer']] += t['count']
    return c

def cmd_recipe(log, name):
    if name not in RECIPES: print(f'unknown recipe {name}; known: {sorted(RECIPES)}'); return 2
    trace, census = parse_log(log)
    c = recipe_counts(trace, census)
    rc = 0
    for rx in RECIPES[name]['nonzero']:
        hit = {k: v for k, v in c.items() if re.search(rx, k) and v > 0}
        print(f"{'OK  ' if hit else 'MISS'} nonzero {rx}: {sum(hit.values())}"); rc |= 0 if hit else 1
    for rx in RECIPES[name]['zero']:
        hit = {k: v for k, v in c.items() if re.search(rx, k) and v > 0}
        print(f"{'OK  ' if not hit else 'HIT '} zero {rx}: {dict(hit) if hit else 0}"); rc |= 1 if hit else 0
    print(f"RECIPE {name} {'PASS' if rc == 0 else 'FAIL'}")
    return rc

def selftest():
    import os, tempfile
    d = tempfile.mkdtemp()
    log = os.path.join(d, 'a.log')
    open(log, 'w').write('\n'.join([
        'LLAMA_QSA_TRACE il=3 producer=cb n_q=4096 n_kv=8192 n_stream=1 k=q8_0 v=q8_0 count=8',
        'LLAMA_QSA_TRACE il=3 producer=av n_q=512 n_kv=2048 n_stream=1 k=q8_0 v=q8_0 count=4',
        'LLAMA_QSA_TRACE il=7 producer=dense n_q=512 n_kv=1024 n_stream=1 k=f16 v=f16 count=2',
        'QSA_CENSUS sa_plan/G4 8', 'QSA_CENSUS sa_attend/G4/q8_0/q8_0/packed/plain 8', 'QSA_CENSUS sa_pack/q8_0 8',
        'QSA_CENSUS sa_plan/G1 4', 'QSA_CENSUS sa_attend/G1/q8_0/q8_0/splits1 4', 'QSA_CENSUS sa_merge 0']) + '\n')
    sel = '\n'.join(['SA_SELECT 4096 8192 1 q8_0 q8_0 0 -> sa_plan/G4 sa_attend/G4/q8_0/q8_0/packed/plain sa_pack/q8_0',
                     'SA_SELECT 512 2048 1 q8_0 q8_0 1 -> sa_plan/G1 sa_attend/G1/q8_0/q8_0/splits1'])
    ok = cmd_expect(log, sel) == 0
    bad = sel.replace('splits1', 'splits2')
    ok &= cmd_expect(log, bad) == 1
    ok &= cmd_keyset([log, log], r'^sa_attend/') == 0
    log2 = os.path.join(d, 'b.log'); open(log2, 'w').write('QSA_CENSUS sa_attend/G4/q8_0/q8_0/packed/gated 3\n')
    ok &= cmd_keyset([log, log2], r'^sa_attend/') == 1
    ok &= cmd_recipe(log, 'Q1-A') == 1          # the av shortcut row is present, Q1-A requires it to be zero
    ok &= cmd_recipe(log, 'P1') == 0
    print('SELFTEST ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['expect', 'keyset', 'recipe', 'selftest'])
    ap.add_argument('--log', action='append', default=[])
    ap.add_argument('--select-cmd'); ap.add_argument('--select-file')
    ap.add_argument('--keys'); ap.add_argument('--recipe')
    a = ap.parse_args()
    if a.mode == 'selftest': return selftest()
    if a.mode == 'expect':
        if len(a.log) != 1 or not (a.select_cmd or a.select_file): ap.error('expect needs one --log and --select-cmd or --select-file')
        if a.select_file: text = open(a.select_file).read()
        else:
            trace, _ = parse_log(a.log[0])
            shapes = sorted({shape_key(t) for t in trace if on_census(t)})
            inp = ''.join(' '.join(map(str, s)) + '\n' for s in shapes)
            text = subprocess.run(a.select_cmd, shell=True, input=inp, capture_output=True, text=True, check=True).stdout
        return cmd_expect(a.log[0], text)
    if a.mode == 'keyset':
        if len(a.log) < 2: ap.error('keyset needs at least two --log')
        return cmd_keyset(a.log, a.keys)
    if not a.recipe or len(a.log) != 1: ap.error('recipe needs one --log and --recipe')
    return cmd_recipe(a.log[0], a.recipe)

if __name__ == '__main__':
    sys.exit(main())
