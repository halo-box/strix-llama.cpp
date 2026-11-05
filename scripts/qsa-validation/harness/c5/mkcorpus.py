# C5 corpora: code = llama.cpp sources of PR tree 94e086847 (src/ then ggml/src/, sorted, *.c *.cpp *.h *.cu *.cuh),
# chat = ho3b holdout chat items (oasst2/Dolly) in sources.jsonl order, rendered with the ChatML tags of the model.
import json, os, sys
root = os.path.expanduser('~/dev/wt_pr219v2')
out = sys.argv[1]
with open(os.path.join(out, 'code.txt'), 'w') as f:
    n = 0
    for sub in ['src', 'ggml/src']:
        for dp, dn, fn in sorted(os.walk(os.path.join(root, sub))):
            dn.sort()
            for x in sorted(fn):
                if x.endswith(('.c', '.cpp', '.h', '.cu', '.cuh')):
                    p = os.path.join(dp, x)
                    s = open(p, errors='replace').read()
                    f.write('// file: %s\n%s\n' % (os.path.relpath(p, root), s)); n += len(s)
                    if n > 3_000_000: break
            if n > 3_000_000: break
hd = os.path.expanduser('~/dev/llm-eval-corpora/work/holdouts/ho3b')
with open(os.path.join(out, 'chat.txt'), 'w') as f:
    for l in open(os.path.join(hd, 'sources.jsonl')):
        d = json.loads(l)
        if d['kind'] != 'chat': continue
        m = json.load(open(os.path.join(hd, d['path'])))['messages']
        for msg in m:
            f.write('<|im_start|>%s\n%s<|im_end|>\n' % (msg['role'], msg['content']))
