#!/usr/bin/env python3
"""P5 MTP cells / AT-W7PERF (b): one running llama-server with the MTP sidecar; per depth d in DEPTHS a prompt of d corpus tokens and n_predict tokens, REPS requests
(rep 0 dropped; the prompt stays cached between reps, so the reps time decode). Prints JSON lines. usage: mtp_perf.py PORT CORPUS OUT DEPTHS REPS N_PREDICT"""
import json, sys, urllib.request
port, corpus, out, depths, reps, npred = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(',')], int(sys.argv[5]), int(sys.argv[6])
def post(path, body):
    r = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=3600))
text = open(corpus, errors='replace').read(2000000)
toks = post('/tokenize', {'content': text})['tokens']
with open(out, 'w') as f:
    for d in depths:
        p = toks[:max(d, 8)]
        for rep in range(reps):
            r = post('/completion', {'prompt': p, 'n_predict': npred, 'cache_prompt': True, 'temperature': 0.0, 'top_k': 1, 'seed': 1, 'stream': False, 'ignore_eos': True})
            t = r.get('timings', {})
            f.write(json.dumps({'depth': d, 'rep': rep, 'tg': t.get('predicted_per_second'), 'n': t.get('predicted_n'), 'draft_n': t.get('draft_n'), 'accepted': t.get('draft_n_accepted'), 'prompt_n': t.get('prompt_n')}) + '\n')
            f.flush()
