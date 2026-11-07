#!/usr/bin/env python3
"""M3 (PLAN-v3 §16): hipMalloc events inside the llama_process windows after warm-up (the shim's LPROC / LSYNC marks). usage: m3_an.py SHIM TAG"""
import sys
ev = [l.split() for l in open(sys.argv[1]) if len(l.split()) >= 5]
win = []; cur = None; first_sync = 0
sync = [i for i, e in enumerate(ev) if e[0] == 'LSYNC']
# a rep = the events between two LSYNC; the warm-up + the first rep windows of the process (rep 0 is dropped by §16) are excluded
start = sync[2] if len(sync) > 2 else len(ev)
m = [e for e in ev[start:] if e[0].startswith('MALLOC')]
print(f"M3 {sys.argv[2]} windows={len(sync)} mallocs_after_warmup={len(m)} bytes={sum(int(e[3]) for e in m)} {'PASS' if not m else 'FAIL'}")
