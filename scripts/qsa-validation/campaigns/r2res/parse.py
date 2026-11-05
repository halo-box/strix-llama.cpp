#!/usr/bin/env python3
# R2-M-LAB-RES parser: per log, the sched_reserve sizes of the probe (main, MTP extra) and of the real contexts,
# plus LAB_NC variant lines (bytes). Usage: parse.py logs/srv_*.log
import re, sys, json
out = {}
for path in sys.argv[1:]:
    tag = re.sub(r'.*srv_(.*)\.log$', r'\1', path)
    phase = 'probe_main'; res = []; cur = {}; lab = []; info = {}
    for line in open(path, errors='replace'):
        if 'getting device memory data for the extra model' in line: phase = 'probe_mtp'
        if 'successfully fit params' in line or 'failed to fit params' in line: phase = 'real'
        m = re.search(r'llama_context: (n_ctx|n_ubatch|n_seq_max|kv_unified)\s+= (\S+)', line)
        if m: info.setdefault(phase, {})[m.group(1)] = m.group(2)
        m = re.search(r'sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+) MiB', line)
        if m:
            cur[m.group(1)] = float(m.group(2))
            if 'ROCm_Host' in cur and 'ROCm0' in cur:
                res.append((phase, dict(cur))); cur = {}
        m = re.search(r'LAB_NC variant: ctx_type=(\d) (\S+) before=(\d+) (after|variant_alone)=(\d+)', line)
        if m: lab.append(dict(phase=phase, ctx=int(m.group(1)), buft=m.group(2), before=int(m.group(3)), kind=m.group(4), after=int(m.group(5))))
        m = re.search(r'LAB_NC variant: ctx_type=(\d) mode=(\d) n_tokens=(\d+) n_seqs=(\d+) n_outputs=(\d+) n_stream=(\d+)', line)
        if m: lab.append(dict(phase=phase, hdr=line.split('LAB_NC variant: ')[1].strip()))
        m = re.search(r'LAB_NC qsa_input il=(\d+) .*n_kv=(\d+) .*n_tps=(\d+) n_stream=(\d+) .*forced=1 scalar=(\d)', line)
        if m: info.setdefault('forced_inputs', set()).add((phase, m.group(0).split('LAB_NC ')[1]))
    # label the reserves: probe main, probe mtp, then real main, real mtp, real mtp (re-reserve after load_model)
    lab_res = []
    nreal = 0
    for ph, d in res:
        if ph == 'real':
            name = ['real_main', 'real_mtp', 'real_mtp_rereserve', 'real_extra'][min(nreal, 3)]; nreal += 1
        else:
            name = ph
        lab_res.append((name, d))
    if 'forced_inputs' in info: info['forced_inputs'] = sorted(info['forced_inputs'])
    out[tag] = dict(reserves=lab_res, lab=lab, info=info)
print(json.dumps(out, indent=1))
