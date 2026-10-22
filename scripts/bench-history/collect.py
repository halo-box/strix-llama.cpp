#!/usr/bin/env python3
"""Convert the raw logs of one bench-gfx1151-history run into <out>/<key>/run.json and <out>/<key>/logs/.

Prints the run key on stdout. Prints nothing and exits 0 when the run never got past input validation.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

SCHEMA = 1

ANSI = re.compile(r'\x1b\[[0-9;]*m')
PERF_BACKEND = re.compile(r'^Backend \d+/\d+: (\S+)\s*$')
PERF_ROW = re.compile(r'^  ([A-Za-z0-9_]+)\((.*)\):\s+(\d+) runs -\s+([0-9.]+) us/run -\s+(.*?)\s*$')
PERF_BW = re.compile(r'^(\d+) kB/run -\s+([0-9.]+) GB/s$')
PERF_FLOP = re.compile(r'^([0-9.]+) ([kMGT])FLOP/run -\s+([0-9.]+) ([kMGT])FLOPS$')
PERF_UNSUPPORTED = re.compile(r'^  ([A-Za-z0-9_]+\(.*\)): not supported')
# a perf case that fails allocation or compute prints one of these instead of its `OP(params):` line
PERF_FAILED = re.compile(r'failed to allocate|graph_compute failed')
SCALE = {'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}
PPL_FINAL = re.compile(r'Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)')
PPL_CHUNK = re.compile(r'\[(\d+)\]([0-9]+\.[0-9]+|nan|inf),')
PPL_SETUP = re.compile(r'calculating perplexity over (\d+) chunks, n_ctx=(\d+), batch_size=(\d+), n_seq=(\d+)')
PPL_PASS = re.compile(r'([0-9.]+) seconds per pass')
OPS_COLUMNS = ['op', 'params', 'runs', 'us_per_run', 'kb_per_run', 'gb_per_s', 'flop_per_run', 'flop_per_s']

# the make logs are large and stay in the Actions artifact only
SKIP_LOG = re.compile(r'^build-.*-make\.log$')
SAFE_NAME = re.compile(r'^[A-Za-z0-9._-]+$')

logger = logging.getLogger('collect')


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(errors='replace').splitlines():
            key, sep, value = line.partition('=')
            if sep:
                out[key.strip()] = value.strip()
    return out


def read_tsv(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    return [line.split('\t') for line in path.read_text(errors='replace').splitlines() if line.strip()]


def to_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, '') else None
    except ValueError:
        return None


def int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(',') if v.strip().isdigit()]


def parse_ops_perf(path: Path, backend: str) -> tuple[list[list[Any]], int, set[str], int]:
    """Return the perf rows of `backend` in OPS_COLUMNS order, the unsupported count, every case name seen and the failure count."""
    rows: list[list[Any]] = []
    unsupported = 0
    failed = 0
    seen: set[str] = set()
    current = None
    for raw in path.read_text(errors='replace').splitlines():
        line = ANSI.sub('', raw)
        m = PERF_BACKEND.match(line)
        if m:
            current = m.group(1)
            continue
        if current != backend:
            continue
        if PERF_FAILED.search(line):
            failed += 1
            continue
        m = PERF_UNSUPPORTED.match(line)
        if m:
            unsupported += 1
            seen.add(m.group(1))
            continue
        m = PERF_ROW.match(line)
        if not m:
            continue
        op, params, runs, us, tail = m.groups()
        seen.add(f'{op}({params})')
        row: list[Any] = [op, params, int(runs), float(us), None, None, None, None]
        bw = PERF_BW.match(tail)
        flop = PERF_FLOP.match(tail)
        if bw:
            row[4] = int(bw.group(1))
            row[5] = float(bw.group(2))
        elif flop:
            row[6] = float(flop.group(1)) * SCALE[flop.group(2)]
            row[7] = float(flop.group(3)) * SCALE[flop.group(4)]
        rows.append(row)
    return rows, unsupported, seen, failed


def parse_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad = 0
    if path.is_file():
        for line in path.read_text(errors='replace').splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def parse_perplexity(path: Path) -> dict[str, Any]:
    """Final PPL, its error, the cumulative per-chunk PPL and the setup line of one llama-perplexity log."""
    out: dict[str, Any] = {'ppl': None, 'ppl_err': None, 'chunk_ppl': [], 'n_chunks': None, 'n_ctx': None,
                           'batch_size': None, 'n_seq': None, 'seconds_per_pass': None}
    if not path.is_file():
        return out
    text = ANSI.sub('', path.read_text(errors='replace'))
    final = PPL_FINAL.findall(text)
    if final:
        out['ppl'], out['ppl_err'] = float(final[-1][0]), float(final[-1][1])
    # stdout and stderr interleave, so take the [i] markers in order and stop at the first gap
    for idx, value in PPL_CHUNK.findall(text):
        if int(idx) == len(out['chunk_ppl']) + 1:
            out['chunk_ppl'].append(float(value))
    setup = PPL_SETUP.search(text)
    if setup:
        out['n_chunks'], out['n_ctx'], out['batch_size'], out['n_seq'] = (int(v) for v in setup.groups())
    per_pass = PPL_PASS.search(text)
    if per_pass:
        out['seconds_per_pass'] = float(per_pass.group(1))
    return out


def first_match(path: Path, pattern: str) -> str | None:
    if not path.is_file():
        return None
    rx = re.compile(pattern)
    for line in path.read_text(errors='replace').splitlines():
        m = rx.search(ANSI.sub('', line))
        if m:
            return m.group(1).strip()
    return None


def exit_status(rc: int | None) -> str:
    if rc is None:
        return 'not-run'
    if rc == 0:
        return 'ok'
    if rc == 124:
        return 'timeout'
    if rc > 128:
        return f'aborted (signal {rc - 128})'
    return f'failed (exit {rc})'


def collect(raw: Path, out: Path, artifact_url: str) -> str | None:
    meta = read_env(raw / 'meta.env')
    if 'commit' not in meta or 'started_utc' not in meta:
        logger.warning('no run metadata in %s; the run stopped before the checkout', raw)
        return None

    env = os.environ
    run_id = env.get('GITHUB_RUN_ID', '0')
    attempt = env.get('GITHUB_RUN_ATTEMPT', '1')
    server = env.get('GITHUB_SERVER_URL', 'https://github.com')
    repo = env.get('GITHUB_REPOSITORY', '')
    stamp = re.sub(r'[^0-9TZ]', '', meta['started_utc'])
    key = f'{stamp}-{run_id}-{attempt}'

    commit_lines = (raw / 'commit.txt').read_text(errors='replace').splitlines() if (raw / 'commit.txt').is_file() else []
    steps = {row[0]: (row[1] if len(row) > 1 else '') for row in read_tsv(raw / 'steps.tsv')}
    results = read_tsv(raw / 'results.tsv')
    ops_requested = meta.get('ops_perf', 'false') == 'true'

    models = []
    # the first record per model is the HIP step's; a Vulkan step download may add a second one
    first_rows = {}
    for row in read_tsv(raw / 'models.tsv'):
        first_rows.setdefault(row[0], row)
    for row in first_rows.values():
        row += [''] * (8 - len(row))
        models.append({
            'key': row[0], 'repo': row[1], 'revision': row[2], 'file': row[3], 'bytes': to_int(row[4]),
            'sha256': row[5], 'sha256_actual': row[6] or None, 'status': row[7],
        })
    model_status = {m['key']: m['status'] for m in models}

    vulkan_requested = meta.get('vulkan_bench', 'false') == 'true'
    bench = []
    # (backend, results.tsv kind, file prefix); the Vulkan run reuses the files the HIP run downloaded
    lanes = [('ROCm', 'llama-bench', 'llama-bench')] + ([('Vulkan', 'llama-bench-vulkan', 'llama-bench-vulkan')] if vulkan_requested else [])
    for backend, kind, prefix in lanes:
        bench_rc: dict[str, str] = {row[1]: row[2] for row in results if row[0] == kind and len(row) >= 3}
        for name in meta.get('models', '').split(','):
            if not name:
                continue
            rows, bad = parse_jsonl(raw / f'{prefix}-{name}.jsonl')
            rc = to_int(bench_rc.get(name))
            status = exit_status(rc)
            if model_status.get(name, 'verified') not in ('verified', ''):
                status = f'skipped ({model_status[name]})'
            elif (bench_rc.get(name) or '').startswith('skipped-'):
                status = f"skipped ({bench_rc[name][len('skipped-'):]})"
            elif bench_rc.get(name) == 'skipped':
                status = 'skipped (no model file)'
            elif backend == 'Vulkan' and rc is None and steps.get('device-vulkan') not in ('success', None):
                status = 'skipped (no Vulkan device)'
            elif status == 'ok' and not rows:
                status = 'no rows'
            bench.append({'model': name, 'backend': backend, 'exit_code': rc, 'status': status, 'bad_lines': bad, 'rows': rows})

    ppl_requested = meta.get('ppl', 'false') == 'true'
    ppl = []
    if ppl_requested:
        ppl_rows = {(row[1], row[2]): row for row in results if row[0] == 'perplexity' and len(row) >= 4}
        for name in meta.get('models', '').split(','):
            if not name:
                continue
            for batch in int_list(meta.get('ppl_batches', '')):
                row = ppl_rows.get((name, str(batch)))
                parsed = parse_perplexity(raw / f'perplexity-{name}-b{batch}.log')
                rc = to_int(row[3]) if row else None
                if row and row[3] == 'no-wikitext':
                    status = 'skipped (no wikitext)'
                elif row is None and model_status.get(name, 'verified') not in ('verified', ''):
                    status = f'skipped ({model_status[name]})'
                else:
                    status = exit_status(rc)
                    if status == 'ok' and parsed['ppl'] is None:
                        status = 'no result'
                ppl.append({'model': name, 'n_batch': batch, 'n_ubatch': batch, 'exit_code': rc, 'status': status,
                            'wall_seconds': float(row[4]) if row and len(row) > 4 and row[4] else None, **parsed})

    ops = []
    if ops_requested:
        ops_rc: dict[str, int | None] = {}
        for row in results:
            if row[0] == 'ops-perf' and len(row) >= 3:
                ops_rc[row[1]] = to_int(row[2])
        listed: list[str] = []
        if (raw / 'perf-cases.txt').is_file():
            listed = [line.strip() for line in (raw / 'perf-cases.txt').read_text().splitlines()
                      if line.strip() and not line.startswith('#')]
        for backend in ('ROCm0', 'Vulkan0'):
            log = raw / f'ops-perf-{backend}.log'
            rows, unsupported, seen, n_failed = parse_ops_perf(log, backend) if log.is_file() else ([], 0, set(), 0)
            rc = ops_rc.get(backend)
            status = exit_status(rc)
            if rc is None and rows:
                status = 'incomplete (step stopped)'
            elif rc is None and steps.get(f'ops-perf-{backend}') == 'skipped':
                status = 'skipped'
            elif status == 'ok' and not rows:
                status = 'no rows'
            groups = [{'group': r[2], 'exit_code': to_int(r[3]), 'n_rows': to_int(r[4])}
                      for r in results if r[0] == 'ops-perf-group' and len(r) >= 5 and r[1] == backend]
            # a listed case is only missing if the list group ran to its end
            list_done = any(g['group'] == 'list' and g['exit_code'] == 0 for g in groups)
            ops.append({
                'backend': backend, 'exit_code': rc, 'status': status, 'step': steps.get(f'ops-perf-{backend}', ''),
                'n_rows': len(rows), 'n_unsupported': unsupported, 'n_failed': n_failed, 'groups': groups,
                'n_listed': len(listed), 'missing_listed': [c for c in listed if c not in seen] if list_done else [],
                'columns': OPS_COLUMNS, 'rows': rows,
            })

    entries = bench + ppl + ops
    if bench and all(e['status'] == 'ok' for e in entries):
        status = 'ok'
    elif any(b['rows'] for b in bench) or any(o['rows'] for o in ops) or any(p['ppl'] is not None for p in ppl):
        status = 'partial'
    else:
        status = 'failed'

    run_dir = out / key
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    for path in sorted(raw.iterdir()):
        if path.is_file() and SAFE_NAME.match(path.name) and not SKIP_LOG.match(path.name):
            shutil.copyfile(path, log_dir / path.name)
            logs.append({'name': path.name, 'bytes': path.stat().st_size})

    mem_kib = to_int(meta.get('mem_total_kib'))
    run: dict[str, Any] = {
        'schema': SCHEMA,
        'key': key,
        'status': status,
        'started_utc': meta.get('started_utc'),
        'finished_utc': meta.get('finished_utc'),
        'ref': meta.get('ref'),
        'commit': meta.get('commit'),
        'commit_date': commit_lines[1] if len(commit_lines) > 1 else None,
        'commit_subject': commit_lines[2] if len(commit_lines) > 2 else None,
        'repository': repo,
        'server_url': server,
        'workflow': {
            'name': env.get('GITHUB_WORKFLOW'),
            'ref': env.get('GITHUB_WORKFLOW_REF'),
            'sha': env.get('GITHUB_WORKFLOW_SHA'),
            'dispatch_ref': env.get('GITHUB_REF_NAME'),
            'run_id': to_int(run_id),
            'run_attempt': to_int(attempt),
            'run_url': f'{server}/{repo}/actions/runs/{run_id}/attempts/{attempt}' if repo else None,
            'artifact_url': artifact_url or None,
            'dispatched_by': env.get('GITHUB_ACTOR'),
        },
        'runner': {'name': meta.get('runner_name'), 'os': meta.get('runner_os'), 'arch': meta.get('runner_arch')},
        'environment': {
            'kernel': meta.get('kernel'),
            'nproc': to_int(meta.get('nproc')),
            'mem_total_gib': round(mem_kib / 1048576, 1) if mem_kib else None,
            'hip_version': meta.get('hip_version') or None,
            'cmake_version': meta.get('cmake_version') or None,
            'rocm_device': first_match(raw / 'devices-hip.txt', r'^\s+ROCm0:\s*(.*)$'),
            'vulkan_device': first_match(raw / 'devices-vulkan.txt', r'^\s+Vulkan0:\s*(.*)$')
            or first_match(raw / 'ops-perf-Vulkan0.log', r'^ggml_vulkan: 0 = (.*)$'),
            'vulkan_driver': first_match(raw / 'vulkan-env.log', r'driverInfo\s*=\s*(.*)$'),
            'vulkan_sdk': meta.get('vulkan_sdk_version') or None,
        },
        'matrix': {
            'models': [m for m in meta.get('models', '').split(',') if m],
            'ubatch': int_list(meta.get('ubatch', '')),
            'depths': int_list(meta.get('depths', '')),
            'reps': to_int(meta.get('reps')),
            'n_prompt': to_int(meta.get('n_prompt')),
            'n_gen': to_int(meta.get('n_gen')),
            'n_batch': to_int(meta.get('n_batch')),
            'ops_perf': ops_requested,
            'vulkan_bench': vulkan_requested,
            'ppl': ppl_requested,
            'ppl_ctx': to_int(meta.get('ppl_ctx')),
            'ppl_chunks': to_int(meta.get('ppl_chunks')),
            'ppl_batches': int_list(meta.get('ppl_batches', '')),
            'wikitext_sha256': meta.get('wikitext_sha256') or None,
        },
        'steps': steps,
        'commands': [{'label': r[0], 'command': r[1]} for r in read_tsv(raw / 'commands.tsv') if len(r) >= 2],
        'models': models,
        'llama_bench': bench,
        'perplexity': ppl,
        'ops_perf': ops,
        'logs': logs,
    }
    (run_dir / 'run.json').write_text(json.dumps(run, indent=1) + '\n')
    write_summary(run)
    return key


def write_summary(run: dict[str, Any]) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if not path:
        return
    lines = [f"### Run `{run['key']}`: {run['status']}", '', '| model | backend | test | ubatch | depth | t/s |',
             '|---|---|---|---:|---:|---:|']
    for entry in run['llama_bench']:
        for row in entry['rows']:
            test = f"pp{row.get('n_prompt')}" if row.get('n_prompt') else f"tg{row.get('n_gen')}"
            lines.append(f"| {entry['model']} | {entry['backend']} | {test} | {row.get('n_ubatch')} | {row.get('n_depth')} | "
                         f"{row.get('avg_ts', 0):.2f} +/- {row.get('stddev_ts', 0):.2f} |")
    lines.append('')
    for entry in run['llama_bench']:
        lines.append(f"- llama-bench {entry['model']} ({entry['backend']}): {entry['status']}, {len(entry['rows'])} rows")
    for entry in run['perplexity']:
        value = f"PPL {entry['ppl']:.4f} +/- {entry['ppl_err']:.5f}" if entry['ppl'] is not None else 'no PPL'
        lines.append(f"- llama-perplexity {entry['model']} b{entry['n_batch']}: {entry['status']}, {value}")
    for entry in run['ops_perf']:
        missing = f", {len(entry['missing_listed'])} listed cases without a result" if entry['missing_listed'] else ''
        if entry.get('n_failed'):
            missing += f", {entry['n_failed']} failed to allocate or compute"
        lines.append(f"- test-backend-ops perf {entry['backend']}: {entry['status']}, {entry['n_rows']} rows{missing}")
    with open(path, 'a') as f:
        f.write('\n'.join(lines) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True, help='directory with the downloaded raw-log artifact')
    parser.add_argument('--out', type=Path, required=True, help='directory that receives <key>/run.json')
    parser.add_argument('--artifact-url', default='', help='URL of the raw-log artifact')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    if not args.raw.is_dir():
        logger.error('%s is not a directory', args.raw)
        return 1
    key = collect(args.raw, args.out, args.artifact_url)
    if key:
        sys.stdout.write(key + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
