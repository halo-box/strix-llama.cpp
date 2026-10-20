#!/usr/bin/env python3
"""Build the static benchmark history site from the runs/*/run.json files of the results branch.

The output needs no network: CSS and the small filter script are inline, there are no external assets.
Local preview:
    git worktree add /tmp/bench-results origin/bench-results
    python3 scripts/bench-history/build_site.py --results /tmp/bench-results --out _site --repository halo-box/strix-llama.cpp
    python3 -m http.server -d _site
"""

import argparse
import datetime
import html
import json
import logging
import os
import re
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA = 1
SAFE_NAME = re.compile(r'^[A-Za-z0-9]+$')

logger = logging.getLogger('build_site')

CSS = """
:root { color-scheme: light dark; --fg: #1f2328; --muted: #59636e; --bg: #ffffff; --panel: #f6f8fa; --border: #d1d9e0;
  --link: #0969da; --ok: #1a7f37; --warn: #9a6700; --bad: #d1242f; }
@media (prefers-color-scheme: dark) { :root { --fg: #e6edf3; --muted: #9198a1; --bg: #0d1117; --panel: #151b23;
  --border: #3d444d; --link: #4493f8; --ok: #3fb950; --warn: #d29922; --bad: #f85149; } }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: var(--fg); background: var(--bg); }
header, main { max-width: 1440px; margin: 0 auto; padding: 0 20px; }
header { padding-top: 18px; }
main { padding-bottom: 40px; }
h1 { font-size: 22px; margin: 4px 0 6px; }
h2 { font-size: 18px; margin: 28px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
h3 { font-size: 15px; margin: 20px 0 6px; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
nav a { margin-right: 16px; }
.lede, .note { color: var(--muted); margin: 4px 0; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 6px 0 12px; }
th, td { border: 1px solid var(--border); padding: 3px 8px; text-align: left; vertical-align: top; white-space: nowrap; }
th { background: var(--panel); font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
th.num { text-align: center; }
tbody tr:nth-child(even) { background: color-mix(in srgb, var(--panel) 70%, transparent); }
code, pre, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; }
pre { background: var(--panel); border: 1px solid var(--border); padding: 8px; margin: 4px 0 10px; white-space: pre-wrap; word-break: break-all; }
.kv th { width: 190px; }
.kv td { white-space: normal; word-break: break-word; }
.sd { color: var(--muted); font-size: 12px; }
.badge { display: inline-block; padding: 0 7px; border: 1px solid currentColor; border-radius: 10px; font-size: 12px; font-weight: 600; }
.ok { color: var(--ok); }
.partial { color: var(--warn); }
.failed { color: var(--bad); }
.filters { margin: 6px 0; }
.filters label { margin-right: 14px; }
td.wrap { white-space: normal; word-break: break-all; min-width: 320px; }
"""

INDEX_JS = """
(function () {
  var table = document.getElementById('results-table');
  if (!table) { return; }
  var selects = document.querySelectorAll('#results-filters select');
  var count = document.getElementById('results-count');
  function apply() {
    var shown = 0;
    table.querySelectorAll('tbody tr').forEach(function (tr) {
      var keep = true;
      selects.forEach(function (s) { if (s.value && tr.dataset[s.name] !== s.value) { keep = false; } });
      tr.hidden = !keep;
      if (keep) { shown++; }
    });
    count.textContent = shown + ' rows';
  }
  selects.forEach(function (s) { s.addEventListener('change', apply); });
  apply();
})();
"""

OPS_JS = """
(function () {
  var input = document.getElementById('ops-filter');
  var rows = document.querySelectorAll('#ops-table tbody tr');
  var count = document.getElementById('ops-count');
  function apply() {
    var q = input.value.toLowerCase();
    var shown = 0;
    rows.forEach(function (tr) {
      var keep = !q || tr.textContent.toLowerCase().indexOf(q) >= 0;
      tr.hidden = !keep;
      if (keep) { shown++; }
    });
    count.textContent = shown + ' rows';
  }
  input.addEventListener('input', apply);
  apply();
})();
"""


def esc(value: Any) -> str:
    return html.escape('' if value is None else str(value), quote=True)


def fmt(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f'{value:.{digits}f}'
    return '-'


def badge(status: str | None) -> str:
    status = status or 'unknown'
    if status == 'ok':
        cls = 'ok'
    elif status.startswith(('partial', 'skipped', 'not-run', 'no rows')):
        cls = 'partial'
    else:
        cls = 'failed'
    return f'<span class="badge {cls}">{esc(status)}</span>'


def page(title: str, body: str, script: str = '') -> str:
    js = f'<script>{script}</script>\n' if script else ''
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{esc(title)}</title>\n<link rel="icon" href="data:,">\n<style>{CSS}</style>\n</head>\n'
            f'<body>\n{body}\n{js}</body>\n</html>\n')


class Site:
    def __init__(self, repository: str, server_url: str, branch: str):
        self.repository = repository
        self.server_url = server_url.rstrip('/')
        self.branch = branch

    def repo_url(self, path: str) -> str | None:
        if not self.repository:
            return None
        return f'{self.server_url}/{self.repository}/{path}'

    def commit_link(self, run: dict[str, Any], short: bool = True) -> str:
        sha = run.get('commit') or ''
        text = f'<code>{esc(sha[:10] if short else sha)}</code>'
        url = self.repo_url(f'commit/{sha}') if sha else None
        return f'<a href="{esc(url)}">{text}</a>' if url else text

    def log_url(self, key: str, name: str) -> str | None:
        return self.repo_url(f'raw/{self.branch}/runs/{key}/logs/{name}')


def test_name(row: dict[str, Any]) -> str | None:
    n_prompt = row.get('n_prompt') or 0
    n_gen = row.get('n_gen') or 0
    if n_prompt and not n_gen:
        return f'pp{n_prompt}'
    if n_gen and not n_prompt:
        return f'tg{n_gen}'
    return None


def cells(run: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for entry in run.get('llama_bench', []):
        for row in entry.get('rows', []):
            name = test_name(row)
            if name is None:
                continue
            samples = [float(x) for x in row.get('samples_ts') or []]
            out.append({
                'model': entry.get('model'), 'test': name, 'kind': name[:2],
                'n_batch': row.get('n_batch'), 'n_ubatch': row.get('n_ubatch'), 'n_depth': row.get('n_depth'),
                'mean': row.get('avg_ts'), 'sd': row.get('stddev_ts'),
                'median': statistics.median(samples) if samples else None,
                'min': min(samples) if samples else None, 'max': max(samples) if samples else None,
                'n': len(samples), 'samples': samples,
            })
    return out


def tps(cell: dict[str, Any] | None) -> str:
    if not cell:
        return '<td class="num">-</td>'
    title = f"{cell['test']}, mean {fmt(cell['mean'])} +/- {fmt(cell['sd'])} t/s, {cell['n']} samples"
    return f'<td class="num" title="{esc(title)}">{fmt(cell["mean"])} <span class="sd">&plusmn;{fmt(cell["sd"])}</span></td>'


def ppl_cell(entry: dict[str, Any] | None) -> str:
    if not entry:
        return '<td class="num">-</td>'
    if entry.get('ppl') is None:
        return f'<td class="num"><span class="sd">{esc(entry.get("status"))}</span></td>'
    title = f"{entry.get('n_chunks')} chunks at c{entry.get('n_ctx')}, b{entry.get('n_batch')} ub{entry.get('n_ubatch')}"
    return f'<td class="num" title="{esc(title)}">{entry["ppl"]:.4f} <span class="sd">&plusmn;{entry["ppl_err"]:.4f}</span></td>'


def ppl_delta(base: dict[str, Any] | None, other: dict[str, Any] | None) -> str:
    if not base or not other or base.get('ppl') is None or other.get('ppl') is None:
        return '<td class="num">-</td>'
    delta = other['ppl'] - base['ppl']
    return f'<td class="num">{delta:+.4f} <span class="sd">({100 * delta / base["ppl"]:+.3f}%)</span></td>'


def load_runs(results: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted((results / 'runs').glob('*/run.json')):
        try:
            run = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning('skipping %s: %s', path, e)
            continue
        if not isinstance(run, dict) or run.get('schema') != SCHEMA or run.get('key') != path.parent.name:
            logger.warning('skipping %s: unknown schema or key', path)
            continue
        runs.append(run)
    runs.sort(key=lambda r: (r.get('started_utc') or '', r['key']), reverse=True)
    return runs


def run_date(run: dict[str, Any]) -> str:
    return (run.get('started_utc') or '').replace('T', ' ').replace('Z', '')


def build_index(site: Site, runs: list[dict[str, Any]], generated: str) -> str:
    parts = []
    branch_url = site.repo_url(f'tree/{site.branch}')
    branch = f'<a href="{esc(branch_url)}">{esc(site.branch)}</a>' if branch_url else esc(site.branch)
    parts.append('<header>\n<h1>gfx1151 benchmark history</h1>\n'
                 f'<p class="lede">{len(runs)} recorded runs of <code>bench-gfx1151-history.yml</code> on the AMD DevLab '
                 f'Dispatch Strix Halo pool. Built {esc(generated)} UTC from the {branch} branch. '
                 'Machine-readable: <a href="data/runs.json">data/runs.json</a>.</p>\n'
                 '<nav><a href="#runs">Runs</a><a href="#trend">Over time</a><a href="#ppl">Perplexity</a><a href="#results">All results</a></nav>\n'
                 '</header>\n<main>')

    parts.append('<section id="runs"><h2>Runs</h2>')
    if not runs:
        parts.append('<p class="note">No runs recorded yet. Dispatch the workflow to add the first one.</p>')
    else:
        parts.append('<div class="scroll"><table><thead><tr><th>Date (UTC)</th><th>Ref</th><th>Commit</th><th>Status</th>'
                     '<th>Models</th><th>ubatch</th><th>Depths</th><th>Runner</th><th></th></tr></thead><tbody>')
        for run in runs:
            matrix = run.get('matrix', {})
            parts.append(
                f'<tr><td>{esc(run_date(run))}</td><td><code>{esc(run.get("ref"))}</code></td><td>{site.commit_link(run)}</td>'
                f'<td>{badge(run.get("status"))}</td><td>{esc(", ".join(matrix.get("models", [])))}</td>'
                f'<td>{esc(",".join(map(str, matrix.get("ubatch", []))))}</td>'
                f'<td>{esc(",".join(map(str, matrix.get("depths", []))))}</td>'
                f'<td class="mono">{esc((run.get("runner") or {}).get("name"))}</td>'
                f'<td><a href="runs/{esc(run["key"])}/index.html">details</a></td></tr>')
        parts.append('</tbody></table></div>')
    parts.append('</section>')

    all_cells = {run['key']: cells(run) for run in runs}

    parts.append('<section id="trend"><h2>Throughput over time</h2>'
                 '<p class="note">Mean tokens per second per cell, newest run first, &plusmn; one standard deviation. '
                 'tg does not depend on ubatch, so the tg cells of one depth repeat one measurement.</p>')
    groups = sorted({(c['model'], c['kind'], c['test']) for cs in all_cells.values() for c in cs},
                    key=lambda g: (str(g[0]), g[1] != 'pp', g[2]))
    for model, _, test in groups:
        cols = sorted({(c['n_depth'], c['n_ubatch']) for cs in all_cells.values() for c in cs
                       if c['model'] == model and c['test'] == test}, key=lambda t: (t[0] or 0, t[1] or 0))
        depths = sorted({d for d, _ in cols}, key=lambda d: d or 0)
        parts.append(f'<h3>{esc(model)} - {esc(test)}</h3><div class="scroll"><table><thead><tr>'
                     '<th rowspan="2">Date (UTC)</th><th rowspan="2">Ref</th><th rowspan="2">Commit</th>')
        for d in depths:
            span = sum(1 for dd, _ in cols if dd == d)
            parts.append(f'<th class="num" colspan="{span}">depth {esc(d)}</th>')
        parts.append('</tr><tr>' + ''.join(f'<th class="num">ub {esc(u)}</th>' for _, u in cols) + '</tr></thead><tbody>')
        for run in runs:
            by_col: dict[tuple[Any, Any], dict[str, Any]] = {}
            for c in all_cells[run['key']]:
                if c['model'] == model and c['test'] == test:
                    by_col.setdefault((c['n_depth'], c['n_ubatch']), c)
            if not by_col:
                continue
            parts.append(f'<tr><td><a href="runs/{esc(run["key"])}/index.html">{esc(run_date(run))}</a></td>'
                         f'<td><code>{esc(run.get("ref"))}</code></td><td>{site.commit_link(run)}</td>'
                         + ''.join(tps(by_col.get(col)) for col in cols) + '</tr>')
        parts.append('</tbody></table></div>')
    if not groups:
        parts.append('<p class="note">No llama-bench results yet.</p>')
    parts.append('</section>')

    parts.append('<section id="ppl"><h2>Perplexity over time</h2>'
                 '<p class="note">llama-perplexity on the pinned Wikitext-2 sample, newest run first, &plusmn; the error the tool reports. '
                 'Both batch sizes score the same tokens, so the delta is the effect of how each chunk is cut into batches.</p>')
    ppl_models = sorted({str(p.get('model')) for run in runs for p in run.get('perplexity') or []})
    for model in ppl_models:
        batches = sorted({p.get('n_batch') or 0 for run in runs for p in run.get('perplexity') or [] if p.get('model') == model})
        parts.append(f'<h3>{esc(model)}</h3><div class="scroll"><table><thead><tr><th>Date (UTC)</th><th>Ref</th><th>Commit</th>'
                     + ''.join(f'<th class="num">b{esc(b)}</th>' for b in batches)
                     + ''.join(f'<th class="num">b{esc(b)} - b{esc(batches[0])}</th>' for b in batches[1:])
                     + '<th>Sample</th></tr></thead><tbody>')
        for run in runs:
            mine = {p.get('n_batch'): p for p in run.get('perplexity') or [] if p.get('model') == model}
            if not mine:
                continue
            matrix = run.get('matrix') or {}
            parts.append(f'<tr><td><a href="runs/{esc(run["key"])}/index.html">{esc(run_date(run))}</a></td>'
                         f'<td><code>{esc(run.get("ref"))}</code></td><td>{site.commit_link(run)}</td>'
                         + ''.join(ppl_cell(mine.get(b)) for b in batches)
                         + ''.join(ppl_delta(mine.get(batches[0]), mine.get(b)) for b in batches[1:])
                         + f'<td>c{esc(matrix.get("ppl_ctx"))} x {esc(matrix.get("ppl_chunks"))} chunks</td></tr>')
        parts.append('</tbody></table></div>')
    if not ppl_models:
        parts.append('<p class="note">No perplexity results yet.</p>')
    parts.append('</section>')

    rows = []
    values: dict[str, set[str]] = {'model': set(), 'ref': set(), 'ub': set(), 'depth': set()}
    for run in runs:
        merged: dict[tuple[Any, ...], dict[str, Any]] = {}
        for c in all_cells[run['key']]:
            slot = merged.setdefault((c['model'], c['n_batch'], c['n_ubatch'], c['n_depth']), {})
            slot.setdefault(c['kind'], c)
        for (model, _, ub, depth), slot in sorted(merged.items(), key=lambda kv: (str(kv[0][0]), kv[0][2] or 0, kv[0][3] or 0)):
            attrs = {'model': str(model), 'ref': str(run.get('ref')), 'ub': str(ub), 'depth': str(depth)}
            for k, v in attrs.items():
                values[k].add(v)
            data = ' '.join(f'data-{k}="{esc(v)}"' for k, v in attrs.items())
            rows.append(f'<tr {data}><td>{esc(run_date(run))}</td><td><code>{esc(run.get("ref"))}</code></td>'
                        f'<td>{site.commit_link(run)}</td><td>{esc(model)}</td><td class="num">{esc(ub)}</td>'
                        f'<td class="num">{esc(depth)}</td>{tps(slot.get("pp"))}{tps(slot.get("tg"))}'
                        f'<td><a href="runs/{esc(run["key"])}/index.html">run</a></td></tr>')

    parts.append('<section id="results"><h2>All results</h2>')
    parts.append('<div class="filters" id="results-filters">')
    labels = {'model': 'Model', 'ref': 'Ref', 'ub': 'ubatch', 'depth': 'Depth'}
    for name, label in labels.items():
        opts = sorted(values[name], key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v))
        parts.append(f'<label>{label} <select name="{name}"><option value="">all</option>'
                     + ''.join(f'<option>{esc(o)}</option>' for o in opts) + '</select></label>')
    parts.append(f'<span class="note" id="results-count">{len(rows)} rows</span></div>')
    parts.append('<div class="scroll"><table id="results-table"><thead><tr><th>Date (UTC)</th><th>Ref</th><th>Commit</th>'
                 '<th>Model</th><th class="num">ubatch</th><th class="num">Depth</th><th class="num">pp t/s</th>'
                 '<th class="num">tg t/s</th><th></th></tr></thead><tbody>')
    parts.extend(rows)
    parts.append('</tbody></table></div></section>\n</main>')
    return page('gfx1151 benchmark history', '\n'.join(parts), INDEX_JS)


def kv_table(items: list[tuple[str, str]]) -> str:
    return ('<table class="kv"><tbody>'
            + ''.join(f'<tr><th>{esc(k)}</th><td>{v or "-"}</td></tr>' for k, v in items)
            + '</tbody></table>')


def link(url: str | None, text: str) -> str:
    return f'<a href="{esc(url)}">{esc(text)}</a>' if url else '-'


def build_run(site: Site, run: dict[str, Any]) -> str:
    key = run['key']
    wf = run.get('workflow') or {}
    env = run.get('environment') or {}
    matrix = run.get('matrix') or {}
    runner = run.get('runner') or {}
    parts = ['<header>\n<nav><a href="../../index.html">All runs</a></nav>',
             f'<h1>Run {esc(key)}</h1>',
             f'<p class="lede">{badge(run.get("status"))} <code>{esc(run.get("ref"))}</code> at {site.commit_link(run)} '
             f'on <span class="mono">{esc(runner.get("name"))}</span></p>',
             '<nav><a href="#run">Run</a><a href="#bench">llama-bench</a><a href="#ppl">llama-perplexity</a><a href="#ops">test-backend-ops</a>'
             '<a href="#commands">Commands</a><a href="#raw">Raw output</a></nav>\n</header>\n<main>']

    commit = site.commit_link(run, short=False)
    if run.get('commit_subject'):
        commit += f' {esc(run.get("commit_subject"))}'
    if run.get('commit_date'):
        commit += f' <span class="sd">({esc(run.get("commit_date"))})</span>'
    workflow = f'{esc(wf.get("name"))} <span class="mono">{esc(wf.get("ref"))}</span>'
    if wf.get('sha'):
        workflow += f' <span class="sd">at {esc(wf.get("sha"))}</span>'
    parts.append('<section id="run"><h2>Run</h2>')
    parts.append(kv_table([
        ('Status', badge(run.get('status'))),
        ('Started (UTC)', esc(run.get('started_utc'))),
        ('Finished (UTC)', esc(run.get('finished_utc'))),
        ('Ref', f'<code>{esc(run.get("ref"))}</code>'),
        ('Commit', commit),
        ('Workflow', workflow),
        ('Actions run', link(wf.get('run_url'), f'run {wf.get("run_id")} attempt {wf.get("run_attempt")}')),
        ('Raw-log artifact', link(wf.get('artifact_url'), 'download (kept 90 days)')),
        ('Runner', f'<span class="mono">{esc(runner.get("name"))}</span> {esc(runner.get("os"))} {esc(runner.get("arch"))}'),
        ('Dispatched by', esc(wf.get('dispatched_by'))),
    ]))
    parts.append('<h3>Environment</h3>')
    parts.append(kv_table([
        ('Kernel', esc(env.get('kernel'))),
        ('CPUs / RAM', f'{esc(env.get("nproc"))} / {esc(env.get("mem_total_gib"))} GiB'),
        ('HIP', esc(env.get('hip_version'))),
        ('CMake', esc(env.get('cmake_version'))),
        ('ROCm0', esc(env.get('rocm_device'))),
        ('Vulkan0', esc(env.get('vulkan_device'))),
        ('Vulkan driver', esc(env.get('vulkan_driver'))),
        ('Vulkan SDK', esc(env.get('vulkan_sdk'))),
    ]))
    parts.append('<h3>Matrix</h3>')
    parts.append(kv_table([
        ('Models', esc(', '.join(matrix.get('models', [])))),
        ('ubatch (-ub)', esc(','.join(map(str, matrix.get('ubatch', []))))),
        ('Depths (-d)', esc(','.join(map(str, matrix.get('depths', []))))),
        ('Repetitions (-r)', esc(matrix.get('reps'))),
        ('pp / tg / batch', f'{esc(matrix.get("n_prompt"))} / {esc(matrix.get("n_gen"))} / {esc(matrix.get("n_batch"))}'),
        ('llama-perplexity', esc(f'Wikitext-2 wiki.test.raw (sha256 {matrix.get("wikitext_sha256")}), -c {matrix.get("ppl_ctx")}, '
                                 f'{matrix.get("ppl_chunks")} chunks, -b/-ub {",".join(map(str, matrix.get("ppl_batches", [])))}')
         if matrix.get('ppl') else 'no'),
        ('test-backend-ops perf', 'yes' if matrix.get('ops_perf') else 'no'),
    ]))
    steps = run.get('steps') or {}
    if steps:
        parts.append('<h3>Steps</h3><table><thead><tr><th>Step</th><th>Outcome</th></tr></thead><tbody>'
                     + ''.join(f'<tr><td>{esc(k)}</td><td>{badge(v or "not-run") if v != "success" else badge("ok")}</td></tr>'
                               for k, v in steps.items())
                     + '</tbody></table>')
    parts.append('</section>')

    models = {m.get('key'): m for m in run.get('models', [])}
    run_cells = cells(run)
    parts.append('<section id="bench"><h2>llama-bench</h2>')
    if not run.get('llama_bench'):
        parts.append('<p class="note">No llama-bench results.</p>')
    for entry in run.get('llama_bench', []):
        name = entry.get('model')
        parts.append(f'<h3>{esc(name)} {badge(entry.get("status"))}</h3>')
        m = models.get(name)
        if m:
            sha_ok = m.get('sha256_actual') == m.get('sha256')
            parts.append(kv_table([
                ('File', f'<code>{esc(m.get("file"))}</code>'),
                ('Hugging Face', link(f'https://huggingface.co/{m.get("repo")}/tree/{m.get("revision")}', f'{m.get("repo")} @ {m.get("revision")}')),
                ('Bytes', esc(m.get('bytes'))),
                ('sha256 (pinned)', f'<code>{esc(m.get("sha256"))}</code>'),
                ('sha256 (downloaded)', f'<code>{esc(m.get("sha256_actual"))}</code> ' + (badge('ok') if sha_ok else badge(m.get('status')))),
            ]))
        rows = entry.get('rows', [])
        if rows:
            first = rows[0]
            parts.append(kv_table([
                ('Model type', esc(first.get('model_type'))),
                ('Size / params', f'{esc(first.get("model_size"))} B / {esc(first.get("model_n_params"))}'),
                ('Backend / GPU', f'{esc(first.get("backends"))} / {esc(first.get("gpu_info"))}'),
                ('Build', f'{esc(first.get("build_commit"))} ({esc(first.get("build_number"))})'),
                ('Settings', esc(f'ngl {first.get("n_gpu_layers")}, fa {first.get("flash_attn")}, '
                                 f'K {first.get("type_k")}, V {first.get("type_v")}, threads {first.get("n_threads")}')),
            ]))
        mine = [c for c in run_cells if c['model'] == name]
        for test in sorted({c['test'] for c in mine}, key=lambda t: (t[:2] != 'pp', t)):
            sub = [c for c in mine if c['test'] == test]
            ubs = sorted({c['n_ubatch'] for c in sub}, key=lambda v: v or 0)
            depths = sorted({c['n_depth'] for c in sub}, key=lambda v: v or 0)
            grid = {(c['n_ubatch'], c['n_depth']): c for c in sub}
            parts.append(f'<p class="note">{esc(test)}, mean t/s &plusmn; sd</p><table><thead><tr><th>ubatch</th>'
                         + ''.join(f'<th class="num">depth {esc(d)}</th>' for d in depths) + '</tr></thead><tbody>')
            for ub in ubs:
                parts.append(f'<tr><th class="num">{esc(ub)}</th>' + ''.join(tps(grid.get((ub, d))) for d in depths) + '</tr>')
            parts.append('</tbody></table>')
        if mine:
            parts.append('<details><summary>All samples</summary><div class="scroll"><table><thead><tr><th>Test</th>'
                         '<th class="num">batch</th><th class="num">ubatch</th><th class="num">depth</th><th class="num">mean</th>'
                         '<th class="num">sd</th><th class="num">median</th><th class="num">min</th><th class="num">max</th>'
                         '<th class="num">n</th><th>samples t/s</th></tr></thead><tbody>')
            for c in sorted(mine, key=lambda c: (c['kind'] != 'pp', c['n_ubatch'] or 0, c['n_depth'] or 0)):
                parts.append(f'<tr><td>{esc(c["test"])}</td><td class="num">{esc(c["n_batch"])}</td><td class="num">{esc(c["n_ubatch"])}</td>'
                             f'<td class="num">{esc(c["n_depth"])}</td><td class="num">{fmt(c["mean"])}</td><td class="num">{fmt(c["sd"])}</td>'
                             f'<td class="num">{fmt(c["median"])}</td><td class="num">{fmt(c["min"])}</td><td class="num">{fmt(c["max"])}</td>'
                             f'<td class="num">{c["n"]}</td><td class="mono">{esc(", ".join(fmt(s) for s in c["samples"]))}</td></tr>')
            parts.append('</tbody></table></div></details>')
    parts.append('</section>')

    parts.append('<section id="ppl"><h2>llama-perplexity (Wikitext-2 sample)</h2>')
    entries = run.get('perplexity') or []
    if not entries:
        parts.append('<p class="note">Not run.</p>')
    else:
        base = {}
        for e in entries:
            base.setdefault(e.get('model'), e)
        parts.append('<table><thead><tr><th>Model</th><th class="num">batch / ubatch</th><th class="num">ctx x chunks</th>'
                     '<th class="num">PPL</th><th class="num">vs first batch</th><th class="num">s/pass</th><th class="num">wall s</th>'
                     '<th>Status</th><th>Raw log</th></tr></thead><tbody>')
        for e in entries:
            name = f'perplexity-{e.get("model")}-b{e.get("n_batch")}.log'
            raw = link(site.log_url(key, name), name) if any(lg.get('name') == name for lg in run.get('logs', [])) else '-'
            delta = ppl_delta(base[e.get('model')], e) if base[e.get('model')] is not e else '<td class="num">-</td>'
            parts.append(f'<tr><td>{esc(e.get("model"))}</td><td class="num">{esc(e.get("n_batch"))} / {esc(e.get("n_ubatch"))}</td>'
                         f'<td class="num">{esc(e.get("n_ctx")) + " x " + esc(e.get("n_chunks")) if e.get("n_ctx") else "-"}</td>{ppl_cell(e)}{delta}'
                         f'<td class="num">{fmt(e.get("seconds_per_pass"))}</td><td class="num">{fmt(e.get("wall_seconds"), 1)}</td>'
                         f'<td>{badge(e.get("status"))}</td><td>{raw}</td></tr>')
        parts.append('</tbody></table>')
        parts.append('<details><summary>Cumulative PPL per chunk</summary><table><tbody>'
                     + ''.join(f'<tr><th>{esc(e.get("model"))} b{esc(e.get("n_batch"))}</th>'
                               f'<td class="mono wrap">{esc(", ".join(f"{v:.4f}" for v in e.get("chunk_ppl") or []))}</td></tr>'
                               for e in entries)
                     + '</tbody></table></details>')
    parts.append('</section>')

    parts.append('<section id="ops"><h2>test-backend-ops perf</h2>')
    ops = run.get('ops_perf') or []
    if not ops:
        parts.append('<p class="note">Not run.</p>')
    else:
        parts.append('<table><thead><tr><th>Backend</th><th>Status</th><th class="num">Exit</th><th class="num">Rows</th>'
                     '<th class="num">Not supported</th><th>Table</th><th>Raw log</th></tr></thead><tbody>')
        for o in ops:
            backend = o.get('backend')
            table = f'<a href="ops-{esc(backend)}.html">all rows</a>' if o.get('rows') else '-'
            raw = link(site.log_url(key, f'ops-perf-{backend}.log'), f'ops-perf-{backend}.log') \
                if any(lg.get('name') == f'ops-perf-{backend}.log' for lg in run.get('logs', [])) else '-'
            parts.append(f'<tr><td>{esc(backend)}</td><td>{badge(o.get("status"))}</td><td class="num">{esc(o.get("exit_code"))}</td>'
                         f'<td class="num">{esc(o.get("n_rows"))}</td><td class="num">{esc(o.get("n_unsupported"))}</td>'
                         f'<td>{table}</td><td>{raw}</td></tr>')
        parts.append('</tbody></table>')
    parts.append('</section>')

    parts.append('<section id="commands"><h2>Commands</h2>')
    for cmd in run.get('commands', []):
        parts.append(f'<h3>{esc(cmd.get("label"))}</h3><pre>{esc(cmd.get("command"))}</pre>')
    if not run.get('commands'):
        parts.append('<p class="note">No commands recorded.</p>')
    parts.append('</section>')

    parts.append('<section id="raw"><h2>Raw output</h2>'
                 '<p class="note">Machine-readable: <a href="run.json">run.json</a>. The logs below live on the '
                 f'<code>{esc(site.branch)}</code> branch; build logs are only in the artifact.</p><ul>')
    for lg in run.get('logs', []):
        url = site.log_url(key, lg.get('name', ''))
        name = esc(lg.get('name'))
        item = f'<a href="{esc(url)}">{name}</a>' if url else name
        parts.append(f'<li>{item} <span class="sd">{esc(lg.get("bytes"))} bytes</span></li>')
    parts.append('</ul></section>\n</main>')
    return page(f'Run {key}', '\n'.join(parts))


def throughput(row: list[Any]) -> str:
    if row[5] is not None:
        return f'{row[5]:.2f} GB/s'
    if row[7] is not None:
        for scale, unit in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M')):
            if row[7] >= scale:
                return f'{row[7] / scale:.2f} {unit}FLOPS'
        return f'{row[7] / 1e3:.2f} kFLOPS'
    return '-'


def build_ops(run: dict[str, Any], ops: dict[str, Any]) -> str:
    cols = ops.get('columns') or []
    expected = ['op', 'params', 'runs', 'us_per_run', 'kb_per_run', 'gb_per_s', 'flop_per_run', 'flop_per_s']
    if cols != expected:
        return page('test-backend-ops perf', '<main><p>Unknown column layout.</p></main>')
    backend = ops.get('backend')
    parts = [f'<header>\n<nav><a href="index.html">Run {esc(run["key"])}</a><a href="../../index.html">All runs</a></nav>',
             f'<h1>test-backend-ops perf - {esc(backend)}</h1>',
             f'<p class="lede">{badge(ops.get("status"))} exit {esc(ops.get("exit_code"))}, {esc(ops.get("n_rows"))} rows, '
             f'{esc(ops.get("n_unsupported"))} not supported. <code>{esc(run.get("ref"))}</code> at <code>{esc((run.get("commit") or "")[:10])}</code>.</p>',
             '</header>\n<main>',
             '<div class="filters"><label>Filter <input id="ops-filter" type="search" size="40"></label>'
             '<span class="note" id="ops-count"></span></div>',
             '<div class="scroll"><table id="ops-table"><thead><tr><th class="num">#</th><th>Op</th><th>Params</th>'
             '<th class="num">Runs</th><th class="num">us/run</th><th class="num">Throughput</th></tr></thead><tbody>']
    for i, row in enumerate(ops.get('rows') or [], 1):
        parts.append(f'<tr><td class="num">{i}</td><td>{esc(row[0])}</td><td class="wrap mono">{esc(row[1])}</td>'
                     f'<td class="num">{esc(row[2])}</td><td class="num">{fmt(row[3])}</td><td class="num">{throughput(row)}</td></tr>')
    parts.append('</tbody></table></div>\n</main>')
    return page(f'test-backend-ops perf {backend} - {run["key"]}', '\n'.join(parts), OPS_JS)


def summary(run: dict[str, Any]) -> dict[str, Any]:
    keep = ('key', 'status', 'started_utc', 'finished_utc', 'ref', 'commit', 'commit_subject', 'matrix')
    out = {k: run.get(k) for k in keep}
    out['runner'] = (run.get('runner') or {}).get('name')
    out['run_url'] = (run.get('workflow') or {}).get('run_url')
    out['results'] = [{k: c[k] for k in ('model', 'test', 'n_batch', 'n_ubatch', 'n_depth', 'mean', 'sd', 'median', 'n')}
                      for c in cells(run)]
    out['perplexity'] = [{k: p.get(k) for k in ('model', 'n_batch', 'n_ubatch', 'n_ctx', 'n_chunks', 'ppl', 'ppl_err', 'status')}
                         for p in run.get('perplexity') or []]
    out['ops_perf'] = [{k: o.get(k) for k in ('backend', 'status', 'exit_code', 'n_rows')} for o in run.get('ops_perf', [])]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='checkout of the results branch')
    parser.add_argument('--out', type=Path, required=True, help='output directory, replaced')
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY', ''), help='owner/name for links')
    parser.add_argument('--server-url', default=os.environ.get('GITHUB_SERVER_URL', 'https://github.com'))
    parser.add_argument('--branch', default='bench-results', help='results branch, for raw log links')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

    epoch = os.environ.get('SOURCE_DATE_EPOCH')
    now = datetime.datetime.fromtimestamp(int(epoch), datetime.timezone.utc) if epoch else datetime.datetime.now(datetime.timezone.utc)
    generated = now.strftime('%Y-%m-%d %H:%M')

    site = Site(args.repository, args.server_url, args.branch)
    runs = load_runs(args.results)
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / 'data').mkdir(parents=True)

    for run in runs:
        run_dir = args.out / 'runs' / run['key']
        run_dir.mkdir(parents=True)
        (run_dir / 'index.html').write_text(build_run(site, run))
        for ops in run.get('ops_perf') or []:
            if ops.get('rows') and SAFE_NAME.match(str(ops.get('backend'))):
                (run_dir / f'ops-{ops["backend"]}.html').write_text(build_ops(run, ops))
        shutil.copyfile(args.results / 'runs' / run['key'] / 'run.json', run_dir / 'run.json')

    (args.out / 'index.html').write_text(build_index(site, runs, generated))
    (args.out / 'data' / 'runs.json').write_text(json.dumps([summary(r) for r in runs], indent=1) + '\n')
    logger.info('built %d runs into %s', len(runs), args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
