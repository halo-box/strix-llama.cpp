#!/usr/bin/env python3
"""The model archs a change of src/models/ concerns, one "<arch> TAB <changed file>" line each. A file is not an arch:
a base file (delta-net-base.cpp) is built on by other models, a model class can serve several archs, and file names do
not always match arch names (ernie4-5.cpp: ernie4_5, ernie4_5-moe, paddleocr). So: the classes each changed file defines,
plus the models.h structs whose declaration changed; every models.h struct that names an affected class (a graph built
on llm_build_delta_net_base, a model deriving from another), until none is added; then the archs the factory switch of
src/llama-model.cpp builds with the affected llama_model_* classes, named as in src/llama-arch.cpp.
A changed file that maps to no arch prints "- TAB <file>".
Usage: model_archs.py <from> <to>   (the classes and the switch are read at <to>)"""

import re
import subprocess
import sys

old, new = sys.argv[1:3]
NAME = re.compile(r"\b((?:llm_build|llama_model)_\w+)")


def emit(*parts):
    """print(): the output of this script is its result, and the repo's flake8 forbids print"""
    sys.stdout.write("\t".join(str(p) for p in parts) + "\n")


def git(*args, check=True):
    """<check>: a bad revision must stop the table, not print an empty one; only a file added or deleted may be missing"""
    r = subprocess.run(["git", "-c", "core.quotePath=false", *args], capture_output=True, text=True, errors="replace")
    if check and r.returncode:
        sys.exit("git %s: %s" % (" ".join(args), r.stderr.strip()))
    return r.stdout


def blocks(text):
    """the top-level structs of models.h: name -> its text (from the struct line to the closing brace at column 0)"""
    out, name = {}, None
    for line in text.splitlines():
        m = re.match(r"struct (\w+)\b[^;]*$", line)
        if name is None and m:
            name, out[m.group(1)] = m.group(1), ""
        if name is not None:
            out[name] += line + "\n"
            if line.startswith("};"):
                name = None
    return out


changed = [f for f in git("diff", "--name-only", "--no-renames", old, new, "--", "src/models/").splitlines() if f.endswith((".cpp", ".h"))]
header = blocks(git("show", "%s:src/models/models.h" % new))
seeds = {}   # class -> the changed files it comes from
for f in changed:
    if f == "src/models/models.h":
        before = blocks(git("show", "%s:src/models/models.h" % old, check=False))
        cls = [n for n in set(header) | set(before) if header.get(n) != before.get(n)]
    else:
        # definitions start at column 0 (an indented llama_model_x::graph(params) is a base class initializer)
        # a deleted file: its classes before
        text = git("show", "%s:%s" % (new, f), check=False) or git("show", "%s:%s" % (old, f), check=False)
        cls = re.findall(r"^[^\s#/][^(]*?\b((?:llm_build|llama_model)_\w+?)(?:<[^>]*>)?::", text, re.M)
    for c in cls:
        seeds.setdefault(c, set()).add(f)

affected = dict(seeds)
grew = True
while grew:
    grew = False
    for top, text in header.items():
        users = [affected[n] for n in set(NAME.findall(text)) if n in affected and n != top]
        files = set().union(*users) if users else set()
        if files - affected.get(top, set()):
            affected[top] = affected.get(top, set()) | files
            grew = True

# the factory: case LLM_ARCH_X: [case LLM_ARCH_Y:] return new llama_model_z(params);
built, cases = {}, []
for line in git("show", "%s:src/llama-model.cpp" % new).splitlines():
    cases += re.findall(r"case (LLM_ARCH_\w+):", line)
    m = re.search(r"return new (llama_model_\w+)\(", line)
    if m:
        for e in cases:
            built.setdefault(m.group(1), []).append(e)
    if m or re.search(r"\b(break|return|default)\b", line):
        cases = []
names = dict(re.findall(r"\{\s*(LLM_ARCH_\w+)\s*,\s*\"([^\"]+)\"\s*\}", git("show", "%s:src/llama-arch.cpp" % new)))

rows, mapped = set(), set()
for c, files in affected.items():
    for e in built.get(c, []):
        if e in names:
            rows |= {(names[e], f) for f in files}
            mapped |= files
for f in changed:
    if f not in mapped:
        rows.add(("-", f))
for a, f in sorted(rows):
    emit(a, f)
