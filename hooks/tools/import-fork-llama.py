#!/usr/bin/env python3
# Copies src/ and include/ of the fork into hooks/llama/fork-src and hooks/llama/fork-include, each file inside
# namespace ggml_fork (closed around its includes, the extern "C" of the headers removed), and generates the
# dispatcher of the public API (hooks/llama/src/llama-fork-dispatch.cpp) and its CMake lists. Run from the root:
#   hooks/tools/import-fork-llama.py [fork ref] [upstream base]
import os
import re
import subprocess
import sys

REF  = sys.argv[1] if len(sys.argv) > 1 else 'origin/master'
BASE = sys.argv[2] if len(sys.argv) > 2 else '9048bad8d'
OPEN = 'namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/llama/fork-src/README.md'
CLOSE = '} // namespace ggml_fork'
# the ggml types named by the fork code: global, so that an elaborated type specifier in the namespace finds them
GLOBAL_DECLS = ('struct ggml_tensor; struct ggml_context; struct ggml_cgraph; struct gguf_context; struct ggml_backend; '
                'struct ggml_backend_buffer; struct ggml_backend_buffer_type; struct ggml_backend_device; '
                'struct ggml_backend_reg; struct ggml_backend_sched; struct ggml_threadpool; struct ggml_opt_context; '
                'struct ggml_backend_event; // hooks: global')


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True, check=True).stdout


FORK = git('rev-parse', '--short=9', REF).strip()


def namespaced(path, src, state):
    lines = src.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    out = [f'// hooks: {path} of the fork ({FORK}, upstream {BASE}, {state} in the fork), verbatim inside',
           '// namespace ggml_fork. Adapted lines carry "hooks:".']
    pragma = [i for i, l in enumerate(lines[:5]) if l.strip() == '#pragma once']
    start = pragma[0] + 1 if pragma else 0
    out += lines[:start]
    out.append(GLOBAL_DECLS)
    out.append(OPEN)
    for l in lines[start:]:
        if re.match(r'\s*#\s*include\b', l):
            out.append(CLOSE)
            out.append(l)
            out.append(OPEN)
        elif re.match(r'^(struct (ggml|gguf)_\w+|typedef struct (ggml|gguf)_\w+\s*\*?\s*\w+)\s*;\s*(//.*)?$', l):
            # a forward declaration of a ggml type: the global one, not a new type in the namespace
            out.append(CLOSE)
            out.append(l + ' // hooks: global')
            out.append(OPEN)
        else:
            out.append(l)
    out.append(CLOSE)
    text = '\n'.join(out) + '\n'
    return text.replace(OPEN + '\n' + CLOSE + '\n', '')


def copy_tree(src_dir, dst_dir):
    files = [f for f in git('ls-tree', '-r', '--name-only', REF, src_dir).split('\n') if f]
    upstream = set(f for f in git('ls-tree', '-r', '--name-only', BASE, src_dir).split('\n') if f)
    for f in files:
        rel = f[len(src_dir):]
        out = os.path.join(dst_dir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        src = git('show', f'{REF}:{f}')
        if f.endswith(('.cpp', '.h', '.hpp')):
            src = namespaced(f, src, 'new' if f not in upstream else 'changed')
        open(out, 'w').write(src)
    return files


copy_tree('src/', 'hooks/llama/fork-src')
copy_tree('include/', 'hooks/llama/fork-include')


def adapt(path, old, new, count=1):
    s = open(path).read()
    assert s.count(old) == count, (path, old, s.count(old))
    open(path, 'w').write(s.replace(old, new))


# C++ linkage inside the namespace: the fork API does not take the C symbols of upstream
for h in ['hooks/llama/fork-include/llama.h']:
    s = open(h).read()
    n = s.count('extern "C" {')
    s = s.replace('extern "C" {', '// hooks: C++ linkage in namespace ggml_fork (extern "C" removed) {')
    s = re.sub(r'#ifdef __cplusplus\n}\n#endif', '#ifdef __cplusplus\n// hooks: end of the removed extern "C"\n#endif', s)
    open(h, 'w').write(s)
    assert n >= 1, h

# explicit global calls of functions that are in the namespace here
for f in ['hooks/llama/fork-src/llama-quant.cpp', 'hooks/llama/fork-src/llama-model.cpp', 'hooks/llama/fork-src/llama-arch.cpp']:
    t = open(f).read()
    t2 = re.sub(r'(?<![\w:>])::(zeros|select_buft|format)\(', r'ggml_fork::\1( /* hooks: namespaced */ ', t)
    assert t2 != t, f
    open(f, 'w').write(t2)
# the EXL3 ops are custom nodes: their kind through ggml_fork_op() (fork-hooks.h)
for root, _, names in os.walk('hooks/llama/fork-src'):
    for n in names:
        f = os.path.join(root, n)
        if not f.endswith(('.cpp', '.h')):
            continue
        t = open(f).read()
        t2 = re.sub(r'([\w\]\[\.]+(?:->[\w\]\[\.]+)*)->op ([!=]=) (GGML_OP_EXL3_\w+)', r'ggml_fork_op(\1) \2 \3 /* hooks: custom node */', t)
        if t2 != t:
            open(f, 'w').write(t2)
p = 'hooks/llama/fork-src/llama-impl.h'
t = open(p).read().rstrip('\n') + '\n#include "fork-hooks.h" // hooks: the hooks side of the fork code\n'
open(p, 'w').write(t)

# ---------------------------------------------------------------- the dispatcher of the public API
def api(text):
    text = re.sub(r'//[^\n]*', '', text)
    out = []
    for m in re.finditer(r'LLAMA_API\b', text):
        i = m.end()
        j = text.index('(', i)
        head = text[i:j]
        name = re.findall(r'(\w+)\s*$', head)[0]
        ret = ' '.join(head[:head.rindex(name)].split())
        d = 0
        for k in range(j, len(text)):
            if text[k] == '(':
                d += 1
            elif text[k] == ')':
                d -= 1
                if d == 0:
                    break
        params = ' '.join(text[j + 1:k].split())
        out.append((name, ret, params))
    return out


up_h = git('show', f'{BASE}:include/llama.h')
fork_h = git('show', f'{REF}:include/llama.h')
up_api = api(up_h)
fork_names = set(n for n, _, _ in api(fork_h))
up_names = set(n for n, _, _ in up_api)


def pnames(params):
    if params in ('', 'void'):
        return []
    names = []
    for p in params.split(','):
        p = p.strip()
        m = re.search(r'(\w+)\s*(\[\s*\])?$', p)
        names.append(m.group(1))
    return names


# where upstream defines each function (hooks_rename on that file)
defs = {}
for f in sorted(git('ls-tree', '-r', '--name-only', BASE, 'src/').split()):
    if not f.endswith('.cpp'):
        continue
    for line in git('show', f'{BASE}:{f}').split('\n'):
        m = re.match(r'^[A-Za-z_][\w\s\*:<>,]*?\b(llama_\w+)\s*\(', line)
        if m and not line.rstrip().endswith(';') and m.group(1) in up_names and m.group(1) not in defs:
            defs[m.group(1)] = f
missing = sorted(up_names - set(defs))

# the functions the libllama hooks already define (renamed *__upstream in their upstream file): llama-hooks.cpp is
# compiled with them renamed *__hooks, and the dispatcher calls those
cm = open('hooks/llama/llama-hooks.cmake').read()
hooked = set()
for m in re.finditer(r'hooks_rename\(llama SOURCE [^\n]* SYMBOLS([^)]*)\)', cm):
    hooked |= set(m.group(1).split())


def fields(text, name):
    text = re.sub(r'//[^\n]*', '', text)
    m = re.search(r'struct ' + name + r'\s*\{', text)
    i = text.index('{', m.start())
    d = 0
    for j in range(i, len(text)):
        if text[j] == '{':
            d += 1
        elif text[j] == '}':
            d -= 1
            if d == 0:
                break
    names = []
    for decl in text[i + 1:j].split(';'):
        decl = decl.strip()
        if decl:
            mm = re.search(r'\(\s*\*\s*(\w+)\s*\)', decl) or re.search(r'(\w+)\s*(\[[^\]]*\])?\s*$', decl)
            names.append(mm.group(1))
    return names


only_fork = sorted(fork_names - up_names)
out = ['// generated by hooks/tools/import-fork-llama.py: the public libllama API with the fork_llama switch',
       '// (GGML_HOOKS=+fork_llama, read once). On: the fork libllama (hooks/llama/fork-src, namespace ggml_fork). Off: the',
       '// upstream functions, renamed *__upstream in their files, or *__hooks for those the libllama hooks define',
       '// (llama-hooks.cpp). The arguments have the same layout in both trees except llama_model_params and',
       '// llama_context_params, converted by field.',
       '',
       '#include "llama.h"',
       '#include "llama-fork-api.h"',
       '#include "ggml-hooks.h"',
       '',
       '#include <cstring>',
       '#include <type_traits>',
       '',
       'namespace ggml_fork {',
       '#include "llama-fork-decls.inc"',
       '}',
       '',
       '// read once: the objects of one side are never passed to the other',
       'bool llama_hook_fork_on() {',
       '    static const bool on = ggml_hook_switch("fork_llama", false);',
       '    return on;',
       '}',
       '',
       'extern "C" {']
for name, ret, params in up_api:
    if name not in missing:
        out.append(f'LLAMA_API {ret} {name}{"__hooks" if name in hooked else "__upstream"}({params});')
out += ['}', '', 'namespace {', '',
        'bool fork_on() {',
        '    return llama_hook_fork_on();',
        '}', '',
        'template <class To, class From> To cv_field(From f) {',
        '    if constexpr (std::is_pointer_v<To> && std::is_pointer_v<From>) {',
        '        return reinterpret_cast<To>(f);',
        '    } else {',
        '        return (To) f;',
        '    }',
        '}', '']
for st, dflt in [('llama_model_params', 'llama_model_default_params'), ('llama_context_params', 'llama_context_default_params')]:
    uf = fields(up_h, st)
    ff = fields(fork_h, st)
    common = [f for f in uf if f in ff]
    out.append(f'{st} cv_up(const ggml_fork::{st} & s) {{')
    out.append(f'    {st} d = {dflt}__upstream();')
    out += [f'    d.{f} = cv_field<decltype(d.{f})>(s.{f});' for f in common]
    out += ['    return d;', '}', '']
    out.append(f'ggml_fork::{st} cv_fork(const {st} & s) {{')
    out.append(f'    ggml_fork::{st} d = ggml_fork::{dflt}(); // the fork-only fields keep their defaults')
    out += [f'    d.{f} = cv_field<decltype(d.{f})>(s.{f});' for f in common]
    out += ['    return d;', '}', '']
out += ['template <class To, class From> To cv(From f) {',
        '    if constexpr (std::is_same_v<To, From>) {',
        '        return f;',
        '    } else if constexpr (std::is_pointer_v<To> && std::is_pointer_v<From>) {',
        '        return reinterpret_cast<To>(f);',
        '    } else if constexpr (std::is_enum_v<To> || std::is_enum_v<From>) {',
        '        return static_cast<To>(f);',
        '    } else if constexpr (std::is_same_v<To, llama_model_params> || std::is_same_v<To, llama_context_params>) {',
        '        return cv_up(f);',
        '    } else if constexpr (std::is_same_v<To, ggml_fork::llama_model_params> || std::is_same_v<To, ggml_fork::llama_context_params>) {',
        '        return cv_fork(f);',
        '    } else {',
        '        static_assert(sizeof(To) == sizeof(From), "a type that differs between the trees");',
        '        To t;',
        '        memcpy((void *) &t, (const void *) &f, sizeof(To));',
        '        return t;',
        '    }',
        '}', '',
        'template <class R, class... FA, class... UA> R call(R (*f)(FA...), UA... a) {',
        '    return f(cv<FA>(a)...);',
        '}', '',
        '} // namespace', '']
for name, ret, params in up_api:
    if name in missing or name not in fork_names:
        continue
    args = ', '.join(pnames(params))
    tail = '__hooks' if name in hooked else '__upstream'
    fargs = (', ' + args) if args else ''
    out.append(f'{ret} {name}({params}) {{')
    if ret == 'void':
        out.append(f'    if (fork_on()) {{ call(&ggml_fork::{name}{fargs}); return; }}')
        out.append(f'    {name}{tail}({args});')
    else:
        out.append(f'    if (fork_on()) {{ return cv<{ret}>(call(&ggml_fork::{name}{fargs})); }}')
        out.append(f'    return {name}{tail}({args});')
    out += ['}', '']
# the functions only the fork has: llama-fork-api.h; without the switch they do nothing
for name, ret, params in api(fork_h):
    if name not in only_fork:
        continue
    args = ', '.join(pnames(params))
    out.append(f'{ret} {name}({params}) {{')
    out.append(f'    if (fork_on()) {{ {"return " if ret != "void" else ""}{"cv<" + ret + ">(" if ret != "void" else ""}call(&ggml_fork::{name}{", " + args if args else ""}){")" if ret != "void" else ""}; {"" if ret != "void" else "return; "}}}')
    if ret != 'void':
        out.append(f'    return {{}};')
    out += ['}', '']
open('hooks/llama/src/llama-fork-dispatch.cpp', 'w').write('\n'.join(out))

# the fork declarations for the dispatcher: the namespaced fork llama.h, without its includes
fh = open('hooks/llama/fork-include/llama.h').read().replace(OPEN, '').replace(CLOSE, '')
fh = re.sub(r'#\s*include\s*[<"][^\n]*', lambda m: '// ' + m.group(0) + ' (hooks: included by the dispatcher)', fh)
fh = fh.replace('#pragma once', '// #pragma once')
fh = fh.replace(GLOBAL_DECLS, '// hooks: the global ggml forward declarations are in the dispatcher')
fh = '\n'.join(l for l in fh.split('\n') if not l.endswith('// hooks: global'))
# the llama types of this header are the fork ones: declared in the namespace before any elaborated specifier finds
# the upstream global ones
local = sorted(set(re.findall(r'struct (llama_\w+)', fh)))
fh = '// hooks: the fork llama types\n' + ''.join(f'struct {t};\n' for t in local) + fh
assert '#ifndef LLAMA_H\n#define LLAMA_H' in fh
fh = fh.replace('#ifndef LLAMA_H\n#define LLAMA_H', '#ifndef LLAMA_FORK_DECLS_H // hooks: not the guard of the upstream llama.h\n#define LLAMA_FORK_DECLS_H')
open('hooks/llama/src/llama-fork-decls.inc', 'w').write(
    '// generated by hooks/tools/import-fork-llama.py: the fork include/llama.h without its includes, for the dispatcher\n' + fh)

# the fork-only declarations, global: llama-fork-api.h
fapi = ['// generated by hooks/tools/import-fork-llama.py: the functions of the fork include/llama.h that upstream does not',
        '// have. Without the fork_llama switch they do nothing.', '#pragma once', '', '#include "llama.h"', '',
        '#ifdef __cplusplus', 'extern "C" {', '#endif']
for name, ret, params in api(fork_h):
    if name in only_fork:
        fapi.append(f'LLAMA_API {ret} {name}({params});')
fapi += ['#ifdef __cplusplus', '}', '#endif', '']
os.makedirs('hooks/llama/include', exist_ok=True)
open('hooks/llama/include/llama-fork-api.h', 'w').write('\n'.join(fapi))

# the CMake lists
by_file = {}
for name in sorted(up_names):
    if name in missing or name in hooked or name not in fork_names:
        continue
    by_file.setdefault(defs[name], []).append(name)
cmake = ['# generated by hooks/tools/import-fork-llama.py: the upstream files of the public functions the dispatcher replaces']
cmake.append('set(LLAMA_FORK_RENAMES')
for f, names in sorted(by_file.items()):
    cmake.append(f'    "{f}|{",".join(names)}"')
cmake.append(')')
cmake.append('set(LLAMA_FORK_HOOKED ' + ' '.join(sorted(hooked)) + ')')
open('hooks/llama/fork-src/symbols.cmake', 'w').write('\n'.join(cmake) + '\n')
print(len(up_api), 'functions;', 'not defined in src:', missing, '; fork only:', only_fork,
      '; missing in the fork:', sorted(up_names - fork_names))
