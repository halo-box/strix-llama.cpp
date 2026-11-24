#!/usr/bin/env python3
# Copies ggml/src/ggml-vulkan of the fork into hooks/ggml/vulkan: the C++ files inside namespace ggml_fork (closed
# around their includes), the shaders and the shader generator as they are, with the adaptations below, each marked
# "hooks:". Run from the repository root:
#   hooks/tools/import-vulkan.py [fork ref] [upstream base]
# The hooks files of the directory (hooks.h, README.md) are not touched.
import os
import re
import subprocess
import sys

REF  = sys.argv[1] if len(sys.argv) > 1 else 'origin/master'
BASE = sys.argv[2] if len(sys.argv) > 2 else '9048bad8d'
DST  = 'hooks/ggml/vulkan'
SRC  = 'ggml/src/ggml-vulkan/'


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True, check=True).stdout


FORK = git('rev-parse', '--short=9', REF).strip()
files = [f for f in git('ls-tree', '-r', '--name-only', REF, SRC).split('\n') if f]
upstream = set(f for f in git('ls-tree', '-r', '--name-only', BASE, SRC).split('\n') if f)

PATHS = [
    ('#include "ggml-vulkan-shaders.hpp"', '#include "ggml-vulkan-hooks-shaders.hpp"'),
]
OPEN = 'namespace ggml_fork { // hooks: the fork code is namespaced, see hooks/ggml/vulkan/README.md'
CLOSE = '} // namespace ggml_fork'

n_files = 0
for f in files:
    rel = f[len(SRC):]
    src = git('show', f'{REF}:{f}')
    out_path = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cxx = (rel.endswith('.cpp') or rel.endswith('.h') or rel.endswith('.hpp')) and not rel.startswith('vulkan-shaders/')
    if not cxx:
        open(out_path, 'w').write(src)  # shaders, the shader generator, CMake files: as they are
        continue
    lines = src.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    state = 'new' if f not in upstream else 'changed'
    out = [f'// hooks: {f} of the fork ({FORK}, upstream {BASE}, {state} in the fork), verbatim inside',
           '// namespace ggml_fork. Adapted lines carry "hooks:".']
    pragma = [i for i, l in enumerate(lines[:5]) if l.strip() == '#pragma once']
    start = pragma[0] + 1 if pragma else 0
    out += lines[:start]
    out.append(OPEN)
    for l in lines[start:]:
        if re.match(r'\s*#\s*include\b', l):
            for a, b in PATHS:
                if l.startswith(a):
                    l = l.replace(a, b) + ' // hooks: the fork shaders'
            out.append(CLOSE)
            out.append(l)
            out.append(OPEN)
        else:
            out.append(l)
    out.append(CLOSE)
    text = '\n'.join(out) + '\n'
    text = text.replace(OPEN + '\n' + CLOSE + '\n', '')
    open(out_path, 'w').write(text)
    n_files += 1
print(n_files, 'files')


def adapt(rel, old, new, count=1):
    p = os.path.join(DST, rel)
    s = open(p).read()
    assert s.count(old) == count, (rel, old, s.count(old))
    open(p, 'w').write(s.replace(old, new))


# the vulkan.hpp dispatcher is the upstream one (global, same instance): its declarations stay outside the namespace
adapt('ggml-vulkan-types.h', '#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1\n',
      CLOSE + ' // hooks: the vulkan.hpp dispatcher of upstream, global\n#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1\n')
adapt('ggml-vulkan-types.h', '#define VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()\n',
      '#define VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()\n' + OPEN + '\n')
adapt('ggml-vulkan.cpp', 'static DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;\n',
      '#if 0 // hooks: the dispatcher of upstream ggml-vulkan.cpp\nstatic DispatchLoaderDynamic ggml_vk_default_dispatcher_instance;\n')
p = os.path.join(DST, 'ggml-vulkan.cpp')
s = open(p).read()
i = s.index('DispatchLoaderDynamic & ggml_vk_default_dispatcher() {')
j = s.index('\n}\n', i) + 3
s = s[:j] + '#endif // hooks\n#define ggml_vk_default_dispatcher_instance (::ggml_vk_default_dispatcher()) // hooks: the upstream dispatcher\n' + s[j:]
open(p, 'w').write(s)

adapt('ggml-vulkan-common.h', 'DispatchLoaderDynamic & ggml_vk_default_dispatcher();\n',
      '// hooks: ggml_vk_default_dispatcher() is the global upstream one (ggml-vulkan-types.h)\n')

# the fork device is built on the upstream instance and logical device (ggml-vulkan-upstream.cpp): one VkDevice, so
# the fork kernels read the upstream buffers and share the queues
adapt('ggml-vulkan.cpp', 'vk_instance.instance = vk::createInstance(instance_create_info);',
      'vk_instance.instance = ::ggml_hook_vk_upstream_instance(); // hooks: the upstream instance')
adapt('ggml-vulkan.cpp', 'device->device = device->physical_device.createDevice(device_create_info);',
      'device->device = ::ggml_hook_vk_upstream_device(idx); // hooks: the upstream logical device, same extensions and features')

# ADL finds both the global declaration of ggml-vulkan.h and the namespaced definition: the fork files define them
# under other names
hdr = re.sub(r'//[^\n]*', '', open('ggml/include/ggml-vulkan.h').read())
api = re.findall(r'GGML_BACKEND_API [^(;]*?\b(ggml_\w+)\s*\(', hdr)
protos = [' '.join(m.group(1).split()) + ';' for m in re.finditer(r'GGML_BACKEND_API\s+([^;]*?\))\s*;', hdr)]
ren = '\n'.join(f'#define {a} ggml_fork_{a.removeprefix("ggml_")} // hooks: the fork definition, not the upstream API' for a in api)
decl = (OPEN + '\n' + '\n'.join(protos) + '\n' + CLOSE + '\n'
        '// hooks: the upstream instance and device (ggml-vulkan-upstream.cpp)\n'
        'vk::Instance ggml_hook_vk_upstream_instance();\n'
        'vk::Device   ggml_hook_vk_upstream_device(size_t idx);\n')
adapt('ggml-vulkan-types.h', '#include "ggml-vulkan.h"\n', '#include "ggml-vulkan.h"\n' + ren + '\n')
adapt('ggml-vulkan-types.h', '#include <vulkan/vulkan.hpp>\n', '#include <vulkan/vulkan.hpp>\n' + decl)

# the hooks side, after the fork types
p = os.path.join(DST, 'ggml-vulkan-types.h')
s = open(p).read().rstrip('\n') + '\n#include "hooks.h" // hooks: the hooks side of the fork code\n'
open(p, 'w').write(s)

# the generated shader symbols go in namespace ggml_fork
gen = 'vulkan-shaders/vulkan-shaders-gen.cpp'
adapt(gen, 'hdr << "#include <cstdint>\\n\\n";', 'hdr << "#include <cstdint>\\n\\nnamespace ggml_fork {\\n\\n"; // hooks: namespaced shader symbols')
adapt(gen, 'src << "#include \\"" << basename(target_hpp) << "\\"\\n\\n";',
      'src << "#include \\"" << basename(target_hpp) << "\\"\\n\\nnamespace ggml_fork {\\n\\n"; // hooks: namespaced shader symbols')
adapt(gen, '        write_file_if_changed(target_hpp, hdr.str());', '        hdr << "} // namespace ggml_fork\\n"; // hooks\n        write_file_if_changed(target_hpp, hdr.str());')
adapt(gen, '        write_binary_file(target_cpp, src.str());', '        src << "} // namespace ggml_fork\\n"; // hooks\n        write_binary_file(target_cpp, src.str());')

# the fork ggml-impl.h for the fork code (ggml_can_fuse_subgraph takes 32 nodes there), next to the fork files
open(os.path.join(DST, 'ggml-impl.h'), 'w').write(
    f'// hooks: ggml/src/ggml-impl.h of the fork ({FORK}, upstream {BASE}), for the fork code only, verbatim\n' +
    git('show', f'{REF}:ggml/src/ggml-impl.h'))
print('adapted')

# GGML_BACKEND_DL: the upstream file has the module entry point (ggml_backend_init)
_p = 'hooks/ggml/vulkan/ggml-vulkan.cpp'
_s = open(_p).read()
_old = 'GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)'
assert _s.count(_old) == 1
open(_p, 'w').write(_s.replace(_old, '// hooks: the module entry point is the upstream one: ' + _old))

# no trailing whitespace in the generated code: the fork has none (its CI checks it), a hooks: marker can leave one
import pathlib as _pathlib  # noqa: E402
_CODE = ('.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh', '.inc', '.comp', '.glsl', '.cmake', '.py')
for _d in ['hooks/ggml/vulkan']:
    for _f in _pathlib.Path(_d).rglob('*'):
        if _f.is_file() and not _f.is_symlink() and (_f.suffix in _CODE or _f.name == 'CMakeLists.txt'):
            _t = _f.read_text(encoding='utf-8', errors='surrogateescape')
            _u = '\n'.join(_l.rstrip(' \t') for _l in _t.split('\n'))
            if _u != _t:
                _f.write_text(_u, encoding='utf-8', errors='surrogateescape')
