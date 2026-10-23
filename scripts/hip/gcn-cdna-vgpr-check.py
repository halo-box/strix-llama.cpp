#!/usr/bin/env python3

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


FIELDS = ('vgprs', 'spill_stores', 'scratch_bytes')
METADATA_FIELDS = ('vgpr_count', 'vgpr_spill_count', 'private_segment_fixed_size', 'wavefront_size')


def parse_assembly(text, architecture):
    target = re.search(r'\.amdgcn_target "[^"\n]+-([^"\n]+)"', text)
    if target is None or target[1] != architecture:
        raise ValueError('missing or unexpected assembly architecture')
    if '.amdgpu_metadata' not in text:
        return {}
    section = text.split('.amdgpu_metadata', 1)[1]
    if '.end_amdgpu_metadata' not in section:
        raise ValueError('unterminated kernel metadata')
    section = section.split('.end_amdgpu_metadata', 1)[0]
    kernels = {}
    for block in re.split(r'(?m)^  - (?=\.)', section)[1:]:
        values = dict(re.findall(r'(?m)^    \.(\w+):[ \t]+(\S+)[ \t]*$', block))
        if not all(key in values for key in ('name',) + METADATA_FIELDS):
            raise ValueError('incomplete kernel metadata')
        name = values['name']
        if not re.fullmatch(r'[A-Za-z0-9_.$]+', name) or name in kernels:
            raise ValueError(f'invalid or duplicate kernel name: {name}')
        numbers = [int(values[key]) for key in METADATA_FIELDS]
        if min(numbers) < 0 or numbers[3] != 32:
            raise ValueError(f'invalid resource count or wave size: {name}')
        kernels[name] = dict(zip(FIELDS, numbers[:3]))
    if not kernels and not re.search(r'(?m)^amdhsa.kernels:[ \t]+\[\][ \t]*$', section):
        raise ValueError('no kernels in metadata section')
    return kernels


def read_assembly(directory, architecture):
    kernels = {}
    units = {}
    for path in sorted(directory.glob('*.lto.s')):
        try:
            records = parse_assembly(path.read_text(), architecture)
        except ValueError as error:
            raise ValueError(f'{path.name}: {error}') from error
        if records:
            unit = path.name.split('.cu-', 1)[0]
            if unit in units:
                raise ValueError(f'duplicate translation unit: {unit}')
            units[unit] = len(records)
        for name, usage in records.items():
            previous = kernels.get(name, usage)
            kernels[name] = {field: max(previous[field], usage[field]) for field in FIELDS}
    return kernels, units


def find_regressions(kernels, units, baseline, compiler_version):
    if baseline['schema_version'] != 1 or baseline['architecture'] != 'gfx1151':
        raise ValueError('unsupported baseline schema or architecture')
    if compiler_version != baseline['compiler_version']:
        raise ValueError('compiler identity changed; review the baseline')
    if not baseline['translation_units'] or not baseline['kernels']:
        raise ValueError('empty baseline coverage or budgets')
    errors = []
    for unit, count in baseline['translation_units'].items():
        if units.get(unit, 0) < count:
            errors.append(f'{unit}: kernel coverage {units.get(unit, 0)} < {count}')
    for name, budget in baseline['kernels'].items():
        if budget['reason'] not in baseline['reasons'] or not baseline['reasons'][budget['reason']]:
            raise ValueError(f'missing baseline reason: {name}')
        if any(type(budget[field]) is not int or budget[field] < 0 for field in FIELDS):
            raise ValueError(f'invalid baseline budget: {name}')
        if budget['vgprs'] > 256 or budget['spill_stores'] == 0:
            raise ValueError(f'invalid baseline spill exception: {name}')
        if name not in kernels:
            errors.append(f'{name}: missing baseline kernel')
    for name, usage in sorted(kernels.items()):
        # gfx1151 can address 256 VGPRs per lane; the shared SIMD pool is larger. See README.md.
        if usage['vgprs'] > 256:
            errors.append(f'{name}: VGPRs {usage["vgprs"]} > architectural limit 256')
        budget = baseline['kernels'].get(name)
        if budget is None:
            if usage['spill_stores']:
                errors.append(f'{name}: new spill ({usage["spill_stores"]} spill saves, {usage["scratch_bytes"]} scratch bytes/lane)')
        else:
            for field in FIELDS:
                if usage[field] > budget[field]:
                    errors.append(f'{name}: {field} {usage[field]} > baseline {budget[field]}')
    return errors


def main():
    parser = argparse.ArgumentParser(description='Check final gfx1151 assembly against per-kernel spill budgets.')
    parser.add_argument('assembly_directory', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--compiler-version', required=True)
    args = parser.parse_args()
    try:
        baseline = json.loads(args.baseline.read_text())
        kernels, units = read_assembly(args.assembly_directory, baseline['architecture'])
        errors = find_regressions(kernels, units, baseline, args.compiler_version)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Error: {error}', file=sys.stderr)  # noqa: NP100
        return 1
    print(f'Parsed {len(kernels)} kernels across {len(units)} translation units from final assembly.')  # noqa: NP100
    print(f'{sum(bool(k["spill_stores"]) for k in kernels.values())} spilling symbols; {len(baseline["kernels"])} bounded baseline entries.')  # noqa: NP100
    print(f'Baseline reasons: {dict(Counter(k["reason"] for k in baseline["kernels"].values()))}')  # noqa: NP100
    for error in errors:
        print(error, file=sys.stderr)  # noqa: NP100
    if errors:
        return 1
    print('PASS: no new spills or increases in per-kernel resource budgets.')  # noqa: NP100
    return 0


if __name__ == '__main__':
    sys.exit(main())
