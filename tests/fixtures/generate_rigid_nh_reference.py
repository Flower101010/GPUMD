"""Extract unchanged LAMMPS scalar bodies for the independent RigidNH test oracle.
Usage: python3 tests/fixtures/generate_rigid_nh_reference.py /path/to/lammps
Only the class name and inline linkage are adapted; no algorithm is rewritten.
"""
from pathlib import Path
import hashlib
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
source = root / 'src/RIGID/fix_rigid_nh_small.cpp'
header = source.with_suffix('.h')


def extract(text, signature):
    start = text.index(signature)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


cpp, hpp = source.read_text(), header.read_text()
license_text = cpp[cpp.index('/*'):cpp.index('#include')]
commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
provenance = '// Generated test oracle; DO NOT use in production.\n'
provenance += '// LAMMPS commit: ' + commit + '\n'
for file in (source, header):
    provenance += '// ' + file.name + ' SHA256: ' + hashlib.sha256(file.read_bytes()).hexdigest() + '\n'
functions = []
for name in ('nhc_temp_integrate', 'nhc_press_integrate', 'nh_epsilon_dot'):
    body = extract(cpp, 'void FixRigidNHSmall::' + name + '(')
    functions.append('inline ' + body.replace('FixRigidNHSmall::', 'Reference::'))
functions.append(extract(hpp, 'inline double FixRigidNHSmall::maclaurin_series(').replace('FixRigidNHSmall::', 'Reference::'))
Path(__file__).with_name('rigid_nh_lammps_functions.h').write_text(
    provenance + license_text + '\n'.join(functions) + '\n')
