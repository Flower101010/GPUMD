"""Static ordinary-NVE NEP force/energy reference for the first and last rigid frames."""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scale_common import iter_xyz, Frame


def write_static_model(frame:Frame, dest:Path):
    n = len(frame.pos)
    vals = frame.lattice.reshape(-1)
    cell = ' '.join(f'{x:.16g}' for x in vals)
    with dest.open('w',encoding='utf-8') as fp:
        fp.write(f'{n}\n')
        fp.write('lattice="'+cell+'" pbc="T T T" '
                 'properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1\n')
        for i in range(n):
            r=frame.pos[i];v=frame.vel[i]
            fp.write(f'{frame.species[i]} {r[0]:.17g} {r[1]:.17g} {r[2]:.17g} '
                     f'{frame.mass[i]:.17g} {v[0]:.17g} {v[1]:.17g} {v[2]:.17g} '
                     f'{frame.group[i]}\n')


def check_frames(folder: Path, gpumd:Path):
    folder = Path(folder)
    frames = iter_xyz(folder/'trajectory.xyz')
    first = next(frames)
    last = first
    for frame in frames:
        last = frame

    references = []
    for label, source in [('first',first),('last',last)]:
        refdir = folder/'static_reference'/label
        refdir.mkdir(parents=True,exist_ok=True)
        write_static_model(source,refdir/'model.xyz')
        shutil.copy2(folder/'nep.txt',refdir/'nep.txt')
        (refdir/'run.in').write_text(
            'potential nep.txt\n'
            'ensemble nve\n'
            'time_step 0\n'
            'dump_xyz 1 reference.xyz precision double mass velocity force potential group_labels\n'
            'run 1\n', encoding='utf-8')
        (refdir/'reference.xyz').unlink(missing_ok=True)
        with (refdir/'gpumd.log').open('w') as log:
            run = subprocess.run([str(gpumd)],cwd=refdir,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode != 0:
            msg=(refdir/'gpumd.log').read_text(errors='replace').splitlines()[-30:]
            raise RuntimeError(f'Static reference {label} failed: '+ '\n'.join(msg))
        reference = next(iter_xyz(refdir/'reference.xyz'))
        coordinate_err = float(np.max(np.linalg.norm(source.pos-reference.pos, axis=1)))
        dF = np.linalg.norm(source.forces-reference.forces, axis=1)
        fmax = float(np.max(np.linalg.norm(reference.forces,axis=1)))
        max_force_err = float(np.max(dF))
        max_peratom_u = float(np.max(np.abs(source.energy_atom-reference.energy_atom)))
        total_u_err = float(abs(np.sum(source.energy_atom)-np.sum(reference.energy_atom)))
        allowed_f = 2e-4 + 2e-5*fmax
        allowed_u_per = 5e-5
        allowed_u_total = 5e-5*len(source.pos)
        passed = (coordinate_err < 1e-8 and max_force_err <= allowed_f
                  and max_peratom_u <= allowed_u_per and total_u_err <= allowed_u_total)
        print(f'Static NEP {label} (t={source.time_fs:g} fs): {"PASS" if passed else "FAIL"}')
        print(f'  max |Δr|: {coordinate_err:.7e} Å')
        print(f'  max |ΔF_i|: {max_force_err:.7e} eV/Å (limit {allowed_f:.4e})')
        print(f'  max |ΔU_i|: {max_peratom_u:.7e} eV')
        print(f'  |ΔU_total|: {total_u_err:.7e} eV')
        references.append({'frame':label,'t_fs':source.time_fs,'passed':passed,
                           'max_coordinate_difference_angstrom':coordinate_err,
                           'max_per_bead_force_difference_ev_per_angstrom':max_force_err,
                           'max_per_bead_energy_difference_ev':max_peratom_u,
                           'total_energy_difference_ev':total_u_err})
    (folder/'static_reference'/'validation.json').write_text(json.dumps(references,indent=2)+'\n')
    return all(ref['passed'] for ref in references)
