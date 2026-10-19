"""LAMMPS compatibility regression: real run.in plus independently captured reference.

Use --lammps to rerun the reference executable; otherwise the checked-in values
are used. Geometry-preserving rigid_npt remains a separate control.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests/gpumd/rigid/lammps_npt_reference_probe'))
import compare

REFERENCE = ROOT/'tests/fixtures/rigid_npt_lammps_run_in.json'


def summarize(states, count):
    first, last = states[0], states[-1]
    pairs = [(j+1, j+2) for j in range(0, count, 3)]
    pairs += [(j+1, j+3) for j in range(0, count, 3)]
    pairs += [(j+2, j+3) for j in range(0, count, 3)]
    d0, d1 = compare.distances(first), compare.distances(last)
    return {"box_A": last['lengths'], "distances_A": {f'{i}-{j}': d1[f'{i}-{j}'] for i,j in pairs},
            "distance_changes_A": {f'{i}-{j}': d1[f'{i}-{j}']-d0[f'{i}-{j}'] for i,j in pairs}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpumd', type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--lammps', type=Path)
    parser.add_argument('--lammps-source', type=Path)
    parser.add_argument('--write-reference', type=Path)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if args.probe:
        probe = subprocess.run([str(args.probe.resolve()), '--probe'], check=True, capture_output=True, text=True)
        if probe.stdout.startswith('SKIP:'):
            print('SKIP: no accessible GPU for LAMMPS rigid NPT compatibility.'); return
    if args.write_reference and not (args.lammps and args.lammps_source):
        parser.error('reference capture requires --lammps and --lammps-source')
    output = args.out_dir.resolve() if args.out_dir else Path(tempfile.mkdtemp(prefix='gpumd-npt-lammps-'))
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(REFERENCE.read_text()) if not args.write_reference else {}
    cases = [('static_1bar', 1, .0001, False, False),
             ('static_1GPa', 1, 1., False, False),
             ('moving_1bar', 1, .0001, True, False),
             ('moving_100steps', 100, .0001, True, False),
             ('interacting_two_bodies', 10, .0001, True, True)]
    report = {'gpumd_sha256': compare.sha256(args.gpumd.resolve()), 'cases': {}}
    if args.lammps:
        report['lammps_sha256'] = compare.sha256(args.lammps.resolve())
    if args.lammps_source:
        report['lammps_source'] = compare.source_info(args.lammps_source.resolve(),
            ['src/RIGID/fix_rigid_nh_small.cpp', 'src/RIGID/fix_rigid_small.cpp'])
    for name, steps, pressure, moving, interacting in cases:
        a = SimpleNamespace(dt=.5, steps=steps, temperature=300., tau_t=100., pressure=pressure, tau_p=1000.)
        p, v, mass, length = compare.initial_state(3.,300.)
        if not moving: v = [[0.,0.,0.] for _ in p]
        if interacting:
            # Second triangle in a tilted plane, with inter-body LJ forces/torques.
            p += [[x+5, 15+(z-15), 15+(y-15)] for x,y,z in p]
            v += [[vx,vz,vy] for vx,vy,vz in v]
        case_root = output/name
        case_root.mkdir()
        if args.lammps:
            c = case_root/'lammps'; c.mkdir()
            text = compare.lammps_input('npt', a, p,v,mass,length,1)
            if interacting:
                text = text.replace('set group all mol 1', 'set group all mol 1\nset atom 4*6 mol 2')
                text = text.replace('pair_style zero 10\npair_coeff * *', 'pair_style lj/cut 9\npair_coeff * * 0.01 2.0')
            (c/'in.lammps').write_text(text)
            compare.run_case([str(args.lammps.resolve()), '-in','in.lammps'], c)
            reference = summarize(compare.frames(c/'state.dump'), len(p))
            thermo = []
            for line in (c/'log.lammps').read_text().splitlines():
                if len(line.split()) == 7:
                    try: thermo.append(list(map(float,line.split())))
                    except ValueError: pass
            reference['pressure_GPa'] = thermo[-1][2]*1e-4
            reference['temperature_K'] = thermo[-1][1]
        else:
            reference = saved['cases'][name]['reference']
        entry = {'reference': reference, 'parameters': vars(a), 'gpumd': {}}
        for mode in ('rigid_npt_lammps', 'rigid_npt'):
            c = case_root/mode; c.mkdir()
            xyz = compare.xyz_text(p,v,mass,length).splitlines()
            xyz[0] = str(len(p))
            if interacting:
                for i in range(5,8): xyz[i] = xyz[i][:-1]+'1'
            (c/'model.xyz').write_text('\n'.join(xyz)+'\n')
            (c/'potential.txt').write_text('lj 1 C\n'+('0.01 2.0 9.0' if interacting else '0.0 1.0 4.0')+'\n')
            (c/'run.in').write_text(f'potential potential.txt\nensemble {mode} 300 300 200 {pressure} 2000 0\n'
                                  f'time_step 0.5\ndump_thermo 1\ndump_xyz 1 trajectory.xyz precision double velocity force\nrun {steps}\n')
            compare.run_case([str(args.gpumd.resolve())], c)
            actual = summarize(compare.gpumd_frames(c/'trajectory.xyz',.5,p,length),len(p))
            rows = [list(map(float,line.split())) for line in (c/'thermo.out').read_text().splitlines()
                    if line.strip() and not line.startswith('#')]
            actual['pressure_GPa'] = sum(rows[-1][3:6])/3
            actual['temperature_K'] = rows[-1][0]
            entry['gpumd'][mode] = actual
            if mode == 'rigid_npt_lammps':
                error = max(abs(actual['distances_A'][key]-value) for key,value in reference['distances_A'].items())
                # Different unit constants and single-precision GPUMD LJ prevent
                # bitwise matching. Bound both absolute error and the dilation signal.
                assert abs(actual['pressure_GPa']-reference['pressure_GPa']) < 1e-8, name
                assert abs(actual['temperature_K']-reference['temperature_K']) < .01, name
                signal = max(abs(x) for x in reference['distance_changes_A'].values())
                limit = max(3e-10, signal*5e-3)
                assert error < limit, (name,error,limit)
                assert max(abs(x-y) for x,y in zip(actual['box_A'],reference['box_A'])) < 2e-5
                for key, expected in reference['distance_changes_A'].items():
                    if abs(expected)>1e-10:
                        assert actual['distance_changes_A'][key]*expected > 0, (name,key)
            else:
                assert max(abs(x) for x in actual['distance_changes_A'].values()) < 1e-10, name
            header = (c/'thermo.out').read_text()
            assert f'# ensemble {mode}\n' in header
            estimator = 'lammps_discrete_impulse' if mode.endswith('_lammps') else 'instantaneous_mechanical'
            assert f'# rigid_constraint_estimator {estimator}\n' in header
        report['cases'][name] = entry
        print(name, 'reference Δr12=',reference['distance_changes_A']['1-2'],
              'compatibility Δr12=',entry['gpumd']['rigid_npt_lammps']['distance_changes_A']['1-2'],flush=True)
    # Exercise parser state reset and the supported free-particle extension too.
    for name, free in [('switch', False), ('free', True)]:
        case = output/name; case.mkdir()
        p,v,m,L = compare.initial_state(3.,300.)
        xyz = compare.xyz_text(p,v,m,L).splitlines()
        if free:
            for i in range(3): xyz[i+2] = xyz[i+2][:-1]+str(i)
        (case/'model.xyz').write_text('\n'.join(xyz)+'\n')
        (case/'potential.txt').write_text('lj 1 C\n0.0 1.0 4.0\n')
        text = 'potential potential.txt\ntime_step 0.5\n'
        modes = ['rigid_npt_lammps'] if free else ['rigid_npt_lammps','rigid_npt','rigid_npt_lammps']
        for mode in modes:
            text += f'ensemble {mode} 300 300 200 .0001 2000 0\ndump_thermo 1\nrun 3\n'
        (case/'run.in').write_text(text)
        compare.run_case([str(args.gpumd.resolve())],case)
        header = (case/'thermo.out').read_text()
        assert header.count('# ensemble rigid_npt_lammps\n') == (1 if free else 2)
        if not free: assert '# ensemble rigid_npt\n' in header
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.write_reference:
        # Keep only independent LAMMPS values; never bless the implementation output.
        fixture = {key:report[key] for key in ('lammps_sha256','lammps_source')}
        fixture['cases'] = {name: {'reference':case['reference'], 'parameters':case['parameters']}
                            for name,case in report['cases'].items()}
        args.write_reference.write_text(json.dumps(fixture,indent=2,allow_nan=False)+'\n')
    print('PASS: LAMMPS remap/virial compatibility and geometry-preserving control. Results:',output)


if __name__ == '__main__': main()
