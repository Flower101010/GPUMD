"""Run the mixed NEP rigid NVT acceptance fixture; smoke is the safe default."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from generate_cases import make
from check import verify
from static_compare import check_frames

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CASE = 'mixed_2700_300'


def main():
    parser = argparse.ArgumentParser(description='Mixed GPUMD NEP rigid NVT acceptance tests')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='2000 steps; structural checks only (DEFAULT)')
    mode.add_argument('--long', action='store_true', help='200000 steps; enforce equilibrium temperature checks')
    parser.add_argument('--steps', type=int, help='Override total steps (including equilibration); divisible by 200')
    parser.add_argument('--dt', type=float, default=0.05, help='Fixed timestep in fs (default 0.05)')
    parser.add_argument('--temperature', type=float, default=300.0, help='Constant target temperature in K')
    parser.add_argument('--initial-temperature', type=float, default=150.0, help='Initial kinetic temperature in K')
    parser.add_argument('--coupling', type=float, default=2000.0, help='NHC coupling in steps; default tau = 100 fs')
    equilibration = parser.add_mutually_exclusive_group()
    equilibration.add_argument('--equilibration-fraction', type=float, help='Discard this initial fraction (default 0.2)')
    equilibration.add_argument('--equilibration-ps', type=float, help='Discard this initial duration in ps instead of a fraction')
    parser.add_argument('--case', choices=[CASE], default=CASE, help='Only the mixed 2700-free/300-body case is supported')
    parser.add_argument('--skip-static', action='store_true', help='Skip first/last frame static NEP cross-check')
    parser.add_argument('--nep', type=Path, default=ROOT / 'potentials/nep/C_2022_NEP4.txt')
    parser.add_argument('--gpumd', type=Path, default=Path(os.environ.get('GPUMD_BIN', ROOT / 'build/gpumd')))
    args = parser.parse_args()
    steps = args.steps if args.steps is not None else (200000 if args.long else 2000)
    if steps < 200 or steps % 200:
        parser.error('--steps must be >= 200 and divisible by 200')
    for name in ('dt', 'temperature', 'initial_temperature', 'coupling'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive and finite')
    if args.coupling < 1:
        parser.error('--coupling must be >= 1 step')
    fraction = args.equilibration_fraction if args.equilibration_fraction is not None else 0.2
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        parser.error('--equilibration-fraction must be in [0, 1)')
    if args.equilibration_ps is not None:
        if not math.isfinite(args.equilibration_ps) or args.equilibration_ps < 0:
            parser.error('--equilibration-ps must be finite and nonnegative')
        equilibration_steps = math.ceil(args.equilibration_ps * 1000 / args.dt)
    else:
        equilibration_steps = int(steps * fraction)
    if equilibration_steps >= steps:
        parser.error('Equilibration must end before the end of the run')
    thermo_interval = steps // (1000 if args.long and steps >= 1000 and steps % 1000 == 0 else 200)
    xyz_interval = steps // (100 if args.long else 20)
    profile = 'long' if args.long else 'smoke'
    if args.steps is not None:
        profile += f'_steps_{steps}_dt_{args.dt:g}'.replace('.', 'p')
    gpumd = args.gpumd.expanduser().resolve()
    nep = args.nep.expanduser().resolve()
    if not gpumd.is_file() or not os.access(gpumd, os.X_OK):
        parser.error(f'GPUMD executable missing/not executable: {gpumd}')
    if not nep.is_file():
        parser.error(f'NEP potential missing: {nep}')
    with nep.open(encoding='utf-8') as stream:
        if stream.readline().split()[:3] != ['nep4', '1', 'C']:
            parser.error('Provide a standard nep4 1 C potential for these synthetic C bead labels')
    folder = HERE / 'results' / profile / CASE
    folder.mkdir(parents=True, exist_ok=True)
    config = make(CASE, folder, args.initial_temperature)
    shutil.copy2(nep, folder / 'nep.txt')
    (folder / 'run.in').write_text(
        'potential nep.txt\n'
        f'ensemble rigid_nvt {args.temperature:.16g} {args.temperature:.16g} {args.coupling:.16g} 0\n'
        f'time_step {args.dt:.16g}\n'
        f'dump_thermo {thermo_interval}\n'
        f'dump_xyz {xyz_interval} trajectory.xyz precision double mass velocity force potential group_labels\n'
        f'run {steps}\n', encoding='utf-8')
    config.update({
        'ensemble': 'rigid_nvt', 'target_temperature_kelvin': args.temperature,
        'coupling_steps': args.coupling, 'tau_fs': args.dt * args.coupling,
        'equilibration_fraction': equilibration_steps / steps,
        'equilibration_steps': equilibration_steps,
        'equilibration_requested_ps': args.equilibration_ps,
        'dt_fs': args.dt, 'steps': steps, 'thermo_interval': thermo_interval,
        'xyz_interval': xyz_interval, 'nep_source': str(nep),
        'nep_sha256': hashlib.sha256(nep.read_bytes()).hexdigest(),
        'gpumd': str(gpumd), 'gpumd_sha256': hashlib.sha256(gpumd.read_bytes()).hexdigest(),
        'profile': profile, 'strict_statistics': args.long,
        'static_check_requested': not args.skip_static,
        'analysis_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                            for name in ('generate_cases.py', 'check.py', 'nvt_statistics.py',
                                         'scale_common.py', 'static_compare.py')},
    })
    (folder / 'manifest.json').write_text(json.dumps(config, indent=2) + '\n')
    for name in ('trajectory.xyz', 'thermo.out', 'gpumd.log', 'validation.json',
                 'summary.json', 'nvt_trace.csv', 'kinetic_partitions.csv', 'nvt_diagnostics.png'):
        (folder / name).unlink(missing_ok=True)
    shutil.rmtree(folder / 'static_reference', ignore_errors=True)
    print(f'Case {CASE}: {config["atoms"]} beads, 300 bodies, 2700 free particles', flush=True)
    print(f'{profile}: dt={args.dt:g} fs, {steps} steps = {steps * args.dt / 1000:g} ps; '
          f'target={args.temperature:g} K, tau={config["tau_fs"]:g} fs', flush=True)
    print(f'One continuous run; discard first {config["equilibration_steps"]} steps for statistics.', flush=True)
    print(f'GPUMD output: {folder / "gpumd.log"}', flush=True)
    start = time.perf_counter()
    with (folder / 'gpumd.log').open('w') as log:
        process = subprocess.run([str(gpumd)], cwd=folder, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    (folder / 'performance.json').write_text(json.dumps({
        'elapsed_seconds': elapsed, 'steps_per_second': steps / elapsed,
        'beads': config['atoms'], 'steps': steps, 'total_simulated_ps': steps * args.dt / 1000,
    }, indent=2) + '\n')
    if process.returncode:
        print('\n'.join((folder / 'gpumd.log').read_text(errors='replace').splitlines()[-35:]), file=sys.stderr)
        raise SystemExit(f'GPUMD exited {process.returncode}; see {folder}')
    errors = []
    equilibrium_acceptance = 'analysis_failed'
    try:
        if not verify(folder):
            errors.append('NVT diagnostics failed; inspect validation.json')
        equilibrium_acceptance = json.loads((folder / 'validation.json').read_text())['equilibrium_acceptance']
    except Exception as exc:
        errors.append(f'Analysis failed: {type(exc).__name__}: {exc}')
    static_passed = None
    if not args.skip_static:
        try:
            static_passed = check_frames(folder, gpumd)
            if not static_passed:
                errors.append('Static NEP reference failed')
        except Exception as exc:
            static_passed = False
            errors.append(f'Static check failed: {type(exc).__name__}: {exc}')
    acceptance = 'long_nvt' if args.long else 'smoke_only'
    if args.skip_static:
        acceptance += '_without_static_reference'
    (folder / 'summary.json').write_text(json.dumps({
        'acceptance_scope': acceptance, 'passed': not errors,
        'equilibrium_acceptance': equilibrium_acceptance,
        'static_reference_passed': static_passed, 'errors': errors,
    }, indent=2) + '\n')
    print(f'Wall time {elapsed:.2f} s. Results: {folder}')
    if errors:
        raise SystemExit('FAIL: ' + '; '.join(errors))
    print('PASS: ' + ('long NVT engineering criteria' if args.long else
                      'smoke checks only; equilibrium/statistical acceptance NOT established'))


if __name__ == '__main__':
    main()
