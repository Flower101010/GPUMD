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


def main(*, ensemble="rigid_nvt", suite_dir=HERE):
    is_npt = ensemble == "rigid_npt"
    suite_dir = Path(suite_dir)
    parser = argparse.ArgumentParser(description=f'Mixed GPUMD NEP {ensemble} acceptance tests')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='2000 steps; structural checks only (DEFAULT)')
    mode.add_argument('--long', action='store_true', help='Enforce production statistics; default 200 ps for NPT, 50 ps for NVT')
    parser.add_argument('--steps', type=int, help='Override total steps (including equilibration); divisible by 200')
    parser.add_argument('--dt', type=float, help='Fixed timestep in fs (default NPT long 0.5; NVT long 2; smoke 0.05)')
    parser.add_argument('--temperature', type=float, default=300.0, help='Constant target temperature in K')
    parser.add_argument('--initial-temperature', type=float, help='Initial kinetic temperature in K (default NPT long / NVT from NPT: target; other modes: 150)')
    parser.add_argument('--coupling', type=float, help='NHC coupling in steps (default NPT long 200; NVT long 50; smoke 2000; tau = 100 fs)')
    equilibration = parser.add_mutually_exclusive_group()
    equilibration.add_argument('--equilibration-fraction', type=float, help='Discard initial fraction (default NVT long 0.4; other modes 0.2)')
    equilibration.add_argument('--equilibration-ps', type=float, help='Discard this initial duration in ps instead of a fraction')
    parser.add_argument('--case', choices=[CASE], default=CASE, help='Only the mixed 2700-free/300-body case is supported')
    parser.add_argument('--skip-static', action='store_true', help='Skip first/last frame static NEP cross-check')
    parser.add_argument('--nep', type=Path, default=ROOT / 'potentials/nep/C_2022_NEP4.txt')
    parser.add_argument('--gpumd', type=Path, default=Path(os.environ.get('GPUMD_BIN', ROOT / 'build/gpumd')))
    parser.add_argument('--output', type=Path, help='Explicit result directory; existing generated outputs are replaced')
    parser.add_argument('--thermo-interval', type=int,
                        help='Thermo interval in steps (default NPT long 20; NVT long 5; smoke 10)')
    initial_state = parser.add_mutually_exclusive_group()
    initial_state.add_argument('--equilibrated-from', type=Path,
                               help='Explicit alternative: use the final frame of an accepted NVT result directory')
    initial_state.add_argument('--fresh-lattice', action='store_true',
                               help='Use the original unrelaxed lattice instead of an existing frame')
    if not is_npt:
        initial_state.add_argument('--npt-source', type=Path,
                                  help='Completed NPT source (default NVT long: sibling NPT results/long_from_nvt/mixed_2700_300)')
        parser.add_argument('--box-scale', type=float,
                            help='Scale box and molecular centers for explicit NVE-frame initialization (default 2; >= 1)')
        initial_state.add_argument('--nve-source', type=Path,
                                  help='Explicit alternative: use an existing validated NVE result directory')
    if is_npt:
        initial_state.add_argument('--nvt-initial-source', type=Path,
                                  help='NVT result directory whose initial model.xyz is reused exactly (default NPT long: 0.5 fs NVT comparison)')
        parser.add_argument('--pressure', type=float, default=0.0001, help='Target pressure in GPa (default 1 bar)')
        parser.add_argument('--pressure-coupling', type=float, help='Piston coupling in steps (default long 2000; smoke 20000; tau = 1 ps)')
        parser.add_argument('--pressure-tolerance', type=float, default=0.01,
                            help='Absolute engineering tolerance for mean pressure in GPa (default 0.01)')
    args = parser.parse_args()
    npt_long = is_npt and args.long
    reuse_nvt_input = is_npt and (args.nvt_initial_source is not None
        or (args.long and not args.fresh_lattice and args.equilibrated_from is None))
    reuse_nve = not is_npt and args.nve_source is not None
    reuse_npt = not is_npt and (args.npt_source is not None
                or (args.long and not args.fresh_lattice and not reuse_nve and args.equilibrated_from is None))
    reuse_equilibrated = (args.equilibrated_from is not None
                          or (is_npt and args.long and not args.fresh_lattice and not reuse_nvt_input))
    if not is_npt:
        if args.box_scale is None:
            args.box_scale = 2.0 if reuse_nve else 1.0
        if not math.isfinite(args.box_scale) or args.box_scale < 1:
            parser.error('--box-scale must be finite and >= 1')
        if not reuse_nve and args.box_scale != 1:
            parser.error('--box-scale requires initialization from an existing NVE frame')
    if args.initial_temperature is None:
        args.initial_temperature = args.temperature if (npt_long or reuse_npt or reuse_nvt_input) else 150.0
    if args.dt is None:
        args.dt = (0.5 if is_npt else 2.0) if args.long else 0.05
    if args.coupling is None:
        args.coupling = (200.0 if is_npt else 50.0) if args.long else 2000.0
    if is_npt and args.pressure_coupling is None:
        args.pressure_coupling = 2000.0 if npt_long else 20000.0
    # Fail before MD if the requested automatic figures cannot be generated.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot
    except ImportError:
        parser.error('Automatic plots require Matplotlib; install numpy and matplotlib before running')
    steps = args.steps if args.steps is not None else ((400000 if is_npt else 25000) if args.long else 2000)
    if steps < 200 or steps % 200:
        parser.error('--steps must be >= 200 and divisible by 200')
    for name in ('dt', 'temperature', 'initial_temperature', 'coupling'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive and finite')
    if args.coupling < 1:
        parser.error('--coupling must be >= 1 step')
    default_fraction = 0.4 if args.long and not is_npt else 0.2
    fraction = args.equilibration_fraction if args.equilibration_fraction is not None else default_fraction
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
    xyz_interval = steps // (100 if args.long else 20)
    thermo_interval = args.thermo_interval if args.thermo_interval is not None else math.gcd((20 if is_npt else 5) if args.long else 10, xyz_interval)
    if thermo_interval < 1 or steps % thermo_interval or xyz_interval % thermo_interval:
        parser.error('--thermo-interval must divide both steps and the XYZ interval (steps/100 long, steps/20 smoke)')
    if is_npt:
        if not math.isfinite(args.pressure):
            parser.error('--pressure must be finite')
        for name in ('pressure_coupling', 'pressure_tolerance'):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(f'--{name.replace("_", "-")} must be positive and finite')
        if args.pressure_coupling < 1:
            parser.error('--pressure-coupling must be >= 1 step')
    profile = 'long' if args.long else 'smoke'
    if reuse_nvt_input:
        profile += '_from_nvt_initial'
    elif reuse_nve:
        profile += '_from_nve'
    elif reuse_npt:
        profile += '_from_npt'
    elif not is_npt and reuse_equilibrated:
        profile += '_from_nvt'
    if args.long and (not is_npt or reuse_nvt_input) and args.steps is None:
        profile += f'_dt_{args.dt:g}fs'.replace('.', 'p')
    if args.steps is not None:
        profile += f'_steps_{steps}_dt_{args.dt:g}'.replace('.', 'p')
    if not is_npt and args.box_scale != 1:
        profile += f'_box_{args.box_scale:g}'.replace('.', 'p')
    gpumd = args.gpumd.expanduser().resolve()
    nep = args.nep.expanduser().resolve()
    if not gpumd.is_file() or not os.access(gpumd, os.X_OK):
        parser.error(f'GPUMD executable missing/not executable: {gpumd}')
    if not nep.is_file():
        parser.error(f'NEP potential missing: {nep}')
    with nep.open(encoding='utf-8') as stream:
        if stream.readline().split()[:3] != ['nep4', '1', 'C']:
            parser.error('Provide a standard nep4 1 C potential for these synthetic C bead labels')
    folder = args.output.expanduser().resolve() if args.output is not None else suite_dir / 'results' / profile / CASE
    folder.mkdir(parents=True, exist_ok=True)
    if reuse_nvt_input:
        from initial_model import from_nvt_initial_model
        source = args.nvt_initial_source or HERE / 'results' / 'long_from_npt_steps_100000_dt_0p5' / CASE
        try:
            config = from_nvt_initial_model(source.expanduser(), folder, args.initial_temperature, args.temperature, nep)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(f'Cannot reuse the NVT initial model: {exc}. '
                         'Provide --nvt-initial-source pointing to the intended NVT result directory.')
    elif reuse_nve:
        from nve_initial_state import from_existing_nve
        source = args.nve_source or HERE.parent / 'scale_validation' / 'results' / 'long' / CASE
        try:
            config = from_existing_nve(source.expanduser(), folder, args.initial_temperature, nep,
                                       box_scale=args.box_scale)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(f'Cannot reuse existing NVE state: {exc}. '
                         'Provide --nve-source pointing to validated results, '
                         'or explicitly use --fresh-lattice for the original fixture.')
    elif reuse_npt:
        from equilibrated_state import from_completed_npt
        source = args.npt_source or HERE.parent / 'scale_validation_NPT' / 'results' / 'long_from_nvt' / CASE
        try:
            config = from_completed_npt(source.expanduser(), folder, args.initial_temperature, args.temperature, nep)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(f'Cannot prepare initial state from completed NPT output: {exc}. '
                         'Provide --npt-source pointing to complete results with valid geometry.')
    elif reuse_equilibrated:
        from equilibrated_state import from_equilibrated_nvt
        source = args.equilibrated_from or HERE / 'results' / 'long' / CASE
        try:
            config = from_equilibrated_nvt(source.expanduser(), folder,
                                          args.initial_temperature, args.temperature, nep)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(f'Cannot prepare equilibrated initial state: {exc}. '
                         'Provide --equilibrated-from pointing to an accepted NVT run, '
                         'or explicitly use --fresh-lattice for the original stress fixture.')
    else:
        config = make(CASE, folder, args.initial_temperature)
        config['initial_state'] = 'unrelaxed_lattice'
    shutil.copy2(nep, folder / 'nep.txt')
    ensemble_line = f'ensemble {ensemble} {args.temperature:.16g} {args.temperature:.16g} {args.coupling:.16g}'
    if is_npt:
        ensemble_line += f' {args.pressure:.16g} {args.pressure_coupling:.16g}'
    ensemble_line += ' 0\n'
    (folder / 'run.in').write_text(
        'potential nep.txt\n'
        f'{ensemble_line}'
        f'time_step {args.dt:.16g}\n'
        f'dump_thermo {thermo_interval}\n'
        f'dump_xyz {xyz_interval} trajectory.xyz precision double mass velocity force potential group_labels\n'
        f'run {steps}\n', encoding='utf-8')
    config.update({
        'ensemble': ensemble, 'target_temperature_kelvin': args.temperature,
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
                                         'scale_common.py', 'static_compare.py', 'analyze.py', 'run_tests.py', 'nve_initial_state.py', 'equilibrated_state.py')},
    })
    if is_npt:
        config.update(target_pressure_GPa=args.pressure,
                      pressure_coupling_steps=args.pressure_coupling,
                      pressure_tau_fs=args.dt * args.pressure_coupling,
                      pressure_tolerance_GPa=args.pressure_tolerance)
        config['analysis_sha256'].update({f'NPT/{p.name}': hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in suite_dir.glob('*.py')})
    (folder / 'manifest.json').write_text(json.dumps(config, indent=2) + '\n')
    for name in ('trajectory.xyz', 'thermo.out', 'gpumd.log', 'validation.json',
                 'summary.json', 'nvt_trace.csv', 'temperature_trace.csv', 'npt_trace.csv', 'kinetic_partitions.csv',
                 'nvt_diagnostics.png', 'production_temperature.png', 'production_temperature.pdf',
                 'npt_diagnostics.png', 'npt_diagnostics.pdf',
                 'production_fluctuations.png', 'production_fluctuations.pdf',
                 'volume_history.png', 'volume_history.pdf'):
        (folder / name).unlink(missing_ok=True)
    shutil.rmtree(folder / 'static_reference', ignore_errors=True)
    print(f'Case {CASE}: {config["atoms"]} beads, 300 bodies, 2700 free particles', flush=True)
    print(f'{profile}: dt={args.dt:g} fs, {steps} steps = {steps * args.dt / 1000:g} ps; '
          f'target={args.temperature:g} K, tau={config["tau_fs"]:g} fs', flush=True)
    print(f'One continuous run; discard first {config["equilibration_steps"]} steps for statistics.', flush=True)
    if is_npt:
        print(f'Target pressure={args.pressure:g} GPa, piston tau={config["pressure_tau_fs"]:g} fs', flush=True)
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
            errors.append(f'{ensemble} diagnostics failed; inspect validation.json')
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
    acceptance = f'long_{ensemble.removeprefix("rigid_")}' if args.long else 'smoke_only'
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
    print('PASS: ' + (f'long {ensemble} engineering criteria' if args.long else
                      'smoke checks only; equilibrium/statistical acceptance NOT established'))


if __name__ == '__main__':
    main()
