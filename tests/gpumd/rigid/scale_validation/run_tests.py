"""Orchestrate synthetic NEP large-system rigid NVE correctness/stability tests."""
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
CASES = ['all_rigid_3000', 'mixed_2700_300']


def main():
    parser=argparse.ArgumentParser(description='Final-scale GPUMD NEP rigid tests')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke',action='store_true',help='2000 integration steps per case (100 fs at default dt)')
    mode.add_argument('--long',action='store_true',help='200000 steps per case (10 ps at default dt; DEFAULT)')
    parser.add_argument('--steps',type=int,help='Override number of steps; divisible by 200')
    parser.add_argument('--dt',type=float,default=0.05,help='Fixed timestep in fs (default 0.05)')
    parser.add_argument('--case',choices=['both']+CASES,default='both')
    parser.add_argument('--skip-static',action='store_true',help='Skip first/last frame static NEP cross-check')
    parser.add_argument('--nep',type=Path,default=ROOT/'potentials/nep/C_2022_NEP4.txt')
    parser.add_argument('--gpumd',type=Path,default=Path(os.environ.get('GPUMD_BIN',ROOT/'build/gpumd')))
    args=parser.parse_args()

    steps=args.steps if args.steps is not None else (2000 if args.smoke else 200000)
    if steps<200 or steps % 200 !=0:
        parser.error('--steps must be >= 200 and divisible by 200')
    if not (math.isfinite(args.dt) and args.dt>0):
        parser.error('--dt must be positive and finite')
    thermo_interval=steps//200
    xyz_interval=steps//20
    profile=('smoke' if args.smoke and args.steps is None else
             'long' if args.steps is None else f'steps_{steps}_dt_{args.dt:g}'.replace('.','p'))
    gpumd=args.gpumd.expanduser().resolve()
    nep=args.nep.expanduser().resolve()
    if not (gpumd.is_file() and os.access(gpumd, os.X_OK)):
        parser.error(f'GPUMD executable missing/not executable: {gpumd}')
    if not nep.is_file():
        parser.error(f'NEP potential missing: {nep}')
    line=nep.open(encoding='utf-8').readline().split()
    if line[:3]!=['nep4','1','C']:
        parser.error('The synthetic test uses only C symbols; provide a standard nep4 1 C potential')
    sha=hashlib.sha256(nep.read_bytes()).hexdigest()
    selected=CASES if args.case=='both' else [args.case]
    errors=[]
    for case in selected:
        folder=HERE/'results'/profile/case
        folder.mkdir(parents=True,exist_ok=True)
        print(f'\n--- Preparing {case} ({profile}) ---', flush=True)
        config=make(case,folder)
        shutil.copy2(nep,folder/'nep.txt')
        (folder/'run.in').write_text(
            'potential nep.txt\n'
            'ensemble rigid_nve 0\n'
            f'time_step {args.dt:.12g}\n'
            f'dump_thermo {thermo_interval}\n'
            f'dump_xyz {xyz_interval} trajectory.xyz precision double mass velocity force potential group_labels\n'
            f'run {steps}\n',encoding='utf-8')
        config.update({'dt_fs':args.dt,'steps':steps,
                       'thermo_interval':thermo_interval,'xyz_interval':xyz_interval,
                       'nep_source':str(nep),'nep_sha256':sha,'gpumd':str(gpumd),
                       'profile':profile})
        (folder/'manifest.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
        for generated in ('trajectory.xyz','thermo.out','gpumd.log','validation.json'):
            (folder/generated).unlink(missing_ok=True)
        shutil.rmtree(folder/'static_reference',ignore_errors=True)
        print(f'  {config["atoms"]} beads = {config["bodies"]} rigid bodies + {config["free_beads"]} free beads')
        print(f'  dt={args.dt:g} fs, steps={steps}, total={args.dt*steps/1000:g} ps')
        print('  Running GPUMD; full stdout -> '+str(folder/'gpumd.log'),flush=True)
        tic=time.perf_counter()
        with (folder/'gpumd.log').open('w') as log:
            proc=subprocess.run([str(gpumd)],cwd=folder,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.perf_counter()-tic
        print(f'  Wall time: {elapsed:.2f} s ({steps/elapsed:.1f} MD steps/s)' if elapsed>0 else '  Finished')
        (folder/'performance.json').write_text(json.dumps({
            'elapsed_seconds':elapsed,'steps_per_second':steps/elapsed if elapsed else None,
            'beads':config['atoms'],'steps':steps,'total_simulated_ps':steps*args.dt/1000,
        },indent=2)+'\n')
        if proc.returncode:
            loglines=(folder/'gpumd.log').read_text(errors='replace').splitlines()
            print('\n'.join(loglines[-35:]),file=sys.stderr)
            errors.append(f'{case}: GPUMD process exit {proc.returncode}')
            continue
        try:
            valid=verify(folder,dt_fs=args.dt,steps=steps,
                         thermo_interval=thermo_interval,xyz_interval=xyz_interval,
                         strict_energy=(not args.smoke and steps>=20000))
            if not valid:
                errors.append(f'{case}: diagnostics failed (inspect validation.json)')
            # Always attempt the static check even when the long-time energy criterion fails.
            if not args.skip_static:
                if not check_frames(folder,gpumd):
                    errors.append(f'{case}: static NEP reference failed')
        except Exception as exc:
            errors.append(f'{case}: {type(exc).__name__}: {exc}')
            print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)

    if errors:
        print('\n=== FINAL TEST: FAIL / NEEDS INVESTIGATION ===')
        for error in errors:
            print(' - '+error)
        raise SystemExit(1)
    print('\n=== ALL SELECTED LARGE-SCALE NEP TESTS PASSED ===')

if __name__ == '__main__':
    main()
