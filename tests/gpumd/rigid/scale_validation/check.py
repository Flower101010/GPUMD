"""Validate large NEP rigid-NVE runs: geometry, P, K, T, E, finite stress.

Important: for periodic systems, total angular momentum is not universally
conserved in the same naive wrapped-coordinate representation; we deliberately
avoid asserting that. The small non-PBC cases already verify angular momentum.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scale_common import (
    KB_GPUMD_EV_PER_K, body_ids, iter_xyz, mic,
    read_initial_xyz, total_ke,
)


def verify(folder, *, dt_fs, steps, thermo_interval, xyz_interval, strict_energy):
    folder = Path(folder)
    lattice, species, r0, m0, v0, groups = read_initial_xyz(folder / 'model.xyz')
    lengths = np.diag(lattice)
    if not np.allclose(lattice, np.diag(lengths)):
        raise ValueError('This test assumes orthorhombic boxes')
    if not all(s == 'C' for s in species):
        raise AssertionError('Expected synthetic C-only NEP bead labels')
    ids, n_free = body_ids(groups)
    n_bodies = len(ids)
    natoms = len(m0)
    dof = 6 * n_bodies + 3 * n_free
    if natoms == 9000:
        assert n_bodies == 3000 and n_free == 0
    elif natoms == 3600:
        assert n_bodies == 300 and n_free == 2700
    else:
        raise AssertionError(f'Unexpected number of beads: {natoms}')
    if np.unique(groups).size != n_bodies + n_free:
        raise AssertionError('Group labels not unique across bodies/singletons')
    if groups.min() < 0 or groups.max() >= natoms:
        raise AssertionError('Invalid group labels')
    v_rms = np.sqrt(np.sum(m0[:,None] * v0*v0)/np.sum(m0))
    p0 = (m0[:,None]*v0).sum(axis=0)
    momentum_scale = max(float(np.sum(m0) * v_rms), 1e-12)

    ref_dists = []
    for a, b in ((0,1),(0,2),(1,2)):
        ref_dists.append(np.linalg.norm(mic(r0[ids[:,a]] - r0[ids[:,b]], lengths), axis=1))

    data = np.atleast_2d(np.loadtxt(folder/'thermo.out'))
    assert data.shape[1] >= 9, 'Unexpected thermo.out column count'
    n_thermo_expected = steps // thermo_interval
    if len(data) != n_thermo_expected:
        raise AssertionError(f'thermo expected {n_thermo_expected} rows, got {len(data)}')
    if not np.all(np.isfinite(data)):
        raise AssertionError('NaN or Inf in thermo.out')
    t_thermo_ps = (np.arange(1,len(data)+1)*dt_fs*thermo_interval)/1000.0
    e = data[:,1] + data[:,2]
    e_per = (e - e[0]) / natoms
    energy_excursion = float(np.max(np.abs(e_per)))
    endpoint_shift = float(e_per[-1])
    slope = float(np.polyfit(t_thermo_ps, e_per, 1)[0]) if len(e)>2 else float('nan')
    if not np.all(np.isfinite(data[:,3:9])):
        raise AssertionError('Nonfinite stress in thermo.out')
    # Save the entire NVE energy trace, not merely the endpoint difference.
    np.savetxt(
        folder / 'energy_trace.csv',
        np.column_stack((t_thermo_ps, data[:,1], data[:,2], e, e_per)),
        delimiter=',',
        header='time_ps,kinetic_energy_eV,potential_energy_eV,total_energy_eV,delta_total_energy_per_bead_eV',
        comments='', fmt='%.14e',
    )
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_thermo_ps, e_per)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Total energy change / bead (eV)')
        ax.set_title(f'{folder.name}: NVE energy stability')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(folder / 'energy_drift.png', dpi=160)
        plt.close(fig)
    except ImportError:
        pass  # matplotlib is optional; CSV and numerical checks still work.

    max_dist_error = 0.0
    max_vel_constraint = 0.0
    max_ke_rel = 0.0
    max_temp_rel = 0.0
    max_pe_per_bead = 0.0
    max_p_rel = 0.0
    max_force = 0.0
    frames_count = 0
    snapshot_times = []
    for f in iter_xyz(folder/'trajectory.xyz'):
        frames_count += 1
        if len(f.pos) != natoms or not np.array_equal(f.group, groups):
            raise AssertionError('Wrong bead count or group labels in dump_xyz')
        if not np.allclose(f.mass, m0, atol=1e-12, rtol=1e-12):
            raise AssertionError('Masses differ between model and trajectory')
        if not np.allclose(f.lattice, lattice, atol=1e-8, rtol=0):
            raise AssertionError('Box changed in fixed-box NVE')
        if not all(f.pbc):
            raise AssertionError('Unexpected nonperiodic axis')
        for arr in (f.pos, f.vel, f.forces, f.energy_atom):
            if not np.all(np.isfinite(arr)):
                raise AssertionError(f'Nonfinite trajectory fields at t={f.time_fs} fs')
        if np.any(f.pos < -1e-6) or np.any(f.pos >= lengths+1e-6):
            raise AssertionError('Trajectory includes unwrapped positions unexpectedly')
        rr = f.pos[ids]
        vv = f.vel[ids]
        for k,(a,b) in enumerate(((0,1),(0,2),(1,2))):
            dr = mic(rr[:,a]-rr[:,b], lengths)
            dv = vv[:,a]-vv[:,b]
            d = np.linalg.norm(dr, axis=1)
            max_dist_error = max(max_dist_error, float(np.max(np.abs(d - ref_dists[k]))))
            max_vel_constraint = max(
                max_vel_constraint,
                float(np.max(np.abs(np.einsum('ij,ij->i',dr,dv)))),
            )
        p = (m0[:,None] * f.vel).sum(axis=0)
        max_p_rel = max(max_p_rel, float(np.linalg.norm(p-p0)/momentum_scale))
        max_force = max(max_force, float(np.max(np.linalg.norm(f.forces,axis=1))))
        # Compare with global values at the same time.
        k = round(f.time_fs/(dt_fs*thermo_interval))-1
        if k < 0 or k >= len(data):
            raise AssertionError(f'Frame at t={f.time_fs} fs has no matching thermo entry')
        expected_time = (k+1)*dt_fs*thermo_interval
        if abs(f.time_fs - expected_time)>1e-4:
            raise AssertionError('XYZ and thermo time grids do not align')
        kinetic = total_ke(m0, f.vel)
        U = float(np.sum(f.energy_atom))
        max_ke_rel = max(max_ke_rel, abs(kinetic-data[k,1])/max(1.0,abs(kinetic)))
        T = 2*kinetic/(KB_GPUMD_EV_PER_K*dof)
        max_temp_rel = max(max_temp_rel, abs(T-data[k,0])/max(1.0,abs(T)))
        max_pe_per_bead = max(max_pe_per_bead, abs(U-data[k,2])/natoms)
        snapshot_times.append(f.time_fs)

    xyz_expected = steps // xyz_interval
    if frames_count != xyz_expected:
        raise AssertionError(f'Expected {xyz_expected} XYZ frames, got {frames_count}')
    if not np.allclose(snapshot_times, np.arange(1,xyz_expected+1)*dt_fs*xyz_interval, atol=1e-4):
        raise AssertionError('XYZ output time intervals are incorrect')

    thresholds = {
        'rigid_distance_angstrom': 2e-6,
        'velocity_constraint_angstrom2_per_fs': 1e-7,
        'momentum_relative': 2e-4,
        'kinetic_energy_relative': 1e-4,
        'temperature_relative': 1e-4,
        'potential_energy_difference_ev_per_bead': 5e-5,
        'max_energy_excursion_ev_per_bead': float(os.getenv('MAX_E_EXCURSION_EV_BEAD', '0.002')),
        'max_secular_slope_ev_per_bead_ps': float(os.getenv('MAX_E_SLOPE_EV_BEAD_PS','0.0002')),
    }
    checks = {
        'Rigid distance': max_dist_error < thresholds['rigid_distance_angstrom'],
        'Rigid velocity constraint': max_vel_constraint < thresholds['velocity_constraint_angstrom2_per_fs'],
        'Total linear momentum': max_p_rel < thresholds['momentum_relative'],
        'Kinetic energy consistency': max_ke_rel < thresholds['kinetic_energy_relative'],
        'Temperature / thermal DOF': max_temp_rel < thresholds['temperature_relative'],
        'Potential energy consistency': max_pe_per_bead < thresholds['potential_energy_difference_ev_per_bead'],
        'Nonzero NEP forces': max_force > 1e-5,
    }
    if strict_energy:
        checks['Energy bounded excursion'] = energy_excursion < thresholds['max_energy_excursion_ev_per_bead']
        checks['No secular energy drift'] = abs(slope) < thresholds['max_secular_slope_ev_per_bead_ps']

    # NumPy comparisons may return numpy.bool_, which json.dump cannot encode.
    # Normalize all check results before emitting the machine-readable report.
    checks = {name: bool(ok) for name, ok in checks.items()}

    print(f'\n=== {folder.name}: NEP LARGE SYSTEM CHECK ===')
    print(f'Atoms={natoms}, bodies={n_bodies}, free={n_free}, groups={n_bodies+n_free}, DOF={dof}')
    print(f'Frames={frames_count}, thermo rows={len(data)}, simulated {steps*dt_fs/1000:.3f} ps')
    print(f'Max rigid distance err (Å):      {max_dist_error:.8e}')
    print(f'Max velocity constraint err:     {max_vel_constraint:.8e}')
    print(f'Max total momentum rel err:      {max_p_rel:.8e}')
    print(f'Max thermo vs bead KE rel err:   {max_ke_rel:.8e}')
    print(f'Max temperature rel err:         {max_temp_rel:.8e}')
    print(f'Max thermo vs per-bead U (eV/N): {max_pe_per_bead:.8e}')
    print(f'Max individual NEP force (eV/Å): {max_force:.8e}')
    print(f'Max |E-E(first)| / bead (eV):    {energy_excursion:.8e}')
    print(f'E(last)-E(first) / bead (eV):   {endpoint_shift:.8e}')
    print(f'Linear energy slope (eV/bead/ps): {slope:+.8e}')
    print(f'Peak-to-peak E / bead (eV):     {np.ptp(e)/natoms:.8e}')
    print('\nAcceptance:')
    for name, ok in checks.items():
        print(f'  {name:32s}: {"PASS" if ok else "FAIL"}')
    if not strict_energy:
        print('  Energy drift thresholds: informational only in smoke mode')
    report = {
        'case':folder.name,'n_atoms':natoms,'n_bodies':n_bodies,'n_free':n_free,
        'thermal_dof':dof,'simulated_ps':steps*dt_fs/1000,
        'max_rigid_distance_error_angstrom':max_dist_error,
        'max_velocity_constraint_error':max_vel_constraint,
        'max_momentum_relative_error':max_p_rel,
        'max_thermo_ke_relative_error':max_ke_rel,
        'max_temperature_relative_error':max_temp_rel,
        'max_potential_difference_ev_per_bead':max_pe_per_bead,
        'max_force_ev_per_angstrom':max_force,
        'energy_excursion_ev_per_bead':energy_excursion,
        'energy_endpoint_delta_ev_per_bead':endpoint_shift,
        'energy_slope_ev_per_bead_per_ps':slope,
        'thresholds':thresholds,'checks':checks,
        'passed':all(checks.values()),
    }
    (folder/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return all(checks.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder',type=Path)
    ap.add_argument('--dt',type=float,required=True)
    ap.add_argument('--steps',type=int,required=True)
    ap.add_argument('--thermo-interval',type=int,required=True)
    ap.add_argument('--xyz-interval',type=int,required=True)
    ap.add_argument('--strict-energy',action='store_true')
    args = ap.parse_args()
    ok = verify(args.folder,dt_fs=args.dt, steps=args.steps,
                thermo_interval=args.thermo_interval,
                xyz_interval=args.xyz_interval, strict_energy=args.strict_energy)
    if not ok:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
