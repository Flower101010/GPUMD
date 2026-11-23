"""Analytic linear rotor and mixed free-particle driver tests."""
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from test_rigid_nvt_run_in import KB, TIME_UNIT_FS, close, frames, thermo_rows


MASSES = (15, 17, 18)
LENGTH = 1.45
CENTER = (29.9, 10, 10)
COM_VELOCITY = (.012, -.004, .007)
AXIS = (1/3, 2/3, 2/3)
ANGULAR_RATE = .025
# Unit angular momentum is (2,-1,0)/sqrt(5), perpendicular to AXIS.
TANGENT = tuple(v / (3*math.sqrt(5)) for v in (-2, -4, 5))
DT = .25
STEPS = 80


def write_case(folder, ensemble, interacting, coincident, dt, steps):
    model = ['3', 'Lattice="30 0 0 0 30 0 0 0 30" pbc="T T T" '
             'Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1']
    for i, offset in enumerate((-LENGTH*MASSES[1]/32, LENGTH*MASSES[0]/32)):
        positions = [(x + (0 if coincident else offset*u)) % 30 for x, u in zip(CENTER, AXIS)]
        # An axial relative velocity is intentionally present and must be projected away.
        velocities = [v + offset*(ANGULAR_RATE*t + .01*u)
                      for v, t, u in zip(COM_VELOCITY, TANGENT, AXIS)]
        model.append('C ' + ' '.join(map(str, (*positions, MASSES[i], *velocities, 0))))
    model.append('C 2.5 10.4 10.3 18 -0.005 0.01 0.004 1')
    (folder/'model.xyz').write_text('\n'.join(model) + '\n')
    (folder/'lj.txt').write_text(f'lj 1 C\n{.006 if interacting else 0} 1.3 4.0\n')
    (folder/'run.in').write_text(
        f'potential lj.txt\ntime_step {dt}\nensemble {ensemble}\n'
        'dump_thermo 1\ndump_xyz 1 trajectory.xyz precision double mass velocity force\n'
        f'run {steps}\n')


def check(folder, mode, steps):
    xyz = folder/'trajectory.xyz'
    trajectory, thermo = frames(xyz), thermo_rows(folder/'thermo.out')
    headers = xyz.read_text().splitlines()[1::5]
    assert len(trajectory) == len(thermo) == len(headers) == steps
    initial_energy = thermo[0][1] + thermo[0][2]
    for step, (frame, row, header) in enumerate(zip(trajectory, thermo, headers), 1):
        box = [row[9], row[13], row[17]]
        a, b, free = frame
        bond = [y-x for x, y in zip(a['pos'], b['pos'])]
        bond = [x-L*round(x/L) for x, L in zip(bond, box)]
        relative_vel = [y-x for x, y in zip(a['vel'], b['vel'])]
        close(sum(v*v for v in bond), LENGTH**2, 2e-10)
        close(sum(r*v for r, v in zip(bond, relative_vel)), 0, 2e-10)
        kinetic = sum(.5*m*sum((v*TIME_UNIT_FS)**2 for v in atom['vel'])
                      for m, atom in zip(MASSES, frame))
        close(row[1], kinetic)
        close(row[0], 2*kinetic/(8*KB))
        if mode == 'nve':
            phase = ANGULAR_RATE*step*DT
            for r, u, t in zip(bond, AXIS, TANGENT):
                close(r, LENGTH*(math.cos(phase)*u + math.sin(phase)*t), 2e-10)
            for actual, start, velocity in zip(free['pos'], (2.5, 10.4, 10.3), (-.005, .01, .004)):
                close(actual, start+step*DT*velocity, 2e-10)
            close(kinetic, initial_energy, 2e-10)
        if mode == 'npt':
            # Independently reconstruct the mechanical pressure after cancelling
            # dimer rotational kinetic pressure with the constraint virial.
            cvel = [(15*x+17*y)/32 for x, y in zip(a['vel'], b['vel'])]
            kinetic_cm = sum(m*sum((v*TIME_UNIT_FS)**2 for v in vel)
                             for m, vel in ((32, cvel), (18, free['vel'])))
            virial = list(map(float, re.search(r'virial="([^"]+)"', header)[1].split()))
            moment = sum(r*(15*y-17*x)/32 for r, x, y in zip(bond, a['forces'], b['forces']))
            pressure = (kinetic_cm + virial[0] + virial[4] + virial[8] - moment)
            close(sum(row[3:6])/3, pressure/(3*math.prod(box))*160.2177, 2e-9)
    if mode.startswith('forced_nve'):
        assert max(abs(f) for frame in trajectory for atom in frame for f in atom['forces']) > 1e-5
        energy = [row[1] + row[2] for row in thermo]
        return max(energy) - min(energy)
    if mode == 'npt':
        assert abs(thermo[-1][9]-30) > 1e-6
        assert max(abs(f) for frame in trajectory for atom in frame for f in atom['forces']) > 1e-5


def main():
    binary, probe = [Path(arg).resolve() for arg in sys.argv[1:]]
    result = subprocess.run([str(probe), '--probe'], capture_output=True, text=True, check=True)
    if result.stdout.startswith('SKIP:'):
        print('SKIP: no accessible GPU for rigid dimer integration test.')
        return
    cases = (
        ('nve', 'rigid_nve 0', False, False, None),
        ('forced_nve', 'rigid_nve 0', True, False, None),
        ('forced_nve_half', 'rigid_nve 0', True, False, None),
        ('nvt', 'rigid_nvt 298.15 298.15 100 0', True, False, None),
        ('npt', 'rigid_npt 298.15 298.15 100 0.002 500 0', True, False, None),
        ('coincident', 'rigid_nve 0', False, True, 'requires distinct positions'),
        ('lammps', 'rigid_npt_lammps 298.15 298.15 100 0.002 500 0', False, False,
         'use rigid_npt for dimers'),
    )
    with tempfile.TemporaryDirectory(prefix='gpumd-rigid-dimer-') as directory:
        energy_span = {}
        for mode, ensemble, interacting, coincident, error in cases:
            folder = Path(directory)/mode
            folder.mkdir()
            dt = {'forced_nve': 2.0, 'forced_nve_half': 1.0}.get(mode, DT)
            steps = round(STEPS*DT/dt)
            write_case(folder, ensemble, interacting, coincident, dt, steps)
            result = subprocess.run([str(binary)], cwd=folder, capture_output=True, text=True, timeout=30)
            log = result.stdout + result.stderr
            if error:
                assert result.returncode != 0 and error in log, (mode, log)
            else:
                assert result.returncode == 0, (mode, log)
                energy_span[mode] = check(folder, mode, steps)
        coarse = energy_span['forced_nve']
        fine = energy_span['forced_nve_half']
        assert 0 < fine < .45*coarse, (coarse, fine)
        print(f'Forced NVE energy spans at dt and dt/2: {coarse:.9g}, {fine:.9g} eV.')
    print('Rigid dimer NVE, NVT, NPT and input validation passed.')


if __name__ == '__main__':
    main()
