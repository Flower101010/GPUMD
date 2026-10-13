import re
import numpy as np

L = 20.0
DT = 0.5
INTERVAL = 10
CONV = 103.64269652680505  # amu A^2/fs^2 -> eV
KB = 8.617333262145e-5      # eV/K
MASS = 12.0
R0 = np.array([18.5, 10.0, 10.0])
V0 = np.array([0.003, 0.001, 0.0])
OMEGA = 0.001
RHO0 = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 2., 0.]])

def mic(d):
    return d - L * np.rint(d / L)

def read_xyz(path):
    positions, velocities, forces, masses, labels, times = [], [], [], [], [], []
    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line)
            assert n == 3
            header = file.readline()
            t_match = re.search(r'\bTime=([-+\d.eE]+)', header)
            p_match = re.search(r'\bProperties=([^\s]+)', header)
            assert t_match and p_match, 'Missing Time or Properties in XYZ'
            times.append(float(t_match.group(1)))
            schema = p_match.group(1).split(':')
            cols, offset = {}, 0
            for k in range(0, len(schema), 3):
                count = int(schema[k + 2])
                cols[schema[k]] = slice(offset, offset + count)
                offset += count
            assert all(k in cols for k in ('pos', 'vel', 'forces', 'mass', 'group'))
            x, v, f, m, g = [], [], [], [], []
            for _ in range(n):
                row = file.readline().split()
                assert len(row) == offset
                x.append([float(z) for z in row[cols['pos']]])
                v.append([float(z) for z in row[cols['vel']]])
                f.append([float(z) for z in row[cols['forces']]])
                m.append(float(row[cols['mass']][0]))
                g.append(int(row[cols['group']][0]))
            positions.append(x)
            velocities.append(v)
            forces.append(f)
            masses.append(m)
            labels.append(g)
    return tuple(np.asarray(a) for a in (positions, velocities, forces, masses, labels, times))

r, v, force, mass, groups, t = read_xyz('trajectory.xyz')
assert len(t) == 100
assert np.allclose(t, DT * INTERVAL * np.arange(1, 101), rtol=0, atol=1e-6)
assert all(np.all(np.isfinite(x)) for x in (r, v, force))
assert np.all(mass == MASS) and np.all(groups == 0)

ang = OMEGA * t
c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
rotated = np.empty((len(t), 3, 3))
rotated[:, :, 0] = c * RHO0[None, :, 0] - s * RHO0[None, :, 1]
rotated[:, :, 1] = s * RHO0[None, :, 0] + c * RHO0[None, :, 1]
rotated[:, :, 2] = RHO0[None, :, 2]

com_ref = R0 + t[:, None] * V0
r_ref = com_ref[:, None, :] + rotated
omega_vec = np.array([0.0, 0.0, OMEGA])
v_ref = V0 + np.cross(omega_vec, rotated)

position_error = np.max(np.linalg.norm(mic(r-r_ref), axis=2))
velocity_error = np.max(np.linalg.norm(v-v_ref, axis=2))
com_error = np.max(np.linalg.norm(np.mean(mic(r-com_ref[:, None, :]), axis=1), axis=1))
distance_error = max(np.max(np.abs(np.linalg.norm(mic(r[:,i] - r[:,j]), axis=1) - np.linalg.norm(RHO0[i]-RHO0[j]))) for i,j in ((0,1),(0,2),(1,2)))
force_error = np.max(np.linalg.norm(force, axis=2))
# Detect that a coordinate genuinely traversed an image boundary.
crossed = bool(np.any(r_ref[:,:,0] >= L))
wrapped_output = bool(np.any(np.abs(r[:,:,0] - r_ref[:,:,0]) > L / 2))

K0 = 0.5 * CONV * MASS * np.sum((V0+np.cross(omega_vec,RHO0))**2)
K_xyz = 0.5 * CONV * MASS * np.sum(v*v, axis=(1,2))
thermo = np.atleast_2d(np.loadtxt('thermo.out'))
assert len(thermo) == len(t)
T_out, K_out, U_out = thermo[:,0], thermo[:,1], thermo[:,2]
kinetic_error = np.max(np.abs(K_xyz-K0))
thermo_kinetic_error = np.max(np.abs(K_out-K_xyz))
temperature_error = np.max(np.abs(T_out - 2*K_xyz/(6*KB)))
potential_error = np.max(np.abs(U_out))

checks = {
    'Boundary actually crossed': crossed,
    'Wrapped coordinates output': wrapped_output,
    'Analytical position': position_error < 1e-7,
    'Analytical velocity': velocity_error < 1e-9,
    'Rigid pair distances (MIC)': distance_error < 1e-8,
    'COM motion': com_error < 1e-8,
    'Zero bead forces': force_error < 1e-10,
    'Kinetic energy': kinetic_error < 1e-6,
    'Thermo kinetic energy': thermo_kinetic_error < 1e-6,
    'Temperature / six DOFs': temperature_error < 1e-3,
    'Zero potential energy': potential_error < 1e-10,
}
print('=== PBC Crossing Test ===')
for k,x in (('Frames',len(t)),('Coordinate error (A)',position_error),('Velocity error (A/fs)',velocity_error),('Pair-distance error (A)',distance_error),('COM error (A)',com_error),('Kinetic-energy error (eV)',kinetic_error),('Thermo KE mismatch (eV)',thermo_kinetic_error),('Temperature error (K)',temperature_error),('Max force (eV/A)',force_error)):
    print(f'{k:30s}: {x}')
print(f'Expected K0: {K0:.12f} eV; T0: {2*K0/(6*KB):.8f} K')
for name, ok in checks.items():
    print(f'{name:30s}: {"PASS" if ok else "FAIL"}')
if not all(checks.values()):
    raise SystemExit(1)
print('All tests passed.')
