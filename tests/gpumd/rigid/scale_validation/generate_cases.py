"""Generate deterministic 3000-body and 2700-singleton + 300-body NEP test inputs."""
from pathlib import Path
import numpy as np
from scale_common import body_ids

NX, NY, NZ = 15, 20, 10  # Exactly 3000 lattice sites
SPACING_A = 4.3
SIDE_A = 1.5
SEED = 20261013


def rotation(rng):
    # Unit quaternion, converted to an orthogonal rotation matrix.
    w, x, y, z = rng.standard_normal(4)
    w, x, y, z = (np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z]))
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def make(case, output):
    rng = np.random.default_rng(SEED + (0 if case == 'all_rigid_3000' else 1))
    lattice = np.array([NX, NY, NZ], float) * SPACING_A
    base_triangle = np.array([
        [-SIDE_A / 2, -SIDE_A/(2*np.sqrt(3)), 0.0],
        [ SIDE_A / 2, -SIDE_A/(2*np.sqrt(3)), 0.0],
        [0, SIDE_A/np.sqrt(3), 0.0],
    ])
    atoms = []
    body_count = 0
    free_count = 0
    indices = [(i,j,k) for i in range(NX) for j in range(NY) for k in range(NZ)]
    rng.shuffle(indices)  # no special spatial segregation of rigid and free sites

    # Determine rigid occupancy so that mixed case has exactly 300 rigid + 2700 singleton sites.
    rigid_sites = set(range(3000)) if case == 'all_rigid_3000' else set(range(0,3000,10))
    for index, (ix, iy, iz) in enumerate(indices):
        center = (np.array([ix+.5, iy+.5, iz+.5]) * SPACING_A
                  + rng.uniform(-0.05,0.05,size=3))
        V = rng.normal(0.0, 0.00045, size=3)
        if index in rigid_sites:
            rho = base_triangle @ rotation(rng).T
            omega = rng.normal(0.0, 0.0005, size=3)  # rad/fs
            vel = V[None,:] + np.cross(omega, rho)
            for i in range(3):
                atoms.append(['C', *(center+rho[i]), 12.0, *vel[i], body_count])
            body_count += 1
        else:
            # A SINGLE bead represents one water molecule; the C is a test-only NEP symbol.
            atoms.append(['C', *center, 18.0, *V, -1])
            free_count += 1

    # Convert singleton -1 placeholders to distinct nonnegative XYZ group labels.
    singleton_label = body_count
    for atom in atoms:
        if atom[-1] == -1:
            atom[-1] = singleton_label
            singleton_label += 1

    # Eliminate system-wide initial COM motion without breaking rigid constraints.
    mass = np.array([x[4] for x in atoms], float)
    vel = np.array([x[5:8] for x in atoms], float)
    vc = np.average(vel, axis=0, weights=mass)
    for atom in atoms:
        atom[5:8] = np.array(atom[5:8]) - vc

    pos = np.array([x[1:4] for x in atoms], float)
    assert np.all(pos > 0) and np.all(pos < lattice)
    groups = np.array([x[8] for x in atoms], int)
    body_arr, nfree = body_ids(groups)
    assert len(body_arr) == body_count and nfree == free_count
    assert (body_count, free_count) == ((3000,0) if case=='all_rigid_3000' else (300,2700))
    assert len(atoms) == 3*body_count + free_count
    # Check rigid-body starting velocities against v_rel dot separation=0.
    rr = pos[body_arr]
    vv = np.asarray([x[5:8] for x in atoms])[body_arr]
    for a,b in ((0,1),(0,2),(1,2)):
        assert np.max(np.abs(np.einsum('ij,ij->i',rr[:,a]-rr[:,b],vv[:,a]-vv[:,b]))) < 1e-14

    output.mkdir(parents=True, exist_ok=True)
    xyz = output / 'model.xyz'
    with xyz.open('w',encoding='utf-8') as fp:
        fp.write(f'{len(atoms)}\n')
        box = ' '.join(f'{x:.12g}' for x in (lattice[0],0,0,0,lattice[1],0,0,0,lattice[2]))
        fp.write(f'lattice="{box}" pbc="T T T" properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1\n')
        for a in atoms:
            fp.write(f'{a[0]} {a[1]:.16g} {a[2]:.16g} {a[3]:.16g} {a[4]:.16g} '
                     f'{a[5]:.16g} {a[6]:.16g} {a[7]:.16g} {int(a[8])}\n')
    return {'case':case, 'atoms':len(atoms), 'bodies':body_count, 'free_beads':free_count,
            'groups':body_count+free_count, 'thermal_dof':6*body_count+3*free_count,
            'box_lengths_angstrom':lattice.tolist(),'seed':SEED,
            'bead_symbol':'C', 'rigid_bead_mass_amu':12.0,'free_water_bead_mass_amu':18.0}
