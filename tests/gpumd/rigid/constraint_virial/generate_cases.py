#!/usr/bin/env python3
"""Generate deterministic inputs for rigid constraint-virial regression tests."""
from pathlib import Path
import math
import numpy as np

HERE=Path(__file__).resolve().parent

def emit(name,coords,vel,dt,steps,interval):
    dest=HERE/name
    dest.mkdir(parents=True,exist_ok=True)
    coords=np.asarray(coords,float)
    vel=np.asarray(vel,float)
    assert coords.shape==(3,3) and vel.shape==(3,3)
    assert np.all(np.isfinite(coords))
    head=['3', 'lattice="40 0 0 0 40 0 0 0 40" pbc="T T T" properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1']
    for r,v in zip(coords,vel):
        head.append('C '+' '.join(f'{x:.17g}' for x in r)+' 12 '+' '.join(f'{x:.17g}' for x in v)+' 0')
    (dest/'model.xyz').write_text('\n'.join(head)+'\n')
    (dest/'zero_lj.txt').write_text('lj 1 C\n0.0 1.0 4.0\n')
    (dest/'run.in').write_text(f'''potential zero_lj.txt\nensemble rigid_nve 0\ntime_step {dt:g}\ndump_thermo {interval}\ndump_xyz {interval} trajectory.xyz precision double mass velocity force virial group_labels\nrun {steps}\n''')

center=np.array([20.0,20.0,20.0])
angles=np.array([0,2*math.pi/3,4*math.pi/3])
rel=np.stack([2*np.cos(angles),2*np.sin(angles),np.zeros(3)],axis=-1)
spin=np.array([0.,0.,0.005])
vr=np.cross(np.broadcast_to(spin,rel.shape),rel)
emit('spin_only',center+rel,vr,0.25,400,4)
V=np.array([0.03,-0.02,0.01])
emit('spin_translate',center+rel,vr+V,0.25,400,4)
# Unequal principal moments: I=(72,24,96) amu A^2 and nonprincipal rotation.
rel=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,2.,0.]])
w=np.array([0.001,0.002,0.003])
emit('asymmetric_rotation',center+rel,np.cross(np.broadcast_to(w,rel.shape),rel),0.1,1000,10)
print('Generated 3 model.xyz + run.in + zero_lj.txt cases')
