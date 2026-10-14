#!/usr/bin/env python3
"""Independent analytic virial/pressure checks for one free triangular rigid body.

Uses bead positions/velocities, rigid-body kinematics and zero external forces.
Does not read internal rigid-manager state or its constraint-virial buffers.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np

# GPUMD's numerical unit constants in src/utilities/common.cuh.
KINETIC_CONVERSION = 103.64269652680505  # amu (A/fs)^2 -> eV
PRESSURE_CONVERSION = 160.2177         # (eV/A^3) -> GPa
BOLTZMANN_GPUMD = 8.617343e-5        # eV/K, GPUMD constant
BOX_L = 40.0
VOLUME = BOX_L**3
DOF = 6


def model_data(path):
    with path.open() as f:
        n = int(f.readline())
        header = f.readline()
        rows = [f.readline().split() for _ in range(n)]
    assert n == 3 and len(rows) == n
    assert 'group:I:1' in header
    mass = np.array([float(row[4]) for row in rows])
    pos = np.array([[float(a) for a in row[1:4]] for row in rows])
    vel = np.array([[float(a) for a in row[5:8]] for row in rows])
    labels = np.array([int(row[8]) for row in rows])
    assert np.array_equal(labels,[0,0,0])
    assert np.all(mass > 0)
    return pos, vel, mass


def load_xyz(path):
    frames = []
    times = []
    with path.open() as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line.strip())
            header = f.readline()
            tm = re.search(r'\bTime=([0-9eE+.-]+)', header)
            pm = re.search(r'\bProperties=([^\s]+)', header)
            if not (tm and pm):
                raise ValueError('Missing Time/Properties in trajectory header')
            times.append(float(tm.group(1)))
            props=pm.group(1).split(':')
            columns={}
            offset=0
            for i in range(0,len(props),3):
                name=props[i]
                count=int(props[i+2])
                columns[name]=slice(offset,offset+count)
                offset+=count
            for required in ('pos','vel','mass','forces','virial','group'):
                if required not in columns:
                    raise ValueError(f'Missing XYZ property: {required}. Available: {list(columns)}')
            payload={key:[] for key in ('pos','vel','mass','forces','virial','group')}
            for _ in range(n):
                line=f.readline()
                if not line:
                    raise ValueError('Unexpected end of XYZ file')
                parts=line.split()
                if len(parts)!=offset:
                    raise ValueError(f'Expected {offset} XYZ columns, got {len(parts)}')
                for key in payload:
                    payload[key].append([float(x) for x in parts[columns[key]]])
            frames.append({key:np.array(value) for key,value in payload.items()})
    if not frames:
        raise ValueError('Empty trajectory')
    return times,frames


def inertia_tensor(rel,mass):
    I=np.zeros((3,3),dtype=float)
    for rho,m in zip(rel,mass):
        I+=m*(np.dot(rho,rho)*np.eye(3)-np.outer(rho,rho))
    return I


def expected_pressure_from_motion(x,v,mass):
    """Zero external force/torque, exact instantaneous free rigid-body kinematics.

    L = I omega, alpha = -I^-1 (omega x L)
    a_i = alpha x rho_i + omega x (omega x rho_i)
    F_constraint,i = mass_i * a_i
    W_constraint = sum rho_i outer F_constraint,i
    P = (sum m_i vi outer vi + W_constraint) / Volume
    """
    M=mass.sum()
    com=np.average(x,axis=0,weights=mass)
    vcom=np.average(v,axis=0,weights=mass)
    rel=x-com
    relv=v-vcom
    I=inertia_tensor(rel,mass)
    L=np.sum(mass[:,None]*np.cross(rel,relv),axis=0)
    omega=np.linalg.solve(I,L)
    alpha=np.linalg.solve(I,-np.cross(omega,L))
    accel=np.cross(np.broadcast_to(alpha,rel.shape),rel)+np.cross(
        np.broadcast_to(omega,rel.shape),
        np.cross(np.broadcast_to(omega,rel.shape),rel)
    )
    # Constraint virial in eV (not per-atom interaction virials).
    Wc=KINETIC_CONVERSION*np.einsum('ia,ib,i->ab',rel,accel,mass)
    Ktensor=KINETIC_CONVERSION*np.einsum('ia,ib,i->ab',v,v,mass)
    pressure=(Ktensor+Wc)*PRESSURE_CONVERSION/VOLUME
    return pressure,Wc,Ktensor,L,omega,com,vcom


def thermo_pressure(row):
    # GPUMD thermo columns: T, KE, PE, xx, yy, zz, yz, xz, xy, cell...
    xx,yy,zz,yz,xz,xy=row[3:9]
    return np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])


def check_case(directory,kind):
    x0,v0,mass=model_data(directory/'model.xyz')
    times,frames=load_xyz(directory/'trajectory.xyz')
    thermo=np.atleast_2d(np.loadtxt(directory/'thermo.out'))
    assert len(frames)==100 and len(thermo)==100, (len(frames),len(thermo))
    assert np.allclose(times,np.arange(1,101),atol=1e-6)
    assert np.all(np.isfinite(thermo))
    P0=np.sum(mass[:,None]*v0,axis=0)
    xc0=np.average(x0,axis=0,weights=mass)
    L0=np.sum(mass[:,None]*np.cross(x0-xc0,v0-np.average(v0,axis=0,weights=mass)),axis=0)
    lengths0=np.array([np.linalg.norm(x0[i]-x0[j]) for i,j in ((0,1),(0,2),(1,2))])
    mass_output_error=0.0
    label_error=0
    pos_err=0.0
    vmomentum_error=0.0
    L_error=0.0
    pressure_error=0.0
    virial_error=0.0
    forced_pressure_error=0.0
    virial_magnitude=0.0
    virial_interaction_magnitude=0.0
    interaction_force_max=0.0
    K_output_error=0.0
    T_output_error=0.0
    offdiag_p_err=0.0

    for i,(frame,row) in enumerate(zip(frames,thermo)):
        x=frame['pos']; v=frame['vel']; mout=frame['mass'][:,0]
        f=frame['forces']; winteraction=frame['virial']; gout=frame['group'][:,0]
        if x.shape!=(3,3) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)):
            raise AssertionError('Wrong frame shape or non-finite coordinates/velocities')
        if not np.all((x>1.0)&(x<39.0)):
            raise AssertionError('This isolated test intentionally avoids wrapping; bad input/trajectory')
        mass_output_error=max(mass_output_error,float(np.max(np.abs(mout-mass))))
        label_error+=int(np.count_nonzero(gout))
        ds=np.array([np.linalg.norm(x[a]-x[b]) for a,b in ((0,1),(0,2),(1,2))])
        pos_err=max(pos_err,float(np.max(np.abs(ds-lengths0))))
        vmomentum_error=max(vmomentum_error,float(np.linalg.norm(np.sum(mass[:,None]*v,axis=0)-P0)))
        p_ref,Wc,Ktensor,L,omega,com,vcom=expected_pressure_from_motion(x,v,mass)
        L_error=max(L_error,float(np.linalg.norm(L-L0)))
        P_measured=thermo_pressure(row)
        pressure_error=max(pressure_error,float(np.max(np.abs(P_measured-p_ref))))
        # Independently infer the constraint-only virial from printed stress.
        W_inferred=P_measured * VOLUME/PRESSURE_CONVERSION-Ktensor
        virial_error=max(virial_error,float(np.max(np.abs(W_inferred-Wc))))
        virial_magnitude=max(virial_magnitude,float(np.max(np.abs(Wc))))
        # Additional closed-form prediction for an equilateral rotor:
        # internal rotational kinetic dyadic cancels constraint virial, so
        # P = M * Vcom outer Vcom / Volume. No Euler reconstruction needed.
        if kind in ('spin_only','spin_translate'):
            P_closed_form=(mass.sum()*KINETIC_CONVERSION*PRESSURE_CONVERSION/VOLUME)*np.outer(vcom,vcom)
            forced_pressure_error=max(forced_pressure_error,float(np.max(np.abs(P_measured-P_closed_form))))
        energy=0.5*KINETIC_CONVERSION*np.sum(mass[:,None]*v*v)
        T=2*energy/(DOF*BOLTZMANN_GPUMD)
        K_output_error=max(K_output_error,abs(float(row[1])-energy))
        T_output_error=max(T_output_error,abs(float(row[0])-T))
        interaction_force_max=max(interaction_force_max,float(np.max(np.abs(f))))
        virial_interaction_magnitude=max(virial_interaction_magnitude,float(np.max(np.abs(winteraction))))
        offdiag_p_err=max(offdiag_p_err,float(np.max(np.abs(P_measured - P_measured.T))))

    E=thermo[:,1]+thermo[:,2]
    energy_drift=float(np.max(np.abs(E-E[0])))
    U_magnitude=float(np.max(np.abs(thermo[:,2])))
    checks={
        'geometry and labels':pos_err<1e-8 and mass_output_error<1e-10 and label_error==0,
        'total linear/angular momentum':vmomentum_error<1e-8 and L_error<1e-8,
        'reported kinetic energy':K_output_error<5e-6,
        'reported temperature (DOF=6)':T_output_error<0.1,
        'no LJ force / per-atom virial':interaction_force_max<1e-10 and virial_interaction_magnitude<1e-10 and U_magnitude<1e-10,
        'nonzero constraint virial':virial_magnitude>0.01,
        'full global pressure tensor':pressure_error<2e-6,
        'inferred constraint virial':virial_error<8e-4,
        'zero-LJ NVE energy drift':energy_drift<3e-5,
    }
    if kind in ('spin_only','spin_translate'):
        checks['equilateral closed-form stress']=forced_pressure_error<2e-6

    print(f'\n=== {kind} ===')
    print(f'frames: {len(frames)}')
    print(f'max distance error (A):             {pos_err:.8e}')
    print(f'angular momentum error:            {L_error:.8e}')
    print(f'max |W_constraint| (eV):           {virial_magnitude:.8e}')
    print(f'max global pressure residual (GPa):{pressure_error:.8e}')
    print(f'max inferred Wc residual (eV):    {virial_error:.8e}')
    if kind in ('spin_only','spin_translate'):
        print(f'closed-form pressure err (GPa):   {forced_pressure_error:.8e}')
    print(f'max KE mismatch (eV):             {K_output_error:.8e}')
    print(f'max T mismatch (K):                {T_output_error:.8e}')
    print(f'max total energy change (eV):     {energy_drift:.8e}')
    for name,passed in checks.items():
        print(f'  {name:35s} {"PASS" if passed else "FAIL"}')
    if not all(checks.values()):
        raise SystemExit(1)
    print(f'PASS: {kind}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('test_dir',type=Path)
    parser.add_argument('case',choices=['spin_only','spin_translate','asymmetric_rotation'])
    args=parser.parse_args()
    check_case(args.test_dir,args.case)
