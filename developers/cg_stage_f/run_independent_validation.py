#!/usr/bin/env python3
"""Independent OpenMM Reference E/F and strain-derivative W vs GPUMD CUDA.

No NEP fitting quality or physical-distribution criterion is used.
Requires numpy/openmm; all run inputs live in temporary directories.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import openmm as mm
from openmm import unit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'cg_stage_a'))
from run_fixture_md import read_xyz

EV = 96.4853321233  # kJ/mol per eV
PARAMETERS = '''gpumd_bonded_parameters 1
harmonic_bond_parameters 2
1.2 5.0
1.4 3.0
harmonic_angle_parameters 1
1.9 2.0
periodic_dihedral_parameters 2
0.2 3 0.4
0.07 1 -0.3
'''
BONDS = [(1.2, 5.0), (1.4, 3.0)]
TORSIONS = [(0.2, 3, 0.4), (0.07, 1, -0.3)]
CHAIN = np.array([[0,0,0],[1.3,.2,.1],[2.4,1.1,-.2],[3.5,1.3,.9],
                  [4.6,2.2,.3],[5.7,1.5,-.6],[6.8,2.2,.4],[7.9,3.1,.2]])
STAR = np.array([[0,0,0],[1.2,.3,.2],[-.5,1.2,.2],[.2,-.9,.8],
                 [-1,-.3,.5],[.1,.4,-1.1],[1,-.7,-.2],[-.7,.5,-.9]])


def topology(n, kind, family):
    bonds = [(i,i+1,i%2) for i in range(n-1)]
    angles = [(i,i+1,i+2,0) for i in range(n-2)]
    torsions = [(i,i+1,i+2,i+3,i%2) for i in range(n-3)]
    torsions += [(0,1,2,3,1)]
    if kind == 'star':
        assert n == 8
        bonds = [(0,i,i%2) for i in range(1,8)]+[(2,3,0),(5,6,1)]
        angles = [(i,0,j,0) for i,j in itertools.combinations(range(1,8),2)]
        torsions = [(1,0,2,3,0),(1,0,2,3,1),(4,0,5,6,0),(6,5,0,7,1)]
    if family == 'bond': angles, torsions = [], []
    elif family == 'angle': torsions = []
    elif family == 'dihedral': bonds, angles = [], []
    return [bonds, angles, torsions]


def packed(n, kind='chain', skew=False, crossing=False):
    assert n % 8 == 0
    groups=n//8; side=int(np.ceil(groups**(1/3)))
    box=np.diag([12*side+16.0]*3)
    if skew: box[1,0]=box[0,0]*.17; box[2,0]=box[0,0]*.11; box[2,1]=box[1,1]*.13
    rng=np.random.default_rng(20261010+n)
    positions=[]; interactions=[[],[],[]]
    for g in range(groups):
        chosen = ('star' if g%2 else 'chain') if kind=='mixed' else kind
        shift=np.array([g%side,(g//side)%side,g//(side*side)])*12+4
        local=(STAR if chosen=='star' else CHAIN)+rng.normal(0,.015,(8,3))+shift
        positions.extend(local)
        for target, terms in zip(interactions,topology(8,chosen,'full')):
            target.extend(tuple(x+8*g for x in row[:-1])+(row[-1],) for row in terms)
    r=np.array(positions)
    if crossing:
        r += np.array([box[0,0]-4.4,box[1,1]-4.6,box[2,2]-4.2])
    # Row-vector cell convention, matching extxyz and OpenMM.
    r=(r@np.linalg.inv(box)%1)@box
    return r,box,interactions


def oracle(r,box,terms):
    system=mm.System()
    for _ in r: system.addParticle(12)
    system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*v/10) for v in box))
    b=mm.HarmonicBondForce(); a=mm.HarmonicAngleForce(); d=mm.PeriodicTorsionForce()
    for i,j,t in terms[0]:
        r0,k=BONDS[t]; b.addBond(i,j,r0/10,k*EV*100)
    for i,j,k,t in terms[1]: a.addAngle(i,j,k,1.9,2*EV)
    for i,j,k,l,t in terms[2]:
        strength,multiplicity,phase=TORSIONS[t]
        d.addTorsion(i,j,k,l,multiplicity,phase,strength*EV)
    for force in (b,a,d):
        force.setUsesPeriodicBoundaryConditions(True); system.addForce(force)
    integrator=mm.VerletIntegrator(.001)
    context=mm.Context(system,integrator,mm.Platform.getPlatformByName('Reference'))
    def evaluate(x,h,forces=False):
        # OpenMM requires reduced triangular cells. Rotate any strained cell
        # to that convention, preserving all geometry and scalar energies.
        q,upper=np.linalg.qr(h.T)
        signs=np.sign(np.diag(upper)); q=q*signs; upper=signs[:,None]*upper
        context.setPeriodicBoxVectors(*(mm.Vec3(*v/10) for v in upper.T))
        context.setPositions((x@q/10)*unit.nanometer)
        state=context.getState(getEnergy=True,getForces=forces)
        energy=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)/EV
        if forces:
            f=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            return energy,np.asarray(f)@q.T/(EV*10)
        return energy
    energy,force=evaluate(r,box,True)
    virials=[]
    for step in (1e-5,5e-6):
        w=np.zeros((3,3))
        for i,j in itertools.product(range(3),repeat=2):
            plus=np.eye(3); minus=np.eye(3)
            plus[i,j]+=step; minus[i,j]-=step
            w[i,j]=-(evaluate(r@plus.T,box@plus.T)-evaluate(r@minus.T,box@minus.T))/(2*step)
        virials.append(w)
    del context,integrator
    return {'energy':energy,'force':force,'virial':virials[1],
            'strain_convergence_per_bead':float(np.max(np.abs(virials[0]-virials[1]))/len(r))}


def write_inputs(work,r,box,terms):
    n=len(r)
    lines=[str(n),'pbc="T T T" Lattice="'+' '.join(format(x,'.17g') for x in box.ravel())+
           '" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3']
    for i,row in enumerate(r):
        lines.append(('C' if i%2==0 else 'O')+' '+' '.join(format(x,'.17g') for x in row)+' 12 0 0 0')
    (work/'model.xyz').write_text('\n'.join(lines)+'\n')
    (work/'parameters.in').write_text(PARAMETERS)
    text=['gpumd_topology 1',f'number_of_atoms {n}']
    for name,rows in zip(('bonds','angles','dihedrals'),terms):
        text.append(f'{name} {len(rows)}');text.extend(' '.join(map(str,row)) for row in rows)
    (work/'topology.in').write_text('\n'.join(text)+'\n')
    (work/'zero_lj.txt').write_text('lj 2 C O\n'+'0 1 3\n'*4)
    (work/'run.in').write_text('potential zero_lj.txt\nmolecular_force parameters.in topology.in\n'
        'time_step 0.00000001\nensemble nve\ndump_xyz 1 result.xyz force potential virial precision double\nrun 1\n')


def gpumd(executable,work):
    output=work/'result.xyz'
    if output.exists(): output.unlink()
    start=time.perf_counter()
    result=subprocess.run([str(executable)],cwd=work,text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,timeout=120)
    (work/'gpumd.log').write_text(result.stdout)
    if result.returncode: raise RuntimeError(result.stdout)
    _,schema,rows=read_xyz(output)[0]
    return {'energy':sum(float(row[schema['energy_atom']][0]) for row in rows),
            'force':np.array([[float(x) for x in row[schema['forces']]] for row in rows]),
            'virial':np.array([[float(x) for x in row[schema['virial']]] for row in rows]).sum(axis=0).reshape(3,3),
            'process_wall_seconds':time.perf_counter()-start}


def compare(actual,reference,n):
    errors={}
    for key,atol,rtol in [('energy',1e-9,1e-9),('force',1e-8,1e-8),('virial',2e-7,2e-7)]:
        left=np.asarray(actual[key]); right=np.asarray(reference[key])
        if key!='force': left=left/n;right=right/n
        difference=np.abs(left-right)
        assert np.isfinite(left).all() and np.isfinite(right).all()
        ratio=float(np.max(difference/(atol+rtol*np.abs(right))))
        errors[key]={'max_error':float(np.max(difference)),'tolerance_ratio':ratio,
                     'normalization':'per_bead' if key!='force' else 'none','atol':atol,'rtol':rtol}
        assert ratio<=1,(key,errors[key])
    assert reference['strain_convergence_per_bead']<2e-7,reference['strain_convergence_per_bead']
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpumd',type=Path)
    parser.add_argument('--sizes',type=int,nargs='+',default=[32,256,2048,8192])
    parser.add_argument('--output',type=Path,default=HERE/'independent_baseline.json')
    parser.add_argument('--public-coordinates',type=Path)
    parser.add_argument('--export-case',type=Path)
    args=parser.parse_args();executable=args.gpumd.resolve()
    if args.export_case and args.export_case.exists(): raise RuntimeError('Refusing to overwrite export directory')
    report={'oracle':'OpenMM native bonded forces, Reference CPU platform',
            'openmm_version':mm.__version__,'numpy_version':np.__version__,
            'gpumd_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
            'cases':[],'scope':'calculation correctness only; no scientific NEP quality criterion'}
    cases=[]
    for family in ('bond','angle','dihedral','full'):
        r,box,_=packed(8);cases.append((f'isolated_{family}',r,box,topology(8,'chain',family)))
    for n in args.sizes:
        for kind,skew,crossing in [('mixed',False,False),('mixed',True,True)]:
            r,box,terms=packed(n,kind,skew,crossing)
            cases.append((f'n{n}_'+('skew_crossing' if skew else 'orthogonal'),r,box,terms))
    r,box,terms=packed(args.sizes[0],'mixed',True,True)
    permutation=np.random.default_rng(17).permutation(len(r))
    inverse=np.argsort(permutation)
    remapped=[[tuple(int(inverse[i]) for i in row[:-1])+(row[-1],) for row in family]
              for family in terms]
    cases.append(('permuted_atoms',r[permutation],box,remapped))
    if args.public_coordinates:
        source=args.public_coordinates.resolve();data=np.load(source,allow_pickle=False)
        assert data.shape==(10000,5,3) and np.isfinite(data).all()
        report['public_geometry']={'file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'shape':list(data.shape),'scope':'published coordinate numbers interpreted as angstrom for identical mathematical tests; not force-label or PMF validation'}
        for index in (0,1,123,999,2500,4999,7500,9999):
            r=np.asarray(data[index],dtype=np.float64);r=r-r.mean(axis=0)+20
            cases.append((f'public_ala2_{index}',r,np.eye(3)*60,topology(5,'chain','full')))
    with tempfile.TemporaryDirectory(prefix='gpumd-stage-f-oracle-') as temporary:
        root=Path(temporary)
        for name,r,box,terms in cases:
            work=root/name;work.mkdir();write_inputs(work,r,box,terms)
            reference=oracle(r,box,terms)
            results=[gpumd(executable,work) for _ in range(3)]
            errors=[compare(actual,reference,len(r)) for actual in results]
            reproducibility=max(float(np.max(np.abs(np.asarray(results[0][key])-np.asarray(other[key]))))
                                for other in results[1:] for key in ('energy','force','virial'))
            item={'name':name,'beads':len(r),'interactions':list(map(len,terms)),
                  'errors':errors[0],'repeat_max_difference':reproducibility,
                  'strain_convergence_per_bead':reference['strain_convergence_per_bead'],
                  'process_wall_seconds':[v['process_wall_seconds'] for v in results]}
            report['cases'].append(item)
            print('PASS',name,'N=',len(r),'E/F/W',flush=True)
            if args.export_case and name==f'n{args.sizes[0]}_skew_crossing':
                import shutil
                shutil.copytree(work,args.export_case)
    report['status']='PASS';args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS independent calculation matrix',len(cases),'cases x 3 runs',flush=True)


if __name__=='__main__': main()
