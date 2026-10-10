#!/usr/bin/env python3
"""Frozen NEP baseline identity vs independent OpenMM labels, without fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from run_independent_validation import PARAMETERS, packed, oracle, write_inputs, gpumd, HERE

CONFIG=('type 2 C O\ncutoff 4 3\nn_max 1 1\nbasis_size 1 1\nl_max 2 0 0\n'
        'neuron 4\nimport_q_scaler 1\nprediction 1\n')


def xyz_frame(r,box,terms,reference):
    cg=[]
    for key,rows in zip(('cg_bonds','cg_angles','cg_dihedrals'),terms):
        cg.append(key+'="'+(';'.join(','.join(map(str,row)) for row in rows) if rows else 'none')+'"')
    header=('pbc="T T T" Lattice="'+' '.join(format(x,'.17g') for x in box.ravel())+'" '
            'Properties=species:S:1:pos:R:3:force:R:3 energy='+format(reference['energy'],'.17g')+
            ' virial="'+' '.join(format(x,'.17g') for x in reference['virial'].ravel())+'" '
            'cg_topology_version=1 '+' '.join(cg))
    lines=[str(len(r)),header]
    for i,(pos,f) in enumerate(zip(r,reference['force'])):
        lines.append(('C' if i%2==0 else 'O')+' '+' '.join(format(x,'.17g') for x in [*pos,*f]))
    return '\n'.join(lines)+'\n'


def outputs(work):
    result={}
    for kind in ('energy','force','virial'):
        for part in ('train','test'):
            key=kind+'_'+part
            rows=np.loadtxt(work/(key+'.out'),ndmin=2)
            assert np.isfinite(rows).all()
            result[key]=rows
    return result


def max_scaled_error(left,right,atol=1e-5,rtol=1e-5):
    ratio=float(np.max(np.abs(left-right)/(atol+rtol*np.abs(right))))
    assert ratio<=1,ratio
    return {'max_absolute_error':float(np.max(np.abs(left-right))),
            'tolerance_ratio':ratio,'atol':atol,'rtol':rtol}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpumd',type=Path);parser.add_argument('nep',type=Path)
    parser.add_argument('exporter',type=Path)
    parser.add_argument('--sizes',type=int,nargs='+',default=[32,256,2048])
    parser.add_argument('--output',type=Path,default=HERE/'nep_interface_baseline.json')
    args=parser.parse_args()
    executables=[p.resolve() for p in (args.gpumd,args.nep,args.exporter)]
    report={'scope':'prediction/loss input plumbing with frozen models; no fit-quality criterion',
            'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in executables},
            'sizes':args.sizes,'paths':[],'md':[]}
    with tempfile.TemporaryDirectory(prefix='gpumd-stage-f-nep-') as temp,(HERE/'nep_interface.log').open('w') as log:
        root=Path(temp)
        def run(command,work):
            result=subprocess.run([str(x) for x in command],cwd=work,text=True,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
            log.write(f'\nRUN {command} cwd={work}\n{result.stdout}');log.flush()
            if result.returncode: raise RuntimeError(result.stdout)
            return result.stdout
        run([executables[2],'--export',root/'models'],root)
        cases=[];texts=[]
        for n in args.sizes:
            r,box,terms=packed(n,'mixed',True,True)
            # Training Structure stores coordinates/cell as float. Oracle and MD
            # use exactly those representable numbers for the comparison.
            r=r.astype(np.float32).astype(np.float64);box=box.astype(np.float32).astype(np.float64)
            ref=oracle(r,box,terms);cases.append((r,box,terms,ref))
            texts.append(xyz_frame(r,box,terms,ref))
        total_xyz=''.join(texts)
        plain=re.sub(r'cg_topology_version=1 cg_bonds="[^"]*" cg_angles="[^"]*" cg_dihedrals="[^"]*"','',total_xyz)
        previous={}
        for model in ('zero','teacher'):
            md_records={'energy':[], 'force':[], 'virial':[]}
            # Production prediction mode deliberately disables specialization
            # (NEP constructor requires prediction==0). CTest separately checks
            # the training evaluator with specialized kernels; do not mislabel
            # a prediction run with nep_compile on as specialized coverage.
            for mode,stream,batch,compiled in [('resident',0,2,'off'),('stream',1,1,'off')]:
                pair={}
                for bonded in (False,True):
                    work=root/f'{model}_{mode}_{bonded}';work.mkdir()
                    shutil.copyfile(root/'models'/f'{model}.txt',work/'nep.txt')
                    (work/'bonded.in').write_text(PARAMETERS)
                    for name in ('train.xyz','test.xyz'):
                        (work/name).write_text(total_xyz if bonded else plain)
                    (work/'nep.in').write_text(CONFIG+f'batch {batch}\nstream_train {stream}\nnep_compile {compiled}\n'+
                                              ('molecular_force bonded.in per_frame\n' if bonded else ''))
                    stdout=run([executables[1]],work)
                    if compiled=='on':
                        assert 'specialized' in stdout.lower() or 'compiled nep backend' in stdout.lower(),stdout
                        assert 'fallback' not in stdout.lower(),stdout
                    pair[bonded]=outputs(work)
                check={}
                for key,actual in pair[True].items():
                    residual=pair[False][key];dim=actual.shape[1]//2
                    assert actual.shape==residual.shape
                    assert np.array_equal(actual[:,dim:],residual[:,dim:]),key
                    baseline=actual[:,dim:]
                    check[key]=max_scaled_error(actual[:,:dim]-residual[:,:dim],baseline)
                    if model=='zero': max_scaled_error(residual[:,:dim],np.zeros_like(baseline))
                # batch/stream may change frame order. Compare independently labelled
                # records rather than assume raw output order is identical.
                for key,values in pair[True].items():
                    sort=np.lexsort(values[:,values.shape[1]//2:].T)
                    ordered=values[sort]
                    if (model,key) in previous: max_scaled_error(ordered,previous[model,key])
                    else: previous[model,key]=ordered
                report['paths'].append({'model':model,'mode':mode,'batch':batch,'stream':stream,
                                        'compiled_required':compiled=='on','baseline_identity':check})
                print('PASS frozen NEP',model,mode,'baseline identity',flush=True)
            # MD residual+baseline difference independently equals OpenMM.
            for n,(r,box,terms,ref) in zip(args.sizes,cases):
                work=root/f'md_{model}_{n}';work.mkdir();write_inputs(work,r,box,terms)
                shutil.copyfile(root/'models'/f'{model}.txt',work/'model_nep.txt')
                script=(work/'run.in').read_text().replace('zero_lj.txt','model_nep.txt')
                (work/'run.in').write_text(script);actual=gpumd(executables[0],work)
                (work/'run.in').write_text(script.replace('molecular_force parameters.in topology.in\n',''))
                residual=gpumd(executables[0],work)
                errors={}
                for key in ('energy','force','virial'):
                    left=np.asarray(actual[key])-np.asarray(residual[key]);right=np.asarray(ref[key])
                    if key!='force':left=left/n;right=right/n
                    errors[key]=max_scaled_error(left,right,1e-8,1e-8)
                report['md'].append({'model':model,'beads':n,'baseline_identity':errors})
                md_records['energy'].append(np.array([[actual['energy']/n,ref['energy']/n]]))
                md_records['force'].append(np.concatenate((actual['force'],ref['force']),axis=1))
                components=[0,4,8,1,5,6]  # xx yy zz xy yz zx
                md_records['virial'].append(np.concatenate((actual['virial'].ravel()[components]/n,
                                                           ref['virial'].ravel()[components]/n))[None,:])
            consistency={}
            for kind,records in md_records.items():
                values=np.concatenate(records);dim=values.shape[1]//2
                # Production text labels use six significant digits. Round the
                # independent keys identically before matching output records.
                keys=np.array([[float(format(x,'.6g')) for x in row]
                               for row in values[:,dim:]])
                ordered=values[np.lexsort(keys.T)]
                prediction=previous[model,kind+'_train']
                consistency[kind]=max_scaled_error(prediction,ordered)
            report.setdefault('prediction_md_consistency',[]).append({'model':model,'errors':consistency})
        # Test the current mandatory energy-field workaround without any fitting:
        # force-only prediction must not depend on a constant placeholder energy.
        force_work=root/'force_only';force_work.mkdir()
        shutil.copyfile(root/'models'/'teacher.txt',force_work/'nep.txt')
        (force_work/'bonded.in').write_text(PARAMETERS)
        (force_work/'nep.in').write_text(CONFIG+'batch 2\nstream_train 1\nnep_compile off\n'
                                       'molecular_force bonded.in per_frame\nlambda_e 0\nlambda_f 1\nlambda_v 0\n')
        force_results=[]
        for placeholder in (0,1):
            data=re.sub(r'energy=[^\s]+',f'energy={placeholder}',total_xyz)
            data=re.sub(r' virial="[^"]*"','',data)
            for name in ('train.xyz','test.xyz'):(force_work/name).write_text(data)
            for file in force_work.glob('*.out'):file.unlink()
            run([executables[1]],force_work);force_results.append(outputs(force_work))
        # Constant total energies can reorder different-size frames (E/N). Compare
        # force rows using their unchanged force reference as keys.
        for part in ('train','test'):
            records=[]
            for result in force_results:
                values=result['force_'+part]; records.append(values[np.lexsort(values[:,3:].T)])
            max_scaled_error(records[0],records[1])
        report['force_only_placeholder_prediction']={'checked_constants':[0,1],
             'virial_labels_omitted':True,'scope':'prediction invariance only; does not certify training fitness ignores placeholder'}
    report['status']='PASS';args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS NEP and MD calculation interfaces',flush=True)


if __name__=='__main__':main()
