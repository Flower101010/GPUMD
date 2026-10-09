#!/usr/bin/env python3
"""Frozen-model identity, total-label synthetic training, and input guard validation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

CONFIG = ("type 2 C O\ncutoff 4 3\nn_max 1 1\nbasis_size 1 1\nl_max 2 0 0\n"
          "neuron 4\nlambda_1 0\nlambda_2 0\n")
BONDED = "molecular_force bonded.in per_frame\n"


def rows(file):
    data = [[float(x) for x in line.split()] for line in file.read_text().splitlines() if line.strip()]
    if not data or any(not math.isfinite(x) for row in data for x in row):
        raise RuntimeError(f"Empty/nonfinite {file}")
    return data


def outputs(work):
    result = {f"{kind}_{part}": rows(work / f"{kind}_{part}.out")
              for part in ("train", "test") for kind in ("energy", "force", "virial")}
    for key, values in result.items():
        kind = key.split("_")[0]
        expected_rows = 10 if kind == "force" else 2
        expected_columns = {"force": 6, "energy": 2, "virial": 12}[kind]
        if len(values) != expected_rows or any(len(row) != expected_columns for row in values):
            raise RuntimeError(f"Incomplete or malformed output: {key}")
    return result


def rmse(data):
    result = {}
    for key, values in data.items():
        dim = len(values[0]) // 2
        errors = [row[i] - row[i + dim] for row in values for i in range(dim)]
        result[key] = math.sqrt(sum(x*x for x in errors) / len(errors))
    return result


def plain(xyz):
    return re.sub(r'cg_topology_version=1 cg_bonds="[^"]*" cg_angles="[^"]*" cg_dihedrals="[^"]*" ', '', xyz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nep", type=Path)
    parser.add_argument("test_bonded_nep", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("training_baseline.json"))
    args = parser.parse_args()
    nep, unit = args.nep.resolve(), args.test_bonded_nep.resolve()
    env = os.environ.copy()
    env.setdefault("CUDACXX", "/usr/local/cuda/bin/nvcc")
    log_file = Path(__file__).with_name("training_validation.log")
    fixture = Path(__file__).resolve().parents[1] / "cg_stage_a/input_draft"
    checks = {}
    with log_file.open("w") as log, tempfile.TemporaryDirectory(prefix="gpumd-stage-d-") as temp:
        root = Path(temp)
        def run(command, work, expected=None):
            result = subprocess.run([str(x) for x in command], cwd=work, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            log.write(f"\nRUN {command} cwd={work}\n{result.stdout}"); log.flush()
            if expected:
                if result.returncode == 0 or expected not in result.stdout:
                    raise RuntimeError(f"Expected rejection {expected}:\n{result.stdout}")
            elif result.returncode:
                raise RuntimeError(result.stdout)
            return result.stdout
        exported = root / "exported"
        text = run([unit, "--export", exported], root)
        checks["strong_baseline_precision"] = {
            "baseline_scale": 1e7,
            "force_residual_recovery_max_abs": float(re.search(r'recovery_max=([^\s]+)', text)[1]),
            "interpretation": "float total cannot preserve this small residual; not a strong-baseline training success"}
        teacher_xyz = (exported / "teacher.xyz").read_text()
        pure_xyz = (fixture / "train.xyz.draft").read_text()
        def setup(name, xyz, model):
            work = root / name; work.mkdir()
            (work / "train.xyz").write_text(xyz)
            (work / "test.xyz").write_text(xyz)
            shutil.copyfile(fixture / "bonded_parameters.in.draft", work / "bonded.in")
            shutil.copyfile(exported / model, work / "nep.txt")
            return work
        def predict(work, batch=2, stream=0, bonded=True):
            for file in work.glob("*.out"): file.unlink()
            (work / "nep.in").write_text(CONFIG + (BONDED if bonded else "") +
                f"prediction 1\nbatch {batch}\nstream_train {stream}\nnep_compile off\n")
            run([nep], work)
            return outputs(work)
        residual_work = setup("residual", plain(teacher_xyz), "teacher.txt")
        residual = predict(residual_work, bonded=False)
        baseline_work = setup("baseline", teacher_xyz, "zero.txt")
        baseline = predict(baseline_work)
        total_work = setup("total", teacher_xyz, "teacher.txt")
        total = predict(total_work)
        identity_max = 0.0
        for key in total:
            dim = len(total[key][0]) // 2
            for r, b, t in zip(residual[key], baseline[key], total[key]):
                assert t[dim:] == r[dim:] == b[dim:], "reference labels changed"
                for i in range(dim): identity_max = max(identity_max, abs(t[i]-r[i]-b[i]))
        if identity_max > 2e-5: raise RuntimeError(f"Baseline identity error {identity_max}")
        variants = {}
        for batch, stream in ((1,0),(1,1),(2,1)):
            variant = predict(total_work, batch, stream)
            difference = max(abs(a-b) for key in total for ra,rb in zip(total[key],variant[key]) for a,b in zip(ra,rb))
            if difference > 2e-5: raise RuntimeError(f"Batch/stream mismatch {difference}")
            variants[f"batch{batch}_stream{stream}"] = difference
        reference_error = rmse(total)
        if max(reference_error.values()) > 8e-6: raise RuntimeError(reference_error)
        checks["frozen_model"] = {"max_identity_output_error":identity_max,
                                   "batch_stream_max_differences": variants,
                                   "teacher_total_rmse":reference_error,
                                   "output_row_counts": {k:len(v) for k,v in total.items()}}
        print("PASS frozen model, fresh train/test outputs, batch and stream identity", flush=True)
        pure = setup("pure_train", pure_xyz, "zero.txt")
        shutil.copyfile(exported / "zero.restart",pure/"nep.restart")
        (pure / "nep.in").write_text(CONFIG+BONDED+"import_q_scaler 1\nbatch 1\nstream_train 1\n"
                                    "generation 10\npopulation 10\noutput_interval 10\nnep_compile off\n")
        run([nep],pure); pure_error=rmse(predict(pure,1,1))
        if max(pure_error.values())>8e-6: raise RuntimeError(pure_error)
        checks["pure_bonded_training"]={"generations":10,"rmse":pure_error,
            "initialization":"zero residual, sigma=1e-4; validates retaining zero residual, not learning topology from scratch"}
        automatic = setup("automatic_scaler", teacher_xyz, "initial.txt")
        shutil.copyfile(exported / "initial.restart", automatic / "nep.restart")
        (automatic / "nep.in").write_text(CONFIG + BONDED +
            "batch 1\nstream_train 1\ngeneration 1\npopulation 10\noutput_interval 1\nnep_compile off\n")
        run([nep], automatic)
        checks["automatic_scaler_smoke"] = {"loss_rows": len(rows(automatic / "loss.out")),
            "finite_prediction_rmse": rmse(predict(automatic, 1, 1))}
        checks["known_residual_training"]={}
        for stream, compiled in ((0,False),(1,False),(1,True)):
            name=f"known_stream{stream}_compiled{int(compiled)}"
            work=setup(name,teacher_xyz,"initial.txt")
            before=rmse(predict(work,1,stream))
            shutil.copyfile(exported/"initial.restart",work/"nep.restart")
            (work/"nep.in").write_text(CONFIG+BONDED+f"import_q_scaler 1\nbatch 1\nstream_train {stream}\n"
                f"generation 200\npopulation 30\noutput_interval 20\nnep_compile {'on' if compiled else 'off'}\n")
            text=run([nep],work)
            if compiled and ("Compile specialized NEP training kernels" not in text or "Warning" in text):
                raise RuntimeError("Specialized training not confirmed: " + text)
            loss=rows(work/"loss.out")
            after=rmse(predict(work,1,stream))
            for kind in ("energy", "force", "virial"):
                key = kind + "_test"
                if after[key] > 0.35 * before[key] or after[key] > 3e-4:
                    raise RuntimeError(f"Insufficient {kind} recovery {name}: before={before}, after={after}")
            checks["known_residual_training"][name]={"generations":200,"population":30,
                "before_rmse":before,"after_rmse":after,"loss_first_last":[loss[0],loss[-1]],
                "initialization":"teacher output weights scaled by 0.6, sigma=0.003"}
            print(f"PASS {name}: force_test {before['force_test']:.4g} -> {after['force_test']:.4g}",flush=True)
        guards={
            "missing_mode":("molecular_force bonded.in\n","requires: parameter_file per_frame"),
            "wrong_mode":("molecular_force bonded.in global\n","requires: parameter_file per_frame"),
            "duplicate":(BONDED+BONDED,"must not be specified more than once"),
            "temperature":(BONDED+"model_type 3\n","plain potential NEP only"),
            "dipole":(BONDED+"model_type 1\n","plain potential NEP only"),
            "polarizability":(BONDED+"model_type 2\n","plain potential NEP only"),
            "charge":(BONDED+"charge_mode 1\n","plain potential NEP only"),
            "vdw":(BONDED+"vdw 1\n","plain potential NEP only"),
            "charge_vdw":(BONDED+"charge_vdw 1\n","plain potential NEP only")}
        checks["input_guards"]=[]
        for name,(suffix,diagnostic) in guards.items():
            work=root/("guard_"+name);work.mkdir()
            shutil.copyfile(fixture/"bonded_parameters.in.draft",work/"bonded.in")
            (work/"nep.in").write_text(CONFIG+suffix)
            run([nep],work,expected=diagnostic);checks["input_guards"].append(name)
        atomic = setup("guard_atomic", teacher_xyz, "teacher.txt")
        atomic_lines = []
        for line in teacher_xyz.splitlines():
            if "Properties=" in line: line += ":adipole:R:3"
            elif line.startswith(("C ", "O ")): line += " 0 0 0"
            atomic_lines.append(line)
        (atomic / "train.xyz").write_text("\n".join(atomic_lines) + "\n")
        (atomic / "nep.in").write_text(CONFIG + BONDED)
        run([nep], atomic, expected="does not support atomic virial labels")
        checks["input_guards"].append("atomic_virial")
    report={"checks":checks,"nep_sha256":hashlib.sha256(nep.read_bytes()).hexdigest(),
            "scope":"two small synthetic frames, shared bead types and coefficients; no physical validation"}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(f"PASS stage-D production validation; report: {args.output}")


if __name__ == "__main__": main()
