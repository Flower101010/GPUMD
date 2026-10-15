"""Characterize rigid geometry in a real LAMMPS executable; no GPUMD code runs.

This is an upstream-reference probe, not an NPT sampling acceptance test.
A zero-force, zero-velocity triangle isolates geometry changes due to dilation.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def frames(path):
    lines = path.read_text().splitlines()
    result = []
    offset = 0
    while offset < len(lines):
        assert lines[offset] == "ITEM: TIMESTEP"
        step, count = int(lines[offset+1]), int(lines[offset+3])
        bounds = [list(map(float, line.split())) for line in lines[offset+5:offset+8]]
        columns = lines[offset+8].split()[2:]
        atoms, from_scaled, unwrapped, images = {}, {}, {}, {}
        reconstruction_error = unwrapping_error = 0.0
        for line in lines[offset+9:offset+9+count]:
            values = dict(zip(columns, map(float, line.split())))
            atom_id = int(values["id"])
            atoms[atom_id] = [values[c] for c in ("x", "y", "z")]
            from_scaled[atom_id] = [lo+values[c]*(hi-lo)
                                   for c, (lo, hi) in zip(("xs", "ys", "zs"), bounds)]
            unwrapped[atom_id] = [values[c] for c in ("xu", "yu", "zu")]
            images[atom_id] = [int(values[c]) for c in ("ix", "iy", "iz")]
            for c, (lo, hi) in enumerate(bounds):
                reconstruction_error = max(reconstruction_error,
                    abs(atoms[atom_id][c]-from_scaled[atom_id][c]))
                unwrapping_error = max(unwrapping_error,
                    abs(unwrapped[atom_id][c]-atoms[atom_id][c]-images[atom_id][c]*(hi-lo)))
        assert max(reconstruction_error, unwrapping_error) <= 1e-12
        result.append({"step": step, "lengths": [b-a for a, b in bounds], "atoms": atoms,
                       "from_scaled": from_scaled, "unwrapped": unwrapped, "images": images,
                       "reconstruction_error_A": reconstruction_error, "unwrapping_error_A": unwrapping_error})
        offset += 9+count
    return result


def distances(frame, coordinates="atoms", minimum_image=True):
    result = {}
    for i, j in itertools.combinations(sorted(frame["atoms"]), 2):
        delta = [a-b for a, b in zip(frame[coordinates][i], frame[coordinates][j])]
        if minimum_image:
            delta = [d-L*round(d/L) for d, L in zip(delta, frame["lengths"])]
        result[f"{i}-{j}"] = math.sqrt(sum(d*d for d in delta))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=HERE/"results")
    parser.add_argument("--source", type=Path, help="Record the checked-out reference commit and source hashes")
    args = parser.parse_args()
    binary = args.lammps.resolve()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = (HERE/"in.lammps").read_text()
    original = "fix rb all rigid/npt/small molecule temp 300 300 1 iso 10000 10000 1"
    external = ("group solvent empty\ncompute physical all temp\n"
                "fix bp solvent npt temp 300 300 1 iso 10000 10000 1\n"
                "fix_modify bp temp physical")
    rigid_nve = "fix rb all rigid/nve/small molecule"
    cases = {
        "nvt_small": "fix rb all rigid/nvt/small molecule temp 300 300 1",
        "npt_small_10000bar": original,
        "npt_large_10000bar": original.replace("rigid/npt/small", "rigid/npt"),
        "npt_small_1bar": original.replace("iso 10000 10000 1", "iso 1 1 1"),
        "npt_small_zero_pressure": original.replace("iso 10000 10000 1", "iso 0 0 1"),
        "nph_small_10000bar": "fix rb all rigid/nph/small molecule iso 10000 10000 1",
        "npt_small_moving": original,
        "npt_before_rigid_nve": external+"\n"+rigid_nve,
        "npt_after_rigid_nve": rigid_nve+"\n"+external,
    }
    report = {"lammps_executable": str(binary), "units": "metal",
              "steps": 1, "timestep_ps": 0.001,
              "interaction": "pair_style zero", "cases": {}}
    with binary.open("rb") as handle:
        report["executable_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    if args.source:
        source = args.source.resolve()
        report["source_commit"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        report["source_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True))
        report["source_sha256"] = {}
        for relative in ("src/RIGID/fix_rigid_nh_small.cpp", "src/RIGID/fix_rigid_small.cpp", "src/fix_nh.cpp"):
            report["source_sha256"][relative] = hashlib.sha256((source/relative).read_bytes()).hexdigest()
    for name, fix_command in cases.items():
        case = output/name
        case.mkdir(exist_ok=True)
        input_text = template.replace(original, fix_command)
        initial_velocity = "1 2 3" if name.endswith("_moving") else "0 0 0"
        input_text = input_text.replace("velocity all set 0 0 0", "velocity all set "+initial_velocity)
        (case/"in.lammps").write_text(input_text)
        run = subprocess.run([str(binary), "-in", "in.lammps"], cwd=case,
                             text=True, capture_output=True, timeout=30)
        (case/"stdout.log").write_text(run.stdout+run.stderr)
        if run.returncode:
            errors = [line for line in (run.stdout+run.stderr).splitlines() if "ERROR" in line]
            report["cases"][name] = {"status": "rejected", "fix_command": fix_command,
                                     "returncode": run.returncode, "errors": errors}
            print(f"{name}: REJECTED: {'; '.join(errors)}")
            continue
        states = frames(case/"state.dump")
        assert [s["step"] for s in states] == [0, 1]
        before, after = map(distances, states)
        from_scaled = distances(states[1], "from_scaled")
        from_unwrapped = distances(states[1], "unwrapped", minimum_image=False)
        expected_scale = math.sqrt(states[1]["lengths"][0]/states[0]["lengths"][0])
        delta = max(abs(after[key]-before[key]) for key in before)
        remap_error = max(abs(after[key]-expected_scale*before[key]) for key in before)
        result = {"status": "completed", "version": run.stdout.splitlines()[0], "fix_command": fix_command,
                  "initial_velocity_A_per_ps": initial_velocity,
                  "box_lengths_before": states[0]["lengths"], "box_lengths_after": states[1]["lengths"],
                  "distances_before_A": before, "distances_after_A": after,
                  "distances_after_from_scaled_A": from_scaled,
                  "distances_after_from_unwrapped_A": from_unwrapped,
                  "max_coordinate_reconstruction_error_A": max(s["reconstruction_error_A"] for s in states),
                  "max_unwrapping_identity_error_A": max(s["unwrapping_error_A"] for s in states),
                  "all_image_flags_zero": all(v == 0 for s in states for img in s["images"].values() for v in img),
                  "max_distance_change_A": delta, "rigid_within_1e-12_A": delta <= 1e-12,
                  "half_box_scale": expected_scale,
                  "max_error_from_half_box_scaling_A": remap_error,
                  "full_box_scale": expected_scale**2,
                  "max_error_from_full_box_scaling_A":
                      max(abs(after[key]-expected_scale**2*before[key]) for key in before)}
        report["cases"][name] = result
        print(f"{name}: r12 {before['1-2']:.16g} -> {after['1-2']:.16g} A; "
              f"rigid={result['rigid_within_1e-12_A']}")
    destination = output/"reference_probe.json"
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Recorded observations: {destination}")


if __name__ == "__main__":
    main()
