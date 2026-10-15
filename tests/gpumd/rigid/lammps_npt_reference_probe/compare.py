"""Run matched equilateral-body NVT/NPT geometry probes in LAMMPS and GPUMD.

Zero interactions isolate dilation and rigid reconstruction; this is not an
ensemble sampling or trajectory equality test. Only the standard library is used.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from run import distances, frames

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PAIRS = ("1-2", "1-3", "2-3")
BEGIN = "<!-- BEGIN EQUILATERAL RESULTS -->"
END = "<!-- END EQUILATERAL RESULTS -->"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_info(source, files):
    return {
        "path": str(source),
        "commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)),
        "sha256": {name: sha256(source/name) for name in files},
    }


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def initial_state(side, temperature):
    # GPUMD constants. Give translation and rotation 3*kT/2 each, without
    # independent bead thermal noise or removal of the single body's COM motion.
    kT = 8.617343e-5 * temperature
    mass, time_unit, length = 12.0, 10.18051, 30.0
    relative = [[-side/2, -side*math.sqrt(3)/6, 0.0],
                [side/2, -side*math.sqrt(3)/6, 0.0],
                [0.0, side*math.sqrt(3)/3, 0.0]]
    vscale = math.sqrt(3*kT/(3*mass))/time_unit/math.sqrt(14)
    vcm = [vscale*x for x in (1, 2, 3)]
    raw_rotation = [cross([1, 2, 3], r) for r in relative]
    wscale = math.sqrt(3*kT/(mass*sum(v*v for row in raw_rotation for v in row)))/time_unit
    positions = [[length/2+r for r in row] for row in relative]
    velocities = [[v+wscale*r for v, r in zip(vcm, row)] for row in raw_rotation]
    return positions, velocities, mass, length


def xyz_text(positions, velocities, mass, length):
    header = (f'Lattice="{length} 0 0 0 {length} 0 0 0 {length}" pbc="T T T" '
              'Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1')
    rows = ["C "+" ".join(repr(x) for x in (*p, mass, *v))+" 0"
            for p, v in zip(positions, velocities)]
    return "3\n"+header+"\n"+"\n".join(rows)+"\n"


def lammps_input(mode, args, positions, velocities, mass, length, interval):
    atoms = "\n".join("create_atoms 1 single "+" ".join(map(repr, p))+" units box" for p in positions)
    velocity_commands = "\n".join(
        f"group atom{i} id {i}\nvelocity atom{i} set "+" ".join(repr(v*1000) for v in row)+" units box"
        for i, row in enumerate(velocities, 1))
    fix = f"fix rb all rigid/{mode}/small molecule temp {args.temperature} {args.temperature} {args.tau_t/1000}"
    if mode == "npt":
        fix += f" iso {args.pressure*10000} {args.pressure*10000} {args.tau_p/1000}"
    fix += " tparam 10 1 3"
    if mode == "npt":
        fix += " pchain 10"
    return f"""units metal
atom_style atomic
atom_modify map array
boundary p p p
region cell block 0 {length} 0 {length} 0 {length} units box
create_box 1 cell
{atoms}
mass 1 {mass}
fix mol all property/atom mol ghost yes
set group all mol 1
pair_style zero 10
pair_coeff * *
comm_modify cutoff 10
{velocity_commands}
{fix}
timestep {args.dt/1000}
thermo {interval}
thermo_style custom step temp press vol lx ly lz
compute_modify thermo_temp extra/dof 0
thermo_modify flush yes format float %.17g
dump state all custom {interval} state.dump id mol x y z xs ys zs xu yu zu ix iy iz vx vy vz
dump_modify state sort id format float %.17g
run {args.steps}
"""


def gpumd_frames(path, dt, positions, length):
    # GPUMD writes at dt*interval, ..., dt*steps. The common input is frame 0.
    result = [{"step": 0, "lengths": [length]*3,
               "atoms": {i: p for i, p in enumerate(positions, 1)}}]
    lines = path.read_text().splitlines()
    offset = 0
    while offset < len(lines):
        count = int(lines[offset])
        header = lines[offset+1]
        lattice = list(map(float, re.search(r'Lattice="([^"]+)"', header)[1].split()))
        assert all(lattice[i] == 0 for i in (1, 2, 3, 5, 6, 7))
        time = float(re.search(r'Time=([\d.eE+-]+)', header)[1])
        atoms = {i: list(map(float, line.split()[1:4]))
                 for i, line in enumerate(lines[offset+2:offset+2+count], 1)}
        result.append({"step": round(time/dt), "lengths": [lattice[i] for i in (0, 4, 8)], "atoms": atoms})
        offset += 2+count
    return result


def analyze(states):
    records, previous = [], None
    for frame in states:
        values = distances(frame)
        assert all(math.isfinite(v) for v in (*frame["lengths"], *values.values()))
        row = {"step": frame["step"], "Lx_A": frame["lengths"][0], **values}
        for pair in PAIRS:
            row["delta_"+pair] = None if previous is None else values[pair]-previous[pair]
        records.append(row)
        previous = values
    baseline = records[0]
    return records, max(abs(row[pair]-baseline[pair]) for row in records for pair in PAIRS)


def run_case(command, case):
    # GPUMD appends dumps, so callers must supply a new case directory.
    result = subprocess.run(command, cwd=case, capture_output=True, text=True, timeout=120,
                            env={**os.environ, "OMP_NUM_THREADS": "1"})
    (case/"stdout.log").write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f"{case.name} exited {result.returncode}; inspect {case/'stdout.log'}")
    return next((line.strip() for line in result.stdout.splitlines()
                 if "LAMMPS (" in line or "Based on version" in line), "see stdout.log")


def markdown(report, output):
    try:
        prefix = output.relative_to(HERE).as_posix()
    except ValueError:
        prefix = str(output)
    lines = [BEGIN, "", f"## 等边三角形：{report['parameters']['steps']} 步实跑记录", "",
             f"本次步数 `{report['parameters']['steps']}`；LAMMPS 源码提交 `{report['lammps_source']['commit']}`；",
             f"GPUMD 工作树提交 `{report['gpumd_source']['commit']}`。可执行文件 SHA256、源码状态和完整参数见",
             f"[{prefix}/comparison.json]({prefix}/comparison.json)。", "",
             "下表距离单位 Å；`Δr = r(当前帧) − r(上一输出帧)`，不是单个积分步的变化。",
             f"第 {report['interval']} 步的上一帧为第 0 步。距离与差值使用 Python binary64 的 round-trip 十进制表示，",
             "不作固定小数位舍入；坐标按 17 位有效数字保存。末尾极小差别包含坐标文本往返误差。", "",
             f"| 运行 | 轨迹文件（相对本目录） | 第 {report['parameters']['steps']} 步 Lx / Å | 采样帧最大内部距离偏离初值 / Å |",
             "| --- | --- | --- | --- |"]
    for name, case in report["cases"].items():
        lines.append(f"| {name} | [{prefix}/{name}/{case['trajectory']}]({prefix}/{name}/{case['trajectory']}) | {case['records'][-1]['Lx_A']!r} | {case['max_distance_change_A']!r} |")
    lines += ["", f"GPUMD 轨迹从第 {report['interval']} 步开始；共同的第 0 步坐标保存在各 GPUMD 目录的 `model.xyz`。",
              "LAMMPS dump 含第 0 步。各目录还保存输入、原始 stdout/log、热力学输出与 `distances.csv`。", ""]
    for name, case in report["cases"].items():
        lines += [f"### {name}", "",
                  "| step | r12 / Å | Δr12 / Å | r13 / Å | Δr13 / Å | r23 / Å | Δr23 / Å |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in case["records"][1:]:
            values = [str(row["step"])]
            for pair in PAIRS:
                values += [repr(row[pair]), repr(row["delta_"+pair])]
            lines.append("| "+" | ".join(values)+" |")
        lines.append("")
    lines += [END]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--lammps-source", type=Path, required=True)
    parser.add_argument("--gpumd", type=Path, default=ROOT/"build/gpumd")
    parser.add_argument("--out-dir", type=Path, default=HERE/"results/equilateral_1000")
    parser.add_argument("--side", type=float, default=3.0, help="Triangle edge length in Angstrom")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.5, help="Time step in fs")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--pressure", type=float, default=0.0001, help="Isotropic target in GPa")
    parser.add_argument("--tau-t", type=float, default=100.0, help="Thermostat time in fs")
    parser.add_argument("--tau-p", type=float, default=1000.0, help="Piston time in fs")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    for key in ("side", "dt", "temperature", "tau_t", "tau_p"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if args.steps < 10 or args.steps % 10 or not math.isfinite(args.pressure):
        parser.error("steps must be a positive multiple of 10; pressure must be finite")
    if args.side >= 10 or min(args.tau_t, args.tau_p) < args.dt:
        parser.error("side must be below 10 A; coupling times must be at least dt")
    binaries = {"lammps": args.lammps.resolve(), "gpumd": args.gpumd.resolve()}
    output = args.out_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; use a new --out-dir to avoid appended trajectories")
    output.mkdir(parents=True, exist_ok=True)
    positions, velocities, mass, length = initial_state(args.side, args.temperature)
    interval = args.steps//10
    report = {"parameters": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
              "interval": interval, "interaction": "zero force / zero model virial",
              "mass_amu": mass, "initial_box_A": length,
              "initial_positions_A": positions, "initial_velocities_A_per_fs": velocities,
              "executable_sha256": {k: sha256(v) for k, v in binaries.items()},
              "executables": {k: str(v) for k, v in binaries.items()},
              "lammps_source": source_info(args.lammps_source.resolve(),
                  ["src/RIGID/fix_rigid_nh_small.cpp", "src/RIGID/fix_rigid_small.cpp"]),
              "gpumd_source": source_info(ROOT,
                  ["src/integrate/ensemble_rigid_npt.cu", "src/integrate/ensemble_rigid_nvt.cu",
                   "src/integrate/rigid_body_integrator.cu",
                   "src/integrate/rigid_body.cu", "src/integrate/rigid_nh.cu"]), "cases": {}}
    for engine, binary in binaries.items():
        for mode in ("nvt", "npt"):
            name = f"{engine}_{mode}"
            case = output/name
            case.mkdir()
            if engine == "lammps":
                (case/"in.lammps").write_text(lammps_input(mode, args, positions, velocities, mass, length, interval))
                version = run_case([str(binary), "-in", "in.lammps"], case)
                trajectory = "state.dump"
                states = frames(case/trajectory)
                reconstruction = max(s["reconstruction_error_A"] for s in states)
                # Read Cartesian coordinates directly; independently reconstruct
                # scaled coordinates using EACH frame's bounds, and unwrap too.
                cartesian_error = max(abs(distances(s)[p]-distances(s, "from_scaled")[p])
                                      for s in states for p in PAIRS)
                unwrapped_error = max(abs(distances(s)[p]-distances(s, "unwrapped", minimum_image=False)[p])
                                      for s in states for p in PAIRS)
                assert max(cartesian_error, unwrapped_error) < 1e-11
            else:
                (case/"model.xyz").write_text(xyz_text(positions, velocities, mass, length))
                (case/"zero_lj.txt").write_text("lj 1 C\n0.0 1.0 4.0\n")
                ensemble = f"ensemble rigid_{mode} {args.temperature} {args.temperature} {args.tau_t/args.dt}"
                if mode == "npt":
                    ensemble += f" {args.pressure} {args.tau_p/args.dt}"
                (case/"run.in").write_text(
                    f"potential zero_lj.txt\n{ensemble} 0\ntime_step {args.dt}\n"
                    f"dump_thermo {interval}\ndump_xyz {interval} trajectory.xyz precision double velocity force group_labels\nrun {args.steps}\n")
                version = run_case([str(binary)], case)
                trajectory = "trajectory.xyz"
                states = gpumd_frames(case/trajectory, args.dt, positions, length)
                reconstruction = cartesian_error = unwrapped_error = None
            assert [s["step"] for s in states] == list(range(0, args.steps+1, interval))
            records, maximum = analyze(states)
            with (case/"distances.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0])
                writer.writeheader()
                writer.writerows(records)
            report["cases"][name] = {
                "version_line": version, "trajectory": trajectory, "records": records,
                "max_distance_change_A": maximum,
                "max_coordinate_reconstruction_error_A": reconstruction,
                "max_distance_from_scaled_difference_A": cartesian_error,
                "max_distance_from_unwrapped_difference_A": unwrapped_error,
            }
            print(f"{name}: max |r-r0| = {maximum!r} A; trajectory {case/trajectory}")
    # The upstream NPT observation is deliberately not a pass/fail target.
    # Fixed-box controls and our rigid geometry must remain within roundoff.
    report["geometry_tolerance_A"] = 1e-10
    report["checks"] = {
        name+"_geometry": report["cases"][name]["max_distance_change_A"] <= 1e-10
        for name in ("lammps_nvt", "gpumd_nvt", "gpumd_npt")}
    (output/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    section = markdown(report, output)
    (output/"tables.md").write_text(section+"\n")
    if args.update_readme:
        path = HERE/"README.md"
        text = path.read_text()
        if BEGIN in text and END in text:
            start, stop = text.index(BEGIN), text.index(END)+len(END)
            text = text[:start]+section+text[stop:]
        else:
            text += "\n"+section+"\n"
        path.write_text(text)
    print(f"Recorded: {output/'comparison.json'}")
    if not all(report["checks"].values()):
        raise SystemExit("FAIL: fixed-box control or GPUMD rigid geometry changed beyond 1e-10 A")


if __name__ == "__main__":
    main()
