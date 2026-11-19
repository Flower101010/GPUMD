#!/usr/bin/env python3
"""Run the NEP/GPUMD composition descriptor checks on a single visible GPU."""
import argparse
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


GEOMETRY = [(1.0, 1.0, 1.0), (2.8, 1.1, 1.0), (1.3, 2.9, 1.1), (2.7, 2.9, 2.0)]
NEP_INPUT = """type 2 O C
cutoff 4 4
n_max 1 1
basis_size 1 1
l_max 2 0 0
neuron 2
batch 2
"""
ZERO_VIRIAL = "0 0 0 0 0 0 0 0 0"


def lattice(length):
    return [(length, 0, 0), (0, length, 0), (0, 0, length)]


def xyz(species, coordinates, cell, training=False, energy=0):
    properties = "species:S:1:pos:R:3"
    if training:
        properties += ":force:R:3"
    cell_text = " ".join(str(x) for vector in cell for x in vector)
    header = f'Lattice="{cell_text}" pbc="T T T" Properties={properties}'
    if training:
        header += f' energy={energy} virial="{ZERO_VIRIAL}"'
    atoms = [
        f"{symbol} {' '.join(map(str, row))}" + (" 0 0 0" if training else "")
        for symbol, row in zip(species, coordinates)
    ]
    return f"{len(species)}\n{header}\n" + "\n".join(atoms) + "\n"


def invoke(executable, folder, expected_error=None):
    result = subprocess.run([executable], cwd=folder, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    (folder / "run.log").write_text(output)
    if expected_error:
        assert expected_error in output, output
    else:
        assert result.returncode == 0, output
    return output


def close(a, b, atol=3e-6, rtol=3e-5):
    assert len(a) == len(b), (len(a), len(b))
    for x, y in zip(a, b):
        assert math.isclose(x, y, abs_tol=atol, rel_tol=rtol), (x, y)


def model(path, local=True, order="OC", composition=True):
    # Two neurons, six local dimensions, optionally one composition dimension.
    ann = {
        "O": [0.3, 0.2, 0.08, 0.1, 0.03, 0.06, 1.0, 0.1, 0.2],
        "C": [0.2, 0.1, 0.18, 0.2, 0.07, 0.03, 1.5, 0.2, 0.4],
    }
    values = []
    for symbol in order:
        neuron = ann[symbol].copy()
        if not local:
            neuron[:6] = [0] * 6
        dim = 6 + composition
        values += neuron[:dim] + [0] * dim + [neuron[7], 0, neuron[8], 0]
    values += [0]  # NEP4 shared energy bias
    pair_coefficients = [0.3, 0.2, 0.2, 0.4]  # OO, OC, CO, CC
    if order == "CO":
        pair_coefficients.reverse()
    values += pair_coefficients * 8  # radial and angular basis coefficients
    values += [1] * dim
    tag = "nep4_composition" if composition else "nep4"
    mapping = "composition_descriptor O C\n" if composition else ""
    path.write_text(
        f"{tag} 2 {' '.join(order)}\n{mapping}"
        "cutoff 4 4 64 64\nn_max 1 1\nbasis_size 1 1\nl_max 2 0 0\nANN 2 0\n"
        + "\n".join(map(str, values)) + "\n"
    )


def rows(path):
    return [list(map(float, line.split())) for line in path.read_text().splitlines()]


def static_md(executable, root, name, potential, species, positions, cell):
    target = root / name
    target.mkdir()
    shutil.copy(potential, target / "nep.txt")
    (target / "model.xyz").write_text(xyz(species, positions, cell))
    (target / "run.in").write_text(
        "potential nep.txt\nvelocity 1\nensemble nve\ntime_step 0\n"
        "dump_xyz 1 out.xyz force potential precision double\nrun 1\n"
    )
    invoke(executable, target)
    _, header, *atoms = (target / "out.xyz").read_text().splitlines()
    energy = float(re.search(r"\benergy=(\S+)", header)[1])
    tensor = list(map(float, re.search(r'\bvirial="([^"]+)"', header)[1].split()))
    virial = [tensor[i] for i in (0, 4, 8, 1, 5, 2)]
    forces = [float(value) for atom in atoms for value in atom.split()[4:7]]
    return energy, forces, virial


def predict(executable, root, potential, cases, stream, name="predict"):
    target = root / f"{name}-{stream}"
    target.mkdir()
    shutil.copy(potential, target / "nep.txt")
    (target / "nep.in").write_text(
        NEP_INPUT + f"prediction 1\nstream_train {stream}\noutput_descriptor 2\n"
    )
    for split in ("train", "test"):
        (target / f"{split}.xyz").write_text(
            "".join(xyz(*case, training=True) for case in cases)
        )
    invoke(executable, target)
    return target


def training_paths(nep, root, cases):
    control = root / "training-control"
    control.mkdir()
    frames = "".join(xyz(*case, training=True, energy=i * 0.01) for i, case in enumerate(cases))
    (control / "train.xyz").write_text(frames)
    (control / "test.xyz").write_text(frames)
    (control / "weights.txt").write_text("0/1 1 0\n1/4 1 2\n1/1 0 1\n1/2 1 2\n")
    training_input = (
        NEP_INPUT + "composition_descriptor O C\ncomposition_force_weights weights.txt\n"
        "validation_force_rmse_by_type 1\ngeneration 2\npopulation 10\n"
        "output_interval 1\nsave_potential 1 0 1\n"
    )
    (control / "nep.in").write_text(training_input)
    invoke(nep, control)
    assert (control / "nep.txt").read_text().splitlines()[-1].strip() == "1.0000000e+00"
    assert (control / "nep.restart").read_text().startswith("nep_restart_composition_v1 O C O C ")
    assert (control / "validation_force_rmse_by_type.csv").is_file()

    for stream, specialized in ((0, 0), (1, 0), (0, 1), (1, 1)):
        target = root / f"resume-{stream}-{specialized}"
        shutil.copytree(control, target)
        # The same optimizer state drives both paths. The saved model supplies the scalers.
        (target / "nep.in").write_text(
            training_input + f"stream_train {stream}\nnep_compile {specialized}\n"
        )
        log = invoke(nep, target)
        if specialized:
            assert "Compile specialized NEP training kernels" in log, log
            assert "specialization disabled" not in log, log
    for filename in ("energy_test.out", "force_test.out", "virial_test.out"):
        first = rows(root / "resume-0-0" / filename)
        for suffix in ("1-0", "0-1", "1-1"):
            second = rows(root / f"resume-{suffix}" / filename)
            assert len(first) == len(second)
            for a, b in zip(first, second):
                close(a, b, atol=2e-5, rtol=1e-4)

    bad = root / "bad-restart"
    shutil.copytree(control, bad)
    (bad / "nep.in").write_text(training_input.replace("composition_descriptor O C", "composition_descriptor C O"))
    invoke(nep, bad, "composition_descriptor mapping or activation differs")
    bad_scaler = root / "bad-scaler"
    shutil.copytree(control, bad_scaler)
    model_lines = (bad_scaler / "nep.txt").read_text().splitlines()
    (bad_scaler / "nep.txt").write_text("\n".join(model_lines[:-1] + ["2"]) + "\n")
    invoke(nep, bad_scaler, "composition descriptor scaler must be 1")


def check(nep, gpumd, root, baseline):
    potential = root / "model.txt"
    model(potential)
    pure_water = ("OOOO", GEOMETRY, lattice(9))
    pure_methanol = ("CCCC", GEOMETRY, lattice(9))
    mixed = ("COOO", GEOMETRY, lattice(9))
    small_box_positions = [(x + 4.7, y, z) for x, y, z in GEOMETRY]
    small_box = ("COCO", small_box_positions, lattice(7))
    cases = [pure_water, mixed, pure_methanol, small_box]
    results = [static_md(gpumd, root, f"md-{i}", potential, *case) for i, case in enumerate(cases)]
    for stream in (0, 1):
        target = predict(nep, root, potential, cases, stream)
        features = rows(target / "descriptor.out")
        close([row[-1] for row in features], [x for x in [0, 0.25, 1, 0.5] * 2 for _ in range(4)])
        for split in ("train", "test"):
            e = rows(target / f"energy_{split}.out")
            f = rows(target / f"force_{split}.out")
            w = rows(target / f"virial_{split}.out")
            for i, (energy, forces, virial) in enumerate(results):
                close([energy / 4], e[i][:1])
                close(forces, [v for row in f[i * 4 : (i + 1) * 4] for v in row[:3]])
                close([v / 4 for v in virial], w[i][:6])

    reversed_model = root / "reversed.txt"
    model(reversed_model, order="CO")
    e, f, w = static_md(gpumd, root, "reverse-types", reversed_model, *mixed)
    close([e, *f, *w], [results[1][0], *results[1][1], *results[1][2]])

    def strain(isotropic, h):
        def transform(row):
            x, y, z = row
            return (x * (1 + h), y * (1 + h), z * (1 + h)) if isotropic else (x + h * y, y, z)
        return [transform(row) for row in GEOMETRY], [transform(row) for row in lattice(9)]

    energy, forces, virial = results[1]
    copies = GEOMETRY + [(x + 9, y, z) for x, y, z in GEOMETRY]
    e, f, w = static_md(gpumd, root, "replica", potential, "COOOCOOO", copies, [(18, 0, 0), (0, 9, 0), (0, 0, 9)])
    close([e], [2 * energy])
    close(f, forces * 2)
    close(w, [2 * value for value in virial])

    for h in (0.002, 0.001):
        energies = {key: [] for key in ("coordinate", "dilation", "shear")}
        for delta in (-h, h):
            displaced = GEOMETRY.copy()
            x, y, z = displaced[0]
            displaced[0] = (x + delta, y, z)
            for key, (positions, cell) in (
                ("coordinate", (displaced, lattice(9))),
                ("dilation", strain(True, delta)),
                ("shear", strain(False, delta)),
            ):
                e, _, _ = static_md(gpumd, root, f"{key}-{delta}", potential, "COOO", positions, cell)
                energies[key].append(e)
        derivatives = [(minus - plus) / (2 * h) for minus, plus in energies.values()]
        close(derivatives, [forces[0], sum(virial[:3]), virial[3]], atol=3e-4, rtol=3e-3)

    state_only = root / "state-only.txt"
    model(state_only, local=False)
    for i, case in enumerate(cases):
        species = case[0]
        fraction = species.count("C") / len(species)
        e, f, w = static_md(gpumd, root, f"state-only-{i}", state_only, *case)
        exact = sum(
            0.2 * math.tanh(fraction - 0.1) if bead == "O" else 0.4 * math.tanh(1.5 * fraction - 0.2)
            for bead in species
        )
        close([e], [exact])
        close(f + w, [0] * (len(f) + len(w)))

    training_paths(nep, root, cases)

    if baseline:
        plain = root / "plain.txt"
        model(plain, composition=False)
        old = predict(baseline, root, plain, cases, 1, name="baseline")
        new = predict(nep, root, plain, cases, 1, name="descriptor-off")
        for path in old.glob("*.out"):
            assert path.read_bytes() == (new / path.name).read_bytes(), path.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nep", type=Path)
    parser.add_argument("gpumd", type=Path)
    parser.add_argument("--baseline-nep", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gpumd-composition-") as directory:
        baseline = args.baseline_nep.resolve() if args.baseline_nep else None
        check(args.nep.resolve(), args.gpumd.resolve(), Path(directory), baseline)
    print("Passed composition descriptor checks.")
