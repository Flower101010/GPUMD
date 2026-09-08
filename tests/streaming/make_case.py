#!/usr/bin/env python3
"""Create deterministic E/F/V streaming test data from a committed fixture."""

import argparse
from pathlib import Path


def read_frames(path: Path):
    lines = path.read_text().splitlines()
    frames = []
    offset = 0
    while offset < len(lines):
        count = int(lines[offset])
        frames.append((lines[offset + 1], lines[offset + 2 : offset + 2 + count]))
        offset += count + 2
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--vary-size", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()

    source = args.source or (
        Path(__file__).parents[2] / "tests_pytest/fixtures/training/train.xyz"
    )
    templates = read_frames(source)
    sizes = (5, 10, 20, 40)
    output = []
    for index in range(args.frames):
        header, atoms = templates[index % len(templates)]
        count = sizes[index % len(sizes)] if args.vary_size else len(atoms)
        energy = -10.0 * count + 0.125 * index
        weight = 1.0 + 0.25 * (index % 3)
        energy_weight = 0.75 + 0.125 * (index % 2)
        virial = " ".join(str(0.01 * (index + 1) * value) for value in range(1, 10))
        fields = [field for field in header.split() if not field.startswith("energy=")]
        fields.extend(
            [
                f"energy={energy}",
                f"weight={weight}",
                f"energy_weight={energy_weight}",
                f'virial="{virial}"',
            ]
        )
        output.extend([str(count), " ".join(fields), *atoms[:count]])
    args.output.write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
