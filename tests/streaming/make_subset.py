#!/usr/bin/env python3
"""Copy the first N complete XYZ frames without changing their labels."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    copied = 0
    with open(args.source) as source, open(args.output, "w") as output:
        while copied < args.frames:
            count_line = source.readline()
            if not count_line:
                break
            count = int(count_line)
            comment = source.readline()
            atoms = [source.readline() for _ in range(count)]
            if not comment or any(not line for line in atoms):
                raise RuntimeError("truncated XYZ frame")
            output.write(count_line)
            output.write(comment)
            output.writelines(atoms)
            copied += 1
    if copied != args.frames:
        raise RuntimeError(f"requested {args.frames} frames, copied {copied}")


if __name__ == "__main__":
    main()
