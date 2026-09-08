#!/usr/bin/env python3
"""Summarize one or more run_nep_performance.sbatch result directories."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PAIR = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def key_values(line: str) -> dict[str, str]:
    return dict(PAIR.findall(line))


def analyze(run: Path) -> dict[str, object]:
    result: dict[str, object] = {"run": str(run)}
    metadata = key_values((run / "metadata.txt").read_text().replace("\n", " "))
    result.update(metadata)

    if (run / "time.txt").exists():
        result.update(key_values((run / "time.txt").read_text().replace("\n", " ")))

    timing: list[dict[str, str]] = []
    log_text = ""
    if (run / "run.log").exists():
        log_text = (run / "run.log").read_text(errors="replace")
        for line in log_text.splitlines():
            if line.startswith("PERF_TIMING "):
                timing.append(key_values(line))
        for label, output_key in (
            ("initialization", "reported_initialization_s"),
            ("training", "reported_training_s"),
        ):
            match = re.search(rf"Time used for {label} = ([0-9.]+) s", log_text)
            if match:
                result[output_key] = match.group(1)
    training = [row for row in timing if row.get("phase") == "training"]
    if training:
        stable = training[-1]
        result["seconds_per_generation"] = stable.get("seconds_per_generation")
        result["configurations_per_second"] = stable.get("configurations_per_second")
    for phase in ("fitness_initialization", "q_scaler", "first_loss"):
        rows = [row for row in timing if row.get("phase") == phase]
        if rows:
            for key, value in rows[-1].items():
                if key != "phase":
                    result[f"{phase}_{key}"] = value
    if training:
        result["validation_compute_total_s"] = training[-1].get("validation_compute_s")
        result["validation_output_total_s"] = training[-1].get("validation_output_s")

    progress_file = run / "loss-progress.txt"
    if progress_file.exists():
        progress = []
        for line in progress_file.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                progress.append((int(fields[0]), int(fields[1])))
        if progress and "start_epoch" in result and "first_loss_elapsed_s" not in result:
            result["first_loss_elapsed_s"] = str(progress[0][0] - int(result["start_epoch"]))
        if len(progress) >= 2 and "seconds_per_generation" not in result:
            elapsed = progress[-1][0] - progress[-2][0]
            reported_steps = progress[-1][1] - progress[-2][1]
            # Each loss.out line represents OUTPUT_INTERVAL generations. Read
            # that value directly from the generated input.
            input_text = (run / "nep.in").read_text()
            match = re.search(r"^output_interval\s+(\d+)", input_text, re.MULTILINE)
            interval = int(match.group(1)) if match else 1
            generations_between = reported_steps * interval
            seconds_per_generation = elapsed / generations_between
            result["seconds_per_generation"] = f"{seconds_per_generation:.9f}"
            result["configurations_per_second"] = (
                f"{float(result['batch']) / seconds_per_generation:.3f}"
            )
    if "seconds_per_generation" not in result and "reported_training_s" in result:
        seconds_per_generation = (
            float(result["reported_training_s"]) / float(result["generations"])
        )
        result["seconds_per_generation"] = f"{seconds_per_generation:.9f}"
        result["configurations_per_second"] = (
            f"{float(result['batch']) / seconds_per_generation:.3f}"
        )

    samples: list[tuple[float, float]] = []
    sample_file = run / "gpu-samples.csv"
    if sample_file.exists():
        for line in sample_file.read_text(errors="replace").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 5 or "/" not in fields[0]:
                continue
            try:
                samples.append((float(fields[3]), float(fields[4])))
            except ValueError:
                pass
    if samples:
        result["gpu_peak_mib"] = f"{max(memory for _, memory in samples):.0f}"
        resident = [(util, memory) for util, memory in samples if memory > 1000]
        if resident:
            result["gpu_util_resident_percent"] = (
                f"{sum(util for util, _ in resident) / len(resident):.1f}"
            )
            active = [util for util, _ in resident if util > 0]
            result["gpu_util_active_percent"] = (
                f"{sum(active) / len(active):.1f}" if active else "0.0"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = [analyze(run) for run in args.runs]
    columns = [
        "variant",
        "batch",
        "generations",
        "exit_status",
        "elapsed_s",
        "reported_initialization_s",
        "reported_training_s",
        "first_loss_elapsed_s",
        "seconds_per_generation",
        "configurations_per_second",
        "gpu_util_resident_percent",
        "gpu_util_active_percent",
        "gpu_peak_mib",
        "validation_compute_total_s",
        "validation_output_total_s",
        "run",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(column, "")) for column in columns))


if __name__ == "__main__":
    main()
