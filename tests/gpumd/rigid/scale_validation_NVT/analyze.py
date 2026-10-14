"""Plot/reassess existing NVT results without MD or overwriting original acceptance reports."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from nvt_statistics import assess_temperature


def select_window(times_ps, start_ps, end_ps):
    times = np.asarray(times_ps)
    if not np.isfinite(start_ps) or not np.isfinite(end_ps) or not 0 <= start_ps < end_ps:
        raise ValueError('Require 0 <= start < end, in ps')
    if len(times) == 0 or end_ps > times[-1] + 1e-8:
        raise ValueError('Requested window exceeds the existing trajectory')
    return (times > start_ps) & (times <= end_ps + 1e-10)


def summarize_windows(time, temperature, starts, end_ps):
    rows = []
    for start in starts:
        mask = select_window(time, start, end_ps)
        values = temperature[mask]
        if len(values) < 2:
            continue
        quarter = max(1, len(values) // 4)
        rows.append(dict(start_ps=float(start), end_ps=float(end_ps), samples=len(values),
                         mean_K=float(np.mean(values)), std_K=float(np.std(values, ddof=1)),
                         first_quarter_K=float(np.mean(values[:quarter])),
                         last_quarter_K=float(np.mean(values[-quarter:])),
                         slope_K_per_ps=float(np.polyfit(time[mask], values, 1)[0])))
    return rows


def draw(folder, trace, partitions, config, start, end, windows, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    t, temperature = trace['time_ps'], trace['temperature_K']
    pt = partitions['time_fs'] / 1000
    target = config['target_temperature_kelvin']
    final = t[-1]
    zoom_start = max(start, end - 0.2 * final)
    mask = select_window(t, zoom_start, end)
    pmask = select_window(pt, zoom_start, end)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout='constrained')
    full, tail, parts_full, parts_tail, energy_axis, windows_axis = axes.ravel()
    full.plot(t, temperature, lw=1, label='Instantaneous total T')
    full.set_title('Full trajectory: thermal relaxation')
    tail.plot(t[mask], temperature[mask], lw=1, label='Instantaneous total T')
    tail.set_title(f'Tail detail: {zoom_start:g}-{end:g} ps')
    width = max(1, len(temperature) // 100)
    if len(temperature) >= width:
        smoothed = np.convolve(temperature, np.ones(width) / width, mode='valid')
        center_time = np.convolve(t, np.ones(width) / width, mode='valid')
        full.plot(center_time, smoothed, lw=1.5, label=f'{width * config["thermo_interval"] * config["dt_fs"] / 1000:g} ps moving mean')
    for axis in (full, tail, parts_full, parts_tail):
        axis.axhline(target, color='black', ls='--', lw=1, label='Target')
        axis.axhspan(.95 * target, 1.05 * target, color='green', alpha=.12, label='Target +/-5% guide')
        axis.set_ylabel('Temperature (K)')
    for key, label in [('T_free_K', 'Free particles'), ('T_body_translation_K', 'Body translation'),
                       ('T_body_rotation_K', 'Body rotation')]:
        parts_full.plot(pt, partitions[key], label=label, lw=1)
        parts_tail.plot(pt[pmask], partitions[key][pmask], label=label, marker='.', ms=3)
    parts_full.set_title('Kinetic partitions: full trajectory')
    parts_tail.set_title('Kinetic partitions: tail detail')
    # Include the target band even when the whole tail remains above it.
    for axis, values in [(tail, temperature[mask]),
                         (parts_tail, np.concatenate([partitions[key][pmask]
                          for key in ('T_free_K', 'T_body_translation_K', 'T_body_rotation_K')]))]:
        if len(values):
            low, high = min(.95 * target, float(np.min(values))), max(1.05 * target, float(np.max(values)))
            padding = max(5, .08 * (high - low))
            axis.set_ylim(low - padding, high + padding)
    for key, label in [('kinetic_energy_eV', 'Kinetic K'), ('potential_energy_eV', 'Potential U'),
                       ('total_energy_eV', 'Physical K+U')]:
        values = trace[key]
        energy_axis.plot(t, (values - values[0]) / config['atoms'], label=label)
    energy_axis.set_title('Energy exchange / structural relaxation (diagnostic)')
    energy_axis.set_ylabel('Change from first output (eV / bead)')
    windows_axis.errorbar([r['start_ps'] for r in windows], [r['mean_K'] for r in windows],
                         yerr=[r['std_K'] for r in windows], fmt='o-', capsize=3,
                         label='Window mean +/- temporal std')
    windows_axis.axhline(target, color='black', ls='--', label='Target')
    windows_axis.axhspan(.95 * target, 1.05 * target, color='green', alpha=.12)
    windows_axis.set_title(f'Windows ending at {end:g} ps: no automatic selection')
    windows_axis.set_xlabel('Discard data before (ps)')
    windows_axis.set_ylabel('Window temperature (K)')
    for axis in (full, parts_full, energy_axis):
        axis.axvline(start, color='grey', ls=':', lw=1, label='Selected statistics start')
    for axis in axes.ravel():
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc='best')
        if axis is not windows_axis:
            axis.set_xlabel('Time (ps)')
    verdict = 'PASS' if all(stats['checks'].values()) else 'NOT EQUILIBRATED / INSUFFICIENT SAMPLING'
    fig.suptitle(f'Mixed rigid NVT | target {target:g} K | selected {start:g}-{end:g} ps: {verdict}', fontsize=12)
    fig.savefig(folder / 'diagnostics.png', dpi=170)
    fig.savefig(folder / 'diagnostics.pdf')
    plt.close(fig)
    draw_production_temperature(folder, trace, config, start, end)


def draw_production_temperature(folder, trace, config, start, end):
    """Show unsmoothed production fluctuations independently of the hot transient."""
    import matplotlib.pyplot as plt
    time, temperature = trace['time_ps'], trace['temperature_K']
    mask = select_window(time, start, end)
    t, values = time[mask], temperature[mask]
    target = config['target_temperature_kelvin']
    mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
    low, high = min(float(values.min()), target), max(float(values.max()), target)
    padding = max(1., .12 * (high - low))
    zoom_start = max(start, end - min(5., (end - start) / 4))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    for axis, cutoff, title in zip(axes, (start, zoom_start),
                                    ('Production window', 'Final-window detail')):
        shown = select_window(t, cutoff, end)
        axis.plot(t[shown], values[shown], color='#2469a0', lw=.85,
                  label='Instantaneous temperature (no smoothing)')
        axis.axhline(target, color='black', ls='--', lw=1, label=f'Target: {target:g} K')
        axis.axhline(mean, color='#c55a11', ls=':', lw=1.2, label=f'Production mean: {mean:.2f} K')
        axis.set(xlabel='Time (ps)', ylabel='Temperature (K)',
                 title=f'{title}: {cutoff:g}-{end:g} ps', ylim=(low - padding, high + padding))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Rigid NVT production temperature | mean {mean:.2f} K | temporal SD {std:.2f} K', fontsize=12)
    fig.savefig(folder / 'production_temperature.png', dpi=180)
    fig.savefig(folder / 'production_temperature.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--start-ps', type=float, help='Explicit production start; default from original manifest')
    parser.add_argument('--end-ps', type=float, help='Explicit production end; default last sample')
    args = parser.parse_args()
    source = args.folder.resolve()
    config = json.loads((source / 'manifest.json').read_text())
    trace = np.atleast_1d(np.genfromtxt(source / 'nvt_trace.csv', delimiter=',', names=True))
    partitions = np.atleast_1d(np.genfromtxt(source / 'kinetic_partitions.csv', delimiter=',', names=True))
    start = args.start_ps if args.start_ps is not None else config['equilibration_steps'] * config['dt_fs'] / 1000
    end = args.end_ps if args.end_ps is not None else float(trace['time_ps'][-1])
    try:
        mask = select_window(trace['time_ps'], start, end)
        pmask = select_window(partitions['time_fs'] / 1000, start, end)
        stats = assess_temperature(
            trace['temperature_K'][mask], np.column_stack([partitions[key][pmask]
                for key in ('T_free_K', 'T_body_translation_K', 'T_body_rotation_K')]),
            target=config['target_temperature_kelvin'], dof=config['thermal_dof'],
            equilibration_fs=start * 1000, production_fs=(end - start) * 1000,
            tau_fs=config['tau_fs'])
    except ValueError as error:
        parser.error(str(error))
    starts = sorted(set([0., start] + [end * f for f in (.2, .4, .6, .8, .9)]))
    windows = summarize_windows(trace['time_ps'], trace['temperature_K'], starts, end)
    output = source / 'analysis' / f'window_{start:g}_{end:g}ps'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'window_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=windows[0].keys())
        writer.writeheader()
        writer.writerows(windows)
    report = {'source': str(source), 'start_ps': start, 'end_ps': end,
              'temperature_statistics': stats, 'diagnostic_windows': windows,
              'temperature_criteria_passed': all(stats['checks'].values()),
              'scope': 'offline_temperature_reassessment_only; original acceptance unchanged',
              'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in ('manifest.json', 'nvt_trace.csv', 'kinetic_partitions.csv')}}
    (output / 'temperature_analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    draw(output, trace, partitions, config, start, end, windows, stats)
    print('Window (ps) | mean T (K) | std (K) | slope (K/ps)')
    for row in windows:
        print(f'{row["start_ps"]:g}-{row["end_ps"]:g} | {row["mean_K"]:.3f} | {row["std_K"]:.3f} | {row["slope_K_per_ps"]:+.3f}')
    for name, passed in stats['checks'].items():
        print(f'{name}: {"PASS" if passed else "FAIL"}')
    print(f'Offline analysis saved to {output}; original validation.json and summary.json unchanged.')
    # Generating diagnostic artifacts is successful even if the trajectory is not equilibrated.


if __name__ == '__main__':
    main()
