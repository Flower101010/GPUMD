"""NPT engineering diagnostics; no NVE conservation or pressure-variance target."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def block_summary(values):
    """Report several block sizes so correlation cannot hide behind one SEM."""
    values = np.asarray(values)
    estimates = {}
    for count in (4, 8, 20):
        if len(values) >= count:
            means = np.array([chunk.mean() for chunk in np.array_split(values, count)])
            estimates[str(count)] = dict(means=means.tolist(),
                                        standard_error=float(means.std(ddof=1) / np.sqrt(count)))
    quarter = max(1, len(values) // 4)
    return dict(mean=float(values.mean()), temporal_std=float(values.std(ddof=1)) if len(values) > 1 else None,
                first_last_quarter_difference=float(values[-quarter:].mean() - values[:quarter].mean()),
                blocks=estimates,
                conservative_block_standard_error=max((e['standard_error'] for e in estimates.values()), default=None))


def assess_pressure(folder, config, data, times_fs, mass):
    pressure = data[:, 3:6].mean(axis=1)  # GPUMD positive-compression stress, GPa.
    cells = data[:, 9:18].reshape(-1, 3, 3)
    volume = np.linalg.det(cells)
    if not np.all(volume > 0):
        raise ValueError('Nonpositive NPT volume')
    if not np.allclose(cells, np.eye(3)[None, :, :] * np.diagonal(cells, axis1=1, axis2=2)[:, None, :], atol=1e-9, rtol=0):
        raise ValueError('Expected orthorhombic NPT boxes')
    scale = np.diagonal(cells, axis1=1, axis2=2) / config['box_lengths_angstrom']
    if not np.allclose(scale, scale[:, :1], atol=1e-9, rtol=1e-8):
        raise ValueError('Expected isotropic NPT box scaling')
    cutoff = config['equilibration_steps'] * config['dt_fs']
    mask = times_fs > cutoff
    if mask.sum() < 2:
        raise ValueError('Need at least two production thermo samples')
    p, v = block_summary(pressure[mask]), block_summary(volume[mask])
    tolerance = config['pressure_tolerance_GPa']
    tau = config['pressure_tau_fs']
    duration = config['steps'] * config['dt_fs'] - cutoff
    checks = {
        'Piston equilibration at least 10 tau': cutoff >= 10 * tau,
        'Piston production at least 100 tau': duration >= 100 * tau,
        'Pressure production samples at least 1000': mask.sum() >= 1000,
        'Pressure blocks span at least 5 piston tau': duration / 20 >= 5 * tau and mask.sum() >= 20,
        'Mean pressure at target (absolute tolerance)': abs(p['mean'] - config['target_pressure_GPa']) <= tolerance,
        'Pressure block uncertainty resolved': p['conservative_block_standard_error'] is not None and p['conservative_block_standard_error'] <= tolerance / 2,
        'No unresolved pressure drift': abs(p['first_last_quarter_difference']) <= tolerance,
        'No large production volume drift': abs(v['first_last_quarter_difference']) / v['mean'] <= .05,
        'Volume block uncertainty resolved': v['conservative_block_standard_error'] is not None and v['conservative_block_standard_error'] / v['mean'] <= .02,
        'Box responds to piston': np.ptp(volume) / volume.mean() > 1e-8,
    }
    density = mass.sum() * 1.66053906660 / volume  # amu / A^3 -> g/cm^3.
    np.savetxt(folder / 'npt_trace.csv', np.column_stack((times_fs / 1000, data[:, 0], pressure,
        volume, density, data[:, 1], data[:, 2], data[:, 3:6])), delimiter=',', comments='',
        header='time_ps,temperature_K,pressure_GPa,volume_A3,density_g_cm3,kinetic_energy_eV,potential_energy_eV,Pxx_GPa,Pyy_GPa,Pzz_GPa', fmt='%.14e')
    print(f"Production mean P={p['mean']:.6g} GPa; target={config['target_pressure_GPa']:g} GPa")
    print(f"Production mean V={v['mean']:.6g} A^3; pressure temporal SD={p['temporal_std']:.6g} GPa")
    return dict(pressure_GPa=p, volume_A3=v, target_GPa=config['target_pressure_GPa'],
                tolerance_GPa=tolerance, production_samples=int(mask.sum()),
                production_piston_tau=duration / tau, equilibration_piston_tau=cutoff / tau,
                uncertainty_scope='Observed block errors at 4/8/20 blocks, not independent-sample confidence intervals',
                checks={name: bool(passed) for name, passed in checks.items()})


def plot_npt(folder, config, *, start_ps=None, end_ps=None, output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    trace = np.atleast_1d(np.genfromtxt(folder / 'npt_trace.csv', delimiter=',', names=True))
    start = config['equilibration_steps'] * config['dt_fs'] / 1000 if start_ps is None else start_ps
    end = float(trace['time_ps'][-1]) if end_ps is None else end_ps
    t = trace['time_ps']
    if not np.isfinite(start) or not np.isfinite(end) or not 0 <= start < end <= t[-1] + 1e-8:
        raise ValueError('Require 0 <= start < end <= final sample time, in ps')
    mask = (t > start) & (t <= end + 1e-10)
    if mask.sum() < 2:
        raise ValueError('Need at least two samples in the requested production window')
    output = folder if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ('temperature_K', 'pressure_GPa', 'volume_A3', 'density_g_cm3')
    labels = ('Temperature (K)', 'Pressure (GPa)', 'Volume (A$^3$)', 'Density (g/cm$^3$)')
    targets = (config['target_temperature_kelvin'], config['target_pressure_GPa'], None, None)
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), layout='constrained')
    for row, (field, label, target) in enumerate(zip(fields, labels, targets)):
        for col, shown in enumerate((t <= end + 1e-10, mask)):
            ax = axes[row, col]
            ax.plot(t[shown], trace[field][shown], lw=.65, label='Instantaneous (no smoothing)')
            if target is not None:
                ax.axhline(target, color='black', ls='--', label=f'Target {target:g}')
            ax.axhline(trace[field][mask].mean(), color='#c55a11', ls=':', label='Production mean')
            if col == 0:
                ax.axvline(start, color='grey', ls=':', label='Selected production start')
            ax.set(xlabel='Time (ps)', ylabel=label,
                   title='Full trajectory' if col == 0 else f'Selected production: {start:g}-{end:g} ps')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle('Rigid NPT: relaxation and production diagnostics')
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'npt_diagnostics.{suffix}', dpi=180)
    plt.close(fig)
    # Production-only axes exclude the initial transient and use their own limits.
    zoom = max(start, end - min(5., (end - start) / 4))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout='constrained')
    for row, (field, label, target) in enumerate(zip(fields[:2], labels[:2], targets[:2])):
        for col, beginning in enumerate((start, zoom)):
            shown = (t > beginning) & (t <= end + 1e-10)
            values = trace[field][shown]
            ax = axes[row, col]
            ax.plot(t[shown], values, lw=.75, label='Instantaneous (no smoothing)')
            ax.axhline(target, color='black', ls='--', label=f'Target {target:g}')
            ax.axhline(trace[field][mask].mean(), color='#c55a11', ls=':', label='Production mean')
            ax.set(xlabel='Time (ps)', ylabel=label,
                   title=f'{"Production" if col == 0 else "Final detail"}: {beginning:g}-{end:g} ps')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Rigid NPT: unsmoothed fluctuations in the selected window')
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'production_fluctuations.{suffix}', dpi=180)
    plt.close(fig)
    plot_volume_history(output, trace, start, end)
    return output


def plot_volume_history(output, trace, start, end):
    """Separate relaxation, selected production and final unsmoothed volume detail."""
    import matplotlib.pyplot as plt
    t, volume = trace['time_ps'], trace['volume_A3']
    production = (t > start) & (t <= end + 1e-10)
    mean = float(volume[production].mean())
    detail_start = max(start, end - min(5., (end - start) / 4))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), layout='constrained')
    windows = ((0., 'Full trajectory'), (start, 'Selected production'),
               (detail_start, 'Final-window detail'))
    for axis, (beginning, title) in zip(axes, windows):
        mask = (t > beginning) & (t <= end + 1e-10)
        axis.plot(t[mask], volume[mask], lw=.75, color='#2469a0',
                  label='Instantaneous volume (no smoothing)')
        guide = float(volume[mask].mean()) if beginning == detail_start else mean
        guide_label = 'Shown-window mean' if beginning == detail_start else 'Selected production mean'
        axis.axhline(guide, color='#c55a11', ls=':',
                     label=f'{guide_label}: {guide:.1f} A^3')
        axis.set(xlabel='Time (ps)', ylabel=r'Volume ($\mathrm{\AA}^3$)',
                 title=f'{title}: {beginning:g}-{end:g} ps')
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0].axvspan(0, start, color='grey', alpha=.15)
    axes[0].axvline(start, color='grey', ls='--', label='Selected production start')
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, end)
    fig.suptitle('Rigid NPT volume history | selected window does not certify equilibrium')
    for suffix in ('png', 'pdf'):
        fig.savefig(Path(output) / f'volume_history.{suffix}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Replot existing NPT data without running MD or changing acceptance')
    parser.add_argument('folder', type=Path)
    parser.add_argument('--start-ps', type=float)
    parser.add_argument('--end-ps', type=float)
    args = parser.parse_args()
    config = json.loads((args.folder / 'manifest.json').read_text())
    if config['ensemble'] != 'rigid_npt':
        parser.error('Expected rigid_npt manifest')
    start = args.start_ps if args.start_ps is not None else config['equilibration_steps'] * config['dt_fs'] / 1000
    end = args.end_ps if args.end_ps is not None else config['steps'] * config['dt_fs'] / 1000
    output = args.folder / 'analysis' / f'window_{start:g}_{end:g}ps'
    try:
        plot_npt(args.folder, config, start_ps=start, end_ps=end, output=output)
    except ValueError as exc:
        parser.error(str(exc))
    (output / 'window.json').write_text(json.dumps(dict(start_ps=start, end_ps=end,
        scope='Plot only; original acceptance unchanged'), indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
