"""Audit Cartesian rigid distances offline, using each frame's actual periodic box.

No MD is launched and original acceptance reports are not modified. A snapshot
copy allows inspection of completed frames from an ongoing append-only dump.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import tempfile
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scale_validation_NVT'))
from scale_common import read_initial_xyz, body_ids, iter_xyz, mic, rigid_distance_tolerance

PAIRS = ((0, 1), (0, 2), (1, 2))


def distances(pos, ids, lengths):
    return np.column_stack([np.linalg.norm(mic(pos[ids[:, a]] - pos[ids[:, b]], lengths), axis=1)
                            for a, b in PAIRS])


def precise_distance(pos, i, j, lengths):
    """50-digit distance of the parsed float64 coordinates, for the worst pair."""
    with localcontext() as context:
        context.prec = 50
        dr = []
        for k in range(3):
            length = Decimal.from_float(float(lengths[k]))
            value = Decimal.from_float(float(pos[i, k])) - Decimal.from_float(float(pos[j, k]))
            value -= (value / length).to_integral_value(rounding=ROUND_HALF_EVEN) * length
            dr.append(value)
        return sum(value * value for value in dr).sqrt()


def audit(folder, *, allow_partial=False, output=None):
    folder = Path(folder).resolve()
    config = json.loads((folder / 'manifest.json').read_text())
    if config['ensemble'] != 'rigid_npt':
        raise ValueError('Expected a rigid_npt run')
    lattice, _, initial, _, _, groups = read_initial_xyz(folder / 'model.xyz')
    lengths0 = np.diag(lattice)
    if not np.allclose(lattice, np.diag(lengths0), atol=1e-9, rtol=0) or np.any(lengths0 <= 0):
        raise ValueError('Expected a positive orthorhombic initial box')
    ids, _ = body_ids(groups)
    reference = distances(initial, ids, lengths0)
    rows = []
    worst = None
    maximum = -1.
    max_box = float(lengths0.max())
    incomplete_tail = False
    with tempfile.TemporaryDirectory(prefix='gpumd-rigidity-audit-') as tmp:
        snapshot = Path(tmp) / 'trajectory.xyz'
        shutil.copyfile(folder / 'trajectory.xyz', snapshot)
        try:
            for frame in iter_xyz(snapshot):
                lengths = np.diag(frame.lattice)
                if not np.array_equal(frame.group, groups):
                    raise ValueError('Frame membership changed')
                if (np.any(lengths <= 0) or not np.allclose(frame.lattice, np.diag(lengths), atol=1e-9, rtol=0)
                        or not np.all(np.isfinite(frame.pos))):
                    raise ValueError('Invalid Cartesian positions or box')
                actual = distances(frame.pos, ids, lengths)
                errors = actual - reference
                ratios = actual / reference
                box_scale = float((np.prod(lengths) / np.prod(lengths0)) ** (1 / 3))
                max_box = max(max_box, float(lengths.max()))
                peak = float(np.max(abs(errors)))
                if peak > maximum:
                    maximum = peak
                    b, pair = np.unravel_index(np.argmax(abs(errors)), errors.shape)
                    a, c = PAIRS[pair]
                    worst = dict(time_ps=frame.time_fs / 1000, body_label=int(groups[ids[b, a]]),
                                 atom_ids_zero_based=[int(ids[b, a]), int(ids[b, c])],
                                 reference_distance_A=float(reference[b, pair]), distance_A=float(actual[b, pair]))
                    with localcontext() as context:
                        context.prec = 50
                        exact_change = precise_distance(frame.pos, ids[b, a], ids[b, c], lengths) - precise_distance(initial, ids[b, a], ids[b, c], lengths0)
                    worst['distance_change_recomputed_50digits_A'] = str(exact_change)
                    worst['precision_scope'] = 'High-precision norm of parsed coordinates; does not undo MD or output rounding'
                rows.append([frame.time_fs / 1000, peak, float(errors.mean()),
                             float(ratios.min()), float(ratios.max()), box_scale])
        except ValueError as exc:
            if allow_partial and str(exc) == 'Unexpected end of frame':
                incomplete_tail = True
            else:
                raise
    if not rows:
        raise ValueError('No complete frames available')
    rows = np.asarray(rows)
    expected_frames = config['steps'] // config['xyz_interval']
    expected_ps = config['steps'] * config['dt_fs'] / 1000
    complete = (not incomplete_tail and len(rows) == expected_frames
                and abs(rows[-1, 0] - expected_ps) <= 1e-7)
    if not complete and not allow_partial:
        raise ValueError('Run is incomplete; use --allow-partial to inspect available frames explicitly')
    # A coordinate-scale reference, not a rigorous floating-point error bound.
    roundoff_reference = 64 * np.finfo(float).eps * max_box
    limit = rigid_distance_tolerance(max_box)
    report = dict(source=str(folder), scope='Offline Cartesian-distance audit of saved frames only',
        run_complete=bool(complete), incomplete_snapshot_tail=incomplete_tail,
        frames=len(rows), expected_frames=expected_frames, last_time_ps=float(rows[-1, 0]),
        max_distance_change_A=maximum, max_relative_distance_change=float(max(abs(rows[:, 3:5] - 1).ravel())),
        signed_mean_error_range_A=[float(rows[:, 2].min()), float(rows[:, 2].max())],
        body_distance_ratio_range=[float(rows[:, 3].min()), float(rows[:, 4].max())],
        box_length_scale_range=[float(rows[:, 5].min()), float(rows[:, 5].max())],
        coordinate_roundoff_reference_A=roundoff_reference,
        roundoff_reference_scope='64 * float64 epsilon * maximum box length; scale guide, not strict bound',
        engineering_distance_limit_A=limit, saved_frame_distance_check_passed=bool(maximum < limit),
        worst_pair=worst)
    output = Path(output) if output is not None else folder / 'analysis' / 'rigidity_audit'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'rigidity_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    np.savetxt(output / 'rigidity_trace.csv', rows, delimiter=',', comments='', fmt='%.16e',
               header='time_ps,max_abs_distance_change_A,mean_signed_distance_change_A,min_distance_ratio,max_distance_ratio,box_length_scale')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.folder, allow_partial=args.allow_partial, output=args.output)
    raise SystemExit(0 if result['saved_frame_distance_check_passed'] else 1)
