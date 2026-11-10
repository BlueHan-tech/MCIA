"""
Build D-035/D-036 masks and preview repetition 6 flags.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.windows_conda_path import ensure_current_env_dll_path

ensure_current_env_dll_path()

import numpy as np

def preview_validation(path, record, quality):
    selected = np.argwhere(quality.flagged & (quality.partitions == 1)[:, None])
    if not len(selected):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from data.waveform import FS
    from utils.waveform_quality import SHORT

    window, channel = selected[0]
    start = quality.starts[window]
    lo, hi = max(0, start - SHORT), min(len(record.raw), start + 2 * SHORT)
    sample = np.arange(lo, hi)
    sample = sample[record.repetitions[sample] == 6]
    time_ms = (sample - start) * 1000 / FS
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, layout='constrained')
    for axis, values, title in zip(axes, (record.raw, record.filtered), ('Raw', 'D-014 bandpass')):
        axis.plot(time_ms, values[sample, channel], linewidth=1)
        axis.axvspan(0, SHORT * 1000 / FS, color='tab:orange', alpha=0.18)
        axis.set_ylabel(title)
    axes[0].set_title(f'Validation repetition 6, electrode {channel + 1}, sample {start}')
    axes[-1].set_xlabel('Time from flagged window start (ms)')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    import argparse
    from utils.waveform_db3 import prepare_db3_exercise, targets, write_quality
    from utils.waveform_pipeline import setup_waveform

    parser = argparse.ArgumentParser(description='Build and preview D-035/D-036 masks')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg, run, loader, _ = setup_waveform(PROJECT_ROOT, smoke=args.smoke)
    figure_dir = run / '02_db3_transfer_completion' / 'figures' / 'quality'
    figure_dir.mkdir(parents=True, exist_ok=True)
    for subject in [5, 6] if args.smoke else targets(cfg):
        exercises = [1] if args.smoke else cfg['exp4_gesture']['exercises']
        for exercise in exercises:
            record, quality = prepare_db3_exercise(cfg, loader, subject, exercise)
            write_quality(run, subject, exercise, record, quality, smoke=args.smoke)
            preview_validation(figure_dir / f'S{subject:02d}_E{exercise}_validation.png', record, quality)
            flags = int(quality.flagged[quality.partitions == 1].sum())
            print(f'S{subject:02d} E{exercise}: {flags} validation electrode-windows flagged', flush=True)


if __name__ == "__main__":
    main()
