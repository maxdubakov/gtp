"""Verify the guitar data pipeline end-to-end.
"""

import argparse
import os

import numpy as np

from gtp import REPO_ROOT
from gtp.data import build_dataset, load_gaps_notes, load_guitarset_notes
from gtp.log import set_verbose

GAPS_DIR = os.path.join(REPO_ROOT, 'data', 'gaps_hf')
GUITARSET_DIR = os.path.join(REPO_ROOT, 'data', 'guitarset')

EXPECTED_KEYS = [
    'waveform',
    'onset_roll', 'offset_roll',
    'reg_onset_roll', 'reg_offset_roll',
    'frame_roll', 'velocity_roll',
    'mask_roll',
]


def print_sample_summary(idx, sample):
    print(f"\n  Sample {idx}:")
    for key in EXPECTED_KEYS:
        arr = sample[key]
        print(f"    {key:20s}  shape={arr.shape}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}")


def verify_shapes_and_ranges(sample, segment_samples, frames_num, classes_num):
    """Assert expected shapes and value ranges. Raises AssertionError on failure."""
    waveform = sample['waveform']
    assert waveform.shape == (segment_samples,), \
        f"waveform shape {waveform.shape} != ({segment_samples},)"

    roll_shape = (frames_num, classes_num)
    for key in ['onset_roll', 'offset_roll', 'frame_roll']:
        arr = sample[key]
        assert arr.shape == roll_shape, f"{key} shape {arr.shape} != {roll_shape}"
        assert arr.min() >= 0.0 and arr.max() <= 1.0, \
            f"{key} out of [0,1]: min={arr.min()}, max={arr.max()}"

    for key in ['reg_onset_roll', 'reg_offset_roll']:
        arr = sample[key]
        assert arr.shape == roll_shape, f"{key} shape {arr.shape} != {roll_shape}"
        assert arr.min() >= 0.0 and arr.max() <= 1.0, \
            f"{key} out of [0,1]: min={arr.min()}, max={arr.max()}"

    vel = sample['velocity_roll']
    assert vel.shape == roll_shape, f"velocity_roll shape {vel.shape} != {roll_shape}"
    assert vel.min() >= 0.0 and vel.max() <= 127.0, \
        f"velocity_roll out of [0,127]: min={vel.min()}, max={vel.max()}"

    mask = sample['mask_roll']
    assert mask.shape == roll_shape, f"mask_roll shape {mask.shape} != {roll_shape}"
    assert set(np.unique(mask)).issubset({0.0, 1.0}), \
        f"mask_roll has non-binary values: {np.unique(mask)}"


def show_verbose_trace(gaps_dir, guitarset_dir):
    """Show detailed trace from raw annotations -> note events -> target rolls."""
    print("\n=== Verbose trace: raw annotation -> note events -> target rolls ===\n")

    # GAPS example
    import csv
    meta_path = os.path.join(gaps_dir, 'gaps_metadata_with_splits.csv')
    first_midi = None
    with open(meta_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'train':
                first_midi = os.path.join(gaps_dir, row['midi_path'])
                break

    if first_midi:
        print(f"--- GAPS: {os.path.basename(first_midi)} ---")
        notes = load_gaps_notes(first_midi)
        print(f"  Total notes: {len(notes)}")
        print("  First 3 note events:")
        for n in sorted(notes, key=lambda x: x['onset_time'])[:3]:
            print(f"    {n}")

    # GuitarSet example
    ann_dir = os.path.join(guitarset_dir, 'annotation')
    first_jams = os.path.join(ann_dir, '05_BN1-129-Eb_comp.jams')
    if not os.path.exists(first_jams):
        first_jams = sorted(
            f for f in os.listdir(ann_dir)
            if f.startswith('05') and f.endswith('.jams')
        )
        if first_jams:
            first_jams = os.path.join(ann_dir, first_jams[0])

    if os.path.exists(first_jams):
        print(f"\n--- GuitarSet: {os.path.basename(first_jams)} ---")
        notes = load_guitarset_notes(first_jams)
        print(f"  Total notes: {len(notes)}")
        print("  First 3 note events:")
        for n in sorted(notes, key=lambda x: x['onset_time'])[:3]:
            print(f"    {n}")

    # Show one sample's rolls
    print("\n--- Target rolls for one segment (verbose trace enabled) ---")
    from gtp.data import TargetProcessor

    tp = TargetProcessor()
    start_time = 0.0

    def _show_roll_stats(source_label, notes):
        rolls = tp.process_notes(start_time, notes)
        print(f"\n  {source_label} segment (start=0s) roll stats:")
        for key, arr in rolls.items():
            nonzero = np.count_nonzero(arr != (1.0 if 'reg' in key else 0.0))
            print(f"    {key:20s}  nonzero/non-default={nonzero:5d}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}")

        active_pitches = np.where(rolls['onset_roll'].sum(axis=0) > 0)[0]
        if len(active_pitches) > 0:
            p = active_pitches[0]
            onset_col = rolls['reg_onset_roll'][:, p]
            event_frames = np.where(rolls['onset_roll'][:, p] > 0)[0]
            if len(event_frames) > 0:
                ef = event_frames[0]
                window = slice(max(0, ef - 5), min(len(onset_col), ef + 6))
                print(f"\n  reg_onset_roll for pitch class {p} around frame {ef}:")
                print(f"    {onset_col[window].round(3)}")

    if first_midi:
        _show_roll_stats("GAPS", load_gaps_notes(first_midi))

    if os.path.exists(first_jams):
        _show_roll_stats("GuitarSet", load_guitarset_notes(first_jams))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable trace logging and show annotation detail')
    parser.add_argument('-n', type=int, default=4,
                        help='Number of samples to load and inspect (default: 4)')
    args = parser.parse_args()

    if args.verbose:
        set_verbose(True)
        show_verbose_trace(GAPS_DIR, GUITARSET_DIR)
        print()

    print("Building training dataset...")
    dataset = build_dataset(GAPS_DIR, GUITARSET_DIR, split='train')

    sample_rate = 16000
    segment_seconds = 10.0
    frames_per_second = 100
    segment_samples = int(segment_seconds * sample_rate)
    frames_num = int(round(segment_seconds * frames_per_second)) + 1
    classes_num = 88

    print(f"\nDataset: {len(dataset)} segments")
    print(f"Expected waveform shape : ({segment_samples},)  = 10s @ 16kHz")
    print(f"Expected roll shape     : ({frames_num}, {classes_num})  = 10s @ 100fps, 88 pitch classes")

    print(f"\nLoading {args.n} sample(s)...")
    for i in range(min(args.n, len(dataset))):
        sample = dataset[i]
        verify_shapes_and_ranges(sample, segment_samples, frames_num, classes_num)
        print_sample_summary(i, sample)

    print(f"\nAll {min(args.n, len(dataset))} samples passed shape/range checks.")

    # Summary across loaded samples
    print("\n--- Key statistics across loaded samples ---")
    samples = [dataset[i] for i in range(min(args.n, len(dataset)))]

    for key in ['onset_roll', 'reg_onset_roll', 'frame_roll', 'velocity_roll', 'mask_roll']:
        vals = np.concatenate([s[key].ravel() for s in samples])
        print(f"  {key:20s}  global min={vals.min():.4f}  max={vals.max():.4f}  "
              f"nonzero={np.count_nonzero(vals)}/{len(vals)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
