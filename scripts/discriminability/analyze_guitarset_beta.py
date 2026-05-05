"""Run the β estimator on every annotated note in GuitarSet; check physics.

For each note in the processed JSON files, extract a 40 ms segment from the
hex-pickup channel corresponding to that note's labeled string, then run the
inharmonic-summation β estimator on it.

Aggregate by (string, fret) class and check two physics laws from Barbancho
et al. 2012:

  (1) Within a string, β scales with fret: β(s, n) = β(s, 0) · 2^(n/6).
      → log(β) is linear in fret with slope ln(2)/6 ≈ 0.1155.

  (2) Across strings at equal pitch, β depends on string thickness (d⁴) and
      length²: thicker strings → larger β. Compare β at equal-pitch positions
      to verify physical ordering.

Conventions:
  - GuitarSet JAMS / our processed JSON: string 1 = high E (MIDI 64),
    string 6 = low E (MIDI 40).
  - Hex channel order: channel 0 = low E ... channel 5 = high E
    (so channel = 6 - string).
  - Our sim arrays (physical_model.py): row 0 = low E ... row 5 = high E
    (so sim_row = 6 - string, matching MATLAB's w0Model).

Usage:
  ./venv/bin/python scripts/discriminability/analyze_guitarset_beta.py
  ./venv/bin/python scripts/discriminability/analyze_guitarset_beta.py --limit-pieces 20
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from inharmonicity import (
    apply_gaussian_window,
    compute_fft,
    hilbert_transform,
    inharmonic_summation,
)
from physical_model import (
    bayesian_classify,
    compute_class_priors,
    obtain_pitch_candidates,
    simulate_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GUITARSET_ROOT = REPO_ROOT / 'data' / 'guitarset'
PROC_DIR = GUITARSET_ROOT / 'processed'
OUTPUT_DIR = REPO_ROOT / 'results' / 'discriminability'

# Audio sources: subdirectory under data/guitarset/, filename suffix to add to
# the JSON's stem. 'hex' is the per-string clean signal (6-channel); 'mic' is
# the mixed mono microphone recording.
AUDIO_SOURCES = {
    'hex': ('audio_hex-pickup_debleeded', '_hex_cln.wav'),
    'mic': ('audio_mono-mic', '_mic.wav'),
}

SR = 44100
DEFAULT_SEGMENT_MS = 40   # MATLAB default. Longer windows give finer frequency
                           # resolution (smaller spectral mainlobe), at the cost
                           # of more polyphonic interference and dropping shorter
                           # notes from the analysis.
N_FFT = 2**19
M = 25
BETA_RES = 1e-5


def segment_len(segment_dur_s: float, sr: int = SR) -> int:
    """Match MATLAB: segment is floor(dur*fs)+2 samples."""
    return int(np.floor(segment_dur_s * sr)) + 2


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def estimate_beta_for_labeled_note(
    audio_channel: np.ndarray,
    fs: int,
    onset_s: float,
    pitch_midi: int,
    sim_row: int,
    fret: int,
    sim: dict,
    segment_dur_s: float = DEFAULT_SEGMENT_MS / 1000,
) -> tuple[float, float]:
    """Estimate β assuming the (string, fret) is known from the label.

    Skips harmonic_summation (we already know f₀ from the MIDI label) and
    uses only the labeled class's β grid (skips candidate enumeration). The
    inharmonic-summation 2D grid still searches f₀ within ±1 Hz to absorb
    minor detuning.

    Returns (β, f₀_estimated). Either may be NaN if the segment is too
    short or the labeled class has an empty β grid.
    """
    if len(audio_channel) < 32:
        return float('nan'), float('nan')

    seg_len = segment_len(segment_dur_s, fs)
    onset_sample = int(np.floor(onset_s * fs))
    seg = audio_channel[onset_sample : onset_sample + seg_len]
    if len(seg) < seg_len // 2:
        return float('nan'), float('nan')

    x = apply_gaussian_window(seg)
    x = hilbert_transform(x)
    _, X = compute_fft(x, fs, N_FFT)

    f0_label = midi_to_hz(pitch_midi)

    b_min = sim['beta_min'][sim_row, fret]
    b_max = sim['beta_max'][sim_row, fret]
    n_steps = int(np.floor((b_max - b_min) / BETA_RES + 1e-12)) if b_max > b_min else 0
    beta_grid = b_min + np.arange(n_steps + 1) * BETA_RES

    f0_est, beta_est, _ = inharmonic_summation(X, f0_label, M, fs, beta_grid, N_FFT)
    return beta_est, f0_est


def classify_note(
    audio_channel: np.ndarray,
    fs: int,
    onset_s: float,
    pitch_midi: int,
    sim: dict,
    segment_dur_s: float = DEFAULT_SEGMENT_MS / 1000,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> tuple:
    """Predict (string, fret) for a note via Hjerrild's candidate-driven β classification.

    Steps (mirrors recreate_plucking_experiment_WASPAA19.m's main loop):
      1. Take a `segment_dur_s` window after the onset, apply Gaussian window,
         Hilbert transform, FFT.
      2. Find equal-pitch (string, fret) candidates from the labeled pitch.
      3. For each candidate, run inharmonic_summation with that class's β grid;
         record max cost.
      4. Pick the candidate with the highest cost.

    Pitch search is anchored at the labeled MIDI pitch (we don't redo
    harmonic_summation here — pitch is given by JAMS, so we focus this analysis
    purely on the (string, fret) discrimination by β).

    Returns (sim_row_pred, fret_pred, beta_pred, f0_pred, candidates_costs).
    `candidates_costs` is a list of dicts:
      {'sim_row', 'fret', 'beta', 'f0', 'cost'}
    sorted by cost descending — useful for understanding margin and confidence.
    Returns (None, None, None, None, []) if the segment is too short or no
    candidates exist.
    """
    if len(audio_channel) < 32:
        return None, None, None, None, []

    seg_len = segment_len(segment_dur_s, fs)
    onset_sample = int(np.floor(onset_s * fs))
    seg = audio_channel[onset_sample : onset_sample + seg_len]
    if len(seg) < seg_len // 2:
        return None, None, None, None, []

    x = apply_gaussian_window(seg)
    x = hilbert_transform(x)
    _, X = compute_fft(x, fs, N_FFT)

    f0_label = midi_to_hz(pitch_midi)
    candidates = obtain_pitch_candidates(f0_label, sim['f0_mean'])
    if not candidates:
        return None, None, None, None, []

    rows = []
    for s_idx, f_idx in candidates:
        b_min = sim['beta_min'][s_idx, f_idx]
        b_max = sim['beta_max'][s_idx, f_idx]
        n_steps = int(np.floor((b_max - b_min) / BETA_RES + 1e-12)) if b_max > b_min else 0
        beta_grid = b_min + np.arange(n_steps + 1) * BETA_RES
        f0_est, beta_est, cost = inharmonic_summation(X, f0_label, M, fs, beta_grid, N_FFT)
        rows.append(
            {
                'sim_row': s_idx,
                'fret': f_idx,
                'beta': float(beta_est),
                'f0': float(f0_est),
                'cost': float(cost),
            }
        )
    rows.sort(key=lambda r: -r['cost'])
    best = rows[0]

    # If Bayesian priors are provided, run MATLAB step 2: phi = (f0, beta) from
    # the max-cost candidate, then apply Gaussian discriminant over candidates.
    # The (β, f0) values keep coming from the max-cost candidate — only the
    # (string, fret) gets re-routed by the classifier.
    if mu is not None and sigma is not None:
        phi = np.array([best['f0'], best['beta']])
        s_bayes, f_bayes = bayesian_classify(phi, candidates, mu, sigma)
        return s_bayes, f_bayes, best['beta'], best['f0'], rows

    return best['sim_row'], best['fret'], best['beta'], best['f0'], rows


def process_piece(json_path: Path, sim: dict, audio_source: str,
                  mode: str = 'extract',
                  segment_dur_s: float = DEFAULT_SEGMENT_MS / 1000,
                  mu: np.ndarray | None = None,
                  sigma: np.ndarray | None = None) -> list[dict]:
    """Estimate β for every well-formed note in the piece. Returns list of records.

    For `audio_source='hex'`: load the 6-channel hex-pickup file and pick the
    labeled string's channel. For `audio_source='mic'`: load the mono mic
    recording and use it directly (polyphonic content is the price of
    realistic conditions).
    """
    audio_dir, suffix = AUDIO_SOURCES[audio_source]
    audio_path = GUITARSET_ROOT / audio_dir / f'{json_path.stem}{suffix}'
    if not audio_path.exists():
        return []
    piece = json.loads(json_path.read_text())

    if audio_source == 'hex':
        audio, sr = sf.read(str(audio_path), always_2d=True)
    else:
        audio_1d, sr = sf.read(str(audio_path))
        if audio_1d.ndim > 1:
            audio_1d = audio_1d.mean(axis=1)
        audio = audio_1d  # 1-D

    if sr != SR:
        return []  # skip pieces at unexpected sample rate

    out = []
    for note in piece['notes']:
        s = int(note['string'])
        f = int(note['fret'])
        pitch = int(note['pitch'])
        # Note must be at least 1.05x the analysis window (small buffer for the
        # decay tail staying above silence at the end of the window).
        if note['end'] - note['start'] < segment_dur_s * 1.05:
            continue
        if not (1 <= s <= 6) or not (0 <= f <= 12):
            continue
        sim_row = 6 - s  # JAMS string -> sim row (string 6 = low E -> row 0)

        if audio_source == 'hex':
            channel = 6 - s  # JAMS string -> hex channel (same mapping)
            if channel < 0 or channel >= audio.shape[1]:
                continue
            signal = audio[:, channel]
        else:
            signal = audio  # mono — same signal for every note

        if mode == 'extract':
            beta, f0 = estimate_beta_for_labeled_note(
                signal,
                sr,
                float(note['start']),
                pitch,
                sim_row,
                f,
                sim,
                segment_dur_s=segment_dur_s,
            )
            out.append(
                {
                    'piece': json_path.stem,
                    'player': json_path.stem[:2],
                    'string': s,
                    'fret': f,
                    'pitch': pitch,
                    'onset': float(note['start']),
                    'beta': float(beta) if np.isfinite(beta) else None,
                    'f0_est': float(f0) if np.isfinite(f0) else None,
                }
            )
        else:  # classify
            row_pred, f_pred, beta_pred, f0_pred, candidates_costs = classify_note(
                signal,
                sr,
                float(note['start']),
                pitch,
                sim,
                segment_dur_s=segment_dur_s,
                mu=mu,
                sigma=sigma,
            )
            string_pred = (6 - row_pred) if row_pred is not None else None
            out.append(
                {
                    'piece': json_path.stem,
                    'player': json_path.stem[:2],
                    'string_true': s,
                    'fret_true': f,
                    'pitch': pitch,
                    'onset': float(note['start']),
                    'string_pred': int(string_pred) if string_pred is not None else None,
                    'fret_pred': int(f_pred) if f_pred is not None else None,
                    'beta_pred': float(beta_pred) if beta_pred is not None and np.isfinite(beta_pred) else None,
                    'f0_pred': float(f0_pred) if f0_pred is not None and np.isfinite(f0_pred) else None,
                    'n_candidates': len(candidates_costs),
                    # Keep top-3 candidate costs for downstream analysis
                    'candidates': [
                        {'string': 6 - r['sim_row'], 'fret': r['fret'], 'beta': r['beta'], 'cost': r['cost']}
                        for r in candidates_costs[:3]
                    ],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Physics-law analyses
# ---------------------------------------------------------------------------


def class_summary(records: list[dict]) -> dict:
    """Per (string, fret) class: count, median β, std β, mean f₀."""
    by_class = defaultdict(list)
    for r in records:
        if r['beta'] is None:
            continue
        by_class[(r['string'], r['fret'])].append(r['beta'])
    out = {}
    for key, betas in by_class.items():
        arr = np.array(betas)
        out[key] = {
            'n': len(arr),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'p10': float(np.percentile(arr, 10)),
            'p90': float(np.percentile(arr, 90)),
        }
    return out


def fret_scaling(class_stats: dict) -> dict:
    """For each string, fit log(β) vs fret. Returns slope, intercept, R² per string.

    Theoretical slope: ln(2)/6 ≈ 0.1155.
    """
    out = {}
    for s in range(1, 7):
        pts = [(f, class_stats[(s, f)]['median']) for f in range(13) if (s, f) in class_stats]
        if len(pts) < 3:
            continue
        x = np.array([p[0] for p in pts])
        y = np.log(np.array([p[1] for p in pts]))
        slope, intercept = np.polyfit(x, y, 1)
        # R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        out[s] = {
            'n_points': len(pts),
            'frets_used': [int(p[0]) for p in pts],
            'slope': float(slope),
            'slope_theoretical': float(np.log(2) / 6),
            'intercept_log_beta_open': float(intercept),
            'beta_open_inferred': float(np.exp(intercept)),
            'r2': float(r2),
        }
    return out


def equal_pitch_groups(class_stats: dict) -> dict:
    """For each pitch with ≥2 (string, fret) positions, list β-per-position.

    Standard tuning: E2=40, A2=45, D3=50, G3=55, B3=59, E4=64.
    """
    tuning = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
    by_pitch = defaultdict(list)
    for (s, f), stats in class_stats.items():
        pitch = tuning[s] + f
        by_pitch[pitch].append({'string': s, 'fret': f, 'median': stats['median'], 'n': stats['n']})
    return {p: sorted(items, key=lambda r: r['string']) for p, items in by_pitch.items() if len(items) >= 2}


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def classification_summary(records: list[dict]) -> dict:
    """Per-(true string, true fret) classification accuracy + overall.

    Includes:
      - per_class: {(string, fret): {n, n_correct, accuracy, top_misclassifications}}
      - per_pitch: {pitch: {n, n_correct, accuracy, n_candidates, chance}}
      - confusion: {(true_s, true_f): {(pred_s, pred_f): count}}
      - overall: {n, n_correct, accuracy, n_singleton}  (singleton = pitch
        with only 1 candidate, classification trivially correct)
    """
    per_class = defaultdict(lambda: {'n': 0, 'n_correct': 0, 'misclass': defaultdict(int)})
    per_pitch = defaultdict(lambda: {'n': 0, 'n_correct': 0, 'n_candidates': 0})
    confusion = defaultdict(lambda: defaultdict(int))

    n_total = 0
    n_correct = 0
    n_singleton = 0

    for r in records:
        if r.get('string_pred') is None:
            continue
        s_t, f_t = r['string_true'], r['fret_true']
        s_p, f_p = r['string_pred'], r['fret_pred']
        is_correct = (s_t == s_p) and (f_t == f_p)

        per_class[(s_t, f_t)]['n'] += 1
        if is_correct:
            per_class[(s_t, f_t)]['n_correct'] += 1
        else:
            per_class[(s_t, f_t)]['misclass'][(s_p, f_p)] += 1

        per_pitch[r['pitch']]['n'] += 1
        per_pitch[r['pitch']]['n_correct'] += int(is_correct)
        per_pitch[r['pitch']]['n_candidates'] = max(
            per_pitch[r['pitch']]['n_candidates'],
            r.get('n_candidates', 1),
        )

        confusion[(s_t, f_t)][(s_p, f_p)] += 1

        n_total += 1
        n_correct += int(is_correct)
        if r.get('n_candidates', 1) == 1:
            n_singleton += 1

    # Convert to plain dicts and add accuracy
    per_class_out = {}
    for k, v in per_class.items():
        per_class_out[k] = {
            'n': v['n'],
            'n_correct': v['n_correct'],
            'accuracy': v['n_correct'] / v['n'] if v['n'] else 0.0,
            'top_misclassifications': sorted(
                v['misclass'].items(),
                key=lambda x: -x[1],
            )[:3],
        }
    per_pitch_out = {}
    for k, v in per_pitch.items():
        chance = 1.0 / max(1, v['n_candidates'])
        per_pitch_out[k] = {
            'n': v['n'],
            'n_correct': v['n_correct'],
            'accuracy': v['n_correct'] / v['n'] if v['n'] else 0.0,
            'n_candidates': v['n_candidates'],
            'chance': chance,
        }
    overall = {
        'n': n_total,
        'n_correct': n_correct,
        'accuracy': n_correct / n_total if n_total else 0.0,
        'n_singleton': n_singleton,
        'n_multi_candidate': n_total - n_singleton,
        'accuracy_excl_singleton': (
            (n_correct - n_singleton) / (n_total - n_singleton) if n_total > n_singleton else 0.0
        ),
    }
    return {
        'overall': overall,
        'per_pitch': per_pitch_out,
        'per_class': per_class_out,
        'confusion': {k: dict(v) for k, v in confusion.items()},
    }


def plot_classification_per_pitch(per_pitch: dict, save_path: Path):
    """Bar chart: accuracy per pitch, with chance baseline."""
    pitches = sorted(per_pitch.keys())
    accs = [per_pitch[p]['accuracy'] for p in pitches]
    chances = [per_pitch[p]['chance'] for p in pitches]
    ns = [per_pitch[p]['n'] for p in pitches]
    n_cands = [per_pitch[p]['n_candidates'] for p in pitches]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(pitches))
    ax.bar(x, accs, color=['tab:green' if c > 1 else 'lightgrey' for c in n_cands], edgecolor='black', linewidth=0.5)
    ax.plot(x, chances, 'r--', linewidth=1, label='chance (1/#candidates)')

    for i, (acc, n, nc) in enumerate(zip(accs, ns, n_cands, strict=False)):
        ax.text(i, acc + 0.01, f'n={n}\nk={nc}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pitches], rotation=0, fontsize=8)
    ax.set_xlabel('pitch (MIDI)')
    ax.set_ylabel('classification accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-pitch (string, fret) classification accuracy (green = multi-candidate; grey = singleton/trivial)')
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_confusion_at_equal_pitch(confusion: dict, per_pitch: dict, save_path: Path):
    """For each multi-candidate pitch, show a small confusion matrix.

    Rows = true (string, fret); cols = predicted (string, fret); values = count.
    Only multi-candidate pitches are shown (others are trivially 100% accurate).
    """
    multi_pitches = [p for p in sorted(per_pitch.keys()) if per_pitch[p]['n_candidates'] > 1]
    if not multi_pitches:
        return

    n_plots = len(multi_pitches)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, pitch in zip(axes_flat, multi_pitches, strict=False):
        # Get all (true, pred) pairs for this pitch
        mat = defaultdict(lambda: defaultdict(int))
        all_classes = set()
        for true_class, pred_dict in confusion.items():
            for pred_class, count in pred_dict.items():
                # Filter to this pitch
                tuning = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
                t_pitch = tuning[true_class[0]] + true_class[1]
                if t_pitch != pitch:
                    continue
                mat[true_class][pred_class] += count
                all_classes.add(true_class)
                all_classes.add(pred_class)

        classes = sorted(all_classes)
        n = len(classes)
        if n == 0:
            ax.axis('off')
            continue
        M = np.zeros((n, n), dtype=int)
        for i, t in enumerate(classes):
            for j, p in enumerate(classes):
                M[i, j] = mat[t].get(p, 0)
        # Row-normalize
        row_sums = M.sum(axis=1, keepdims=True)
        M_norm = M / np.maximum(row_sums, 1)

        ax.imshow(M_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(n):
            for j in range(n):
                if M[i, j] > 0:
                    ax.text(
                        j,
                        i,
                        f'{M[i, j]}',
                        ha='center',
                        va='center',
                        fontsize=8,
                        color='white' if M_norm[i, j] > 0.5 else 'black',
                    )
        ax.set_xticks(range(n))
        ax.set_xticklabels([f's{c[0]},f{c[1]}' for c in classes], fontsize=7, rotation=45)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f's{c[0]},f{c[1]}' for c in classes], fontsize=7)
        ax.set_title(f'pitch {pitch}', fontsize=10)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')

    for ax in axes_flat[len(multi_pitches) :]:
        ax.axis('off')
    fig.suptitle(
        'Per-pitch confusion: rows = true (string, fret), cols = predicted, cells = count (color = row-normalized)'
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_beta_vs_fret(class_stats: dict, scaling: dict, save_path: Path):
    """One line per string: β vs fret, log y, with theoretical slope reference."""
    fig, ax = plt.subplots(figsize=(10, 6))
    string_names = {1: 'E4 (high E)', 2: 'B3', 3: 'G3', 4: 'D3', 5: 'A2', 6: 'E2 (low E)'}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for s in range(1, 7):
        pts = [(f, class_stats[(s, f)]['median']) for f in range(13) if (s, f) in class_stats]
        if not pts:
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        ax.plot(x, y, 'o-', color=colors[s - 1], label=f'string {s} ({string_names[s]})')

        # Theoretical line: beta(s, 0) extrapolated * 2^(n/6)
        if s in scaling:
            beta0 = scaling[s]['beta_open_inferred']
            x_theory = np.arange(13)
            y_theory = beta0 * 2 ** (x_theory / 6)
            ax.plot(x_theory, y_theory, '--', color=colors[s - 1], alpha=0.4, linewidth=1)

    ax.set_yscale('log')
    ax.set_xlabel('fret')
    ax.set_ylabel('β (median per class)')
    ax.set_title('β vs fret per string (solid = measured median, dashed = β(s,0)·2^(n/6))')
    ax.set_xticks(range(13))
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_equal_pitch(eq_groups: dict, save_path: Path):
    """For each pitch with multiple positions, plot β per position."""
    pitches_with_3plus = sorted(p for p in eq_groups if len(eq_groups[p]) >= 3)
    if not pitches_with_3plus:
        # Fall back to 2-position pitches
        pitches_with_3plus = sorted(eq_groups.keys())[:8]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    axes = axes.flatten()
    for ax, pitch in zip(axes, pitches_with_3plus[:8], strict=False):
        items = eq_groups[pitch]
        x_labels = [f'({it["string"]},{it["fret"]})' for it in items]
        y = [it['median'] for it in items]
        ns = [it['n'] for it in items]
        ax.bar(x_labels, y, color='steelblue')
        for i, (_xv, yv, nv) in enumerate(zip(x_labels, y, ns, strict=False)):
            ax.text(i, yv, f'n={nv}', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'pitch={pitch} (MIDI)')
        ax.set_yscale('log')
        ax.tick_params(axis='x', labelsize=8)
    # Hide unused
    for ax in axes[len(pitches_with_3plus[:8]) :]:
        ax.axis('off')
    fig.suptitle('β at equal pitch across (string, fret) positions')
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--audio-source',
        choices=list(AUDIO_SOURCES),
        default='hex',
        help='hex: per-string clean audio (gold standard). mic: mono '
        'mixed mic recording (realistic conditions, polyphonic noise).',
    )
    ap.add_argument(
        '--mode',
        choices=['extract', 'classify'],
        default='extract',
        help='extract: measure β at the labeled (string, fret) for physics. '
        'classify: predict (string, fret) among equal-pitch candidates by β.',
    )
    ap.add_argument('--limit-pieces', type=int, default=None, help='Process only first N pieces (for fast iteration)')
    ap.add_argument(
        '--output-dir', default=None, help='Override output directory (default: results/discriminability/<source>/)'
    )
    ap.add_argument(
        '--n-realizations', type=int, default=500, help='Monte Carlo realizations for the physical-model simulation'
    )
    ap.add_argument(
        '--segment-ms', type=int, default=DEFAULT_SEGMENT_MS,
        help='Analysis-window length in ms (default: 40, MATLAB default). Longer = '
             'finer freq resolution but more polyphonic interference + drops shorter notes.'
    )
    ap.add_argument(
        '--track-filter', choices=['all', 'solo', 'comp'], default='all',
        help='Filter pieces by track type from filename suffix. solo = melodic '
             '(mostly monophonic), comp = chord comping (polyphonic), all = both.'
    )
    ap.add_argument(
        '--classifier', choices=['max-cost', 'bayesian'], default='bayesian',
        help='classify mode only. max-cost: pick candidate with highest β-grid cost. '
             'bayesian: also apply MATLAB step 2 (Gaussian discriminant on extracted φ). '
             'Default bayesian to match MATLAB.'
    )
    args = ap.parse_args()

    segment_dur_s = args.segment_ms / 1000.0

    out_dir = Path(args.output_dir) if args.output_dir else (OUTPUT_DIR / args.audio_source)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Mode:         {args.mode}')
    print(f'Audio source: {args.audio_source}')
    print(f'Segment len:  {args.segment_ms} ms')
    print(f'Output dir:   {out_dir}')

    print(f'Simulating physical-model features ({args.n_realizations} realizations)...')
    sim = simulate_features(n_realizations=args.n_realizations, seed=0)

    mu, sigma = (None, None)
    if args.mode == 'classify' and args.classifier == 'bayesian':
        print('Computing class priors (μ, Σ) for Bayesian classifier...')
        mu, sigma = compute_class_priors(sim)

    json_files = sorted(PROC_DIR.glob('*.json'))
    if args.track_filter != 'all':
        json_files = [p for p in json_files if f'_{args.track_filter}' in p.stem]
    if args.limit_pieces:
        json_files = json_files[: args.limit_pieces]
    print(f'Track filter: {args.track_filter}')
    print(f'Processing {len(json_files)} pieces from {PROC_DIR}...')

    t0 = time.time()
    all_records: list[dict] = []
    for i, jp in enumerate(json_files):
        recs = process_piece(jp, sim, args.audio_source, args.mode,
                             segment_dur_s, mu=mu, sigma=sigma)
        all_records.extend(recs)
        if (i + 1) % 20 == 0 or i + 1 == len(json_files):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(json_files) - i - 1)
            print(f'  [{i + 1}/{len(json_files)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s ETA. {len(all_records)} records')

    print(f'\nTotal records: {len(all_records)}')

    suffix = '' if args.mode == 'extract' else '_classify'
    records_path = out_dir / f'all_records{suffix}.json'
    records_path.write_text(json.dumps(all_records, indent=1))
    print(f'Saved records: {records_path}')

    if args.mode == 'extract':
        run_extract_analysis(all_records, out_dir)
    else:
        run_classify_analysis(all_records, out_dir)


def run_extract_analysis(all_records: list[dict], out_dir: Path) -> None:
    stats = class_summary(all_records)
    print('\nPer-(string, fret) summary:')
    print(f'  {"str":>3} {"fret":>4} {"n":>5}  {"med β":>11}  {"std/med":>8}')
    for key in sorted(stats):
        s, f = key
        st = stats[key]
        print(
            f'  {s:>3d} {f:>4d} {st["n"]:>5d}  {st["median"]:>11.3e}  '
            f'{st["std"] / st["median"] * 100 if st["median"] else 0:>7.1f}%'
        )

    scaling = fret_scaling(stats)
    print('\nPhysics check 1: β(s, n) = β(s, 0) · 2^(n/6)')
    print(f'  Theoretical slope of log(β) vs fret = ln(2)/6 = {np.log(2) / 6:.4f}')
    print(f'  {"string":>6}  {"slope":>8}  {"slope/theory":>13}  {"β(s,0) inferred":>18}  {"R²":>5}  {"frets":>6}')
    theory = np.log(2) / 6
    for s in sorted(scaling):
        sc = scaling[s]
        ratio = sc['slope'] / theory
        print(
            f'  {s:>6d}  {sc["slope"]:>8.4f}  {ratio:>13.3f}  '
            f'{sc["beta_open_inferred"]:>18.3e}  {sc["r2"]:>5.3f}  {sc["n_points"]:>6d}'
        )

    eq = equal_pitch_groups(stats)
    print('\nPhysics check 2: β at equal pitch across positions')
    print('  Pitches with ≥2 positions in our data:')
    for pitch in sorted(eq):
        items = eq[pitch]
        items_str = ', '.join(f'(s{it["string"]},f{it["fret"]})={it["median"]:.2e} [n={it["n"]}]' for it in items)
        print(f'  pitch {pitch}: {items_str}')

    plot_beta_vs_fret(stats, scaling, out_dir / 'beta_vs_fret.png')
    plot_equal_pitch(eq, out_dir / 'beta_equal_pitch.png')

    summary = {
        'n_records': len(all_records),
        'n_valid_beta': sum(1 for r in all_records if r.get('beta') is not None),
        'class_stats': {f'{s},{f}': v for (s, f), v in stats.items()},
        'fret_scaling': scaling,
        'equal_pitch_groups': {str(k): v for k, v in eq.items()},
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nPlots and summary written to {out_dir}/')


def run_classify_analysis(all_records: list[dict], out_dir: Path) -> None:
    cls = classification_summary(all_records)
    o = cls['overall']
    print('\nClassification overall:')
    print(f'  total notes:       {o["n"]}')
    print(f'  correct:           {o["n_correct"]}  ({o["accuracy"] * 100:.2f}%)')
    print(f'  singleton (k=1):   {o["n_singleton"]} (trivially correct)')
    print(f'  multi-candidate:   {o["n_multi_candidate"]}')
    print(f'  accuracy excl. singletons: {o["accuracy_excl_singleton"] * 100:.2f}%')

    print('\nPer-pitch accuracy (only multi-candidate pitches shown):')
    print(f'  {"pitch":>6} {"k":>3} {"n":>6} {"chance":>7} {"acc":>7} {"lift":>7}')
    for pitch in sorted(cls['per_pitch']):
        p = cls['per_pitch'][pitch]
        if p['n_candidates'] < 2:
            continue
        lift = p['accuracy'] - p['chance']
        print(
            f'  {pitch:>6d} {p["n_candidates"]:>3d} {p["n"]:>6d} '
            f'{p["chance"] * 100:>6.1f}% {p["accuracy"] * 100:>6.1f}% '
            f'{lift * 100:>+6.1f}pp'
        )

    print('\nPer-class accuracy (top misclassifications shown):')
    print(f'  {"true (s,f)":>11} {"n":>5} {"acc":>7}  top wrong predictions')
    for key in sorted(cls['per_class']):
        c = cls['per_class'][key]
        miscls = c['top_misclassifications']
        miscls_str = ', '.join(f'({mk[0]},{mk[1]})={cnt}' for mk, cnt in miscls) if miscls else '-'
        print(f'  ({key[0]},{key[1]:>2d})    {c["n"]:>5d} {c["accuracy"] * 100:>6.1f}%  {miscls_str}')

    plot_classification_per_pitch(cls['per_pitch'], out_dir / 'classify_per_pitch.png')
    plot_confusion_at_equal_pitch(cls['confusion'], cls['per_pitch'], out_dir / 'classify_confusion.png')

    # Save summary (convert tuple keys for JSON)
    confusion_serializable = {
        f'{k[0]},{k[1]}': {f'{p[0]},{p[1]}': v for p, v in conf.items()} for k, conf in cls['confusion'].items()
    }
    per_class_serializable = {
        f'{k[0]},{k[1]}': {
            **v,
            'top_misclassifications': [
                {'pred_string': mk[0], 'pred_fret': mk[1], 'count': cnt} for mk, cnt in v['top_misclassifications']
            ],
        }
        for k, v in cls['per_class'].items()
    }
    summary = {
        'overall': cls['overall'],
        'per_pitch': cls['per_pitch'],
        'per_class': per_class_serializable,
        'confusion': confusion_serializable,
    }
    (out_dir / 'summary_classify.json').write_text(json.dumps(summary, indent=2))
    print(f'\nClassification summary written to {out_dir}/')


if __name__ == '__main__':
    main()
