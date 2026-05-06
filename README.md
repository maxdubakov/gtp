# GTP — Guitar Tab Predictor
Audio-to-MIDI → MIDI-to-Tablature

Implemented in 2 stages:
1. **Stage 1 (Audio → MIDI):** Fine-tuned Kong et al. CRNN piano transcription model adapted for guitar
2. **Stage 2 (MIDI → Tabs):** Trained-from-scratch Fretting-Transformer (T5-based) that assigns string/fret positions to MIDI notes

## Results

### Stage 1: Audio → MIDI

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Pretrained piano baseline | 74.9% | 42.3% | 51.8% |
| Fine-tuned on GAPS + GuitarSet | 92.0% | 95.2% | **93.5%** |

*Evaluated on GuitarSet player 5 (held out from training), onset-only F1 at 50ms tolerance.*

### Stage 2: MIDI → Tabs (baseline run, 60k steps, batch=64, ~2.24M params)

| Metric | val |
|---|---:|
| `tab_strict` (exact (string, fret) match) | **86.7%** |
| `tab_equivalent` (accepts piece-consistent alternate fingerings) | **94.3%** |
| Pitch accuracy after post-processing | 100% |

`tab_strict` matches the paper's metric. `tab_equivalent` is our fairer metric: it counts notes where the model picked a *different but musically valid* (string, fret) realization for an entire piece (e.g. playing the song one position higher on the neck) as correct. Error analysis showed ~7.5% of notes are these consistent-alternate-fingering choices, not real errors.

Per-source val (`tab_strict`):

| Source | Note count | tab_strict |
|---|---:|---:|
| GuitarToday | 24,532 | **98.7%** |
| DadaGP | 540,852 | 86.5% |
| GuitarSet | 6,135 | 77.9% |
| Leduc | 13,034 | 77.8% |

Per-genre val (DadaGP, top/bottom buckets):

| Best | tab_strict | Worst | tab_strict |
|---|---:|---|---:|
| electronic | 98.9% | country | 74.1% |
| reggae | 97.6% | classical | 76.3% |
| folk | 93.0% | jazz | 76.3% |
| punk | 88.6% | blues | 78.2% |

## Data Pipeline

### Stage 1 Training Data

1. **GAPS** — audio + aligned MIDI. 270 train + 30 test tracks.
2. **GuitarSet** — audio + JAMS annotations from hexaphonic pickups. 300 train + 60 test tracks (player 5 held out).

### Stage 2 Training Data

| Source | Pieces | Notes | Method |
|---|---:|---:|---|
| DadaGP | 5,662 | 5,671,658 | pyguitarpro on GP3/4/5 files (acoustic-track filter) |
| GuitarToday | 616 | 245,124 | Soundslice JSON via Playwright |
| GuitarSet | 360 | 62,476 | JAMS per-string annotations (hexaphonic pickup ground truth) |
| Leduc | 181 | 124,075 | alphaTab on GP7/8 files |
| **Total** | **6,819** | **6,103,333** | |

After capo augmentation (8 variants per piece where playable), the train split has **47,514 pieces / 233,763 sub-sequences / 61M decoder tokens** for training.

All sources are processed to a common JSON format:
```json
{
  "source": "guitarset",
  "tuning": [64, 59, 55, 50, 45, 40],
  "tempo": 120,
  "capo": 0,
  "genre": "jazz",
  "notes": [
    {"pitch": 43, "string": 6, "fret": 3, "start": 0.232, "end": 0.812}
  ]
}
```
*`string` 1 = high E, 6 = low E (guitar convention). `genre` is one of 14 coarse buckets (rock/metal/pop/folk/blues/classical/jazz/punk/country/reggae/electronic/hip_hop/funk/unknown).*

## Architecture

**Stage 1**: Kong et al.'s onset+offset+frame regression CRNN, fine-tuned on guitar audio. Input: 16 kHz mel-spectrogram. Output: MIDI piano-roll style annotations.

**Stage 2**: T5 encoder-decoder, halved t5-small dims (`d_model=128, d_ff=1024, 3 layers, 4 heads`, ~2.24M params). 553-token vocabulary (567 with optional GENRE conditioning).

Encoder input prefix: `[GENRE<X>] [TEMPO<bpm>] CAPO<n> <TUNING_START> NOTE_ON×6 <TUNING_END>` followed by note events `(TIME_SHIFT, NOTE_ON, NOTE_OFF)`.

Decoder output: `(TIME_SHIFT, TAB<string,fret>)` per note.

Post-processing (paper-faithful): for each input pitch, search ±5 neighbor predictions for a tab producing the same pitch; if none, fall back to either `first_viable_tab` (paper) or `nearest_viable_tab` (deviation, anchors on raw model output — tested empirically and slightly better).

## Quick Start

### Prerequisites
- Python 3.12 · PyTorch 2.6.0 · Node.js (for Leduc GP7/8 parsing via alphaTab)

### Setup
```bash
pip install -e .
```

### Stage 1 — Train + Evaluate
```bash
# Fine-tune on guitar data
python scripts/stage1/train.py --device cuda --num-workers 4

# Evaluate on full GuitarSet
python scripts/stage1/eval_guitarset.py \
    --checkpoint models/finetuned/step_0070000_final.pth -j 4
```

### Stage 2 — Build dataset, train, evaluate

```bash
# 1) Build per-source processed JSONs (one-time)
python scripts/stage2/data/dadagp/build_dataset.py
python scripts/stage2/data/guitarset/build_dataset.py
bash   scripts/stage2/data/leduc/build_dataset.sh
python scripts/stage2/data/guitartoday/build_dataset.py

# 2) Build the augmented training set (capo variants + train/val/test split)
python scripts/stage2/build_aug_dataset.py

# 3) Train baseline (no genre conditioning, ~14h on RTX 4090)
python scripts/stage2/train.py \
    --output-dir runs/stage2_baseline \
    --device cuda --batch-size 64 --max-steps 60000 \
    --num-workers 2 \
    --experiment-label "Baseline"

# 3b) Or with genre conditioning (15% classifier-free dropout)
python scripts/stage2/train.py \
    --output-dir runs/stage2_genre \
    --device cuda --batch-size 64 --max-steps 60000 \
    --genre-conditioning --genre-dropout 0.10

# 4) Evaluate (autoregressive + post-processing)
python scripts/stage2/eval.py \
    --checkpoint runs/stage2_baseline/checkpoints/step_0060000_final.pth \
    --include-test --output results/eval.json
```

### Stage 1+2 Demo (audio → tabs)
```bash
python scripts/demo.py --audio path/to/recording.wav --capo 3 --genre rock
```

## Repo Structure

```
gtp/
├── src/gtp/                         # Importable package
│   ├── log.py                       # Timestamped logging helpers
│   ├── stage1/                      # Audio→MIDI (Kong CRNN, fine-tuned)
│   │   ├── data.py / inference.py / postprocess.py
│   │   └── model/{kong,losses,utils}.py
│   └── stage2/                      # MIDI→Tab (T5)
│       ├── config.py                # RunConfig dataclasses, JSON I/O
│       ├── data.py                  # TabDataset, augmentation, splits
│       ├── genres.py                # 14-bucket coarse genre taxonomy
│       ├── inference.py             # generate, post-process, anchor priming
│       ├── metrics.py               # tab_strict/equivalent, drift buckets
│       ├── model.py                 # halved-t5-small builder
│       ├── postprocess.py           # ±5 window + first/nearest_viable fallback
│       └── tokenizer.py             # Vocabulary + tokenize_piece
├── scripts/
│   ├── stage1/                      # train, eval, transcribe, listen
│   ├── stage2/
│   │   ├── train.py                 # T5 trainer with config.json + metrics.jsonl
│   │   ├── eval.py                  # autoregressive eval sweep
│   │   ├── build_aug_dataset.py     # capo augmentation + stratified split
│   │   ├── backfill_metrics.py      # re-eval old checkpoints into metrics.jsonl
│   │   ├── data/<source>/           # per-source processing scripts
│   │   ├── error_analysis/          # dump_eval_predictions, enrich_errors,
│   │   │                            # analyze_errors, plot_errors, etc.
│   │   └── setup/                   # pack_*.sh for shipping to remote training
│   └── demo.py                      # End-to-end audio → tabs demo
├── data/                            # Datasets (gitignored content)
│   ├── <source>/processed/*.json    # Per-source canonical pieces
│   └── stage2_aug/{train,val,test}.jsonl
└── runs/                            # Training outputs (gitignored content)
    └── <run_id>/
        ├── config.json              # Run hyperparams + git SHA + GPU info
        ├── metrics.jsonl            # Per-eval-cycle val_loss / tab_acc
        ├── final_eval.json          # End-of-training summary
        ├── train.log                # Timestamped training stdout
        └── checkpoints/
            └── step_*.pth
```

## References

- Kong et al. 2021 — [High-resolution piano transcription with pedals](https://arxiv.org/abs/2010.01815) (Stage 1: base model)
- Riley et al. 2024 — [High resolution guitar transcription via domain adaptation](https://arxiv.org/abs/2402.15258) (Stage 1: guitar fine-tuning recipe)
- Riley et al. 2024 — [GAPS dataset](https://arxiv.org/abs/2408.08653) (Stage 1: GAPS dataset)
- Hamberger et al. 2025 — [Fretting-Transformer](https://arxiv.org/abs/2506.14223) (Stage 2: model architecture, post-processing algorithm)
- Sarmento et al. 2021 — [DadaGP dataset](https://arxiv.org/abs/2107.14653) (Stage 2: main bulk of training data)
