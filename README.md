# GTP — Guitar Tab Predictor

Audio → MIDI → Tablature pipeline for guitar transcription.

Two-stage architecture:

1. **Stage 1 (Audio → MIDI):** Fine-tuned Kong et al. CRNN piano transcription model adapted for guitar
2. **Stage 2 (MIDI → Tabs):** Fretting-Transformer (T5-based) that assigns string/fret positions to MIDI notes

## Results

### Stage 1: Audio → MIDI

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Pretrained piano baseline | 74.9% | 42.3% | 51.8% |
| Fine-tuned on GAPS + GuitarSet | 92.0% | 95.2% | **93.5%** |

Evaluated on GuitarSet player 5 (held out from training), onset-only F1 at 50ms tolerance.

### Stage 2: MIDI → Tabs

*In progress — tokenizer and model training.*

## Project Structure

```
gtp/
├── src/gtp/                    # Library code (importable)
│   ├── model/
│   │   ├── kong.py             # Kong et al. CRNN architecture
│   │   ├── losses.py           # Training loss functions
│   │   └── utils.py            # PyTorch utilities
│   ├── data.py                 # Stage 1 training data pipeline (GAPS + GuitarSet)
│   ├── inference.py            # Audio → MIDI inference (handles chunking, post-processing)
│   ├── postprocess.py          # Model activations → note events (regression post-processor)
│   └── log.py                  # Verbose tracing for data flow debugging
│
├── scripts/                    # Runnable entry points
│   ├── train.py                # Stage 1 fine-tuning (Kong CRNN on guitar data)
│   ├── eval_guitarset.py       # Evaluate model on GuitarSet (mir_eval F1)
│   ├── transcribe.py           # Transcribe any audio file → MIDI + overlay WAV
│   │
│   ├── data/                   # Data acquisition & processing pipelines
│   │   ├── dadagp/
│   │   │   ├── filter_acoustic.py       # Scan DadaGP GP files, find acoustic guitar tracks
│   │   │   ├── filter_acoustic_strict.py # Stricter filtering (exclude electric names, min notes)
│   │   │   └── build_dataset.py         # Parse GP files → JSON + MIDI
│   │   ├── guitarset/
│   │   │   └── build_dataset.py         # JAMS annotations → JSON + MIDI (hex pickup ground truth)
│   │   ├── guitartoday/
│   │   │   ├── parse_patreon_posts.py   # Extract catalog CSV from Patreon batch JSONs
│   │   │   ├── fetch_soundslice.py      # Playwright: fetch Soundslice notation JSONs
│   │   │   ├── soundslice_to_midi.py    # Convert Soundslice JSON → note events
│   │   │   └── build_dataset.py         # Batch convert all slices → JSON + MIDI
│   │   └── leduc/
│   │       ├── alphatab/
│   │       │   └── parse_gp.mjs         # Node.js: parse GP7/8 files via alphaTab
│   │       └── build_dataset.sh         # Batch convert GP files → JSON + MIDI
│   │
│   ├── listen/                 # Human verification tools
│   │   ├── listen_to_predictions.py     # Compare best/worst predictions side-by-side
│   │   └── verify_transcription.py      # Overlay MIDI on audio (stereo WAV)
│   │
│   └── setup/                  # Environment setup & verification
│       ├── verify.py           # Full setup verification (24 checks)
│       ├── test_model_load.py  # Quick model loading test
│       ├── test_data_pipeline.py # Data pipeline shape/value checks
│       └── pack.sh             # Create archive for RunPod upload
│
├── models/                     # Model checkpoints (gitignored)
│   ├── pretrained/             # Kong piano checkpoint (164MB)
│   ├── finetuned/              # Guitar fine-tuned checkpoints
│   └── soundfonts/             # For MIDI → audio rendering
│
├── data/                       # Datasets (gitignored)
│   ├── gaps_hf/                # GAPS dataset from HuggingFace (404 audio + MIDI pairs)
│   ├── guitarset/              # GuitarSet (360 audio + JAMS annotations)
│   │   ├── audio_mono-mic/     # Raw audio files
│   │   ├── annotation/         # JAMS files with per-string note data
│   │   └── processed/          # Extracted tab JSON + MIDI
│   ├── DadaGP-v1.1/            # Raw DadaGP dataset (26K GP files)
│   ├── dadagp/                 # DadaGP processed
│   │   ├── acoustic_tracks.csv # Catalog of acoustic guitar tracks
│   │   └── processed/          # Extracted tab JSON + MIDI (~5,643 files)
│   ├── guitartoday/            # GuitarToday (Patreon + Soundslice)
│   │   ├── patreon_posts/      # Raw Patreon batch JSONs
│   │   ├── posts.csv           # Parsed catalog
│   │   ├── slices/             # Fetched Soundslice JSONs
│   │   └── processed/          # Extracted tab JSON + MIDI (~624 files)
│   └── leduc/                  # François Leduc jazz transcriptions
│       ├── gp_files/           # GP7/8 files (184 files)
│       └── processed/          # Extracted tab JSON + MIDI (~183 files)
│
└── results/                    # Evaluation results (gitignored)
    ├── baseline_guitarset.csv  # Pretrained piano baseline
    ├── finetuned_*.csv         # Fine-tuned model results
    └── listen/                 # Verification audio files
```

## Data Pipeline

### Stage 1 training data (Audio → MIDI)

```
GAPS (HuggingFace)     → audio + aligned MIDI          → 270 train / 30 test
GuitarSet              → audio + JAMS annotations       → 300 train / 60 val (by player)
```

### Stage 2 training data (MIDI → Tabs)

All datasets are processed to a common JSON format:

```json
{
  "tuning": [64, 59, 55, 50, 45, 40],
  "notes": [
    {"pitch": 43, "string": 6, "fret": 3, "start": 0.232, "end": 0.812},
    ...
  ]
}
```

Where `string` 1 = high E, 6 = low E (guitar convention).

| Source | Pieces | Notes | Method |
|---|---|---|---|
| DadaGP | 5,643 | 5,654,300 | pyguitarpro on GP3/4/5 files |
| GuitarToday | 624 | 242,598 | Soundslice JSON via Playwright |
| GuitarSet | 360 | 62,476 | JAMS per-string annotations (hexaphonic pickup) |
| Leduc | 183 | ~120,000 | alphaTab on GP7/8 files |
| **Total** | **6,810** | **~6,079,374** | |

## Quick Start

### Prerequisites

- Python 3.12 with venv
- PyTorch 2.6.0
- Node.js (for Leduc GP7/8 parsing via alphaTab)

### Stage 1: Transcribe audio

```bash
source venv/bin/activate
python scripts/transcribe.py path/to/guitar_recording.wav
```

Outputs: `_predicted.mid`, `_overlay.wav` (stereo verification), `_guitar.wav` (synthesized).

### Stage 1: Train

```bash
# Fine-tune on guitar data (requires GAPS + GuitarSet in data/)
python scripts/train.py --device cuda --num-workers 4

# Resume from checkpoint
python scripts/train.py --resume runs/finetune_001/step_XXXXX.pth
```

### Stage 1: Evaluate

```bash
# Full GuitarSet evaluation
python scripts/eval_guitarset.py --checkpoint models/finetuned/step_0070000_final.pth -j 4

# Held-out player only
python scripts/eval_guitarset.py --checkpoint models/finetuned/step_0070000_final.pth --split 05
```

## Key Papers

- Kong et al. 2021 — [High-resolution piano transcription with pedals](https://arxiv.org/abs/2010.01815) (Stage 1 base model)
- Riley et al. 2024 — [High resolution guitar transcription via domain adaptation](https://arxiv.org/abs/2402.15258) (fine-tuning recipe)
- Riley et al. 2024 — [GAPS dataset](https://arxiv.org/abs/2408.08653) (guitar training data)
- Hamberger et al. 2025 — [Fretting-Transformer](https://arxiv.org/abs/2506.14223) (Stage 2 model)
- Sarmento et al. 2021 — [DadaGP dataset](https://arxiv.org/abs/2107.14653) (tab training data)

