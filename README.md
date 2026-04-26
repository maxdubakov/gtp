# GTP — Guitar Tab Predictor
Audio-to-MIDI --> MIDI-to-Tablature

Implemented in 2 stages:
1. **Stage 1 (Audio → MIDI):** Fine-tuned Kong et al. CRNN piano transcription model adapted for guitar
2. **Stage 2 (MIDI → Tabs):** Trained from scratch Fretting-Transformer (T5-based) that assigns string/fret positions to MIDI notes

## Results

### Stage 1: Audio → MIDI

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Pretrained piano baseline | 74.9% | 42.3% | 51.8% |
| Fine-tuned on GAPS + GuitarSet | 92.0% | 95.2% | **93.5%** |

*Evaluated on GuitarSet player 5 (held out from training), onset-only F1 at 50ms tolerance.*

### Stage 2: MIDI → Tabs

*In progress — tokenizer and model training.*

## Data Pipeline

### Stage 1 Training Data

1. **GAPS** - audio + aligned MIDI. Contains 270 train tracks + 30 test tracks 
2. **GuitarSet** - audio + JAMS annotations. Contains 300 train tracks + 60 test tracks (player 5 recordings)

### Stage 2 Training Data

| Source | Pieces | Notes | Method |
|---|---|---|---|
| DadaGP | 5,643 | 5,654,300 | pyguitarpro on GP3/4/5 files |
| GuitarToday | 624 | 242,598 | Soundslice JSON via Playwright |
| GuitarSet | 360 | 62,476 | JAMS per-string annotations (hexaphonic pickup) |
| Leduc | 183 | ~120,000 | alphaTab on GP7/8 files |
| **Total** | **6,810** | **~6,079,374** | |

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
*Where `string` 1 = high E, 6 = low E (guitar convention).*

## Quick Start

### Prerequisites

- Python 3.12
- PyTorch 2.6.0
- Node.js (for Leduc GP7/8 parsing via alphaTab)

### Stage 1: Train

```bash
# Fine-tune on guitar data (requires GAPS + GuitarSet in data/)
python scripts/stage1/train.py --device cuda --num-workers 4

# Resume from checkpoint
python scripts/stage1/train.py --resume runs/finetune_001/step_XXXXX.pth
```

### Stage 1: Evaluate

```bash
# Full GuitarSet evaluation
python scripts/stage1/eval_guitarset.py --checkpoint models/finetuned/step_0070000_final.pth -j 4

# Held-out player only
python scripts/stage1/eval_guitarset.py --checkpoint models/finetuned/step_0070000_final.pth --split 05
```

## References

- Kong et al. 2021 — [High-resolution piano transcription with pedals](https://arxiv.org/abs/2010.01815) (Stage 1: base model)
- Riley et al. 2024 — [High resolution guitar transcription via domain adaptation](https://arxiv.org/abs/2402.15258) (Stage 1: Guitar fine-tuning recipe)
- Riley et al. 2024 — [GAPS dataset](https://arxiv.org/abs/2408.08653) (Stage 1: GAPS dataset)
- Hamberger et al. 2025 — [Fretting-Transformer](https://arxiv.org/abs/2506.14223) (Stage 2: Model architecture)
- Sarmento et al. 2021 — [DadaGP dataset](https://arxiv.org/abs/2107.14653) (Stage 2: DadaGP dataset - main bulk of training data)

