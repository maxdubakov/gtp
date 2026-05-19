# GTP

Guitar tab predictor: audio -> MIDI -> tablature.

Two-stage setup:
- Stage 1: audio -> MIDI. Kong-style CRNN fine-tuned on guitar audio.
- Stage 2: MIDI -> tabs. Small T5/Fretting-Transformer trained from scratch.

## Results

### Stage 1

Held-out GuitarSet player 5, onset-only F1 at 50 ms.

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Piano baseline | 74.9% | 42.3% | 51.8% |
| Fine-tuned on GAPS + GuitarSet | 92.0% | 95.2% | **93.5%** |

### Stage 2

Baseline: 60k steps, batch 64, ~2.24M params.

| Metric | val |
|---|---:|
| `tab_strict` | **86.7%** |
| `tab_equivalent` | **94.3%** |
| pitch after post-processing | 100% |

`tab_strict` is exact `(string, fret)` match. `tab_equivalent` also accepts piece-consistent alternate fingerings, which are musically valid but fail strict tab matching.

Per-source val, `tab_strict`:

| Source | Notes | Score |
|---|---:|---:|
| GuitarToday | 24,532 | **98.7%** |
| DadaGP | 540,852 | 86.5% |
| GuitarSet | 6,135 | 77.9% |
| Leduc | 13,034 | 77.8% |

## Data

Stage 1:
- GAPS: 270 train + 30 test tracks.
- GuitarSet: 300 train + 60 test tracks, player 5 held out.

Stage 2:

| Source | Pieces | Notes | Parser |
|---|---:|---:|---|
| DadaGP | 5,662 | 5,671,658 | pyguitarpro, acoustic-track filter |
| GuitarToday | 616 | 245,124 | Soundslice JSON |
| GuitarSet | 360 | 62,476 | JAMS per-string annotations |
| Leduc | 181 | 124,075 | alphaTab GP7/8 parse |
| **Total** | **6,819** | **6,103,333** | |

After capo augmentation, train has 47,514 pieces, 233,763 sub-sequences, and **~61M** decoder tokens.

## References

- Kong et al. 2021 — [High-resolution piano transcription with pedals](https://arxiv.org/abs/2010.01815) (Stage 1: base model)
- Riley et al. 2024 — [High resolution guitar transcription via domain adaptation](https://arxiv.org/abs/2402.15258) (Stage 1: guitar fine-tuning recipe)
- Riley et al. 2024 — [GAPS dataset](https://arxiv.org/abs/2408.08653) (Stage 1: GAPS dataset)
- Hamberger et al. 2025 — [Fretting-Transformer](https://arxiv.org/abs/2506.14223) (Stage 2: model architecture, post-processing algorithm)
- Sarmento et al. 2021 — [DadaGP dataset](https://arxiv.org/abs/2107.14653) (Stage 2: main bulk of training data)
