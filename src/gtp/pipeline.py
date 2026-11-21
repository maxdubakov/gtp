"""The demo pipeline: audio → Stage 1 notes → string evidence → Stage 2 tabs, with the models loaded once.

Each note's string comes from Stage 2's beam search, rescored with string evidence heard in the recording: the
string classifiers, the notes' inharmonicity and their harmonic levels, fitted to the guitar without labels
(gtp.stage1.string_evidence). Earlier takes of the same guitar (earlier_takes) can be fitted together with the new one.

Used by scripts/demo.py and scripts/riff/server.py.
"""

import glob
import tempfile
import time
from functools import cached_property
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from gtp import REPO_ROOT, takes
from gtp.device import auto_device
from gtp.stage1 import strings
from gtp.stage1.inference import SAMPLE_RATE as STAGE1_SR
from gtp.stage1.inference import PianoTranscription
from gtp.stage1.string_evidence import classifier_probs, fuse_recordings, physical, settings
from gtp.stage1.tagger import TechniqueTagger
from gtp.stage2.fusion import decode_beam, log_ratio, piece_rows, sorted_notes, valid_tabs
from gtp.stage2.genres import UNKNOWN
from gtp.stage2.inference import build_anchor_prefix, generate_tabs, load_checkpoint
from gtp.stage2.metrics import difficulty_score
from gtp.stage2.postprocess import correct_tabs
from gtp.stage2.tokenizer import tokenize_piece_for_inference
from gtp.track import git_state

STAGE1_DEFAULT = REPO_ROOT / 'models' / 'finetuned' / 'z8_step_0045000.pth'
STAGE2_DEFAULT = (
    REPO_ROOT / 'runs' / 'stage2' / '2026-11-08_11-58-31_p05_baseline' / 'checkpoints' / 'step_0060000_final.pth'
)
CLASSIFIER_RUNS_DEFAULT = 'results/strings/runs/*goatexp-di_s*'
TECHNIQUE_RUNS_DEFAULT = 'results/techniques/runs/*_demo_s?'
TECHNIQUE_PRIOR_DEFAULT = 'results/techniques/priors/2026-11-21_16-48-47_tab_prior_5k'
STD_TUNING = [64, 59, 55, 50, 45, 40]
SR = 44100


def repo_relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def parse_anchor(anchor):
    """'string:fret,string:fret,...' → [(string, fret)], or None."""
    if not anchor:
        return None
    return [tuple(int(x) for x in pair.strip().split(':')) for pair in anchor.split(',')]


def notes_to_piece(note_events, tuning, capo, tempo, genre=UNKNOWN):
    notes = sorted(
        (
            {
                'pitch': int(e['midi_note']),
                'start': float(e['onset_time']),
                'end': float(e['offset_time']),
                'velocity': int(e['velocity']),
            }
            for e in note_events
        ),
        key=lambda n: (n['start'], n['pitch']),
    )
    return {'tuning': tuning, 'tempo': tempo, 'capo': capo, 'genre': genre, 'notes': notes}


def stage2(piece, model, vocab, max_seq_len, device, anchor_tabs=None, fallback='first_viable'):
    """Returns (corrected_tabs, raw_tabs, enc_subseqs, dec_subseqs, sources).

    If `anchor_tabs` is provided (list of (string, fret) for the first N notes),
    the decoder is primed with those tabs (with proper TIME_SHIFTs between them)
    so generation continues with that style.

    `sources` is a parallel list to `corrected_tabs` with values
    'unchanged' / 'window_swap' / 'fallback' — see correct_tabs for meanings.

    `fallback` selects the post-processing fallback strategy: 'first_viable'
    (paper-faithful) or 'nearest_viable' (deviation; Manhattan-nearest to raw).
    """
    print('Stage 2: notes → tabs...')
    enc_subseqs = tokenize_piece_for_inference(piece, vocab, max_seq_len=max_seq_len)

    decoder_prefix = build_anchor_prefix(piece, vocab, anchor_tabs) if anchor_tabs else None
    if decoder_prefix is not None:
        print(f'  anchoring first {len(anchor_tabs)} notes ({len(decoder_prefix) - 1} prefix tokens)')

    raw_tabs, dec_subseqs = generate_tabs(
        model, vocab, enc_subseqs, device, max_seq_len=max_seq_len, return_raw=True, decoder_prefix=decoder_prefix
    )

    sorted_notes = sorted(piece['notes'], key=lambda x: (x['start'], x['pitch']))
    input_pitches = [n['pitch'] for n in sorted_notes]
    corrected_tabs, sources = correct_tabs(
        input_pitches,
        raw_tabs,
        piece['tuning'],
        return_sources=True,
        fallback=fallback,
    )

    n_unchanged = sum(1 for s in sources if s == 'unchanged')
    n_swap = sum(1 for s in sources if s == 'window_swap')
    n_fallback = sum(1 for s in sources if s == 'fallback')
    print(
        f'  encoder sub-seqs: {len(enc_subseqs)}  '
        f'raw tabs: {len(raw_tabs)}  corrected: {len(corrected_tabs)}  '
        f'unplayable: {sum(1 for t in corrected_tabs if t is None)}'
    )
    print(f'  postproc sources: unchanged={n_unchanged}  window_swap={n_swap}  fallback={n_fallback}')

    d_raw = difficulty_score(raw_tabs)
    d_pp = difficulty_score(corrected_tabs)
    if d_raw is not None and d_pp is not None:
        print(f'  difficulty: raw={d_raw:.3f}  pp={d_pp:.3f}')
    else:
        print('  difficulty: (need ≥2 valid tabs)')
    return corrected_tabs, raw_tabs, enc_subseqs, dec_subseqs, sources


def earlier_takes(takes_dir, guitar):
    """[(audio path, notes)] of the takes recorded on `guitar`: their edited notes if saved, else the predicted ones."""
    out = []
    for d in sorted(Path(takes_dir).iterdir()) if guitar and Path(takes_dir).exists() else []:
        if not (d / 'prediction.json').exists():
            continue
        take = takes.load(d)
        notes = (take['edited'] or take['prediction'])['notes']
        if take['meta'].get('guitar', '').strip().lower() == guitar.strip().lower() and notes:
            out.append((d / 'audio.wav', sorted(notes, key=lambda n: (n['start'], n['pitch']))))
    return out


def string_evidence(wav_path, piece, classifiers, pooled):
    """(n_notes, 6) fused string probabilities for sorted_notes(piece) and the classifier's pitch prior.

    `pooled` [(audio path, notes)] are other recordings of the same guitar: the string constants and envelopes are
    fitted over them and this take together.
    """
    print(f'String evidence: {len(classifiers)} classifiers, {len(pooled)} earlier takes of this guitar...')
    profiles, physics, envelope = settings()
    parts = []
    for audio, notes in [(wav_path, sorted_notes(piece)), *pooled]:
        onsets, pitches = [n['start'] for n in notes], [n['pitch'] for n in notes]
        probs = classifier_probs(audio, onsets, pitches, classifiers)
        parts.append((probs, *physical(audio, onsets, [n['end'] for n in notes], pitches, profiles, envelope), pitches))
    probs, curves, levels, pitches = zip(*parts, strict=True)
    fused, _, family = fuse_recordings(probs, curves, levels, pitches, profiles, physics, envelope)
    print(f'  {sum(len(p) for p in pitches)} notes fitted together, {family} string set')
    return fused[0], classifiers[0].prior(pitches[0])


def stage2_fused(piece, model, vocab, max_seq_len, fused, prior, lam, beams, device):
    """Choose each note's string by beam search along the skeleton its notes fix, rescored with the string
    evidence (weight lam); returns tabs in sorted_notes(piece) order."""
    print('Stage 2 (with string evidence): notes → tabs...')
    tuning = piece['tuning']
    # tokenize_piece needs a string/fret per note; these placeholders are replaced by the model's choices.
    notes = []
    for n in piece['notes']:
        s = int(valid_tabs(n['pitch'], tuning, vocab)[0][0])
        notes.append({**n, 'string': s, 'fret': n['pitch'] - tuning[s - 1]})
    piece = {**piece, 'notes': notes}
    ordered = sorted_notes(piece)
    ratios = [
        log_ratio(f[None], q, valid_tabs(n['pitch'], tuning, vocab)[0], 1e-4)
        for n, f, q in zip(ordered, fused, prior, strict=True)
    ]
    rows = piece_rows(piece, vocab, max_seq_len, list(range(len(ordered))), ratios)
    picks = decode_beam(model, rows, device, 64, lam, beams=beams)
    tabs = [(picks[i], note['pitch'] - tuning[picks[i] - 1]) for i, note in enumerate(ordered)]
    agree = np.mean([s == f.argmax() + 1 for (s, _), f in zip(tabs, fused, strict=True)])
    print(f'  Stage 2 agrees with the audio evidence on {agree:.0%} of {len(tabs)} notes')
    return tabs, [r.enc for r in rows]


class Pipeline:
    """The note model, Stage 2 and (on first use) the string classifiers, loaded once; run() transcribes a take."""

    def __init__(
        self,
        stage1_checkpoint=STAGE1_DEFAULT,
        stage2_checkpoint=STAGE2_DEFAULT,
        classifier_runs=CLASSIFIER_RUNS_DEFAULT,
        technique_runs=TECHNIQUE_RUNS_DEFAULT,
        technique_prior=TECHNIQUE_PRIOR_DEFAULT,
        device=None,
    ):
        self.device = device or auto_device()
        self.stage1_checkpoint, self.stage2_checkpoint = stage1_checkpoint, stage2_checkpoint
        self.classifier_runs = classifier_runs
        self.technique_runs, self.technique_prior = technique_runs, technique_prior
        # Stage 1's recurrent layers run several times faster on the CPU than on Apple's GPU.
        self.transcriptor = PianoTranscription(
            checkpoint_path=str(stage1_checkpoint), device='cpu' if self.device == 'mps' else self.device
        )
        self.model, self.vocab, iteration, self.max_seq_len = load_checkpoint(str(stage2_checkpoint), self.device)
        print(f'Stage 2 checkpoint iteration: {iteration}  max_seq_len: {self.max_seq_len}')
        if hasattr(self.model, 'embed'):
            raise ValueError(
                'This Stage 2 checkpoint reads evidence in its encoder; the pipeline fuses evidence at decoding: '
                'use a plain one'
            )

    @cached_property
    def runs(self):
        runs = sorted(glob.glob(str(REPO_ROOT / self.classifier_runs)))
        if not runs:
            raise FileNotFoundError(f'No classifier runs match {self.classifier_runs}')
        return runs

    @cached_property
    def classifiers(self):
        return [strings.StringClassifier.from_run(run, self.device) for run in self.runs]

    @cached_property
    def tagger(self):
        runs = sorted(glob.glob(str(REPO_ROOT / self.technique_runs)))
        if not runs:
            raise FileNotFoundError(f'No technique classifier runs match {self.technique_runs}')
        # The prior's recurrent layers do not run on Apple's GPU.
        device = self.device if self.device == 'cuda' else 'cpu'
        return TechniqueTagger(runs, self.technique_prior, device=device)

    def stage1(self, audio):
        """Note events of `audio` (mono, at SR)."""
        print('Stage 1: audio → MIDI...')
        result = self.transcriptor.transcribe(librosa.resample(audio, orig_sr=SR, target_sr=STAGE1_SR))
        note_events = result['note_events']
        print(f'  detected {len(note_events)} notes')
        return note_events

    def run(self, audio, sr, wav_path=None, **options):
        """Transcribe mono `audio` at rate `sr` into a tab.

        The string evidence reads the audio from a file: it is written at SR to `wav_path` (a temporary file if
        None). `options` are transcribe()'s. Returns a dict: `placed` notes (with string, fret, evidence),
        `unplaced` notes (no tab), `pipeline` (provenance for the take), `tuning` (including the capo), and the
        intermediate `note_events`, `piece`, `tabs`, `raw_tabs`, `enc_subseqs`, `dec_subseqs`, `sources` for the
        debug log.
        """
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = wav_path or Path(tmp) / 'audio.wav'
            sf.write(str(wav_path), audio, SR)
            return self.transcribe(audio, wav_path, **options)

    def transcribe(
        self,
        audio,
        wav_path,
        tuning=STD_TUNING,
        capo=0,
        tempo=None,
        genre=UNKNOWN,
        lam=1.3,
        beams=8,
        evidence=True,
        anchor=None,
        fallback='first_viable',
        pooled=(),
        techniques=False,
    ):
        """`audio` at SR, also in the file `wav_path`. `anchor` is 'string:fret,...' for the first notes (Stage 2
        alone then); `pooled` comes from earlier_takes(). With `techniques`, each placed note also gets the marks
        of its playing technique (TechniqueTagger.tab_marks)."""
        anchor_tabs = parse_anchor(anchor)
        sha, diff = git_state()
        pipeline = {
            'stage1_checkpoint': repo_relative(self.stage1_checkpoint),
            'stage2_checkpoint': repo_relative(self.stage2_checkpoint),
            'stage2_evidence': None,
            'no_evidence': not evidence,
            'classifier_runs': [],
            'pooled_takes': [],
            'genre': genre,
            'anchor': anchor,
            'fallback': None,
            'device': self.device,
            'git_sha': sha,
            'git_dirty': bool(diff),
        }
        tuning_with_capo = [t + capo for t in tuning]  # our convention: tuning includes capo
        out = {
            'placed': [],
            'unplaced': [],
            'pipeline': pipeline,
            'tuning': tuning_with_capo,
            'note_events': self.stage1(audio),
        }
        if not out['note_events']:
            print('No notes detected.')
            return out

        piece = notes_to_piece(out['note_events'], tuning_with_capo, capo, tempo, genre=genre)
        model, vocab, max_seq_len, device = self.model, self.vocab, self.max_seq_len, self.device
        t0 = time.time()
        if evidence and not anchor_tabs:
            if capo or list(tuning) != STD_TUNING:
                print('  warning: the string evidence assumes standard tuning without a capo')
            unplaced = [n for n in piece['notes'] if not len(valid_tabs(n['pitch'], piece['tuning'], vocab)[0])]
            piece['notes'] = [
                {**n, 'end': max(n['end'], n['start'] + 0.01)}
                for n in piece['notes']
                if len(valid_tabs(n['pitch'], piece['tuning'], vocab)[0])
            ]
            fused, prior = string_evidence(wav_path, piece, self.classifiers, pooled)
            pipeline |= {
                'classifier_runs': [repo_relative(r) for r in self.runs],
                'pooled_takes': [a.parent.name for a, _ in pooled],
                'lam': lam,
                'beams': beams,
            }
            tabs, enc_subseqs = stage2_fused(piece, model, vocab, max_seq_len, fused, prior, lam, beams, device)
            raw_tabs, dec_subseqs, sources = tabs, [], ['unchanged'] * len(tabs)
            pipeline['stage2_evidence'] = True
            placed = [
                {**n, 'string': s, 'fret': f, 'evidence': e}
                for n, (s, f), e in zip(sorted_notes(piece), tabs, fused, strict=True)
            ]
        else:
            tabs, raw_tabs, enc_subseqs, dec_subseqs, sources = stage2(
                piece, model, vocab, max_seq_len, device, anchor_tabs=anchor_tabs, fallback=fallback
            )
            pipeline |= {'stage2_evidence': False, 'fallback': fallback}
            notes = sorted_notes(piece)
            padded_tabs = list(tabs) + [None] * (len(notes) - len(tabs))
            placed = [{**n, 'string': t[0], 'fret': t[1]} for n, t in zip(notes, padded_tabs, strict=True) if t]
            unplaced = [n for n, t in zip(notes, padded_tabs, strict=True) if t is None]
        print(f'  stage 2 elapsed: {time.time() - t0:.1f}s')
        if techniques and placed:
            t0 = time.time()
            for n, marks in zip(placed, self.tagger.tab_marks(audio, SR, placed), strict=True):
                n['techniques'] = marks
            pipeline |= {
                'technique_runs': [repo_relative(r) for r in sorted(glob.glob(str(REPO_ROOT / self.technique_runs)))],
                'technique_prior': self.technique_prior,
            }
            print(f'  techniques elapsed: {time.time() - t0:.1f}s')
        return out | {
            'placed': placed,
            'unplaced': unplaced,
            'piece': piece,
            'tabs': tabs,
            'raw_tabs': raw_tabs,
            'enc_subseqs': enc_subseqs,
            'dec_subseqs': dec_subseqs,
            'sources': sources,
        }
