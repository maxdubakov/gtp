"""Name the playing technique of each note of a recording: trained technique classifiers and the tab prior, applied
to notes that come from anywhere (labels, or the note model and the string step)."""

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.stage1.strings import SAMPLE_RATE, spectrogram
from gtp.stage1.tab_prior import load_prior, log_softmax, prior_over
from gtp.stage1.techniques import pitch_windows, string_context


class TechniqueTagger:
    """Classifier runs (scripts/techniques/train.py; their log-probabilities are averaged) and optionally a tab
    prior (train_prior.py) that counts `weight` times in the heads `prior_heads`, as fuse_prior.py fuses them."""

    def __init__(self, runs, prior=None, weight=1.0, prior_heads=('attack', 'glide', 'pitch'), device='cpu'):
        self.device, self.weight, self.prior_heads = device, weight, tuple(prior_heads)
        self.models = []
        for run in map(Path, runs):
            cfg = OmegaConf.load(run / '.hydra' / 'config.yaml')
            cache = json.loads((run / 'provenance.json').read_text())['cache']
            given = (cfg.feature, bool(cfg.context), cfg.get('attack'), cfg.get('prior'), list(cfg.more_views))
            if given != ('cqt36', True, None, None, []) or not cache.get('pitch_relative') or cfg.pitch_harmonics:
                raise ValueError(f'{run.name}: only a pitch-following cqt36 view with the note before is supported')
            heads = {name: list(head.classes) for name, head in cfg.heads.items()}
            layout, relative = cache['layout'], cache['pitch_relative']
            window = (layout['pre_frames'], layout['max_post_frames'], relative['semitones'], relative['harmonics'])
            shape = (
                relative['harmonics'],
                6 * relative['semitones'] + 1,
                layout['pre_frames'] + layout['max_post_frames'],
            )
            model = instantiate(cfg.model, views=[shape], n_classes=sum(len(c) for c in heads.values()), context=4)
            model.load_state_dict(torch.load(run / 'model.pt', map_location=device, weights_only=True))
            scale = np.load(run / 'classifier.npz')
            stats = [torch.from_numpy(scale[k]).to(device) for k in ('mean0', 'std0')]
            self.models.append((model.to(device).eval(), stats, window, heads))
        self.heads = self.models[0][3]
        if any(m[3] != self.heads or m[2] != self.models[0][2] for m in self.models):
            raise ValueError('the runs answer different heads or read different windows')
        self.prior = load_prior(REPO_ROOT / prior, self.heads, device) if prior else None

    def log_probs(self, audio, sr, notes):
        """{head: (notes, classes)} for the notes (onset, offset, pitch, string, fret) of one recording."""
        notes = notes.reset_index(drop=True).assign(rec='recording')
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        spec = spectrogram(audio[None], 'cqt36')
        windows = pitch_windows(spec, notes['onset'].to_numpy(), notes['pitch'].to_numpy(), *self.models[0][2])
        x = torch.from_numpy(windows.astype(np.float16)).to(self.device)
        pitch = torch.tensor(notes['pitch'].to_numpy(), device=self.device)
        context = torch.tensor(string_context(notes), device=self.device)
        sizes_of = [len(c) for c in self.heads.values()]
        total = [0.0] * len(sizes_of)
        with torch.no_grad():
            for model, (mean, std), _, _ in self.models:
                logits = torch.cat(
                    [
                        model([(x[i : i + 512].float() - mean) / std], pitch[i : i + 512], context[i : i + 512])
                        for i in range(0, len(x), 512)
                    ]
                )
                for h, part in enumerate(logits.split(sizes_of, dim=1)):
                    total[h] = total[h] + torch.log_softmax(part, dim=1) / len(self.models)
        out = {name: part.cpu().numpy() for name, part in zip(self.heads, total, strict=True)}
        if self.prior:
            gains, _ = prior_over(self.prior, notes, self.heads, self.device)
            for name in self.prior_heads:
                out[name] = out[name] + self.weight * gains[name]
        return {name: log_softmax(part) for name, part in out.items()}

    def tab_marks(self, audio, sr, notes):
        """Per note (dicts with start, end, pitch, string, fret), the marks a tab prints on it: palm_mute, dead,
        bend, slide (the pitch slides into the note), hammer_on and pull_off (not picked; which of the two is read
        from the fret of the note before on the string, and a note with none before it gets neither)."""
        frame = pd.DataFrame(notes).rename(columns={'start': 'onset', 'end': 'offset'})
        named = {
            name: [self.heads[name][k] for k in part.argmax(axis=1)]
            for name, part in self.log_probs(audio, sr, frame).items()
        }
        ordered = frame.sort_values(['string', 'onset'], kind='stable')
        step = (ordered['fret'] - ordered.groupby('string')['fret'].shift()).reindex(frame.index).to_numpy()
        out = []
        for i in range(len(frame)):
            marks = [named[h][i] for h in ('muting', 'pitch') if named[h][i] != self.heads[h][0]]
            if named['glide'][i] == 'slide':
                marks.append('slide')
            elif named['attack'][i] == 'connected' and step[i] and not np.isnan(step[i]):
                marks.append('hammer_on' if step[i] > 0 else 'pull_off')
            out.append(marks)
        return out
