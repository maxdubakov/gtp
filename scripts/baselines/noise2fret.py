"""Noise2Fret (Simionato & Bigo, ISMIR 2026), with its released checkpoints and its own code, on GOAT and our index.

Their code (Noise2Fret_Aux_Losses) is imported from the clone at `repo`; nothing of it is copied here.
  task=goat       their GOAT test: their preprocessing on the test items, their test loop, their tab_metrics.
  task=guitarset  their GuitarSet test: their preprocessing of player 00, WindowedGuitarSet as their training script
                  calls it, their tab_metrics.
  task=index      the index recordings of `guitars`:
                  (a) their two protocols, windows placed from the labels: one 100 ms window from each labelled
                      event (GOAT), or the non-empty cells of a fixed grid (GuitarSet); scored by their tab_metrics,
                      which compares each window's predicted event slots with its labelled ones;
                  (b) audio alone: windows from `audio_windows`, events turned into notes, scored as
                      scripts/e2e/index.py does (mir_eval onsets within `onset_tolerance`, offsets ignored; a tab
                      note is right when its string is right).
                  Our own run `ours` is scored the same way on the same recordings and put beside it.

Run in the virtualenv next to their clone, with our src on the path:
  PYTHONPATH=src ~/Projects/third_party/noise2fret/.venv/bin/python scripts/baselines/noise2fret.py task=goat
  PYTHONPATH=src ~/Projects/third_party/noise2fret/.venv/bin/python scripts/baselines/noise2fret.py task=index
"""

import glob
import importlib
import os
import sys
from pathlib import Path

import hydra
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.log import info
from gtp.takes import match, prf

OmegaConf.register_new_resolver('repo_root', lambda: str(REPO_ROOT), replace=True)

OPEN = [64, 59, 55, 50, 45, 40]  # our string 1 (high e) .. 6 (low E)
GOAT_SR, MODEL_SR = 44100, 16000


def import_theirs(repo):
    root = Path(repo) / 'Noise2Fret_Aux_Losses'
    sys.path[:0] = [
        str(root / 'src'),
        str(root / 'data_preprocess' / 'GOAT'),
        str(root / 'data_preprocess' / 'GuitarSet'),
    ]
    # DiffusionModel imports `Embeddings`, but the file in their repo is spelled Embeddigns.py.
    sys.modules['Embeddings'] = importlib.import_module('Embeddigns')
    return {m: importlib.import_module(m) for m in (
        'U_NET_Token', 'DiffusionModel', 'FeaturesExtractor', 'GOATDataset', 'tab_metrics', 'windowed_guitarset',
        'AlignFrames', 'TimeTabExtraction')}  # fmt: skip


class Checkpoint:
    """A released checkpoint in their DiffusionModel; `strings[i]` is our string number of their string slot i."""

    def __init__(self, theirs, path, device):
        ck = torch.load(path, map_location=device, weights_only=True)
        sd = ck['model_state_dict']
        self.slots, dim = sd['pos_emb.weight'].shape
        n_classes, embed = ck['embedding_state_dict']['weight'].shape
        net = theirs['U_NET_Token'].TokenUNet(
            in_channels=dim, base_channels=64, inject_feature_dim=sd['c_conv.weight'].shape[1], max_len=self.slots
        )
        self.diffusion = (
            theirs['DiffusionModel']
            .DiffusionModel(model=net, noise_steps=1, embed_dim=embed, n_classes=n_classes, device=device)
            .to(device)
        )
        self.diffusion.model.load_state_dict(sd)
        self.diffusion.encoder.load_state_dict(ck['time_embedding_state_dict'])
        self.diffusion.embeddings.load_state_dict(ck['embedding_state_dict'])
        self.diffusion.eval()
        self.device = device
        self.goat = n_classes != 21  # GuitarSet's midi_to_numpy layout has 21 classes, GOAT's checkpoints 24
        # GOAT tokens keep Guitar Pro's numbering (s1 = high e, as ours); GuitarSet's arrays start at low E.
        self.strings = [1, 2, 3, 4, 5, 6] if self.goat else [6, 5, 4, 3, 2, 1]

    def fret(self, c):
        """Fret of an output class, None for a string not played. GOAT: class 0 muted, class 1 + f fret f.
        GuitarSet: their targets read midi_to_numpy's layout (class 20 = not played) with mute_index=0, so class 20
        marks a silent string, class 0 an empty slot, and an open string has no class of its own."""
        if self.goat:
            return c - 1 if c else None
        return c if 0 < c < 20 else None

    @torch.no_grad()
    def predict(self, audio, cond, batch):
        """audio (N, L, 1), cond (N, frames, F) -> (N, slots, 6) class ids; one DDIM step, as their test loop."""
        out = []
        for i in range(0, len(audio), batch):
            a, c = audio[i : i + batch].to(self.device), cond[i : i + batch].to(self.device)
            shape = torch.zeros(len(a), self.slots, 6, dtype=torch.long, device=self.device)
            z = self.diffusion.sample(input=shape, audio=a, cond=c, num_steps=1)
            out.append(self.diffusion.decode(z).argmax(-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, self.slots, 6, dtype=torch.long)


def goat_features(theirs, audio, batch=512):
    f = theirs['FeaturesExtractor'].compute_audio_features
    parts = [f(audio[i : i + batch], sr=MODEL_SR) for i in range(0, len(audio), batch)]
    return torch.cat([torch.cat([p['stft_mag'], p['spectral_flux'], p['brightness']], dim=-1) for p in parts])


def their_scores(theirs, gt, pred):
    m = theirs['tab_metrics'].tab_metrics(gt, pred)
    keys = ['pitch_precision', 'pitch_recall', 'pitch_f_measure', 'tab_precision', 'tab_recall', 'tab_f_measure',
            'tab_disamb', 'false_neg_rate', 'false_pos_rate']  # fmt: skip
    return {k: m[k] for k in keys}


def goat_windows(theirs, root, slots):
    """Their GOATFrameDataset over `root`: 16 kHz audio and label ids of every window with at most `slots` events."""
    ds = theirs['GOATDataset'].GOATFrameDataset(root_dir=str(root), data_dir=str(root), max_events=slots)
    n_events = ds.meta['chords'].apply(ds._count_events).to_numpy()
    keep = np.flatnonzero(n_events <= slots)
    items = [ds[i] for i in keep]
    audio = torch.stack([a for a, _ in items])
    gt = torch.stack([t for _, t in items]).argmax(-1)
    return audio, gt, int((n_events > slots).sum())


def link(dst, src):
    if not dst.exists():
        dst.symlink_to(src)


def run_goat(cfg, theirs, ckpts, out):
    """Their GOAT test set: the standard-tuned items of GOAT's test split, DI audio, their preprocessing."""
    goat = REPO_ROOT / cfg.goat_dir
    meta = pd.read_csv(goat / 'metadata.csv')
    root = Path(cfg.work) / 'goat' / 'test'
    for item in meta[(meta['split'] == 'test') & (meta['tuning_class'] == 'standard')]['item']:
        (root / item).mkdir(parents=True, exist_ok=True)
        for ext in ('.wav', '.gp5'):
            link(root / item / f'{item}{ext}', goat / 'data' / item / f'{item}{ext}')
    if not (root / 'all_note_frame_meta.csv').exists():
        theirs['TimeTabExtraction'].process_dataset_extraction(str(root))
        theirs['AlignFrames'].process_dataset_align(str(root), frame_duration=cfg.goat_window_s)
    rows = []
    for name, ck in ckpts.items():
        if not ck.goat:
            continue
        audio, gt, dropped = goat_windows(theirs, root, ck.slots)
        torch.manual_seed(cfg.seed)
        pred = ck.predict(audio, goat_features(theirs, audio), cfg.batch)
        rows.append({'checkpoint': name, 'windows': len(gt), 'dropped': dropped, **their_scores(theirs, gt, pred)})
        info(f'{name}: {rows[-1]}')
    pd.DataFrame(rows).to_csv(out / 'goat.csv', index=False)


def gs_dataset(theirs, files, gs, slots):
    return theirs['windowed_guitarset'].WindowedGuitarSet(
        files, sr=gs.sr, hop=gs.hop, win_s=gs.win_s, mute_index=gs.mute_index, cluster_frames=gs.cluster_frames,
        target='events', drop_empty=True, max_events=slots,
    )  # fmt: skip


def gs_windows(theirs, ds, ck, batch):
    """Label ids and predictions of every window of a WindowedGuitarSet, batched with their collate."""
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, collate_fn=theirs['windowed_guitarset'].window_collate)
    gt, pred = [], []
    for _, _, token, _, _, _, stft, sf, b, audio, _ in loader:
        gt.append(token.argmax(-1))
        pred.append(ck.predict(audio.unsqueeze(-1), torch.cat([stft, sf, b], dim=-1), batch))
    return torch.cat(gt), torch.cat(pred)


def run_guitarset(cfg, theirs, ckpts, out):
    """Their GuitarSet test: player 00 held out (split_files), 4-bar chunks from their midi_to_numpy, their windows."""
    work = Path(cfg.work) / 'guitarset'
    work.mkdir(parents=True, exist_ok=True)
    # jams_to_midi reads the tempo as the third '-' field of the annotation's whole path, which their own
    # path ('.../Tab-estimator/...') satisfies; a link with that name gives ours the same shape.
    link(work / 'Tab-estimator', REPO_ROOT / cfg.guitarset_dir)
    os.chdir(work)
    os.environ['GUITARSET_DIR'] = 'Tab-estimator'
    jams_to_midi, midi_to_numpy = (importlib.import_module(m) for m in ('jams_to_midi', 'midi_to_numpy'))
    if not Path('data/midi/original').exists():
        jams_to_midi.main(16, 'Tab-estimator')
    kwargs = {'note_resolution': 16, 'down_sampling_rate': cfg.gs.sr, 'bins_per_octave': 24, 'n_bins': 192,
              'hop_length': cfg.gs.hop}  # fmt: skip
    Path('data/npz/original').mkdir(parents=True, exist_ok=True)
    for m in sorted(glob.glob(f'data/midi/original/{cfg.gs.test_player}_*.mid')):
        if not Path(f'data/npz/original/{Path(m).stem}.npz').exists():
            midi_to_numpy.main(m, kwargs)
    files = sorted(glob.glob(str(work / 'data/npz/original/split/*.npz')))
    test = theirs['windowed_guitarset'].split_files(files, test_player=cfg.gs.test_player)[2]
    rows = []
    for name, ck in ckpts.items():
        if ck.goat:
            continue
        ds = gs_dataset(theirs, test, cfg.gs, ck.slots)
        torch.manual_seed(cfg.seed)
        gt, pred = gs_windows(theirs, ds, ck, cfg.batch)
        np.savez_compressed(out / f'guitarset_{name}.npz', gt=gt.numpy(), pred=pred.numpy())
        rows.append({'checkpoint': name, 'files': len(test), 'windows': len(gt), 'dropped': ds.n_over_cap,
                     'mute_index': cfg.gs.mute_index, **their_scores(theirs, gt, pred)})  # fmt: skip
        info(f'{name}: {rows[-1]}')
    pd.DataFrame(rows).to_csv(out / 'guitarset.csv', index=False)


def chord_starts(onsets, strings, tol):
    """Each note's event start: notes on different strings within `tol` of an event's first note join it."""
    out, start, used = np.empty(len(onsets)), None, set()
    for i in np.argsort(onsets, kind='stable'):
        if start is None or onsets[i] - start > tol or strings[i] in used:
            start, used = onsets[i], set()
        used.add(strings[i])
        out[i] = start
    return out


def goat_items(cfg, theirs, recs, notes):
    """Each (dataset, chain) as a GOAT test folder: per recording a link to its audio and our labels in their
    notes CSV (TimeTabExtraction's columns), cut into windows by their AlignFrames.process_item."""
    fields = theirs['TimeTabExtraction'].FIELDNAMES
    roots = {}
    for key, g in recs.groupby(['dataset', 'chain']):
        root = Path(cfg.work) / 'index' / 'goat' / '_'.join(key)
        for i, r in enumerate(g.itertuples()):
            item = root / f'item_{i}'
            item.mkdir(parents=True, exist_ok=True)
            link(item / f'item_{i}.wav', REPO_ROOT / r.audio)
            n = notes.get_group(r.rec)
            start = chord_starts(n['onset'].to_numpy(), n['string'].to_numpy(), cfg.chord_tol_s)
            table = pd.DataFrame({
                'item': item.name, 'string': n['string'], 'fret': n['fret'], 'midi_pitch': n['pitch'],
                'onset_s': start.round(6), 'duration_s': (n['offset'] - start).round(6), 'type': 'normal',
                'token': [f'clean0:note:s{s}:f{f}' for s, f in zip(n['string'], n['fret'], strict=True)],
            }).reindex(columns=fields, fill_value=0)  # fmt: skip
            table.sort_values(['onset_s', 'string']).to_csv(item / f'item_{i}_notes.csv', index=False)
            theirs['AlignFrames'].process_item(str(item), cfg.goat_window_s)
        roots[key] = (root, list(g['rec']))
    return roots


def gs_arrays(cfg, path, notes):
    """A recording in their midi_to_numpy layout (whole recording as one chunk): normalised 22050 Hz audio, STFT,
    flux and brightness from their process_stft_sf_b, frame labels with class 20 = not played, arrays from low E."""
    gs = cfg.gs
    y, sr = librosa.load(path, sr=None, mono=True)
    stft, flux, bright = importlib.import_module('midi_to_numpy').process_stft_sf_b(y, sr, down_sampling_rate=gs.sr)
    audio = librosa.resample(y=librosa.util.normalize(y.astype(float)), orig_sr=sr, target_sr=gs.sr)
    n = len(audio) // gs.hop
    tab, onset = (np.zeros((n, 6, 21)) for _ in range(2))
    tab[..., 20] = onset[..., 20] = 1
    rate = gs.sr / gs.hop
    kept = notes[notes['fret'] <= 19]
    for s, f, on, off in zip(6 - kept['string'], kept['fret'], kept['onset'], kept['offset'], strict=True):
        a, b = round(on * rate), round(off * rate)
        tab[a:b, s] = np.eye(21)[f]
        if a < n:
            onset[a, s] = np.eye(21)[f]
    arrays = {'cqt': np.zeros((n, 1)), 'stft': stft.T[:n], 'sf': flux[:n, None], 'b': bright[:n, None],
              'audio': audio[: n * gs.hop], 'frame_tab': tab, 'frame_tab_onset': onset, 'tempo': 0.0}  # fmt: skip
    return arrays, len(notes) - len(kept)


def gs_items(cfg, recs, notes):
    root = Path(cfg.work) / 'index' / 'guitarset'
    root.mkdir(parents=True, exist_ok=True)
    files, dropped = {}, {}
    for r in recs.itertuples():
        path = root / f'{r.rec.replace("/", "_")}.npz'
        arrays, dropped[r.rec] = gs_arrays(cfg, REPO_ROOT / r.audio, notes.get_group(r.rec))
        np.savez_compressed(path, **arrays)
        files[r.rec] = str(path)
    return files, dropped


def protocol_a(cfg, theirs, ckpts, recs, notes, gs_files, gs_dropped):
    """Their windows and metric on our recordings, per (dataset, chain): each checkpoint under its own protocol."""
    roots = goat_items(cfg, theirs, recs, notes)
    rows = []
    for name, ck in ckpts.items():
        for (dataset, chain), (root, group) in roots.items():
            torch.manual_seed(cfg.seed)
            if ck.goat:
                audio, gt, dropped = goat_windows(theirs, root, ck.slots)
                pred = ck.predict(audio, goat_features(theirs, audio), cfg.batch)
                lost = 0
            else:
                ds = gs_dataset(theirs, [gs_files[r] for r in group], cfg.gs, ck.slots)
                gt, pred = gs_windows(theirs, ds, ck, cfg.batch)
                dropped, lost = ds.n_over_cap, sum(gs_dropped[r] for r in group)
            rows.append({'checkpoint': name, 'protocol': 'goat' if ck.goat else 'guitarset', 'dataset': dataset,
                         'chain': chain, 'recordings': len(group), 'windows': len(gt), 'windows_over_cap': dropped,
                         'notes_beyond_fret_19': lost, **their_scores(theirs, gt, pred)})  # fmt: skip
            info(f'(a) {name} {dataset} {chain}: tab F1 {rows[-1]["tab_f_measure"]:.3f} over {len(gt)} windows')
    return pd.DataFrame(rows)


def audio_windows(cfg, theirs, ck, path, gs_file, kind):
    """Windows from the audio alone -> (16 kHz or 22050 Hz audio, conditioning features, note time per window)."""
    if ck.goat:
        y, sr = librosa.load(path, sr=None, mono=True)
        y = librosa.resample(y=y, orig_sr=sr, target_sr=GOAT_SR) if sr != GOAT_SR else y
        size = round(cfg.goat_window_s * GOAT_SR)
        if kind == 'grid':
            starts = np.arange(0, len(y), size)
        else:
            starts = librosa.onset.onset_detect(y=y, sr=GOAT_SR, units='samples')
        y = np.pad(y, (0, size))
        chunks = torch.from_numpy(np.stack([y[s : s + size] for s in starts]).astype(np.float32))
        audio = torchaudio.functional.resample(chunks, GOAT_SR, MODEL_SR).unsqueeze(-1)
        offset = size / 2 if kind == 'grid' else 0
        return audio, goat_features(theirs, audio), (starts + offset) / GOAT_SR
    z, gs = np.load(gs_file), cfg.gs
    y, cond = z['audio'], np.concatenate([z['stft'], z['sf'], z['b']], axis=1)
    frames = round(gs.win_s * gs.sr / gs.hop)
    if kind == 'grid':
        first = np.arange(0, len(cond) - frames + 1, frames)
    else:
        onsets = librosa.onset.onset_detect(y=y, sr=gs.sr, hop_length=gs.hop)
        first = onsets[onsets + frames <= len(cond)]
    cond = torch.from_numpy(np.stack([cond[f : f + frames] for f in first]).astype(np.float32))
    audio = torch.from_numpy(np.stack([y[f * gs.hop : (f + frames) * gs.hop] for f in first]).astype(np.float32))
    offset = frames / 2 if kind == 'grid' else 0
    return audio.unsqueeze(-1), cond, (first + offset) * gs.hop / gs.sr


def to_notes(ck, pred, times, slots):
    """Every played string of a window's first `slots` events becomes a note at the window's time; a (string, fret)
    repeated within one window is one note, as their merge='notes' does for the labels."""
    notes = []
    for t, events in zip(times, pred[:, :slots].tolist(), strict=True):
        seen = set()
        for event in events:
            for i, c in enumerate(event):
                fret = ck.fret(c)
                if fret is None or (i, fret) in seen:
                    continue
                seen.add((i, fret))
                s = ck.strings[i]
                notes.append({'start': float(t), 'end': float(t) + 0.1, 'pitch': OPEN[s - 1] + fret, 'string': s})
    return notes


def score(ref, est, tolerance):
    pairs = match(ref, est, tolerance)
    tab = sum(ref[r]['string'] == est[e]['string'] for r, e in pairs)
    return {'n_ref': len(ref), 'n_est': len(est), 'n_matched': len(pairs), 'n_tab': tab}


def ours(cfg, recs):
    """Our run's per-recording counts on the same recordings, as scripts/e2e/bootstrap.py reads them."""
    run = REPO_ROOT / cfg.ours
    notes = pd.read_csv(run / 'notes.csv.gz')
    notes = notes[(notes['input'] == 'stage1') & notes['rec'].isin(recs['rec'])]
    counts = pd.read_csv(run / 'counts.csv')
    counts = counts[(counts['input'] == 'stage1') & counts['rec'].isin(recs['rec'])]
    ok = (notes[cfg.ours_method] == notes['ref_string']).groupby(notes['rec']).sum().rename('n_tab')
    per_rec = counts[['rec', 'dataset', 'chain', 'n_ref', 'n_est', 'n_matched']].join(ok, on='rec').fillna({'n_tab': 0})
    assert len(per_rec) == len(recs), f'{cfg.ours} has {len(per_rec)} of the {len(recs)} recordings'
    return per_rec.assign(model=f'ours_{cfg.ours_method}', windows='stage1')


def summarize(per_rec, by):
    """Tab and note F1 per group: the mean of the recordings' F1, and pooled over the group's notes (bootstrap.py)."""
    per_rec = per_rec.assign(
        note_f1=2 * per_rec['n_matched'] / (per_rec['n_est'] + per_rec['n_ref']).clip(lower=1),
        tab_f1=2 * per_rec['n_tab'] / (per_rec['n_est'] + per_rec['n_ref']).clip(lower=1),
    )
    rows = []
    groups = [*per_rec.groupby(['model', 'windows', *by]), *(((*k, *['all'] * len(by)), g)
              for k, g in per_rec.groupby(['model', 'windows']))]  # fmt: skip
    for key, g in groups:
        s = g[['n_ref', 'n_est', 'n_matched', 'n_tab']].sum()
        rows.append(dict(zip(['model', 'windows', *by], key, strict=True)) | {
            'recordings': len(g), 'tab_f1_mean': g['tab_f1'].mean(), 'note_f1_mean': g['note_f1'].mean(),
            'tab_f1_pooled': prf(s['n_tab'], s['n_est'], s['n_ref'])[2],
            'note_f1_pooled': prf(s['n_matched'], s['n_est'], s['n_ref'])[2], **s.to_dict(),
        })  # fmt: skip
    return pd.DataFrame(rows), per_rec


def run_index(cfg, theirs, ckpts, out):
    index = REPO_ROOT / cfg.index_dir
    recs = pd.read_csv(index / 'recordings.csv')
    recs = recs[recs['guitar'].isin(list(cfg.guitars))]
    if cfg.limit:
        recs = recs.groupby(['dataset', 'chain']).head(cfg.limit)
    recs = recs.reset_index(drop=True)
    notes = pd.read_csv(index / 'notes.csv.gz')
    notes = notes[notes['rec'].isin(recs['rec'])].groupby('rec')
    gs_files, gs_dropped = gs_items(cfg, recs, notes)
    a = protocol_a(cfg, theirs, ckpts, recs, notes, gs_files, gs_dropped)
    a.to_csv(out / 'protocol_a.csv', index=False)

    per_rec, predicted = [], []
    for name, ck in ckpts.items():
        for kind in cfg.audio_windows:
            torch.manual_seed(cfg.seed)
            for r in recs.itertuples():
                audio, cond, times = audio_windows(cfg, theirs, ck, REPO_ROOT / r.audio, gs_files[r.rec], kind)
                est = to_notes(ck, ck.predict(audio, cond, cfg.batch), times, ck.slots if kind == 'grid' else 1)
                ref = [{'start': n.onset, 'end': n.offset, 'pitch': int(n.pitch), 'string': int(n.string)}
                       for n in notes.get_group(r.rec).itertuples()]  # fmt: skip
                row = {'model': name, 'windows': kind, 'rec': r.rec, 'dataset': r.dataset, 'chain': r.chain}
                per_rec.append(row | score(ref, est, cfg.onset_tolerance))
                predicted += [row | n for n in est]
            info(f'(b) {name} {kind}: {sum(p["n_tab"] for p in per_rec[-len(recs) :])} right tab notes')
    per_rec = pd.concat([pd.DataFrame(per_rec), ours(cfg, recs)], ignore_index=True)
    summary, per_rec = summarize(per_rec, ['dataset', 'chain'])
    by_chain, _ = summarize(per_rec, ['chain'])
    per_rec.to_csv(out / 'per_recording.csv', index=False)
    summary.to_csv(out / 'summary.csv', index=False)
    by_chain.to_csv(out / 'summary_by_chain.csv', index=False)
    pd.DataFrame(predicted).to_csv(out / 'predicted_notes.csv.gz', index=False)
    show = ['model', 'windows', 'dataset', 'chain', 'tab_f1_mean', 'note_f1_mean', 'tab_f1_pooled', 'note_f1_pooled']
    info(f'(b) and ours\n{summary[show].round(3).to_string(index=False)}')
    cols = ['checkpoint', 'dataset', 'chain', 'windows', 'tab_f_measure', 'pitch_f_measure', 'tab_disamb']
    info(f'(a)\n{a[cols].round(3).to_string(index=False)}')


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'baselines'), config_name='noise2fret')
def main(cfg):
    torch.set_num_threads(cfg.threads)
    theirs = import_theirs(cfg.repo)
    ckpts = {name: Checkpoint(theirs, Path(cfg.weights) / p, cfg.device) for name, p in cfg.checkpoints.items()}
    out = Path(HydraConfig.get().runtime.output_dir)
    {'goat': run_goat, 'guitarset': run_guitarset, 'index': run_index}[cfg.task](cfg, theirs, ckpts, out)
    info(f'Saved {out}')


if __name__ == '__main__':
    main()
