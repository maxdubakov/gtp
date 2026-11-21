"""Which DadaGP files hold the pieces of GOAT, so that a tab prior trained on DadaGP has not read GOAT's tabs.

GOAT's tabs carry no song names, so pieces are matched by content: every run of `run_notes` consecutive notes of a
GOAT tab, tied notes left out, is looked up among the runs of every corpus track (build_tab_corpus.py), once by
strings and frets and once by the pitch steps between the notes. A run found with more than `common_artists` artists
says nothing about which song it is and is not counted. A corpus file holds a piece when one of its tracks has at
least `min_share` of the piece's runs, read either way; so does every other version of that song in the corpus
(same artist folder and title, whatever the number in brackets and the format).

Writes <corpus_dir>/goat_matches.csv (per GOAT item: the largest share of its runs in one file, by frets, by steps
and by either, and that file) and goat_files.csv (every file holding a piece: file, item, share; no item for a
version found by its name).

Usage:
  uv run scripts/techniques/match_goat_tabs.py
"""

import re

import hydra
import numpy as np
import pandas as pd

from gtp import REPO_ROOT
from gtp.log import info
from gtp.stage2.guitarpro import MARKS, parse_gp_track

TIE = np.uint32(1 << MARKS.index('tie'))


def runs(code, n):
    """A hash per run of `n` consecutive codes."""
    if len(code) < n:
        return np.zeros(0, dtype=np.uint64)
    out = np.zeros(len(code) - n + 1, dtype=np.uint64)
    for k in range(n):
        out = out * np.uint64(1_000_003) + code[k : len(code) - n + 1 + k].astype(np.uint64)
    return out


def codes(string, fret, tuning):
    """Two readings of a tab's notes: the string and fret of each, and the pitch step from each note to the next.
    The second also finds a piece that is fingered elsewhere on the neck or written in another tuning."""
    pitch = np.take_along_axis(tuning, string[:, None].astype(np.int64) - 1, axis=1)[:, 0] + fret
    return {'frets': string.astype(np.int64) * 128 + fret + 1, 'steps': np.diff(pitch) + 128}


def song(file):
    """A corpus file's artist folder and title, alike for 'X - Song (2).gp3' and 'X - Song.gp4'."""
    folder, name = file.rsplit('/', 1)
    title = re.sub(r'(\.gp\w*)+$', '', name)
    return folder + '/' + re.sub(r'[^a-z0-9]', '', re.sub(r'\(\d+\)', '', title.lower()))


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'techniques'), config_name='match_goat_tabs')
def main(cfg):
    corpus = REPO_ROOT / cfg.corpus_dir
    tracks = pd.read_csv(corpus / 'tracks.csv')
    notes = np.load(corpus / 'notes.npz')
    played = notes['marks'] & TIE == 0
    track = notes['track'][played]
    artist = tracks['artist'].factorize()[0]
    tunings = np.array([list(map(int, t.split())) for t in tracks['tuning']])
    tables = {}
    for kind, code in codes(notes['string'][played], notes['fret'][played].astype(np.int64), tunings[track]).items():
        hashes = runs(code, cfg.run_notes)
        within = track[: len(hashes)] == track[cfg.run_notes - 1 :][: len(hashes)]
        hashes, of_track = hashes[within], track[: len(hashes)][within]
        order = np.argsort(hashes, kind='stable')
        tables[kind] = hashes[order], of_track[order]
    info(f'{len(tables["frets"][0])} runs of {cfg.run_notes} notes in {len(tracks)} tracks')

    goat = REPO_ROOT / cfg.goat_dir
    items, files = [], []
    for r in pd.read_csv(goat / 'metadata.csv').itertuples():
        try:
            tab = parse_gp_track(goat / 'data' / r.gp5_path.split('/', 1)[1], 0, techniques=True)
        except ValueError:  # unreadable for build_goat_index.py too, so the piece is in no index
            continue
        played = sorted(
            (n for n in tab['notes'] if 'tie' not in n['techniques']), key=lambda n: (n['start'], n['string'])
        )
        string, fret = (np.array([n[k] for n in played], dtype=np.int64) for k in ('string', 'fret'))
        row, per_file = {'item': r.item, 'player': r.player, 'split': r.split, 'notes': len(played)}, []
        for kind, code in codes(string, fret, np.tile(tab['tuning'], (len(played), 1))).items():
            hashes, of_track = tables[kind]
            mine = np.unique(runs(code, cfg.run_notes))
            lo, hi = np.searchsorted(hashes, mine, 'left'), np.searchsorted(hashes, mine, 'right')
            # A run counts once per track, however often the track repeats it.
            hits = [np.unique(of_track[a:b]) for a, b in zip(lo, hi, strict=True)]
            rare = [h for h in hits if len(np.unique(artist[h])) <= cfg.common_artists]
            hits = np.concatenate(rare or [of_track[:0]])
            share = pd.Series(np.bincount(hits, minlength=len(tracks)) / max(len(mine), 1))
            per_file.append(share.groupby(tracks['file']).max())
            row[kind] = per_file[-1].max()
        per_file = pd.concat(per_file, axis=1).max(axis=1).sort_values(ascending=False)
        items.append({**row, 'best': per_file.iloc[0], 'best_file': per_file.index[0]})
        files += [{'file': f, 'item': r.item, 'share': s} for f, s in per_file[per_file >= cfg.min_share].items()]
    items, files = pd.DataFrame(items), pd.DataFrame(files)
    songs = pd.Series(tracks['file'].unique())
    versions = songs[songs.map(song).isin(set(files['file'].map(song))) & ~songs.isin(files['file'])]
    files = pd.concat([files, pd.DataFrame({'file': versions})])
    items.to_csv(corpus / 'goat_matches.csv', index=False)
    files.to_csv(corpus / 'goat_files.csv', index=False)
    bands = [-1, 0.1, 0.3, 0.6, 0.9, 1.0]
    counts = pd.DataFrame({k: pd.cut(items[k], bands).value_counts().sort_index() for k in ('frets', 'steps', 'best')})
    info(f'{len(items)} GOAT items by the share of their runs in the best file:\n{counts}')
    info(f'{files["file"].nunique()} files hold one of {files["item"].nunique()} pieces')
    with pd.option_context('display.width', 250, 'display.max_colwidth', 70, 'display.float_format', '{:.2f}'.format):
        info(items.sort_values('best', ascending=False).head(30).to_string(index=False))


if __name__ == '__main__':
    main()
