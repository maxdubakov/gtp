"""Per-note string evidence for Stage 2: the audio string classifier's distribution over the 6 strings.

GuitarSet notes get real out-of-fold classifier outputs, looked up by (track, onset, string), which
survive capo and tuning augmentation since those keep string and fret.

Notes of other sources have no audio, so their outputs are simulated. A piece picks one donor GuitarSet
track (with its classifier seed), and each note borrows the output of a donor note on the same string and
fret: from the donor track if it has one, else from the donor's player and seed at the nearest fret. Strings
that cannot play the note in the piece's tuning are then dropped. One donor per piece keeps its errors
correlated: on real recordings whole tracks are hard, not scattered notes.
"""

import glob
import re
from dataclasses import dataclass

import numpy as np
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.stage1.strings import OPEN_STRINGS, PLAYERS, split_players
from gtp.stage2.tokenizer import MAX_FRET

N_STRINGS = 6


def sweep_runs(sweep_dir):
    """{(player, seed): run_dir} for every finished run of a string-classifier sweep."""
    runs = {}
    for path in glob.glob(str(REPO_ROOT / sweep_dir / '*' / '*' / 'test_probs.npz')):
        run_dir = path.removesuffix('/test_probs.npz')
        cfg = OmegaConf.load(f'{run_dir}/.hydra/config.yaml')
        runs[(PLAYERS[cfg.test_player], int(cfg.seed))] = run_dir
    return runs


@dataclass
class Pool:
    """Real classifier outputs, one row per (note, seed)."""

    player: np.ndarray
    seed: np.ndarray
    track: np.ndarray
    onset: np.ndarray
    string: np.ndarray
    fret: np.ndarray
    probs: np.ndarray  # (N, 6)

    @classmethod
    def load(cls, runs, calibrated=True):
        """`runs`: {(player, seed): run_dir}; each run contributes its test player's notes."""
        cols = {k: [] for k in ('player', 'seed', 'track', 'onset', 'string', 'fret', 'probs')}
        for (player, seed), run_dir in sorted(runs.items()):
            z = np.load(f'{run_dir}/test_probs.npz')
            assert all(re.match(player, t) for t in z['track']), run_dir
            n = len(z['track'])
            cols['player'].append(np.full(n, player))
            cols['seed'].append(np.full(n, seed))
            cols['track'].append(z['track'])
            cols['onset'].append(z['onset'])
            cols['string'].append(z['string'])
            cols['fret'].append(z['pitch'] - OPEN_STRINGS[z['string'] - 1])
            cols['probs'].append(z['probs_calibrated' if calibrated else 'probs'])
        return cls(**{k: np.concatenate(v) for k, v in cols.items()})


def nearest_fret_table(frets):
    """(MAX_FRET + 1,) table: fret -> the nearest fret present in `frets`."""
    present = np.asarray(frets)
    query = np.arange(MAX_FRET + 1)
    return present[np.abs(present[None, :] - query[:, None]).argmin(axis=1)]


def restrict(probs, pitches, tuning):
    """Zero the strings that cannot play each pitch in `tuning` and renormalize; NaN rows where no mass is left."""
    fret = np.asarray(pitches)[:, None] - np.asarray(tuning)[None, :]
    p = np.where((fret >= 0) & (fret <= MAX_FRET), probs, 0.0)
    total = p.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, p / total, np.nan)


class StringEvidence:
    """Per-note string distributions as (n_notes, 6) arrays, with NaN rows where a note has none."""

    def __init__(self, real: Pool, donors: Pool):
        self.real_probs = real.probs
        self.real = {}
        for i, key in enumerate(zip(real.track, np.round(real.onset, 4), real.string, strict=True)):
            self.real.setdefault(key, {})[int(real.seed[i])] = i
        self.real_seeds = sorted(set(real.seed.tolist()))

        self.donor_probs = donors.probs
        keys = list(
            zip(
                donors.player.tolist(),
                donors.seed.tolist(),
                donors.track.tolist(),
                donors.string.tolist(),
                donors.fret.tolist(),
                strict=True,
            )
        )
        bins, frets = {}, {}
        for i, (player, seed, track, string, fret) in enumerate(keys):
            bins.setdefault((player, seed, track, string, fret), []).append(i)
            bins.setdefault((player, seed, string, fret), []).append(i)
            frets.setdefault((player, seed, string), set()).add(fret)
        self.bins = {k: np.array(v) for k, v in bins.items()}
        self.nearest = {k: nearest_fret_table(sorted(v)) for k, v in frets.items()}
        self.donor_tracks = sorted({k[:3] for k in keys})

    def sample(self, piece, rng: np.random.Generator):
        notes = piece['notes']
        if len(piece['tuning']) != N_STRINGS:
            return np.full((len(notes), N_STRINGS), np.nan)
        rows, table = (self._real if piece['source'] == 'guitarset' else self._simulated)(piece, rng)
        probs = np.full((len(notes), N_STRINGS), np.nan)
        found = rows >= 0
        probs[found] = table[rows[found]]
        return restrict(probs, [n['pitch'] for n in notes], piece['tuning'])

    def _real(self, piece, rng):
        track, seed = piece['filename'].removesuffix('.json'), int(rng.choice(self.real_seeds))
        rows = [self.real.get((track, round(n['start'], 4), n['string']), {}).get(seed, -1) for n in piece['notes']]
        return np.array(rows, dtype=int), self.real_probs

    def _simulated(self, piece, rng):
        player, seed, track = self.donor_tracks[rng.integers(len(self.donor_tracks))]
        u = rng.random(len(piece['notes']))
        rows = np.full(len(piece['notes']), -1)
        for i, n in enumerate(piece['notes']):
            s = n['string']
            candidates = self.bins.get((player, seed, track, s, n['fret']))
            if candidates is None:
                nearest = self.nearest.get((player, seed, s))
                if nearest is None:
                    continue
                candidates = self.bins[player, seed, s, int(nearest[min(n['fret'], MAX_FRET)])]
            rows[i] = candidates[int(u[i] * len(candidates))]
        return rows, self.donor_probs


@dataclass
class Perturbation:
    """Training-time noise on a piece's evidence, so Stage 2 learns how far to trust it.

    Each piece gets a temperature T = exp(U(-log_temperature, log_temperature)), applied as p ** (1 / T);
    all its evidence is dropped with `piece_drop`, and each remaining note's with `note_drop`.
    """

    log_temperature: float = 0.0
    piece_drop: float = 0.0
    note_drop: float = 0.0

    def __call__(self, probs, rng):
        if rng.random() < self.piece_drop:
            return np.full_like(probs, np.nan)
        p = probs ** np.exp(-rng.uniform(-self.log_temperature, self.log_temperature))
        p = p / p.sum(axis=1, keepdims=True)
        p[rng.random(len(p)) < self.note_drop] = np.nan
        return p


def rows_or_none(probs):
    return [None if np.isnan(row[0]) else row for row in probs]


def with_evidence(piece, evidence: StringEvidence, rng, perturbation: Perturbation | None = None):
    """A copy of `piece` whose notes carry an 'evidence' (6,) string distribution, or None."""
    probs = evidence.sample(piece, rng)
    if perturbation is not None:
        probs = perturbation(probs, rng)
    return {**piece, 'notes': [{**n, 'evidence': p} for n, p in zip(piece['notes'], rows_or_none(probs), strict=True)]}


def load_evidence(cfg):
    """StringEvidence from an evidence config: real outputs for every player, donors from Stage 2's train players."""
    test = PLAYERS[cfg.test_player]
    heldin = sweep_runs(cfg.heldin_sweep)
    test_runs = {k: v for k, v in sweep_runs(cfg.test_sweep).items() if k[0] == test}
    assert test_runs and not any(p == test for p, _ in heldin), (sorted(heldin), sorted(test_runs))
    train_players = split_players(cfg.test_player)['train']
    donors = {k: v for k, v in heldin.items() if k[0] in train_players}
    return StringEvidence(Pool.load({**heldin, **test_runs}, cfg.calibrated), Pool.load(donors, cfg.calibrated))
