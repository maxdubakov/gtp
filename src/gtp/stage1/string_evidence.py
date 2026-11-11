"""String evidence for the notes of real recordings: what the end-to-end scripts and the demo share.

A recording's notes get the classifier ensemble's string probabilities, an inharmonicity curve and their harmonic
levels. Recordings of one guitar through one chain are then fused together (inharmonicity.fuse_guitar): the string
constants and the string envelopes are fitted to those recordings themselves, without labels, so the more of a
guitar's playing is pooled, the better they are known (about 1,300 notes is enough).
"""

import librosa
import numpy as np
from omegaconf import OmegaConf

from gtp import REPO_ROOT
from gtp.stage1 import strings as st
from gtp.stage1.bprofile import grid, recording_profiles
from gtp.stage1.envelope import note_levels
from gtp.stage1.inharmonicity import fuse_guitar

CONF = REPO_ROOT / 'conf' / 'strings'


def settings(profiles=None, physics=None, envelope=None):
    """The curve, fusion and envelope settings (conf/strings), each merged with its overrides."""
    files = ('b_profiles.yaml', 'physics_fusion.yaml', 'harmonic_levels.yaml')
    return tuple(
        OmegaConf.merge(OmegaConf.load(CONF / f), o or {})
        for f, o in zip(files, (profiles, physics, envelope), strict=True)
    )


def physical(audio, onsets, ends, pitches, profiles, envelope):
    """A recording's inharmonicity curves (notes, grid) and harmonic levels (notes, harmonics)."""
    y, sr = librosa.load(audio, sr=profiles.sample_rate, mono=True)
    return (
        recording_profiles(y, sr, onsets, ends, pitches, profiles).sum(axis=1),
        note_levels(y, sr, onsets, ends, pitches, envelope),
    )


def classifier_probs(audio, onsets, pitches, classifiers):
    """(notes, 6) string probabilities, averaged over the classifiers."""
    y, _ = librosa.load(audio, sr=st.SAMPLE_RATE, mono=True)
    spec = st.spectrogram(y[None], classifiers[0].feature)
    return np.mean([c.predict(spec, onsets, pitches) for c in classifiers], axis=0)


def fuse_recordings(probs, curves, levels, pitches, profiles, physics, envelope):
    """Fuse the recordings of one guitar and chain (lists, one entry per recording).

    Returns the fused probabilities per recording, the string constants and the string-set family.
    """
    fused, const, family = fuse_guitar(
        np.concatenate(probs),
        np.concatenate(curves),
        grid(profiles),
        np.concatenate([np.asarray(p) for p in pitches]),
        physics,
        np.concatenate(levels),
        envelope,
    )
    return np.split(fused, np.cumsum([len(p) for p in probs])[:-1]), const, family
