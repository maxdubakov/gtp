"""Match a tab's notes (score time) to the notes of a MIDI aligned to the audio (performance time)."""

import numpy as np


def clusters(notes, start, gap):
    out = []
    for n in notes:
        if out and start(n) - start(out[-1][-1]) <= gap:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def strings_by_alignment(tab, midi, cfg):
    """{index of a MIDI note: tab note} for the MIDI notes the tab's chords align to."""
    chords = clusters(sorted(tab, key=lambda n: (n['start'], n['pitch'])), lambda n: n['start'], 1e-4)
    order = sorted(range(len(midi)), key=lambda k: (midi[k].start, midi[k].pitch))
    onsets = clusters(order, lambda k: midi[k].start, cfg.cluster_gap_s)
    pa = [{n['pitch'] for n in c} for c in chords]
    pb = [{midi[k].pitch for k in c} for c in onsets]
    n, m = len(chords), len(onsets)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    step = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        for j in range(max(1, i - cfg.band), min(m, i + cfg.band) + 1):
            options = (cost[i - 1, j - 1] + 1 - len(pa[i - 1] & pb[j - 1]) / len(pa[i - 1] | pb[j - 1]),
                       cost[i - 1, j] + 0.5, cost[i, j - 1] + 0.5)  # fmt: skip
            step[i, j] = int(np.argmin(options))
            cost[i, j] = options[step[i, j]]
    matched, i, j = {}, n, m
    while i > 0 and j > 0:
        if step[i, j] == 0:
            pool = {}
            for t in chords[i - 1]:
                pool.setdefault(t['pitch'], []).append(t)
            for k in onsets[j - 1]:
                if pool.get(midi[k].pitch):
                    matched[k] = pool[midi[k].pitch].pop(0)
            i, j = i - 1, j - 1
        elif step[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return matched
