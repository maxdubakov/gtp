"""Where the model's tabs of home-recording takes differ from the user's corrections (see gtp.takes).

Every take with an edited.json is compared with its prediction two ways:
  * through source_id: each prediction note is kept (string, fret/pitch or onset maybe changed) or
    deleted (a Stage 1 false positive), and edited notes without a source were added (Stage 1 misses).
    String accuracy is over kept notes whose pitch was right, where Stage 2's choice is what counts.
  * by onset matching (same pitch, onsets within onset_tolerance, as scripts/e2e/guitarset.py):
    Stage 1 note P/R/F1, and tab P/R/F1 counting matched notes on the right string.

Writes notes.csv (one row per note: kind, predicted and edited pitch/string/fret/onset, classifier
evidence, recording metadata) and takes.csv (per take and overall) to out_dir/<timestamp>/.

Usage:
  uv run scripts/takes/report.py
  uv run scripts/takes/report.py verified_only=true
"""

from datetime import datetime

import hydra
import pandas as pd

from gtp import REPO_ROOT, takes
from gtp.log import info

SUMS = ['pred', 'edited', 'kept', 'deleted', 'added', 'pitch_changed', 'string_changed', 'onset_moved', 'changed']
SUMS += ['string_ok', 'string_scored', 'matched', 'tab_tp']


def rates(row):
    row['string_acc'] = row['string_ok'] / max(row['string_scored'], 1)
    row['note_p'], row['note_r'], row['note_f1'] = takes.prf(row['matched'], row['pred'], row['edited'])
    row['tab_p'], row['tab_r'], row['tab_f1'] = takes.prf(row['tab_tp'], row['pred'], row['edited'])
    return row


@hydra.main(version_base='1.3', config_path=str(REPO_ROOT / 'conf' / 'takes'), config_name='report')
def main(cfg):
    root = REPO_ROOT / cfg.takes_dir
    notes, per_take = [], []
    for take_id in takes.take_ids(root):
        take = takes.load(root / take_id)
        prediction, edited, meta = take['prediction'], take['edited'], take['meta']
        if not edited or (cfg.verified_only and edited['status'] != 'verified'):
            continue
        rows = takes.compare(prediction, edited)
        recording = {k: meta.get(k, '') for k in ('guitar', 'pickup', 'mic', 'input_device', 'room', 'player')}
        notes += [{'take': take_id, 'status': edited['status'], **recording, **r} for r in rows]
        counts = takes.counts(rows) | takes.scores(prediction, edited, cfg.onset_tolerance)
        per_take.append(rates({'take': take_id, 'status': edited['status'], **counts}))
    if not per_take:
        info(f'No {"verified" if cfg.verified_only else "edited"} takes in {root}')
        return

    table = pd.DataFrame(per_take)
    overall = rates({'take': 'all', 'status': '', **table[SUMS].sum().to_dict()})
    table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)
    out = REPO_ROOT / cfg.out_dir / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out.mkdir(parents=True)
    pd.DataFrame(notes).convert_dtypes().to_csv(out / 'notes.csv', index=False)
    table.to_csv(out / 'takes.csv', index=False)

    cols = ['take', 'status', 'pred', 'edited', 'kept', 'string_acc', 'string_changed', 'pitch_changed']
    cols += ['onset_moved', 'deleted', 'added', 'note_p', 'note_r', 'note_f1', 'tab_f1']
    info(
        f'{len(per_take)} takes, {int(overall["pred"])} predicted notes\n{table[cols].round(3).to_string(index=False)}'
    )
    info(f'Wrote {out}/notes.csv and takes.csv')


if __name__ == '__main__':
    main()
