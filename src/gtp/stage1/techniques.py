"""Playing-technique labels of notes."""

import numpy as np


def class_of(techniques, classes, other_marks):
    """Index into `classes` per note, or -1 for a note to leave out."""
    out = np.full(len(techniques), -1)
    for i, marks in enumerate(techniques.fillna('').str.split(',')):
        named = [classes.index(m) for m in marks if m in classes]
        if named:
            out[i] = max(named)
        elif not set(marks) & set(other_marks):
            out[i] = classes.index('plain')
    return out
