"""Canonical train/val/test split boundaries — the single source of truth.

Phase 0 (2026-08-11): before this file existed, the same train/val/test boundary was
independently hardcoded in four places — configs/assets.yaml's `date_range` block (as
inclusive "2023-12-31"), configs/assets.yaml's `training` block (as exclusive
"2024-01-01"), configs/encoder.yaml (a duplicate of the latter), and rl/env.py (as
tz-aware pd.Timestamps). configs/encoder.yaml's copy went stale at a two-generations-old
boundary for a full retrain cycle before a manual audit caught it (see CLAUDE.md's
"walk-forward split" history). Every module that needs a split boundary now imports the
constants below instead of defining its own.

Boundaries use exclusive-cutoff semantics throughout: a row belongs to a split if its
timestamp is >= *_START and < *_END_EXCLUSIVE. TRAIN_END_EXCLUSIVE == VAL_START and
VAL_END_EXCLUSIVE == TEST_START by construction — the three splits are contiguous with
no gap and no overlap.
"""

from __future__ import annotations

TRAIN_START = "2018-01-01"
TRAIN_END_EXCLUSIVE = "2024-01-01"
VAL_START = "2024-01-01"
VAL_END_EXCLUSIVE = "2025-01-01"
TEST_START = "2025-01-01"
TEST_END_EXCLUSIVE = "2026-01-01"


class SplitConfigError(RuntimeError):
    """The boundaries defined in this file are internally inconsistent."""


class TestSetContaminationError(RuntimeError):
    """A component used data at or after TEST_START outside of run_test_set.py.

    Raised by encoder/train.py, regime/hmm.py, and rl/env.py after they apply their
    slice of the canonical split, to verify the *resulting* slice actually respects the
    boundary — not just that the slicing code was written correctly. The 2025 test set
    is sacred (see CLAUDE.md's walk-forward split section): it must never reach encoder
    training/validation, HMM/PCA fitting, or RL training/validation, only the dedicated
    run_test_set.py evaluation.
    """


# ISO "YYYY-MM-DD" strings sort lexicographically in chronological order, so plain
# string comparison is sufficient here — deliberately no pandas dependency in this
# otherwise-trivial module. Raised as an explicit exception (not `assert`) so `python -O`
# cannot silently disable it — this is an infrastructure invariant, not a debug check.
if TRAIN_END_EXCLUSIVE != VAL_START:
    raise SplitConfigError(
        f"TRAIN_END_EXCLUSIVE={TRAIN_END_EXCLUSIVE!r} != VAL_START={VAL_START!r} — "
        "train and val must be contiguous (no gap, no overlap)."
    )
if VAL_END_EXCLUSIVE != TEST_START:
    raise SplitConfigError(
        f"VAL_END_EXCLUSIVE={VAL_END_EXCLUSIVE!r} != TEST_START={TEST_START!r} — "
        "val and test must be contiguous (no gap, no overlap)."
    )
if not (TRAIN_START < TRAIN_END_EXCLUSIVE < VAL_END_EXCLUSIVE < TEST_END_EXCLUSIVE):
    raise SplitConfigError(
        "Split boundaries are not strictly increasing: "
        f"TRAIN_START={TRAIN_START!r}, TRAIN_END_EXCLUSIVE={TRAIN_END_EXCLUSIVE!r}, "
        f"VAL_END_EXCLUSIVE={VAL_END_EXCLUSIVE!r}, TEST_END_EXCLUSIVE={TEST_END_EXCLUSIVE!r}"
    )
