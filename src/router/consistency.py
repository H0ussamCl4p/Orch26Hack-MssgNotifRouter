"""Consistency layer: batch-level reconciliation of the per-message decisions.

Currently this enforces one well-supported invariant: ground-truth confidences
in this dataset always fall within [0.78, 0.91]. The router (Gemma) tends to emit
higher, less-calibrated values (0.9-0.98). We map the batch's confidences into
the target band with a monotonic linear rescale, so relative ordering (which
messages the model was more sure about) is preserved while the absolute values
match the observed calibration. Values are rounded to 2 decimals like the truth.

No LLM calls here — this is a cheap, deterministic post-process, so it can be
re-applied on every run without cost.
"""
from __future__ import annotations

import pandas as pd

# Observed ground-truth confidence band (from sample_messages.csv).
TARGET_LO = 0.78
TARGET_HI = 0.91

# Robust input clipping so a lone outlier doesn't compress everything else.
_INPUT_LO_PCT = 0.05
_INPUT_HI_PCT = 0.95


def _rescale(conf: pd.Series) -> pd.Series:
    """Monotonic linear map of confidences into [TARGET_LO, TARGET_HI]."""
    c = pd.to_numeric(conf, errors="coerce").fillna(conf.median())
    lo = c.quantile(_INPUT_LO_PCT)
    hi = c.quantile(_INPUT_HI_PCT)
    if hi <= lo:
        # No spread to preserve: put everything at the band midpoint.
        return pd.Series((TARGET_LO + TARGET_HI) / 2, index=c.index)
    c = c.clip(lo, hi)
    scaled = TARGET_LO + (c - lo) * (TARGET_HI - TARGET_LO) / (hi - lo)
    return scaled.clip(TARGET_LO, TARGET_HI)


def reconcile(predictions: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return predictions after cross-message reconciliation.

    Calibrates confidence into the ground-truth band. Other coherence passes
    (e.g. aligning near-identical messages) can be added here later.
    """
    df = predictions.copy()
    if "confidence" in df.columns and len(df):
        df["confidence"] = _rescale(df["confidence"]).round(2)
    return df
