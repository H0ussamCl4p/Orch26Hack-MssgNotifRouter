"""Shared configuration: paths, enums, output contract.

Single source of truth for anything that must stay consistent across
ingestion, routing, and validation. Keeping the enums here means run.py
and validate.py can never drift apart.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
# ROOT is the repo root. This module lives at src/router/config.py, so the
# repo root is three levels up (config.py -> router -> src -> repo root).
# Everything is resolved from here so the pipeline runs the same regardless of
# the caller's working directory.
ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
RESULTS_DIR = ROOT / "results"

# Pipeline write target. Defaults to results/output.csv (gitignored); override
# with the OUTPUT_PATH env var. The graded submission lives separately at
# results/submission_output.csv and is FROZEN — the pipeline never writes there.
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH") or (RESULTS_DIR / "output.csv"))

# Load .env (if present) so secrets live in the environment, never in code.
load_dotenv(ROOT / ".env")


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from the environment. Never hardcode keys."""
    return os.environ.get(name, default)


# --- Output contract -------------------------------------------------------
# Exact column order required by the output CSV. Do not reorder.
OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

# Allowed enum values from problem_statement.md.
ACTIONS = {"notify", "digest", "mute"}

MESSAGE_TYPES = {
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
}

# Expected number of rows to route (== data rows in dataset/messages.csv).
# Derived at runtime so the contract tracks the actual dataset; falls back to
# the documented 110 when the dataset is absent (e.g. a fresh clone with no
# dataset, running only the test suite).
_EXPECTED_ROW_COUNT_FALLBACK = 110


def _count_messages() -> int:
    """Count data rows in messages.csv, or return the documented fallback."""
    path = DATASET_DIR / "messages.csv"
    if not path.exists():
        return _EXPECTED_ROW_COUNT_FALLBACK
    try:
        with path.open(encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)  # minus the header row
    except OSError:
        return _EXPECTED_ROW_COUNT_FALLBACK


EXPECTED_ROW_COUNT = _count_messages()

# Per-message failure fallback: written to the output when routing raises
# unexpectedly, so one bad message never sinks the batch. It is NOT cached, so
# the next run retries the failed message.
PLACEHOLDER = {
    "action": "digest",
    "message_type": "unknown",
    "reason": "placeholder",
    "confidence": 0.84,
    "evidence_message_ids": "none",
}
