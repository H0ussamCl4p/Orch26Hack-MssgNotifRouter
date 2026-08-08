"""Ingestion layer: load every input CSV into named pandas DataFrames.

Responsibility: read from disk and hand back clean, named tables. No routing
logic, no filtering, no business rules. Downstream modules build context and
candidates from what load_all() returns.
"""
from __future__ import annotations

import pandas as pd

from .config import DATASET_DIR

# Logical name -> filename. output.csv is intentionally absent: it is the
# write target, not an input. These are the 12 context/input CSVs.
INPUT_FILES = {
    "messages": "messages.csv",
    "sample_messages": "sample_messages.csv",
    "users": "users.csv",
    "groups": "groups.csv",
    "group_members": "group_members.csv",
    "business_accounts": "business_accounts.csv",
    "user_business_history": "user_business_history.csv",
    "message_history": "message_history.csv",
    "message_events": "message_events.csv",
    "images": "images.csv",
    "voice_notes": "voice_notes.csv",
    "daily_notification_summary": "daily_notification_summary.csv",
}


def load_csv(name: str) -> pd.DataFrame:
    """Load a single named CSV as a DataFrame."""
    if name not in INPUT_FILES:
        raise KeyError(f"Unknown dataset table: {name!r}. Known: {sorted(INPUT_FILES)}")
    return pd.read_csv(DATASET_DIR / INPUT_FILES[name])


def load_all() -> dict[str, pd.DataFrame]:
    """Load every input CSV.

    Returns a dict keyed by logical table name (see INPUT_FILES), e.g.
    tables["messages"], tables["users"], tables["message_history"].
    """
    return {name: load_csv(name) for name in INPUT_FILES}


if __name__ == "__main__":
    tables = load_all()
    for name, df in tables.items():
        print(f"{name:28s} {len(df):5d} rows  {list(df.columns)}")
