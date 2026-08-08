"""Output validator: fail loudly if output.csv breaks the contract.

Checks:
  - exactly EXPECTED_ROW_COUNT data rows
  - columns present, in the exact required order
  - action within the allowed enum
  - message_type within the allowed enum
  - confidence parseable as float in [0, 1]
  - evidence_message_ids non-empty (use "none" when there is no evidence)
  - one row per message_id in messages.csv, no duplicates

Run from the terminal:

    python -m router.validate                              # validates OUTPUT_PATH
    python -m router.validate results/submission_output.csv

Exit code 0 = valid, 1 = invalid (prints every problem found).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import (
    ACTIONS,
    DATASET_DIR,
    EXPECTED_ROW_COUNT,
    MESSAGE_TYPES,
    OUTPUT_COLUMNS,
    OUTPUT_PATH,
)
from .ingest import load_csv


def validate(path: Path | str | None = None, skip_id_crosscheck: bool = False) -> list[str]:
    """Return a list of contract violations. Empty list == valid.

    path defaults to OUTPUT_PATH. The per-message_id coverage check needs
    dataset/messages.csv; it is skipped (not failed) when the dataset is absent
    or skip_id_crosscheck is set, so the contract can be validated standalone.
    """
    errors: list[str] = []

    target = Path(path) if path is not None else OUTPUT_PATH
    if not target.exists():
        return [f"output file missing: {target}"]

    # Read everything as string first so we can validate confidence ourselves
    # rather than letting pandas silently coerce or drop bad values.
    df = pd.read_csv(target, dtype=str, keep_default_na=False)

    # --- columns: presence and exact order ---
    if list(df.columns) != OUTPUT_COLUMNS:
        errors.append(
            f"columns must be exactly {OUTPUT_COLUMNS} in order, got {list(df.columns)}"
        )
        # Column order is foundational; further checks would be noise.
        return errors

    # --- row count ---
    if len(df) != EXPECTED_ROW_COUNT:
        errors.append(f"expected {EXPECTED_ROW_COUNT} rows, got {len(df)}")

    # --- action enum ---
    bad_actions = sorted(set(df["action"]) - ACTIONS)
    if bad_actions:
        errors.append(f"invalid action values: {bad_actions} (allowed: {sorted(ACTIONS)})")

    # --- message_type enum ---
    bad_types = sorted(set(df["message_type"]) - MESSAGE_TYPES)
    if bad_types:
        errors.append(f"invalid message_type values: {bad_types}")

    # --- confidence in [0, 1] ---
    for i, raw in enumerate(df["confidence"]):
        try:
            c = float(raw)
        except ValueError:
            errors.append(f"row {i}: confidence not a number: {raw!r}")
            continue
        if not (0.0 <= c <= 1.0):
            errors.append(f"row {i}: confidence out of [0,1]: {c}")

    # --- evidence non-empty ---
    empty_evidence = int((df["evidence_message_ids"].str.strip() == "").sum())
    if empty_evidence:
        errors.append(f"{empty_evidence} rows have empty evidence_message_ids (use 'none')")

    # --- one row per input message_id, no dupes ---
    dupes = sorted(df.loc[df["message_id"].duplicated(), "message_id"].unique())
    if dupes:
        errors.append(f"duplicate message_id rows: {dupes}")

    # --- one row per input message_id (needs the dataset) ---
    if not skip_id_crosscheck and (DATASET_DIR / "messages.csv").exists():
        try:
            expected_ids = set(load_csv("messages")["message_id"])
            got_ids = set(df["message_id"])
            missing = sorted(expected_ids - got_ids)
            extra = sorted(got_ids - expected_ids)
            if missing:
                errors.append(f"missing predictions for message_ids: {missing}")
            if extra:
                errors.append(f"predictions for unknown message_ids: {extra}")
        except Exception as exc:  # messages.csv unreadable; report but don't crash
            errors.append(f"could not cross-check message_ids against messages.csv: {exc}")

    return errors


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Validate an output CSV against the contract.")
    ap.add_argument("path", nargs="?", default=None,
                    help="CSV to validate (default: results/output.csv)")
    args = ap.parse_args()

    errors = validate(path=args.path)
    if errors:
        print(f"INVALID — {len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"VALID — {EXPECTED_ROW_COUNT} rows, contract satisfied.")


if __name__ == "__main__":
    main()
