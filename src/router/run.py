"""Pipeline entry point.

Reads dataset/messages.csv, routes every message (concurrently), reconciles the
batch, and writes dataset/output.csv with the exact 6-column contract, one row
per message. Run from the terminal:

    python code/run.py
    ROUTING_WORKERS=8 python code/run.py     # tune concurrency

Routing is cached per message, so re-runs are near-instant and only re-route
what is missing. Failures are isolated per message (a placeholder is used and
NOT cached, so the next run retries it) — one bad message never sinks the batch.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .config import OUTPUT_COLUMNS, OUTPUT_PATH, PLACEHOLDER
from .consistency import reconcile
from .ingest import load_all
from .routing import route_one

WORKERS = int(os.environ.get("ROUTING_WORKERS", "6"))


def main() -> None:
    tables = load_all()
    messages = tables["messages"]
    rows = [row for _, row in messages.iterrows()]

    done = {"n": 0}

    def work(message: pd.Series) -> dict:
        mid = message["message_id"]
        try:
            decision = route_one(message, tables)
        except Exception as exc:  # isolate: never let one message sink the batch
            print(f"  ERROR {mid}: {type(exc).__name__}: {exc}  -> placeholder (not cached)")
            decision = {"message_id": mid, **PLACEHOLDER}
        done["n"] += 1
        print(f"  [{done['n']:3d}/{len(rows)}] {mid} -> {decision['action']}/{decision['message_type']}")
        return decision

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(work, rows))

    # Reassemble in the original message order.
    by_id = {d["message_id"]: d for d in results}
    ordered = [by_id[row["message_id"]] for row in rows]
    predictions = pd.DataFrame(ordered, columns=OUTPUT_COLUMNS)

    predictions = reconcile(predictions, tables)
    predictions = predictions[OUTPUT_COLUMNS]
    assert len(predictions) == len(messages), (
        f"row count mismatch: {len(predictions)} predictions for {len(messages)} messages"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(predictions)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
