"""Evaluation layer: score the pipeline against sample_messages.csv (30 labels).

The 30 sample messages (sample_msg_xxx) are DISTINCT from the 110 (msg_xxx) and
have never been routed, so the first eval run routes them through the real
pipeline (~30 Gemma calls) and caches the decisions; later runs are API-free.

Metrics: accuracy + macro-F1 and confusion matrix for action and message_type
(with per-class support), evidence exact-match + id precision/recall (treating
"none" as a first-class value), confidence calibration (Brier + per-bucket
accuracy), and an action x message_type coherence check against pairs the sample
never uses. Also writes an errors CSV for case-by-case reading.

30 examples is small: numbers are shown at low precision on purpose, with class
counts, so nobody over-reads a 0.03 wobble.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .config import DATASET_DIR, ROOT
from .ingest import load_all
from .routing import route_one

SAMPLE_PATH = DATASET_DIR / "sample_messages.csv"
ERRORS_CSV = ROOT / "results" / "eval_errors.csv"
INPUT_COLS = [
    "message_id", "user_id", "conversation_type", "group_id", "business_id",
    "sender_user_id", "created_at", "message_text", "media_type", "media_id",
    "forwarded_count",
]


# --- prediction ------------------------------------------------------------
def _predict_sample(workers: int = 3, force: bool = False) -> pd.DataFrame:
    """Route the 30 sample messages through the pipeline (cached)."""
    tables = load_all()
    sample = pd.read_csv(SAMPLE_PATH)
    eval_tables = {**tables, "messages": sample[INPUT_COLS].copy()}
    rows = [r for _, r in eval_tables["messages"].iterrows()]

    def work(row):
        return route_one(row, eval_tables, force=force)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        preds = list(pool.map(work, rows))
    pred_df = pd.DataFrame(preds).set_index("message_id")
    gold = sample.set_index("message_id")
    # reset_index so message_id is a column again (coherence/errors iterate rows).
    return gold.join(pred_df, rsuffix="_pred").reset_index()


# --- metric helpers --------------------------------------------------------
def _prf(y_true: list[str], y_pred: list[str]) -> dict[str, tuple]:
    labels = sorted(set(y_true) | set(y_pred))
    out = {}
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = (prec, rec, f1, support)
    return out


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _report_field(name: str, y_true: list[str], y_pred: list[str]) -> None:
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    prf = _prf(y_true, y_pred)
    macro_f1 = sum(v[2] for v in prf.values()) / len(prf)
    print(f"\n### {name}: accuracy {_pct(acc)} ({sum(1 for t,p in zip(y_true,y_pred) if t==p)}/{n})"
          f" | macro-F1 {macro_f1:.2f} (over {len(prf)} classes)")
    print(f"{'class':16s} {'F1':>5s} {'P':>5s} {'R':>5s} {'support':>8s}")
    for c, (p, r, f1, sup) in sorted(prf.items(), key=lambda kv: -kv[1][3]):
        print(f"{c:16s} {f1:5.2f} {_pct(p):>5s} {_pct(r):>5s} {sup:8d}")
    print(f"\nconfusion — {name} (rows=expected, cols=predicted):")
    cm = pd.crosstab(
        pd.Series(y_true, name="exp"), pd.Series(y_pred, name="pred"), dropna=False
    )
    print(cm.to_string())


# --- evidence --------------------------------------------------------------
def _parse_ev(s) -> set[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return set()
    txt = str(s).strip()
    if not txt or txt.lower() == "none":
        return set()
    return {t for t in re.split(r"[;,\s]+", txt) if t}


def _report_evidence(df: pd.DataFrame) -> None:
    gold = [_parse_ev(x) for x in df["evidence_message_ids"]]
    pred = [_parse_ev(x) for x in df["evidence_message_ids_pred"]]
    n = len(df)

    exact = sum(1 for g, p in zip(gold, pred) if g == p)
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    fp = sum(len(p - g) for g, p in zip(gold, pred))
    fn = sum(len(g - p) for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    g_none = [len(g) == 0 for g in gold]
    p_none = [len(p) == 0 for p in pred]
    both_none = sum(1 for a, b in zip(g_none, p_none) if a and b)
    exp_none_pred_ids = sum(1 for a, b in zip(g_none, p_none) if a and not b)
    exp_ids_pred_none = sum(1 for a, b in zip(g_none, p_none) if not a and b)

    print(f"\n### evidence_message_ids ({n} rows)")
    print(f"exact match (set-equal incl. none): {_pct(exact / n)} ({exact}/{n})")
    print(f"id precision {_pct(prec)}  recall {_pct(rec)}  (TP={tp} FP={fp} FN={fn})")
    print(f"none — expected {sum(g_none)}, predicted {sum(p_none)}")
    print(f"     both none (correct): {both_none}")
    print(f"     expected none but predicted ids: {exp_none_pred_ids}")
    print(f"     expected ids but predicted none: {exp_ids_pred_none}")


# --- calibration -----------------------------------------------------------
def _report_calibration(df: pd.DataFrame) -> None:
    conf = pd.to_numeric(df["confidence_pred"], errors="coerce").fillna(0.8)
    correct = (df["action"] == df["action_pred"]).astype(int)
    brier = float(((conf - correct) ** 2).mean())
    print("\n### calibration (confidence = routed value, pre consistency.py rescale)")
    print(f"Brier score (action-correct): {brier:.2f}  (lower is better)")
    edges = [0.0, 0.85, 0.90, 0.95, 1.01]
    print(f"{'bucket':14s} {'n':>3s} {'action_acc':>11s}")
    for lo, hi in zip(edges, edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        k = int(mask.sum())
        if k == 0:
            continue
        acc = correct[mask].mean()
        print(f"[{lo:.2f},{hi:.2f}) {k:>5d} {_pct(acc):>11s}")


# --- coherence -------------------------------------------------------------
def _report_coherence(df: pd.DataFrame) -> None:
    gold_pairs = set(zip(df["action"], df["message_type"]))
    print("\n### coherence: predicted action×type pairs the sample NEVER uses")
    offenders = {}
    for _, r in df.iterrows():
        pair = (r["action_pred"], r["message_type_pred"])
        if pair not in gold_pairs:
            offenders.setdefault(pair, []).append(r["message_id"])
    if not offenders:
        print("none — every predicted pair exists in the sample")
        return
    for pair, ids in sorted(offenders.items(), key=lambda kv: -len(kv[1])):
        print(f"  {pair[0]:7s}+{pair[1]:16s} x{len(ids):2d}  e.g. {', '.join(ids[:3])}")


# --- errors csv ------------------------------------------------------------
def _write_errors(df: pd.DataFrame) -> int:
    recs = []
    for _, r in df.iterrows():
        a_ok = r["action"] == r["action_pred"]
        t_ok = r["message_type"] == r["message_type_pred"]
        e_ok = _parse_ev(r["evidence_message_ids"]) == _parse_ev(r["evidence_message_ids_pred"])
        if a_ok and t_ok and e_ok:
            continue
        text = str(r["message_text"]) if not pd.isna(r["message_text"]) else ""
        recs.append({
            "message_id": r["message_id"],
            "wrong": ",".join(w for w, ok in
                              [("action", a_ok), ("type", t_ok), ("evidence", e_ok)] if not ok),
            "action_exp": r["action"], "action_pred": r["action_pred"],
            "type_exp": r["message_type"], "type_pred": r["message_type_pred"],
            "evidence_exp": r["evidence_message_ids"],
            "evidence_pred": r["evidence_message_ids_pred"],
            "text": text[:160],
        })
    pd.DataFrame(recs).to_csv(ERRORS_CSV, index=False)
    return len(recs)


# --- public API ------------------------------------------------------------
def evaluate(workers: int = 3, force: bool = False) -> None:
    df = _predict_sample(workers=workers, force=force)
    print("=" * 68)
    print(f"EVALUATION vs sample_messages.csv — {len(df)} labelled examples")
    print("small sample: read F1/accuracy as directional, not exact.")
    print("=" * 68)

    _report_field("action", list(df["action"]), list(df["action_pred"]))
    _report_field("message_type", list(df["message_type"]), list(df["message_type_pred"]))
    _report_evidence(df)
    _report_calibration(df)
    _report_coherence(df)

    n_err = _write_errors(df)
    print(f"\nWrote {n_err} error rows to {ERRORS_CSV}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Score the pipeline vs sample_messages.csv.")
    ap.add_argument("--force", action="store_true", help="re-route samples (ignore cache)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    evaluate(workers=args.workers, force=args.force)
