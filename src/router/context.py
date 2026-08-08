"""Context layer: assemble a compact, serializable dossier per message.

For a given message_id, build_context() gathers everything a router needs to
judge the message *except* the historical candidate messages (that is
candidates.py): the receiving user's profile and quiet hours, the relevant
group- or business-relationship, spoofing signals, and recent notification
load. Output is a plain dict of JSON-serializable primitives, ready to inject
into a prompt. No LLM here.

Two sharp edges handled explicitly:
  1. do_not_disturb_window wraps past midnight ("22:00-07:00"). A naive
     start <= t <= end is always false for wrap windows. See in_dnd().
  2. Sender-domain spoofing: official_domain vs domain_used_by_sender, and a
     recently-registered sender domain on a long-lived account. Exposed as the
     boolean domain_mismatch.
"""
from __future__ import annotations

import math
from datetime import datetime, time
from typing import Any

import pandas as pd

# A sender domain this young on an account this old reads as impersonation.
_OLD_ACCOUNT_DAYS = 365
_YOUNG_DOMAIN_DAYS = 90


# --- primitives ------------------------------------------------------------
def _to_time(value: str) -> time:
    """Parse 'HH:MM' into a datetime.time."""
    hh, mm = value.strip().split(":")
    return time(int(hh), int(mm))


def in_dnd(window: str, t: time) -> bool:
    """Is time t inside the do-not-disturb window?

    window is 'START-END' as 'HH:MM'. Handles windows that wrap past midnight:
    when start > end (e.g. '22:00-07:00'), the window is the union
    [start, 24:00) u [00:00, end]. Endpoints are inclusive.
    """
    if not window or not isinstance(window, str) or "-" not in window:
        return False
    start_s, end_s = window.split("-", 1)
    start, end = _to_time(start_s), _to_time(end_s)
    if start <= end:
        # Same-day window, e.g. '09:00-17:00'.
        return start <= t <= end
    # Wrap-around window, e.g. '22:00-07:00'.
    return t >= start or t <= end


def _clean(value: Any) -> Any:
    """Coerce numpy/pandas scalars to JSON-serializable primitives; NaN -> None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    # numpy scalar types expose .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _lookup(df: pd.DataFrame, **filters: Any) -> pd.Series | None:
    """Return the first row matching all filters, or None."""
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col] == val
    hit = df[mask]
    return None if hit.empty else hit.iloc[0]


def _fields(row: pd.Series | None, cols: list[str]) -> dict[str, Any]:
    """Extract named columns from a row as clean primitives; {} if row is None."""
    if row is None:
        return {}
    return {c: _clean(row[c]) for c in cols if c in row.index}


# --- sub-dossiers ----------------------------------------------------------
def _user_block(user_id: str, created_at: datetime, users: pd.DataFrame) -> dict[str, Any]:
    row = _lookup(users, user_id=user_id)
    if row is None:
        return {"user_id": user_id, "found": False}
    window = row.get("do_not_disturb_window")
    return {
        "user_id": user_id,
        "do_not_disturb_window": _clean(window),
        "message_arrives_in_dnd": in_dnd(window, created_at.time())
        if isinstance(window, str)
        else False,
        # Raw 30d engagement counts (users.csv has no denominator; see load block
        # for a true dismissal rate).
        "messages_opened_30d": _clean(row.get("messages_opened_30d")),
        "messages_replied_30d": _clean(row.get("messages_replied_30d")),
        "notifications_dismissed_30d": _clean(row.get("notifications_dismissed_30d")),
        "messages_reported_30d": _clean(row.get("messages_reported_30d")),
    }


def _group_block(
    group_id: str, user_id: str, groups: pd.DataFrame, members: pd.DataFrame
) -> dict[str, Any]:
    g = _fields(
        _lookup(groups, group_id=group_id),
        ["group_type", "member_count", "admin_count", "messages_30d"],
    )
    m = _fields(
        _lookup(members, group_id=group_id, user_id=user_id),
        [
            "role",
            "group_muted_by_user",
            "messages_read_30d",
            "replies_sent_30d",
            "notifications_dismissed_30d",
        ],
    )
    return {"group_id": group_id, **g, "membership": m}


def _business_block(
    business_id: str,
    user_id: str,
    accounts: pd.DataFrame,
    history: pd.DataFrame,
) -> dict[str, Any]:
    acct = _lookup(accounts, business_id=business_id)
    a = _fields(
        acct,
        [
            "display_name",
            "brand_name",
            "category",
            "verified",
            "official_domain",
            "domain_used_by_sender",
            "account_age_days",
            "domain_used_by_sender_age_days",
            "user_reports_30d",
        ],
    )

    # --- spoofing signal ---
    official = a.get("official_domain")
    used = a.get("domain_used_by_sender")
    acct_age = a.get("account_age_days")
    dom_age = a.get("domain_used_by_sender_age_days")

    name_mismatch = bool(official and used and official != used)
    age_suspicious = bool(
        isinstance(acct_age, (int, float))
        and isinstance(dom_age, (int, float))
        and acct_age >= _OLD_ACCOUNT_DAYS
        and dom_age < _YOUNG_DOMAIN_DAYS
    )
    a["domain_name_mismatch"] = name_mismatch
    a["domain_age_gap_days"] = (
        acct_age - dom_age
        if isinstance(acct_age, (int, float)) and isinstance(dom_age, (int, float))
        else None
    )
    a["domain_age_suspicious"] = age_suspicious
    a["domain_mismatch"] = name_mismatch or age_suspicious

    rel = _fields(
        _lookup(history, user_id=user_id, business_id=business_id),
        [
            "why_user_knows_account",
            "allows_promotions",
            "promotions_opted_out_at",
            "activity_count_180d",
            "messages_dismissed_30d",
            "last_activity_at",
        ],
    )
    return {"business_id": business_id, **a, "relationship": rel}


def _load_block(user_id: str, summary: pd.DataFrame) -> dict[str, Any]:
    """Recent notification load for the user, with a true dismissal rate."""
    rows = summary[summary["user_id"] == user_id]
    if rows.empty:
        return {"days_tracked": 0}
    sent = float(rows["notifications_sent"].sum())
    dismissed = float(rows["notifications_dismissed"].sum())
    days = int(len(rows))
    return {
        "days_tracked": days,
        "avg_sent_per_day": round(sent / days, 2) if days else 0.0,
        "avg_dismissed_per_day": round(dismissed / days, 2) if days else 0.0,
        "dismissal_rate": round(dismissed / sent, 3) if sent else None,
    }


# --- public API ------------------------------------------------------------
def build_context(message_id: str, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Assemble the compact context dossier for one message_id.

    Returns a JSON-serializable dict. Only the relevant conversation block
    (group vs business vs personal) is populated.
    """
    messages = tables["messages"]
    msg = _lookup(messages, message_id=message_id)
    if msg is None:
        raise KeyError(f"message_id not found: {message_id!r}")

    user_id = msg["user_id"]
    conv = msg["conversation_type"]
    created_at = pd.to_datetime(msg["created_at"]).to_pydatetime()

    dossier: dict[str, Any] = {
        "message_id": message_id,
        "conversation_type": _clean(conv),
        "created_at": str(msg["created_at"]),
        "media_type": _clean(msg.get("media_type")),
        "forwarded_count": _clean(msg.get("forwarded_count")),
        "user": _user_block(user_id, created_at, tables["users"]),
    }

    if conv == "group" and pd.notna(msg.get("group_id")):
        dossier["group"] = _group_block(
            msg["group_id"], user_id, tables["groups"], tables["group_members"]
        )
    elif conv == "business" and pd.notna(msg.get("business_id")):
        dossier["business"] = _business_block(
            msg["business_id"],
            user_id,
            tables["business_accounts"],
            tables["user_business_history"],
        )
    elif conv == "personal":
        dossier["sender_user_id"] = _clean(msg.get("sender_user_id"))

    dossier["notification_load"] = _load_block(
        user_id, tables["daily_notification_summary"]
    )
    return dossier


# --- self-tests ------------------------------------------------------------
def _selftest_dnd() -> None:
    """The two required wrap-around cases, plus a same-day sanity check."""
    assert in_dnd("22:00-07:00", time(23, 30)) is True   # late night -> in
    assert in_dnd("22:00-07:00", time(6, 0)) is True     # early morning -> in
    assert in_dnd("22:00-07:00", time(12, 0)) is False   # midday -> out
    assert in_dnd("22:00-07:00", time(22, 0)) is True    # start inclusive
    assert in_dnd("22:00-07:00", time(7, 0)) is True     # end inclusive

    assert in_dnd("00:00-06:30", time(3, 15)) is True    # inside wrap
    assert in_dnd("00:00-06:30", time(6, 30)) is True    # end inclusive
    assert in_dnd("00:00-06:30", time(6, 31)) is False   # just after
    assert in_dnd("00:00-06:30", time(9, 0)) is False    # daytime -> out

    assert in_dnd("09:00-17:00", time(12, 0)) is True    # same-day window
    assert in_dnd("09:00-17:00", time(20, 0)) is False
    print("in_dnd self-test: OK")


if __name__ == "__main__":
    import json

    _selftest_dnd()

    from router.ingest import load_all

    tables = load_all()
    # Show one dossier per conversation type for a quick eyeball.
    seen: set[str] = set()
    for _, m in tables["messages"].iterrows():
        conv = m["conversation_type"]
        if conv in seen:
            continue
        seen.add(conv)
        print(f"\n=== {conv} :: {m['message_id']} ===")
        print(json.dumps(build_context(m["message_id"], tables), indent=2, ensure_ascii=False))
        if len(seen) == 3:
            break
