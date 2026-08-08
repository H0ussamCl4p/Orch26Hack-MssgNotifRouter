"""candidates + context edge cases: domain_mismatch, _jaccard, best_evidence.

All fixtures are built inline — no dataset, no network, no API calls.
"""
import pandas as pd

from router.candidates import _jaccard, best_evidence
from router.context import build_context

# --- shared empty tables ---------------------------------------------------
_EMPTY = {
    "groups": ["group_id", "group_type", "member_count", "admin_count", "messages_30d"],
    "group_members": [
        "group_id", "user_id", "role", "group_muted_by_user",
        "messages_read_30d", "replies_sent_30d", "notifications_dismissed_30d",
    ],
    "user_business_history": [
        "user_id", "business_id", "why_user_knows_account", "allows_promotions",
        "promotions_opted_out_at", "activity_count_180d", "messages_dismissed_30d",
        "last_activity_at",
    ],
    "daily_notification_summary": ["user_id", "notifications_sent", "notifications_dismissed"],
    "message_history": [
        "message_id", "user_id", "conversation_type", "group_id", "business_id",
        "sender_user_id", "media_type", "forwarded_count", "message_text", "created_at",
    ],
    "message_events": [
        "message_id", "user_id", "message_opened", "message_replied",
        "notification_dismissed", "muted_after_message", "message_reported",
    ],
}


def _empty_tables() -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame(columns=cols) for name, cols in _EMPTY.items()}


def _user_row(user_id="u1"):
    return {
        "user_id": user_id, "do_not_disturb_window": None,
        "messages_opened_30d": 5, "messages_replied_30d": 2,
        "notifications_dismissed_30d": 1, "messages_reported_30d": 0,
    }


# --- domain_mismatch (via build_context) -----------------------------------
def _business_tables(business_overrides: dict) -> dict[str, pd.DataFrame]:
    tables = _empty_tables()
    tables["messages"] = pd.DataFrame([{
        "message_id": "m1", "user_id": "u1", "conversation_type": "business",
        "business_id": "b1", "group_id": None, "sender_user_id": None,
        "created_at": "2026-08-01T10:00:00", "message_text": "hi",
        "media_type": None, "media_id": None, "forwarded_count": 0,
    }])
    tables["users"] = pd.DataFrame([_user_row()])
    tables["business_accounts"] = pd.DataFrame([{
        "business_id": "b1", "display_name": "HDFC Bank", "brand_name": "HDFC",
        "category": "bank", "verified": True, "user_reports_30d": 0,
        **business_overrides,
    }])
    return tables


def test_domain_mismatch_name_case():
    tables = _business_tables({
        "official_domain": "hdfc.com", "domain_used_by_sender": "hdfc-secure.com",
        "account_age_days": 400, "domain_used_by_sender_age_days": 300,
    })
    assert build_context("m1", tables)["business"]["domain_mismatch"] is True


def test_domain_mismatch_young_domain_on_old_account():
    tables = _business_tables({
        "official_domain": "hdfc.com", "domain_used_by_sender": "hdfc.com",
        "account_age_days": 400, "domain_used_by_sender_age_days": 30,
    })
    assert build_context("m1", tables)["business"]["domain_mismatch"] is True


def test_domain_mismatch_clean_case():
    tables = _business_tables({
        "official_domain": "hdfc.com", "domain_used_by_sender": "hdfc.com",
        "account_age_days": 400, "domain_used_by_sender_age_days": 300,
    })
    assert build_context("m1", tables)["business"]["domain_mismatch"] is False


# --- _jaccard --------------------------------------------------------------
def test_jaccard_identical_is_one():
    assert _jaccard("payment due tomorrow", "payment due tomorrow") == 1.0


def test_jaccard_disjoint_is_zero():
    assert _jaccard("hello world", "foo bar baz") == 0.0


def test_jaccard_partial_is_between():
    assert 0.0 < _jaccard("payment invoice due", "invoice pending amount") < 1.0


def test_jaccard_symmetric():
    a, b = "transfer amount upi", "upi amount transfer request"
    assert _jaccard(a, b) == _jaccard(b, a)


# --- best_evidence ---------------------------------------------------------
def _history_tables(msg_text: str, hist_texts: list[str]):
    tables = _empty_tables()
    tables["users"] = pd.DataFrame([_user_row("u_test")])
    tables["business_accounts"] = pd.DataFrame([{
        "business_id": "b_test", "display_name": "X", "brand_name": "X",
        "category": "bank", "verified": True, "official_domain": "x.com",
        "domain_used_by_sender": "x.com", "account_age_days": 500,
        "domain_used_by_sender_age_days": 500, "user_reports_30d": 0,
    }])
    tables["messages"] = pd.DataFrame([{
        "message_id": "m_test", "user_id": "u_test", "conversation_type": "business",
        "business_id": "b_test", "group_id": None, "sender_user_id": None,
        "created_at": "2026-08-01T10:00:00", "message_text": msg_text,
        "media_type": None, "media_id": None, "forwarded_count": 0,
    }])
    hist_rows = [
        {
            "message_id": f"h{i}", "user_id": "u_test", "conversation_type": "business",
            "business_id": "b_test", "group_id": None, "sender_user_id": None,
            "media_type": None, "forwarded_count": 0, "message_text": t,
            "created_at": f"2026-07-{i + 1:02d}T10:00:00",
        }
        for i, t in enumerate(hist_texts)
    ]
    # Keep the schema even when empty, so the retrieval filter has its columns.
    tables["message_history"] = pd.DataFrame(hist_rows, columns=_EMPTY["message_history"])
    valid_ids = {f"h{i}" for i in range(len(hist_texts))}
    return tables, valid_ids


def test_best_evidence_id_always_from_pool():
    # The structural guarantee: a returned id is either "none" or a candidate id,
    # so a hallucinated evidence id is impossible by construction.
    tables, valid_ids = _history_tables(
        "Please pay your electricity bill",
        ["Electricity bill due payment", "Hello how are you", "Invoice pending amount"],
    )
    result = best_evidence("m_test", tables, k=1)
    assert result == "none" or result in valid_ids


def test_best_evidence_picks_most_similar():
    tables, _ = _history_tables(
        "UPI payment failed please retry the transfer",
        ["UPI payment failed retry transfer", "random greeting hello", "monthly statement"],
    )
    assert best_evidence("m_test", tables, k=1) == "h0"


def test_best_evidence_deterministic():
    tables, _ = _history_tables(
        "UPI payment failed retry",
        ["UPI payment failed", "account statement", "fraud alert warning"],
    )
    assert best_evidence("m_test", tables, k=1) == best_evidence("m_test", tables, k=1)


def test_best_evidence_first_contact_returns_none():
    tables, _ = _history_tables("first ever message", [])
    assert best_evidence("m_test", tables, k=1) == "none"
