"""validate(): the output contract accepts a good file and rejects bad ones.

skip_id_crosscheck=True keeps these tests independent of the dataset.
"""
import pandas as pd

from router.validate import validate

_GOOD_ROW = {
    "message_id": "msg_001", "action": "notify", "message_type": "payment",
    "reason": "a reason", "confidence": "0.85", "evidence_message_ids": "hist_001",
}
_COLUMNS = [
    "message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids",
]


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _valid_rows(n=110):
    rows = []
    for i in range(1, n + 1):
        r = dict(_GOOD_ROW)
        r["message_id"] = f"msg_{i:03d}"
        rows.append(r)
    return rows


def test_valid_output_passes(tmp_path):
    p = tmp_path / "output.csv"
    _write(p, _valid_rows())
    assert validate(path=p, skip_id_crosscheck=True) == []


def test_bad_action_enum_rejected(tmp_path):
    rows = _valid_rows()
    rows[0]["action"] = "escalate"
    p = tmp_path / "output.csv"
    _write(p, rows)
    errors = validate(path=p, skip_id_crosscheck=True)
    assert any("action" in e.lower() for e in errors)


def test_missing_column_rejected(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "reason"} for r in _valid_rows()]
    p = tmp_path / "output.csv"
    _write(p, rows)
    errors = validate(path=p, skip_id_crosscheck=True)
    assert any("column" in e.lower() for e in errors)


def test_confidence_out_of_range_rejected(tmp_path):
    rows = _valid_rows()
    rows[0]["confidence"] = "1.5"
    p = tmp_path / "output.csv"
    _write(p, rows)
    errors = validate(path=p, skip_id_crosscheck=True)
    assert any("confidence" in e.lower() for e in errors)


def test_missing_file_reported(tmp_path):
    errors = validate(path=tmp_path / "nope.csv", skip_id_crosscheck=True)
    assert any("missing" in e.lower() for e in errors)
