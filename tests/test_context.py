"""context.in_dnd(): do-not-disturb windows that wrap past midnight.

All 14 DND windows in the real data wrap midnight (start > end). A naive
`start <= t <= end` returns False for every one of them, silently disabling
quiet hours everywhere. This is the single highest-value test in the repo.
"""
from datetime import time

from router.context import in_dnd


def test_wrap_late_night_is_inside():
    assert in_dnd("22:00-07:00", time(23, 30)) is True


def test_wrap_early_morning_is_inside():
    assert in_dnd("22:00-07:00", time(2, 0)) is True


def test_wrap_midday_is_outside():
    assert in_dnd("22:00-07:00", time(12, 0)) is False


def test_wrap_endpoints_inclusive():
    assert in_dnd("22:00-07:00", time(22, 0)) is True  # start inclusive
    assert in_dnd("22:00-07:00", time(7, 0)) is True   # end inclusive


def test_same_day_window():
    assert in_dnd("09:00-17:00", time(12, 0)) is True
    assert in_dnd("09:00-17:00", time(20, 0)) is False


def test_malformed_window_is_false():
    assert in_dnd("", time(3, 0)) is False
    assert in_dnd(None, time(3, 0)) is False  # type: ignore[arg-type]
