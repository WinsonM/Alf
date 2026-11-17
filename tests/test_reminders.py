from datetime import datetime, timedelta

from alf import db
from alf.reminders import Reminders

from .conftest import NOW, TZ


def test_one_off_fires_once_then_is_done(conn):
    rem = Reminders(conn, TZ)
    r = rem.add("交作业", NOW + timedelta(minutes=5), now=int(NOW.timestamp()))
    assert rem.due_for_notification(NOW) == []
    later = NOW + timedelta(minutes=6)
    [due] = rem.due_for_notification(later)
    assert rem.claim(due, later)
    assert rem.due_for_notification(later) == [] and rem.get(r.id).status == "done"


def test_lead_time_moves_the_notification(conn):
    rem = Reminders(conn, TZ)
    rem.add("开会", NOW.replace(hour=15), now=int(NOW.timestamp()), lead_minutes=30)
    assert rem.due_for_notification(NOW.replace(hour=14, minute=29)) == []
    assert len(rem.due_for_notification(NOW.replace(hour=14, minute=30))) == 1


def test_recurring_advances_on_wall_clock_across_dst(conn):
    rem = Reminders(conn, TZ)
    r = rem.add("吃药", datetime(2026, 10, 3, 8, 0, tzinfo=TZ), now=int(NOW.timestamp()), recurrence="DAILY")
    fired_at = datetime(2026, 10, 3, 8, 0, 20, tzinfo=TZ)
    assert rem.claim(rem.due_for_notification(fired_at)[0], fired_at)
    nxt = rem.get(r.id)
    assert (nxt.due.day, nxt.due.hour, nxt.status) == (4, 8, "pending")
    assert nxt.due.utcoffset() == timedelta(hours=11)


def test_recurring_missed_for_days_fires_once_and_skips_ahead(conn):
    rem = Reminders(conn, TZ)
    r = rem.add("吃药", datetime(2026, 9, 18, 8, 0, tzinfo=TZ), now=0, recurrence="DAILY")
    [due] = rem.due_for_notification(NOW)
    assert rem.claim(due, NOW)
    assert rem.get(r.id).due == datetime(2026, 9, 23, 8, 0, tzinfo=TZ)


def test_two_processes_cannot_both_claim_the_same_reminder(tmp_path):
    """常驻服务和终端版同时开着：两边都读到了同一条到期提醒，只有一边能认领成功。"""
    path = tmp_path / "alf.db"
    a, b = Reminders(db.connect(path), TZ), Reminders(db.connect(path), TZ)
    a.add("倒垃圾", NOW - timedelta(minutes=1), now=0)
    a.add("吃药", NOW - timedelta(minutes=1), now=0, recurrence="DAILY")
    seen_by_a, seen_by_b = a.due_for_notification(NOW), b.due_for_notification(NOW)
    assert len(seen_by_a) == len(seen_by_b) == 2
    assert [a.claim(r, NOW) for r in seen_by_a] == [True, True]
    assert [b.claim(r, NOW) for r in seen_by_b] == [False, False]
