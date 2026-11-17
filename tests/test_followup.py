"""提醒发出后的跟进（场景 14）：没回应 → 追问一次 → 第二天早报再提一次 → 不再追。"""
from datetime import timedelta

import pytest

from alf import followup, jobs, timeline
from alf.decide import Decider
from alf.heartbeat import JobContext
from alf.reminders import Reminders

from .conftest import NOW, TZ, Scripted


@pytest.fixture(autouse=True)
def no_desktop_notifications(monkeypatch):
    monkeypatch.setattr(jobs, "notify_desktop", lambda *a: None)


def ctx_at(conn, cfg, now, **prefs):
    return JobContext(conn, cfg, now, Decider(Scripted(), conn), None, prefs=prefs)


def posted(conn):
    return [(r["kind"], r["text"]) for r in timeline.fetch(conn)]


def says(conn, text, now):
    conn.execute("INSERT INTO episodes(session_id, role, text, created_at) VALUES ('s', 'user', ?, ?)", (text, int(now.timestamp())))


def fire(conn, cfg, title, at):
    Reminders(conn, TZ).add(title, at, now=0)
    jobs.reminders_job(ctx_at(conn, cfg, at + timedelta(seconds=5)))


def test_no_answer_gets_one_question_then_one_mention_in_the_briefing_then_silence(conn, cfg):
    fire(conn, cfg, "交作业", NOW)
    assert followup.followup_job(ctx_at(conn, cfg, NOW + timedelta(minutes=29))) is None          # 还没到半小时
    assert followup.followup_job(ctx_at(conn, cfg, NOW + timedelta(minutes=31))) == "追问了 1 件事"
    assert posted(conn) == [("reminder", "交作业（10:00）"), ("followup", "「交作业」做了吗？")]
    assert followup.followup_job(ctx_at(conn, cfg, NOW + timedelta(hours=3))) is None             # 只追问一次
    morning = (NOW + timedelta(days=1)).replace(hour=8, minute=0)
    assert jobs.briefing_job(ctx_at(conn, cfg, morning)) == "播报了 0 个安排"
    assert posted(conn)[-1] == ("briefing", "之前提醒过、还没回音的：交作业。做了的话跟我说一声～")
    assert jobs.briefing_job(ctx_at(conn, cfg, morning + timedelta(days=1))) is None               # 之后不再提
    assert followup.followup_job(ctx_at(conn, cfg, morning + timedelta(days=1))) is None


def test_any_reply_after_the_reminder_ends_it(conn, cfg):
    fire(conn, cfg, "吃药", NOW)
    says(conn, "做了", NOW + timedelta(minutes=10))
    assert followup.followup_job(ctx_at(conn, cfg, NOW + timedelta(minutes=40))) is None
    fire(conn, cfg, "交房租", NOW + timedelta(hours=1))
    assert followup.followup_job(ctx_at(conn, cfg, NOW + timedelta(hours=1, minutes=40))) == "追问了 1 件事"
    says(conn, "交了交了", NOW + timedelta(hours=2))                                               # 追问之后才回：早报里就不用再提
    assert jobs.briefing_job(ctx_at(conn, cfg, (NOW + timedelta(days=1)).replace(hour=8, minute=0))) is None
    assert [kind for kind, _ in posted(conn) if kind != "chat"] == ["reminder", "reminder", "followup"]   # 不算他自己说的两句


def test_nobody_is_chased_at_night_or_when_it_is_switched_off(conn, cfg):
    late = NOW.replace(hour=21, minute=50)
    fire(conn, cfg, "关窗", late)
    assert followup.followup_job(ctx_at(conn, cfg, late + timedelta(minutes=35))) is None          # 22:25，安静时段
    assert followup.followup_job(ctx_at(conn, cfg, late + timedelta(hours=11))) is None            # 第二天上午也不补问
    assert jobs.briefing_job(ctx_at(conn, cfg, (NOW + timedelta(days=1)).replace(hour=8, minute=0))) == "播报了 0 个安排"
    assert posted(conn)[-1][1] == "之前提醒过、还没回音的：关窗。做了的话跟我说一声～"
    fire(conn, cfg, "喝水", NOW + timedelta(days=2))
    off = ctx_at(conn, cfg, NOW + timedelta(days=2, minutes=40), **{"followup.enabled": False})
    assert followup.followup_job(off) is None and [kind for kind, _ in posted(conn)].count("followup") == 0
