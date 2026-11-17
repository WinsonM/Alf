from dataclasses import replace
from datetime import timedelta

import pytest

from alf import jobs, timeline
from alf.config import EmailConfig
from alf.connectors import Item
from alf.decide import Decider, DecisionError
from alf.heartbeat import JobContext
from alf.reminders import Reminders

from .conftest import NOW, TZ, Scripted


@pytest.fixture(autouse=True)
def no_desktop_notifications(monkeypatch):
    sent = []
    monkeypatch.setattr(jobs, "notify_desktop", lambda title, body: sent.append((title, body)))
    return sent


def ctx_for(conn, cfg, backend=None, cursor=None, now=NOW):
    return JobContext(conn, cfg, now, Decider(backend or Scripted(), conn), cursor)


def texts(conn):
    return [(r["kind"], r["text"]) for r in timeline.fetch(conn)]


def test_reminders_job_posts_each_due_reminder_once(conn, cfg, no_desktop_notifications):
    rem = Reminders(conn, TZ)
    rem.add("交作业", NOW - timedelta(seconds=30), now=0)
    rem.add("开会", NOW + timedelta(minutes=30), now=0, lead_minutes=30)
    rem.add("吃药", NOW - timedelta(hours=3), now=0)
    rem.add("以后的事", NOW + timedelta(days=1), now=0)
    assert jobs.reminders_job(ctx_for(conn, cfg)) == "发出 3 条提醒"
    assert texts(conn) == [("reminder", "吃药（07:00，补发）"), ("reminder", "交作业（09:59）"),
                           ("reminder", "开会（半小时后 10:30）")]
    assert [body for _, body in no_desktop_notifications] == ["吃药", "交作业", "开会"]
    assert jobs.reminders_job(ctx_for(conn, cfg)) is None and len(texts(conn)) == 3


def test_briefing_lists_what_is_left_today_and_stays_quiet_otherwise(conn, cfg):
    assert jobs.briefing_job(ctx_for(conn, cfg)) is None and texts(conn) == []
    rem = Reminders(conn, TZ)
    rem.add("已经过去的", NOW - timedelta(hours=1), now=0)
    rem.add("开会", NOW.replace(hour=15), now=0)
    rem.add("明天的事", NOW + timedelta(days=1), now=0)
    assert jobs.briefing_job(ctx_for(conn, cfg)) == "播报了 1 个安排"
    assert texts(conn) == [("briefing", "今天还有 1 个安排：\n  #2  今天 9月22日（周二）15:00  开会")]


class FakeConnector:
    name, label, every = "email", "邮箱", 300

    def __init__(self, items):
        self.items, self.calls = items, []

    def fetch_new(self, cursor):
        self.calls.append(cursor)
        return ([], "v:0") if cursor is None else (self.items, f"v:{len(self.items)}")


def mail(n, title, sender="李老师 <li@uni.edu>"):
    return Item(f"<{n}@x>", sender, title, "正文开头", 1_700_000_000)


URGENT = lambda q, state: 0.9 if "作业" in state["subject"] else 0.1   # noqa: E731
BY_SUBJECT = lambda q, state: "study_work" if "作业" in state["subject"] else "promotion"   # noqa: E731


def test_first_run_only_sets_the_starting_point(conn, cfg):
    run = jobs.inbox_job(FakeConnector([mail(1, "旧邮件")]), triage=True, threshold=0.6)
    ctx = ctx_for(conn, cfg)
    assert run(ctx) == "已接入，从现在起留意新消息"
    assert ctx.cursor == "v:0" and texts(conn) == []


def test_only_mail_that_needs_attention_is_surfaced(conn, cfg, no_desktop_notifications):
    backend = Scripted(needs_attention=URGENT, needs_now=URGENT, category=BY_SUBJECT)
    connector = FakeConnector([mail(1, "作业截止提醒"), mail(2, "双十一大促", "shop <ad@shop.com>")])
    ctx = ctx_for(conn, cfg, backend, cursor="v:0")
    assert jobs.inbox_job(connector, triage=True, threshold=0.6)(ctx) == "2 封新邮件：1 封立刻告诉了你，1 封广告和通知只计数"
    assert texts(conn) == [("email", "李老师来了封信：作业截止提醒（邮箱）")]       # 没配生成式模型：说一声谁来了封什么信
    assert no_desktop_notifications == [("Alf · 邮箱", "李老师 <li@uni.edu>：作业截止提醒")]
    rows = conn.execute("SELECT title, attention, category, notified FROM inbox_items ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("作业截止提醒", 0.9, "study_work", 1), ("双十一大促", 0.1, "promotion", 0)]
    assert ctx.cursor == "v:2"
    # Jev 只看到发件人、主题和正文开头
    state, _ = backend.calls[0]
    assert state == {"from": "李老师 <li@uni.edu>", "subject": "作业截止提醒", "body_start": "正文开头"}


def test_same_mail_is_never_processed_twice(conn, cfg):
    backend = Scripted(needs_attention=0.9, needs_now=0.9, category="personal")
    run = jobs.inbox_job(FakeConnector([mail(1, "你好")]), triage=True, threshold=0.6)
    run(ctx_for(conn, cfg, backend, cursor="v:0"))
    assert run(ctx_for(conn, cfg, backend, cursor="v:0")) is None
    assert len(texts(conn)) == 1 and len(backend.calls) == 1


def test_triage_off_notifies_everything_without_calling_jev(conn, cfg):
    backend = Scripted()
    run = jobs.inbox_job(FakeConnector([mail(1, "双十一大促")]), triage=False, threshold=0.6)
    run(ctx_for(conn, cfg, backend, cursor="v:0"))
    assert texts(conn) == [("email", "李老师来了封信：双十一大促（邮箱）")] and backend.calls == []


def test_jev_failure_keeps_the_cursor_so_mail_is_retried(conn, cfg):
    class DownAfterFirst:
        name, n = "down", 0

        def system_one(self, state, questions):
            self.n += 1
            if self.n > 1:
                raise DecisionError("Jev 限流了，稍后再试")
            return Scripted(needs_attention=0.9, needs_now=0.9, category="personal").system_one(state, questions)

    run = jobs.inbox_job(FakeConnector([mail(1, "第一封"), mail(2, "第二封")]), triage=True, threshold=0.6)
    ctx = ctx_for(conn, cfg, DownAfterFirst(), cursor="v:0")
    with pytest.raises(DecisionError):
        run(ctx)
    assert ctx.cursor == "v:0" and len(texts(conn)) == 1
    # 下一轮 Jev 恢复：第一封靠去重跳过，只补第二封
    ctx = ctx_for(conn, cfg, Scripted(needs_attention=0.9, needs_now=0.9, category="personal"), cursor="v:0")
    assert run(ctx) == "1 封新邮件：1 封立刻告诉了你"
    assert [t for _, t in texts(conn)] == ["李老师来了封信：第一封（邮箱）", "李老师来了封信：第二封（邮箱）"] and ctx.cursor == "v:2"


def test_build_jobs_follows_config(cfg):
    assert [j.name for j in jobs.build_jobs(cfg, [])] == ["reminders", "followups", "calendar", "calendar_alerts", "sources"]
    assert {j.name for j in jobs.build_jobs(cfg, [])} & jobs.FAST_JOBS == {"reminders", "followups", "calendar_alerts"}   # 只查本机库的走快车道
    from datetime import time

    full = replace(cfg, briefing_at=time(8, 0), email=EmailConfig("imap.x", "me@x", "secret", every=120))
    built = jobs.build_jobs(full, [FakeConnector([])])
    assert [(j.name, j.schedule_text()) for j in built] == [
        ("reminders", "每 10 秒"), ("followups", "每 1 分钟"), ("calendar", "每 5 分钟"), ("calendar_alerts", "每 30 秒"),
        ("briefing", "每天 08:00"), ("sources", "每 30 分钟"), ("email", "每 2 分钟"), ("mail_report", "每天 18:00")]
    assert "secret" not in repr(full)


# --- 每晚的记忆汇总 -------------------------------------------------------------------

def remember(conn, text, at):
    conn.execute("INSERT INTO facts(category, text, valid_from, recorded_at) VALUES ('other', ?, ?, ?)", (text, at, at))
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def test_digest_lists_todays_new_memories_once(conn, cfg):
    import json

    evening = NOW.replace(hour=21, minute=30)
    remember(conn, "昨天记的", int((NOW - timedelta(days=1)).timestamp()))
    a = remember(conn, "对花生过敏", int(NOW.timestamp()))
    b = remember(conn, "2026年10月搬到 Glebe", int(NOW.timestamp()) + 60)
    gone = remember(conn, "今天记了又被纠正掉的", int(NOW.timestamp()) + 120)
    conn.execute("UPDATE facts SET valid_to = ? WHERE id = ?", (int(NOW.timestamp()) + 200, gone))

    ctx = ctx_for(conn, cfg, now=evening)
    assert jobs.digest_job(ctx) == "汇总了 2 条新记忆"
    assert texts(conn) == [("digest", "今天记了这些，有不对的跟我说：\n1. 对花生过敏\n2. 2026年10月搬到 Glebe")]
    assert json.loads(ctx.cursor) == {"until": int(evening.timestamp()), "ids": [a, b]}

    tomorrow = evening + timedelta(days=1)
    quiet = ctx_for(conn, cfg, cursor=ctx.cursor, now=tomorrow)
    assert jobs.digest_job(quiet) is None and quiet.cursor == ctx.cursor and len(texts(conn)) == 1   # 没记新东西就不出声
    remember(conn, "这学期在上 COMP5318", int(tomorrow.timestamp()) - 3600)
    again = ctx_for(conn, cfg, cursor=ctx.cursor, now=tomorrow)
    assert jobs.digest_job(again) == "汇总了 1 条新记忆" and texts(conn)[-1][1] == "今天记了这些，有不对的跟我说：\n1. 这学期在上 COMP5318"


def test_digest_only_exists_when_a_generative_model_is_configured(cfg):
    from datetime import time

    from alf.config import LLMConfig

    with_time = replace(cfg, digest_at=time(21, 30))
    assert "digest" not in [j.name for j in jobs.build_jobs(with_time, [])]
    built = jobs.build_jobs(replace(with_time, llm=LLMConfig(api_key="sk-test")), [])
    assert [(j.name, j.label, j.schedule_text()) for j in built] == [
        ("reminders", "到期提醒", "每 10 秒"), ("followups", "提醒后的追问", "每 1 分钟"), ("calendar", "日历同步", "每 5 分钟"),
        ("calendar_alerts", "日程开始前提醒", "每 30 秒"), ("moments", "把当天的对话归纳成经历", "每天 21:20"),   # 比汇总早 10 分钟
        ("impressions", "从当天的话里形成印象", "每天 21:25"),                                                          # 夹在两者之间
        ("digest", "今日记忆汇总", "每天 21:30"),
        ("sources", "信息汇总（MCP）", "每 30 分钟"), ("proactive", "主动找你（活跃）", "每 15 分钟")]


def test_proactive_posts_go_to_the_push_channel(conn, cfg):
    from alf.config import IMessageConfig

    Reminders(conn, TZ).add("喝水", NOW - timedelta(seconds=5), now=0)
    jobs.reminders_job(ctx_for(conn, replace(cfg, imessage=IMessageConfig(owners=("+61412345678",)))))
    assert tuple(conn.execute("SELECT kind, channel, delivered_at FROM episodes").fetchone()) == ("reminder", "imessage", None)


def test_digest_shows_what_a_new_memory_replaced(conn, cfg):
    evening = NOW.replace(hour=21, minute=30)
    old = remember(conn, "这学期在上 COMP5318", int((NOW - timedelta(days=3)).timestamp()))
    new = remember(conn, "这学期在上 COMP5310", int(NOW.timestamp()))
    conn.execute("UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?", (int(NOW.timestamp()), new, old))
    remember(conn, "对花生过敏", int(NOW.timestamp()) + 1)
    jobs.digest_job(ctx_for(conn, cfg, now=evening))
    assert texts(conn)[0][1] == "今天记了这些，有不对的跟我说：\n1. 这学期在上 COMP5310（换掉了：这学期在上 COMP5318）\n2. 对花生过敏"
