import time as clock
from datetime import datetime, time, timedelta

from alf import timeline
from alf.connectors import ConnectorError
from alf.decide import Decider, OfflineBackend
from alf.heartbeat import Heartbeat, Job, is_due, job_status, load_state, next_run

from .conftest import NOW, TZ


def make(cfg, jobs, on_post=None):
    return Heartbeat(cfg, jobs, make_decider=lambda c: Decider(OfflineBackend(), c), on_post=on_post)


def test_interval_job_is_due_first_time_and_after_its_interval():
    job = Job("j", "J", lambda ctx: None, every=300)
    assert is_due(job, None, NOW)
    assert not is_due(job, NOW - timedelta(seconds=299), NOW)
    assert is_due(job, NOW - timedelta(seconds=300), NOW)


def test_daily_job_runs_once_per_day_and_catches_up_after_sleep():
    job = Job("j", "J", lambda ctx: None, daily_at=time(8, 0))
    at_7, at_8, at_13 = (NOW.replace(hour=h, minute=0) for h in (7, 8, 13))
    yesterday_8 = at_8 - timedelta(days=1)
    assert not is_due(job, yesterday_8, at_7)        # 还没到点
    assert is_due(job, yesterday_8, at_8)            # 到点
    assert is_due(job, yesterday_8, at_13)           # 8 点时电脑在睡觉，13 点醒来补跑
    assert not is_due(job, at_8, at_13)              # 今天跑过了
    assert next_run(job, at_8, at_13) == at_8 + timedelta(days=1)
    assert Job("j", "J", lambda c: None, every=300).schedule_text() == "每 5 分钟" and job.schedule_text() == "每天 08:00"


def test_tick_runs_due_jobs_and_persists_state(cfg):
    calls = []
    hb = make(cfg, [Job("a", "A", lambda ctx: calls.append(ctx.now) or "做了一件事", every=60)])
    hb.tick(NOW)
    hb.tick(NOW + timedelta(seconds=30))    # 没到点
    hb.tick(NOW + timedelta(seconds=60))
    assert calls == [NOW, NOW + timedelta(seconds=60)]
    state = load_state(hb._conn, "a")
    assert (state.runs, state.ok, state.result) == (2, True, "做了一件事")
    # 重启（换一个 Heartbeat 实例）后不会马上重跑：状态在库里
    hb2 = make(cfg, [Job("a", "A", lambda ctx: calls.append("again"), every=60)])
    hb2.tick(NOW + timedelta(seconds=90))
    assert "again" not in calls


def test_failing_job_does_not_stop_the_others_and_keeps_its_cursor(cfg):
    def flaky(ctx):
        ctx.cursor = "moved"
        raise ConnectorError("连不上邮件服务器")

    def steady(ctx):
        ctx.cursor = "42"
        ctx.post("reminder", "喝水")

    posted = []
    hb = make(cfg, [Job("flaky", "F", flaky, every=60), Job("steady", "S", steady, every=60)],
              on_post=lambda kind, text: posted.append((kind, text)))
    hb.tick(NOW)
    flaky_state, steady_state = load_state(hb._conn, "flaky"), load_state(hb._conn, "steady")
    assert (flaky_state.ok, flaky_state.error, flaky_state.cursor) == (False, "连不上邮件服务器", None)
    assert (steady_state.ok, steady_state.cursor) == (True, "42")
    assert posted == [("reminder", "喝水")]
    [row] = timeline.fetch(hb._conn)
    assert (row["role"], row["kind"], row["text"]) == ("alf", "reminder", "喝水")
    status = {j["name"]: j for j in job_status(hb._conn, hb.jobs, NOW + timedelta(seconds=10))}
    assert status["flaky"]["error"] == "连不上邮件服务器"
    assert status["steady"]["next_run"] == int((NOW + timedelta(seconds=60)).timestamp())


def test_requested_job_runs_before_it_is_due(cfg):
    calls = []
    hb = make(cfg, [Job("a", "A", lambda ctx: calls.append(1), every=3600)])
    hb.tick(NOW)
    assert hb.request("a") and not hb.request("nope")
    hb.tick(NOW + timedelta(seconds=5))
    hb.tick(NOW + timedelta(seconds=10))
    assert len(calls) == 2


def test_thread_ticks_until_stopped(cfg, tmp_path):
    from dataclasses import replace

    calls = []
    hb = make(replace(cfg, heartbeat_tick=0.02), [Job("a", "A", lambda ctx: calls.append(1), every=0)])
    hb.start()
    deadline = clock.monotonic() + 3
    while len(calls) < 3 and clock.monotonic() < deadline:
        clock.sleep(0.01)
    hb.stop()
    hb.join(3)
    assert len(calls) >= 3 and not hb.is_alive()
    assert isinstance(hb.last_tick, datetime) and hb.last_tick.tzinfo == TZ


def test_repeated_failures_back_off_and_recover(cfg):
    attempts, healthy = [], []

    def job(ctx):
        attempts.append(int((ctx.now - NOW).total_seconds()))
        if not healthy:
            raise ConnectorError("邮箱登录失败")

    hb = make(cfg, [Job("email", "邮箱", job, every=300)])
    for seconds in range(0, 3 * 3600 + 1, 60):       # 三小时，每分钟一次心跳
        if seconds == 2 * 3600:
            healthy.append(True)
        hb.tick(NOW + timedelta(seconds=seconds))
    failing = [a for a in attempts if a < 2 * 3600]
    # 间隔依次是 10、20、40 分钟，之后封顶在 60 分钟，而不是每 5 分钟撞一次
    assert failing == [0, 600, 1800, 4200]
    recovered = [a for a in attempts if a >= 2 * 3600]
    assert recovered[0] == 7800 and recovered[1] - recovered[0] == 300   # 成功一次后恢复正常间隔
    assert hb.request("email")
