"""心跳：一个常驻线程，每隔几秒醒一次，看哪些任务到点了就跑。

"全天候运行"拆成两层：进程不死由 launchd 负责（见 service.py），进程里"每隔多久
做什么"由这里负责。没有任务到点时，一次心跳只是几条本地 SQLite 查询，不调模型、
不联网，所以心跳可以很密而不花钱。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Callable

from . import db, prefs, timeline
from .config import Settings
from .connectors import ConnectorError
from .decide import Decider, DecisionError

log = logging.getLogger("alf.heartbeat")

MAX_BACKOFF = 12   # 连续失败时，重试间隔最多拉长到原来的 12 倍（5 分钟的任务 → 1 小时）


@dataclass
class Job:
    """一个周期任务。every（每隔几秒）和 daily_at（每天几点）二选一。"""

    name: str
    label: str                                   # 状态面板上显示的名字
    run: Callable[[JobContext], str | None]      # 返回一句结果摘要；没什么可说就返回 None
    every: float | None = None
    daily_at: time | None = None

    def schedule_text(self) -> str:
        if self.daily_at is not None:
            return f"每天 {self.daily_at:%H:%M}"
        if self.every >= 3600:
            return f"每 {self.every / 3600:g} 小时"
        if self.every >= 60:
            return f"每 {self.every / 60:g} 分钟"
        return f"每 {self.every:g} 秒"


@dataclass
class JobContext:
    """任务运行时拿到的东西。conn 和 decider 属于心跳线程，任务不要跨线程保存它们。"""

    conn: sqlite3.Connection
    cfg: Settings
    now: datetime
    decider: Decider
    cursor: str | None                           # 任务自己的进度；改了它，成功后会被保存
    on_post: Callable[[str, str], None] | None = None
    prefs: dict = field(default_factory=dict)    # 界面上的设置（见 prefs.py）；空 = 全用默认
    llm: Any = None                              # 生成式模型（LLM），没配置时是 None

    def post(self, kind: str, text: str) -> int:
        """往时间线发一条消息（网页会读到）。"""
        row = timeline.post(self.conn, kind, text, int(self.now.timestamp()), self.cfg.push_channel)
        if self.on_post:
            self.on_post(kind, text)
        return row


@dataclass
class JobState:
    name: str
    last_run_at: int | None = None
    ok: bool | None = None
    error: str | None = None
    result: str | None = None
    runs: int = 0
    cursor: str | None = None


def load_state(conn: sqlite3.Connection, name: str) -> JobState:
    r = conn.execute("SELECT * FROM job_state WHERE name = ?", (name,)).fetchone()
    if not r:
        return JobState(name)
    return JobState(name, r["last_run_at"], None if r["ok"] is None else bool(r["ok"]),
                    r["error"], r["result"], r["runs"], r["cursor"])


def save_state(conn: sqlite3.Connection, s: JobState) -> None:
    with conn:
        conn.execute(
            "INSERT INTO job_state(name, last_run_at, ok, error, result, runs, cursor) VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET last_run_at = excluded.last_run_at, ok = excluded.ok,"
            " error = excluded.error, result = excluded.result, runs = excluded.runs, cursor = excluded.cursor",
            (s.name, s.last_run_at, s.ok, s.error, s.result, s.runs, s.cursor))


def is_due(job: Job, last_run: datetime | None, now: datetime) -> bool:
    if job.daily_at is None:
        return last_run is None or (now - last_run).total_seconds() >= job.every
    # 每天一次：过了今天的点、且今天这个点之后还没跑过。电脑当时在睡觉的话，醒来会补跑。
    at = now.replace(hour=job.daily_at.hour, minute=job.daily_at.minute, second=0, microsecond=0)
    return now >= at and (last_run is None or last_run < at)


def next_run(job: Job, last_run: datetime | None, now: datetime) -> datetime:
    if job.daily_at is None:
        return now if last_run is None else last_run + timedelta(seconds=job.every)
    at = now.replace(hour=job.daily_at.hour, minute=job.daily_at.minute, second=0, microsecond=0)
    if is_due(job, last_run, now):
        return now
    return at if now < at else at + timedelta(days=1)


def job_status(conn: sqlite3.Connection, jobs: list[Job], now: datetime) -> list[dict]:
    """给状态面板用：每个任务上次什么时候跑的、成没成、下次什么时候跑。"""
    out = []
    for job in jobs:
        s = load_state(conn, job.name)
        last = datetime.fromtimestamp(s.last_run_at, now.tzinfo) if s.last_run_at else None
        out.append({"name": job.name, "label": job.label, "schedule": job.schedule_text(),
                    "last_run": s.last_run_at, "ok": s.ok, "error": s.error, "result": s.result,
                    "runs": s.runs, "next_run": int(next_run(job, last, now).timestamp())})
    return out


class Heartbeat(threading.Thread):
    def __init__(self, cfg: Settings, jobs: list[Job] | Callable[[dict], list[Job]],
                 make_decider: Callable[[sqlite3.Connection], Decider],
                 on_post: Callable[[str, str], None] | None = None,
                 make_llm: Callable[[sqlite3.Connection], Any] | None = None, name: str = "alf-heartbeat"):
        """jobs 可以是固定的任务表，也可以是"按当前设置生成任务表"的函数。
        后者每次心跳都重新生成，所以在界面上改了时间或开关，下一次心跳就生效。"""
        super().__init__(daemon=True, name=name)
        self.cfg, self.make_decider, self.on_post, self.make_llm = cfg, make_decider, on_post, make_llm
        self._llm = None
        self._jobs_for = jobs if callable(jobs) else None
        self.jobs: list[Job] = jobs(prefs.defaults(cfg)) if callable(jobs) else jobs
        self.last_tick: datetime | None = None
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._requested: set[str] = set()
        self._failures: dict[str, int] = {}   # 任务名 -> 连续失败次数
        self._conn: sqlite3.Connection | None = None
        self._decider: Decider | None = None

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                self.tick(datetime.now(self.cfg.tz))
            except Exception:
                log.exception("这一轮心跳失败，下一轮继续")   # 比如数据库一时被锁；不能让线程死掉
            self._halt.wait(self.cfg.heartbeat_tick)

    def stop(self) -> None:
        self._halt.set()

    def request(self, name: str) -> bool:
        """让某个任务在下一次心跳立刻跑一遍（不等它到点）。任务仍然在心跳线程里执行。"""
        if not any(j.name == name for j in self.jobs):
            return False
        with self._lock:
            self._requested.add(name)
        return True

    def tick(self, now: datetime) -> None:
        if self._conn is None:   # 在第一次心跳时才建：SQLite 连接不能跨线程用
            self._conn = db.connect(self.cfg.db_path)
            self._decider = self.make_decider(self._conn)
            self._llm = self.make_llm(self._conn) if self.make_llm else None
        with self._lock:
            forced, self._requested = self._requested, set()
        current = prefs.load(self._conn, self.cfg)
        if self._jobs_for:
            self.jobs = self._jobs_for(current)
        for job in self.jobs:
            state = load_state(self._conn, job.name)
            last = datetime.fromtimestamp(state.last_run_at, now.tzinfo) if state.last_run_at else None
            if job.name not in forced:
                if not is_due(job, last, now):
                    continue
                # 连续失败就把间隔翻倍。最要紧的场景是邮箱密码填错：不退避的话一天会用错密码
                # 登录几百次，有的邮箱会因此临时封号。手动点"立即运行"不受退避限制。
                wait = job.every * min(2 ** self._failures.get(job.name, 0), MAX_BACKOFF) if job.every else 0
                if last and (now - last).total_seconds() < wait:
                    continue
            ctx = JobContext(self._conn, self.cfg, now, self._decider, state.cursor, self.on_post, current, self._llm)
            try:
                result = job.run(ctx)
            except Exception as e:
                # 一个任务出错（邮箱连不上、Jev 限流……）不能拖垮其他任务。游标不动，下次从原处重来。
                state.ok = False
                state.error = str(e) if isinstance(e, (ConnectorError, DecisionError)) else f"{type(e).__name__}: {e}"
                self._failures[job.name] = self._failures.get(job.name, 0) + 1
                log.warning("任务 %s 失败（连续第 %d 次）：%s", job.name, self._failures[job.name], state.error)
            else:
                self._failures.pop(job.name, None)
                state.ok, state.error, state.cursor = True, None, ctx.cursor
                if result:
                    state.result = result
                    log.info("任务 %s：%s", job.name, result)
            state.last_run_at = int(now.timestamp())
            state.runs += 1
            save_state(self._conn, state)
        self.last_tick = now
