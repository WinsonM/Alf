"""把「日历」里未来几天的日程同步到本地一张表。

有了这张表，三件事才做得到：
- 发现用户在日历里新增、改期、取消了什么，几分钟内告诉他；
- 在日程开始前提醒（提醒走每 30 秒一次的快车道，不用每次都去读日历）；
- 对话时让模型知道日历上有什么。

"新增"的判定要小心一处：同步的范围是"从今天起 7 天"，每过一天就有新的一天滑进来，那一天的日程
并不是用户刚加的。所以只有开始时间落在"上一次已经同步过的范围"里的，才算新增。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from . import apple
from .connectors import ConnectorError
from .heartbeat import JobContext
from .prefs import in_quiet_hours

SYNC_DAYS = 7
LATE_ALERT = 120      # 日程已经开始超过这么多秒就不再补发"开始前提醒"（电脑当时在睡觉）


def event_line(start: datetime, end: datetime, all_day: bool, title: str, location: str | None, now: datetime) -> str:
    """"明天 14:00–15:00 组会 @ ABS 1110"。"""
    days = (start.date() - now.date()).days
    day = {0: "今天", 1: "明天", 2: "后天"}.get(days, f"{start.month}月{start.day}日（周{'一二三四五六日'[start.weekday()]}）")
    when = f"{day} 全天" if all_day else f"{day} {start:%H:%M}–{end:%H:%M}"
    return f"{when} {title}" + (f" @ {location}" if location else "")


def row_line(r: sqlite3.Row, now: datetime) -> str:
    tz = now.tzinfo
    return event_line(datetime.fromtimestamp(r["start_at"], tz), datetime.fromtimestamp(r["end_at"], tz),
                      bool(r["all_day"]), r["title"], r["location"], now)


def upcoming(conn: sqlite3.Connection, now: datetime, days: int = SYNC_DAYS, limit: int = 30) -> list[sqlite3.Row]:
    end = (now + timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    return conn.execute("SELECT * FROM calendar_events WHERE end_at > ? AND start_at < ? ORDER BY start_at LIMIT ?",
                        (int(now.timestamp()), int(end.timestamp()), limit)).fetchall()


def remember_own(conn: sqlite3.Connection, event_id: str, title: str, start: datetime, end: datetime, all_day: bool, location: str) -> None:
    """Alf 自己刚写进日历的日程：马上记进表里，下一次同步就不会把它当成"用户新增的"再播报一遍。"""
    with conn:
        conn.execute("INSERT OR REPLACE INTO calendar_events(key, event_id, title, start_at, end_at, all_day, location, by_alf)"
                     " VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                     (f"{event_id}@{int(start.timestamp())}", event_id, title, int(start.timestamp()), int(end.timestamp()), all_day, location))


def forget_own(conn: sqlite3.Connection, event_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM calendar_events WHERE event_id = ?", (event_id,))


def apply(conn: sqlite3.Connection, events: list[dict], now: datetime, known_until: int | None) -> list[str]:
    """用刚读到的日程替换表里的，返回要告诉用户的变动（每条一句）。known_until 是上一次同步覆盖到的时间；
    None 表示第一次同步——那时表里是空的，所有日程都"是新的"，一条都不播报。"""
    tz, ts = now.tzinfo, int(now.timestamp())
    fresh: dict[str, dict] = {}
    for e in events:
        try:
            start, end = datetime.fromisoformat(e["start"]), datetime.fromisoformat(e["end"])
        except (KeyError, ValueError, TypeError):
            continue
        event_id = str(e.get("id") or f"{e.get('title', '')}|{e.get('calendar', '')}")
        fresh[f"{event_id}@{int(start.timestamp())}"] = {
            "event_id": event_id, "title": str(e.get("title") or "（无标题）"), "start_at": int(start.timestamp()),
            "end_at": int(end.timestamp()), "all_day": bool(e.get("all_day")), "location": e.get("location") or "",
            "calendar": e.get("calendar") or "", "by_alf": bool(e.get("by_alf"))}
    old = {r["key"]: r for r in conn.execute("SELECT * FROM calendar_events")}
    added = [k for k in fresh if k not in old]
    gone = [k for k, r in old.items() if k not in fresh and r["start_at"] > ts]   # 已经开始或结束的日程自然会从结果里消失，不算取消

    def at(epoch: int) -> datetime:
        return datetime.fromtimestamp(epoch, tz)

    def line(e) -> str:
        return event_line(at(e["start_at"]), at(e["end_at"]), bool(e["all_day"]), e["title"], e["location"], now)

    changes: list[str] = []
    if known_until is not None:
        moved: set[str] = set()
        for k in gone:
            # 同一个日程只有一处消失、一处出现：是改期，不是"取消一个又新增一个"。
            twins = [n for n in added if fresh[n]["event_id"] == old[k]["event_id"]]
            if len(twins) == 1 and sum(old[g]["event_id"] == old[k]["event_id"] for g in gone) == 1:
                changes.append(f"改期：{old[k]['title']} 从 {line(old[k]).split(' ' + old[k]['title'])[0]} 改到 {line(fresh[twins[0]]).split(' ' + fresh[twins[0]]['title'])[0]}")
                moved |= {k, twins[0]}
        for k in added:
            e = fresh[k]
            if k not in moved and not e["by_alf"] and ts < e["start_at"] < known_until:
                changes.append(f"新增：{line(e)}")
        for k in gone:
            if k not in moved and old[k]["start_at"] < known_until:
                changes.append(f"取消：{line(old[k])}")
    with conn:
        conn.execute("DELETE FROM calendar_events WHERE end_at < ?", (ts - 86400,))
        for k in gone:
            conn.execute("DELETE FROM calendar_events WHERE key = ?", (k,))
        for k, e in fresh.items():
            conn.execute(
                "INSERT INTO calendar_events(key, event_id, title, start_at, end_at, all_day, location, calendar, by_alf)"
                " VALUES (:key, :event_id, :title, :start_at, :end_at, :all_day, :location, :calendar, :by_alf)"
                " ON CONFLICT(key) DO UPDATE SET title = excluded.title, end_at = excluded.end_at, all_day = excluded.all_day,"
                " location = excluded.location, calendar = excluded.calendar, by_alf = MAX(by_alf, excluded.by_alf)",
                {"key": k, **e})
    return changes


def calendar_job(ctx: JobContext, read=apple.read_events) -> str | None:
    """读日历、更新本地表、把变动告诉用户。夜里发现的变动攒着，过了安静时段再说。"""
    state = json.loads(ctx.cursor or "{}")
    data = read(SYNC_DAYS)
    if "error" in data:
        if data["error"] == apple.NEEDS_PERMISSION["event"]:
            return f"还没同步：{data['error']}"     # 用户还没点授权，这是预期中的状态，不算任务失败
        raise ConnectorError(data["error"])
    changes = apply(ctx.conn, data.get("events", []), ctx.now, state.get("until"))
    until = (ctx.now + timedelta(days=SYNC_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0)
    pending = [*state.get("pending", []), *changes]
    told = 0
    if pending and not in_quiet_hours(ctx.now, ctx.prefs):
        ctx.post("calendar", "日历有变动：\n" + "\n".join(pending[:12]))
        told, pending = len(pending), []
    ctx.cursor = json.dumps({"until": int(until.timestamp()), "pending": pending[:30]}, ensure_ascii=False)
    count = len(data.get("events", []))
    if told:
        return f"同步了 {count} 条日程，告诉了你 {told} 处变动"
    return f"同步了 {count} 条日程" + (f"，{len(pending)} 处变动等早上再说" if pending else "")


def calendar_alerts_job(ctx: JobContext) -> str | None:
    """日程开始前提醒。只查本地表，所以可以每半分钟查一次；全天日程不提醒（早报里会有）。"""
    if not ctx.prefs.get("calendar.alerts", True):
        return None
    lead, ts = int(ctx.prefs.get("calendar.lead_minutes", 15)) * 60, int(ctx.now.timestamp())
    rows = ctx.conn.execute("SELECT * FROM calendar_events WHERE all_day = 0 AND alerted_at IS NULL"
                            " AND start_at - ? <= ? AND start_at + ? > ? ORDER BY start_at", (lead, ts, LATE_ALERT, ts)).fetchall()
    sent = 0
    for r in rows:
        with ctx.conn:   # 带条件的 UPDATE：两个进程同时查到也只有一个能认领，不会响两遍
            claimed = ctx.conn.execute("UPDATE calendar_events SET alerted_at = ? WHERE key = ? AND alerted_at IS NULL", (ts, r["key"])).rowcount
        if not claimed:
            continue
        tz = ctx.now.tzinfo
        start, end = datetime.fromtimestamp(r["start_at"], tz), datetime.fromtimestamp(r["end_at"], tz)
        minutes = round((r["start_at"] - ts) / 60)
        lead_text = f"{minutes} 分钟后" if minutes >= 1 else "现在开始"
        ctx.post("calendar", f"{lead_text}：{r['title']}（{start:%H:%M}–{end:%H:%M}" + (f" @ {r['location']}" if r["location"] else "") + "）")
        sent += 1
    return f"发出 {sent} 条日程提醒" if sent else None
