"""Reminder storage. Times are UTC epoch seconds; recurrences advance on the local wall clock."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .timeparse import next_occurrence


@dataclass
class Reminder:
    id: int
    title: str
    due: datetime       # local, tz-aware
    notify: datetime    # due - lead
    lead_minutes: int
    recurrence: str | None
    status: str
    source: str

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Reminder:
        tz = ZoneInfo(r["tz"])
        return cls(r["id"], r["title"], datetime.fromtimestamp(r["due_at"], tz),
                   datetime.fromtimestamp(r["notify_at"], tz), r["lead_minutes"], r["recurrence"],
                   r["status"], r["source"])


class Reminders:
    def __init__(self, conn: sqlite3.Connection, tz: ZoneInfo):
        self.conn = conn
        self.tz = tz

    def add(self, title: str, due: datetime, *, now: int, lead_minutes: int = 0, recurrence: str | None = None,
            source: str = "text", attachment_id: str | None = None, episode_id: int | None = None) -> Reminder:
        due_at = int(due.timestamp())
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO reminders(title, due_at, lead_minutes, notify_at, tz, recurrence, source,"
                " source_attachment, source_episode, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (title, due_at, lead_minutes, due_at - lead_minutes * 60, self.tz.key, recurrence, source,
                 attachment_id, episode_id, now))
        return self.get(cur.lastrowid)

    def get(self, reminder_id: int) -> Reminder | None:
        row = self.conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,)).fetchone()
        return Reminder.from_row(row) if row else None

    def pending(self, start: datetime | None = None, end: datetime | None = None, limit: int = 50) -> list[Reminder]:
        s = int(start.timestamp()) if start else None
        e = int(end.timestamp()) if end else None
        rows = self.conn.execute(
            "SELECT * FROM reminders WHERE status = 'pending'"
            " AND (? IS NULL OR due_at >= ?) AND (? IS NULL OR due_at < ?) ORDER BY due_at LIMIT ?",
            (s, s, e, e, limit))
        return [Reminder.from_row(r) for r in rows]

    def update(self, reminder_id: int, *, title: str | None = None, due: datetime | None = None,
               lead_minutes: int | None = None, recurrence: str | None = None) -> Reminder | None:
        """改一条待办提醒；没给的字段保持原样。改了时间就重新算通知时间，并让它可以再次触发。"""
        r = self.get(reminder_id)
        if not r or r.status != "pending":
            return None
        due_at = int((due or r.due).timestamp())
        lead = r.lead_minutes if lead_minutes is None else lead_minutes
        with self.conn:
            self.conn.execute(
                "UPDATE reminders SET title = ?, due_at = ?, lead_minutes = ?, notify_at = ?, recurrence = ?,"
                " notified_at = NULL WHERE id = ?",
                (title or r.title, due_at, lead, due_at - lead * 60, recurrence or r.recurrence, reminder_id))
        return self.get(reminder_id)

    def cancel(self, reminder_id: int) -> Reminder | None:
        with self.conn:
            cur = self.conn.execute(
                "UPDATE reminders SET status = 'cancelled' WHERE id = ? AND status = 'pending'", (reminder_id,))
        return self.get(reminder_id) if cur.rowcount else None

    def due_for_notification(self, now: datetime) -> list[Reminder]:
        rows = self.conn.execute(
            "SELECT * FROM reminders WHERE status = 'pending' AND notified_at IS NULL AND notify_at <= ?"
            " ORDER BY notify_at", (int(now.timestamp()),))
        return [Reminder.from_row(r) for r in rows]

    def claim(self, r: Reminder, now: datetime) -> bool:
        """认领一条到期提醒：返回 True 的调用方负责把它发出去。

        常驻服务和终端版可能同时开着，各有一个心跳在查同一张表。UPDATE 带着"它还是我
        读到时的样子"这个条件，所以两边同时抢也只有一个能改成功，提醒不会响两遍。
        一次性提醒认领后即完成；重复提醒推进到下一次还没到的时间。
        """
        with self.conn:
            if r.recurrence:
                after = max(r.due, now.astimezone(r.due.tzinfo))
                nxt = int(next_occurrence(r.recurrence, r.due.hour, r.due.minute, after=after).timestamp())
                cur = self.conn.execute(
                    "UPDATE reminders SET due_at = ?, notify_at = ? WHERE id = ? AND status = 'pending' AND due_at = ?",
                    (nxt, nxt - r.lead_minutes * 60, r.id, int(r.due.timestamp())))
            else:
                cur = self.conn.execute(
                    "UPDATE reminders SET notified_at = ?, status = 'done'"
                    " WHERE id = ? AND status = 'pending' AND notified_at IS NULL",
                    (int(now.timestamp()), r.id))
        return cur.rowcount == 1
