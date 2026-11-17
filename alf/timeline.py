"""时间线：episodes 表的读写。对话和 Alf 主动发的消息排在同一条线上，前端只需要盯着 id 往后读。"""
from __future__ import annotations

import sqlite3

SYSTEM_SESSION = "system"   # 心跳任务发的消息不属于任何一次对话


def post(conn: sqlite3.Connection, kind: str, text: str, now: int, channel: str | None = None) -> int:
    """Alf 主动发一条消息（提醒到点、新邮件、每日播报）。带 channel 的会由那个渠道的线程送出去。"""
    with conn:
        cur = conn.execute(
            "INSERT INTO episodes(session_id, role, kind, text, created_at, channel) VALUES (?, 'alf', ?, ?, ?, ?)",
            (SYSTEM_SESSION, kind, text, now, channel))
    return cur.lastrowid


def fetch(conn: sqlite3.Connection, after: int = 0, limit: int = 50) -> list[dict]:
    """after=0 取最近 limit 条（刚打开页面）；否则取 id 大于 after 的（轮询新消息）。都按时间正序返回。"""
    if after:
        rows = conn.execute(
            "SELECT id, role, kind, text, source, created_at FROM episodes WHERE id > ? ORDER BY id LIMIT ?",
            (after, limit)).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, role, kind, text, source, created_at FROM episodes ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()[::-1]
    return [{"id": r["id"], "role": r["role"], "kind": r["kind"], "text": r["text"],
             "source": r["source"], "ts": r["created_at"]} for r in rows]
