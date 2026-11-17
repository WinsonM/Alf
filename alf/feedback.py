"""用户对 Alf 主动发的消息的评价：在 iMessage 里点 👍👎，或者回一句"没用"。

7 天试用的一条线是"主动消息里觉得没用的不超过 20%"，按这里的记录算。每一次"没用"同时也存成一个例子
（见 lessons.note_rating）：他嫌没用的那几条，模型下次要不要主动开口时能看到。
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime

# 算作"主动消息"的类型。用户亲口设的提醒（reminder）不算：那是他要的。
PROACTIVE_KINDS = ("nudge", "briefing", "digest", "email", "calendar", "followup")
# 「信息」里的回应类型：2000 喜欢 2001 赞 2002 踩 2003 哈哈 2004 强调 2005 疑问；3000 起是取消对应的回应。
UP, DOWN = {2000, 2001}, {2002}
USELESS = re.compile(r"^(?:这条?|这个|这种|刚才那条)?(?:消息|提醒|新闻)?(?:对我)?(?:真的?|完全)?(?:没用|没啥用|没什么用|不需要|别发这种|不用发这种|useless)[了的啊呀吧。.!！~～ ]*$", re.I)
USEFUL = re.compile(r"^(?:这条?|这个)?(?:消息)?(?:挺|很|蛮|真)?(?:有用|有帮助)[了的啊呀。.!！~～ ]*$")


def rate(conn: sqlite3.Connection, episode_id: int, rating: int, source: str, now: int) -> None:
    with conn:
        conn.execute("INSERT INTO feedback(episode_id, rating, source, created_at) VALUES (?, ?, ?, ?)"
                     " ON CONFLICT(episode_id) DO UPDATE SET rating = excluded.rating, source = excluded.source, created_at = excluded.created_at",
                     (episode_id, rating, source, now))


def from_tapback(conn: sqlite3.Connection, reaction: int, episode_id: int | None, now: int) -> str | None:
    """处理一次点按回应。返回 "up"/"down"/"cleared"，不相干的回应（哈哈、强调……）返回 None。"""
    if episode_id is None:
        return None
    if reaction in UP or reaction in DOWN:
        rate(conn, episode_id, 1 if reaction in UP else -1, "tapback", now)
        return "up" if reaction in UP else "down"
    if reaction - 1000 in UP | DOWN:
        with conn:
            conn.execute("DELETE FROM feedback WHERE episode_id = ? AND source = 'tapback'", (episode_id,))
        return "cleared"
    return None


def from_text(conn: sqlite3.Connection, text: str, before_episode: int, now: int, target: int | None = None) -> tuple[int, int] | None:
    """用户回了一句"没用"（或"有用"）：记成对某一条消息的评价，返回（评价，那条消息的编号）。

    说的是哪一条：他引用回复了 Alf 的某条消息（target），就是那一条；没引用，就得紧挨着的上一条是 Alf 主动发的才算——
    不然一句"没用"可能是在说别的。
    """
    rating = -1 if USELESS.match(text.strip()) else 1 if USEFUL.match(text.strip()) else 0
    if not rating:
        return None
    if target is not None:
        quoted = conn.execute("SELECT id FROM episodes WHERE id = ? AND role = 'alf'", (target,)).fetchone()
        if not quoted:
            return None
        rate(conn, quoted["id"], rating, "text", now)
        return rating, quoted["id"]
    last = conn.execute("SELECT id, role, kind FROM episodes WHERE id < ? ORDER BY id DESC LIMIT 1", (before_episode,)).fetchone()
    if not last or last["role"] != "alf" or last["kind"] not in PROACTIVE_KINDS:
        return None
    rate(conn, last["id"], rating, "text", now)
    return rating, last["id"]


def stats(conn: sqlite3.Connection, now: datetime, days: int = 7) -> dict:
    since = int(now.timestamp()) - days * 86400
    marks = ",".join("?" * len(PROACTIVE_KINDS))
    sent = conn.execute(f"SELECT COUNT(*) FROM episodes WHERE role = 'alf' AND kind IN ({marks}) AND created_at >= ?",
                        (*PROACTIVE_KINDS, since)).fetchone()[0]
    rows = conn.execute(f"SELECT f.rating, COUNT(*) AS n FROM feedback f JOIN episodes e ON e.id = f.episode_id"
                        f" WHERE e.kind IN ({marks}) AND e.created_at >= ? GROUP BY f.rating", (*PROACTIVE_KINDS, since)).fetchall()
    counts = {r["rating"]: r["n"] for r in rows}
    down = counts.get(-1, 0)
    return {"days": days, "sent": sent, "down": down, "up": counts.get(1, 0), "useless_rate": round(down / sent, 3) if sent else None}
