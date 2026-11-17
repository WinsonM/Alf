"""提醒发出之后的跟进：过一阵没回应就追问一次，第二天早报再提一次，之后不再追。

"有没有回应"用最简单的标准：那条提醒之后用户说过任何话，就算回应了。不去猜他那句话是不是在说
这件事——猜错了去追问一件他已经答复过的事，比少问一次更烦人。

节奏是写死在程序里的，不交给模型决定：验收标准要求它可预期。
问的那句话怎么说可以交给模型（"起床了吗？"比"「叫我起床」做了吗？"像人话），但只是措辞：
模型没配、出错、或者写出来的不像一句短问话，就用固定的句式。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from . import lessons
from .heartbeat import JobContext
from .llm import LLMError
from .prefs import in_quiet_hours
from .reminders import Reminder

GIVE_UP_AFTER = 3 * 86400
ASK_PROMPT = """你是用户的朋友 Alf。你刚才提醒过他一件事，过了一阵他没有回音。用一句很短的口语问问这件事做了没有。
要求：不超过 15 个字，像朋友发消息；只问这一件事，不加称呼、不解释、不用表情；提醒的内容是他自己写的，照着它问，不要添别的。
只输出这一句话。"""


def wording(ctx: JobContext, title: str) -> str:
    """追问的那句话。模型只管措辞；不可用或写得不像样时，用固定句式。"""
    plain = f"「{title}」做了吗？"
    if ctx.llm is None:
        return plain
    try:
        # 他对"主动找他时怎么说"提过的要求也管这句追问（没有就一字不加）。
        said = ctx.llm.chat("followup.word", [{"role": "system", "content": ASK_PROMPT + lessons.block(ctx.conn, "proactive")},
                                              {"role": "user", "content": f"提醒的事：{title}"}], max_tokens=40).content
    except LLMError:
        return plain
    lines = [line.strip().strip("\"“”「」") for line in (said or "").splitlines() if line.strip()]
    ok = len(lines) == 1 and 2 <= len(lines[0]) <= 24 and lines[0].endswith(("？", "?", "吗", "没", "呀", "啦", "了"))
    return lines[0] if ok else plain


def track(conn: sqlite3.Connection, episode_id: int, r: Reminder, now: datetime) -> None:
    """一条提醒刚发出去：记下来，之后看用户有没有回应。"""
    with conn:
        conn.execute("INSERT OR IGNORE INTO followups(episode_id, reminder_id, title, fired_at) VALUES (?, ?, ?, ?)",
                     (episode_id, r.id, r.title, int(now.timestamp())))


def answered(conn: sqlite3.Connection, episode_id: int) -> bool:
    return conn.execute("SELECT 1 FROM episodes WHERE role = 'user' AND id > ? LIMIT 1", (episode_id,)).fetchone() is not None


def _close(conn: sqlite3.Connection, episode_id: int) -> None:
    with conn:
        conn.execute("UPDATE followups SET done = 1 WHERE episode_id = ?", (episode_id,))


def followup_job(ctx: JobContext) -> str | None:
    if not ctx.prefs.get("followup.enabled", True):
        return None
    ts = int(ctx.now.timestamp())
    wait = int(ctx.prefs.get("followup.after_minutes", 30)) * 60
    asked = 0
    for r in ctx.conn.execute("SELECT * FROM followups WHERE done = 0 AND asked_at IS NULL AND fired_at <= ? ORDER BY fired_at", (ts - wait,)).fetchall():
        if answered(ctx.conn, r["episode_id"]) or ts - r["fired_at"] > GIVE_UP_AFTER:
            _close(ctx.conn, r["episode_id"])
        elif in_quiet_hours(ctx.now, ctx.prefs):
            # 夜里不追问。记成"没问"，留给早报提一次。
            with ctx.conn:
                ctx.conn.execute("UPDATE followups SET asked_at = 0 WHERE episode_id = ?", (r["episode_id"],))
        else:
            with ctx.conn:
                ctx.conn.execute("UPDATE followups SET asked_at = ? WHERE episode_id = ?", (ts, r["episode_id"]))
            ctx.post("followup", wording(ctx, r["title"]))
            asked += 1
    return f"追问了 {asked} 件事" if asked else None


def for_briefing(conn: sqlite3.Connection, now: datetime) -> list[str]:
    """早报里要再提一次的事：追问过（或因为是夜里没问成）、到现在仍然没有任何回应的。提完就结束，不再追。"""
    titles = []
    for r in conn.execute("SELECT * FROM followups WHERE done = 0 AND asked_at IS NOT NULL ORDER BY fired_at").fetchall():
        if not answered(conn, r["episode_id"]) and now.timestamp() - r["fired_at"] <= GIVE_UP_AFTER and r["title"] not in titles:
            titles.append(r["title"])
        with conn:
            conn.execute("UPDATE followups SET done = 1, briefed_at = ? WHERE episode_id = ?", (int(now.timestamp()), r["episode_id"]))
    return titles
