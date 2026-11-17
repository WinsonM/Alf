"""约定：他和 Alf 说好的事。

谁答应的（who）决定之后怎么用：
- user  他答应要做的事（"这周日前把简历改完""每天背 20 个单词"）。Alf 惦记着，合适的时候问一句。
- alf   Alf 答应他的一件具体的事（"你考过了我给你写首打油诗"）。到时候得兑现。
- both  两边说好的习惯和暗号（"他说『老规矩』就是明早七点叫他""每周日晚上一起复盘"）。

和别的几样的分界，免得一句话存到两处：
- 到点叫他                     → 提醒（reminders）
- 他单方面要求 Alf 怎么说话做事 → 要求（lessons）
- 他挂在心上、没人答应过什么的事 → 惦记的事（memory 里的 open loops）

只在对话里当场记，而且得是真的说好了（Jev 判断，门槛见 chat.py 的 PACT_FLOOR）；记下时附回执——之后 Alf 会拿这句话
来问他，记成了什么他得看得到。问他的分寸和"惦记的事"一样：今天聊过就不再提，提了两次他都没接话就不再提。
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime

WHO = {"user": "他答应的", "alf": "你答应他的", "both": "你们说好的"}          # 给模型看的说法
WHO_LABELS = {"user": "你答应的", "alf": "Alf 答应你的", "both": "你们说好的"}   # 给用户看的说法
STATUS_LABELS = {"active": "有效", "kept": "兑现了", "dropped": "作废了", "lapsed": "不了了之"}
TEXT_CHARS = 120
SAID_CHARS = 120
SHOWN = 12               # 一次最多给模型看这么多条有效的约定（新的在前）
QUIET = 20 * 3600        # 这么久之内聊到过：今天不用再提
MAX_ASKS = 2             # Alf 提了这么多次他都没接话：不再提
LAPSE_DAYS = 7           # 一次性的约定过了期限这么多天还没了结：算不了了之，不再带着
STALE_DAYS = 30          # 没有期限的一次性约定，这么久没人提：同上
_SQUASH = re.compile(r"[\W_]+")


def _cut(text, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def get(conn: sqlite3.Connection, pact_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM pacts WHERE id = ?", (pact_id,)).fetchone()


def active(conn: sqlite3.Connection, limit: int = SHOWN) -> list[sqlite3.Row]:
    """还有效的约定，旧的在前。"""
    return conn.execute("SELECT * FROM pacts WHERE status = 'active' ORDER BY id DESC LIMIT ?", (limit,)).fetchall()[::-1]


def add(conn: sqlite3.Connection, text: str, who: str, *, now: int, standing: bool = False, due_at: int | None = None,
        said: str | None = None, episode_id: int | None = None) -> tuple[sqlite3.Row, bool]:
    """记下一条约定。返回（这一条，是不是新的）：字面上一样的已经有了就不再记一遍。"""
    text = _cut(text, TEXT_CHARS)
    squashed = _SQUASH.sub("", text).lower()
    for r in active(conn, limit=200):
        if _SQUASH.sub("", r["text"]).lower() == squashed:
            return r, False
    with conn:
        cur = conn.execute("INSERT INTO pacts(text, who, standing, due_at, said, episode_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (text, who, bool(standing), due_at, _cut(said, SAID_CHARS) or None, episode_id, now))
    return get(conn, cur.lastrowid), True


def settle(conn: sqlite3.Connection, pact_id: int, status: str, now: int, outcome: str | None = None) -> sqlite3.Row | None:
    """了结一条约定：kept 兑现了 | dropped 作废。返回了结之前的那一行；没有这条、或者已经了结了，返回 None。"""
    r = conn.execute("SELECT * FROM pacts WHERE id = ? AND status = 'active'", (pact_id,)).fetchone()
    if not r or status not in ("kept", "dropped"):
        return None
    with conn:
        conn.execute("UPDATE pacts SET status = ?, outcome = ?, settled_at = ? WHERE id = ?",
                     (status, _cut(outcome, SAID_CHARS) or None, now, pact_id))
    return r


def touch(conn: sqlite3.Connection, pact_id: int, now: int, *, by_user: bool) -> None:
    """聊到这条约定了。他自己说起的，没接话的次数清零；只是 Alf 提了，记一次。"""
    with conn:
        conn.execute("UPDATE pacts SET touched_at = ?, asks = CASE WHEN ? THEN 0 ELSE asks + 1 END WHERE id = ?",
                     (now, by_user, pact_id))


def lapse(conn: sqlite3.Connection, now: int) -> int:
    """一次性的约定拖得太久就算不了了之，不再带进对话。一直有效的那种（习惯、暗号）不会。"""
    with conn:
        cur = conn.execute(
            "UPDATE pacts SET status = 'lapsed', settled_at = ? WHERE status = 'active' AND standing = 0 AND ("
            " (due_at IS NOT NULL AND due_at < ?) OR (due_at IS NULL AND COALESCE(touched_at, created_at) < ?))",
            (now, now - LAPSE_DAYS * 86400, now - STALE_DAYS * 86400))
    return cur.rowcount


def quiet(r: sqlite3.Row, now: int) -> bool:
    """这条今天聊过了，或者提了两次他都没接话：现在不该再提。"""
    return r["asks"] >= MAX_ASKS or (r["touched_at"] is not None and now - r["touched_at"] < QUIET)


def worth_raising(r: sqlite3.Row, now: int) -> bool:
    """值不值得 Alf 主动问一句。只有他要做的事才问；Alf 自己答应的是去兑现，暗号和习惯没什么可问的。"""
    return r["who"] != "alf" and not (r["who"] == "both" and r["standing"]) and not quiet(r, now)


def due_note(r: sqlite3.Row, now: datetime) -> str:
    """期限写成"还有几天"：天数由程序算，不让模型换算。没有期限就是空串。"""
    if r["due_at"] is None:
        return ""
    due = datetime.fromtimestamp(r["due_at"], now.tzinfo)
    left = (due.date() - now.date()).days
    when = f"{due.month}月{due.day}日"
    return f"（期限 {when}，" + ("就是今天）" if left == 0 else f"还有 {left} 天）" if left > 0 else f"已经过了 {-left} 天）")


def line(r: sqlite3.Row, now: datetime) -> str:
    """给模型看的一行。编号写成"约3"，和记忆的 #3、提议的 P3 分开。"""
    return f"约{r['id']} {WHO[r['who']]}：{r['text']}" + ("（一直有效）" if r["standing"] else "") + due_note(r, now)


def context_lines(conn: sqlite3.Connection, now: datetime) -> tuple[list[str], list[sqlite3.Row]]:
    """聊天时带的：有效的约定全列出来（他问"我们约定过什么"要答得上来），今天不该再提的注明。"""
    ts = int(now.timestamp())
    rows = active(conn)
    return [line(r, now) + ("（今天聊过了，或者他没接话：不用再提）" if r["who"] != "alf" and quiet(r, ts) and not (r["who"] == "both" and r["standing"]) else "")
            for r in rows], rows


def situation_lines(conn: sqlite3.Connection, now: datetime) -> list[str]:
    """主动找他时带的：只带现在提正合适的——他答应的、还没聊过的；和 Alf 自己答应的。"""
    ts = int(now.timestamp())
    return [line(r, now) for r in active(conn) if worth_raising(r, ts) or r["who"] == "alf"]


def view(conn: sqlite3.Connection, now: int) -> dict:
    """记忆页上「约定」那一栏：有效的，和最近了结的几条。"""
    def row(r: sqlite3.Row) -> dict:
        return {"id": r["id"], "text": r["text"], "who": r["who"], "who_label": WHO_LABELS[r["who"]], "standing": bool(r["standing"]),
                "due_at": r["due_at"], "said": r["said"], "status": r["status"], "status_label": STATUS_LABELS[r["status"]],
                "outcome": r["outcome"], "at": r["created_at"], "settled_at": r["settled_at"]}

    lapse(conn, now)
    done = conn.execute("SELECT * FROM pacts WHERE status != 'active' ORDER BY settled_at DESC, id DESC LIMIT 10").fetchall()
    return {"active": [row(r) for r in active(conn, limit=100)][::-1], "settled": [row(r) for r in done]}
