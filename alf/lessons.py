"""学到的东西：用户的话怎样变成 Alf 以后的做法。

用户的要求（2026-11-17）："让他听了我的话后会改进"。改进放在数据库里，不放在程序和固定提示词里；一句话按性质走三条路：

1. 有现成开关的 → 改设置。聊天里的 change_setting 工具负责，每次改动记在 setting_changes（改了什么、原来是什么、他的原话）。
2. 关于怎么说话做事的 → 记成要求。就是记忆里 category=feedback 的那些，加了一个范围（scope）：哪里都管、只管聊天、
   只管说邮件、只管主动找他。每个要让模型写话的地方（聊天、主动搭话、早报、邮件提要、追问）都来这里取管得到自己的那几条。
3. 针对某一次的纠正 → 存成例子（examples）："这封不用马上告诉我""这条没用""这句太长了"。判断同类情况时拿出来看；
   同类的攒够 ENOUGH 个，晚上汇总时提议定成一条规矩（rule_proposals），他点头才算。

碰不到的那一层：说到做到的核对、发信前的确认、外部内容不能触发动作——这些在程序里，聊天改不了。
"""
from __future__ import annotations

import json
import sqlite3

from . import prefs
from .config import Settings
from .memory import FEEDBACK, Fact

SCOPES = {"all": "哪里都管", "chat": "聊天时", "mail": "说邮件时", "proactive": "主动找你时"}
RULES_MAX = 15            # 一个地方最多带这么多条要求；超了取最新的
EXAMPLES_SHOWN = 6        # 给模型和 Jev 看的例子，每类最多这么多个（最新的）
ENOUGH = 2                # 同类的例子攒到这么多个，就提议定成规矩
PROPOSAL_DAYS = 3         # 提议放这么久没回音就作废
ABOUT_CHARS, SAID_CHARS = 120, 80
AREAS = {"mail": "邮件", "proactive": "主动发的消息", "chat": "聊天"}
# 他对一封信表的态 → 之后同类的信怎么办（也就是分到哪一档：now 立刻说 | later 每日汇总 | skip 只计数）。
WISHES = {"now": "立刻告诉你", "later": "留到每天的汇总", "skip": "不用告诉你"}


def _cut(text: str | None, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


# --- 要求 -------------------------------------------------------------------------

def scope_of(value) -> str | None:
    """模型或页面给的范围 → 存进库里的写法。"哪里都管"存成空。认不出来的也当哪里都管：宁可多管，不要漏管。"""
    v = str(value or "").strip().lower()
    return v if v in SCOPES and v != "all" else None


def rules(conn: sqlite3.Connection, where: str | None = None, *, only: bool = False) -> list[Fact]:
    """现在生效的要求，旧的在前。给了 where 就只要管得到那里的：哪里都管的 + 专管那里的；only=True 只要专管那里的。"""
    rows = conn.execute("SELECT * FROM facts WHERE valid_to IS NULL AND category = ? ORDER BY recorded_at DESC, id DESC", (FEEDBACK,))
    out = [Fact.from_row(r) for r in rows]
    if where:
        out = [f for f in out if f.scope == where or (not only and f.scope is None)]
    return sorted(out[:RULES_MAX], key=lambda f: f.id)


def block(conn: sqlite3.Connection, where: str) -> str:
    """接在一段提示词后面的话：他提过的、管得到这里的要求。没有就是空串——提示词和以前逐字相同。"""
    lines = [f.text for f in rules(conn, where)]
    return "" if not lines else "\n\n他对你提过这些要求（照做）：\n" + "\n".join("- " + t for t in lines)


def set_scope(conn: sqlite3.Connection, fact_id: int, scope) -> bool:
    with conn:
        cur = conn.execute("UPDATE facts SET scope = ? WHERE id = ? AND category = ? AND valid_to IS NULL",
                           (scope_of(scope), fact_id, FEEDBACK))
    return cur.rowcount > 0


# --- 例子 -------------------------------------------------------------------------

def add_example(conn: sqlite3.Connection, area: str, verdict: str, about: str, said: str, *, now: int,
                inbox_id: int | None = None, target: int | None = None, episode_id: int | None = None) -> tuple[int, bool]:
    """记一个例子，返回（编号，是不是新记的）。同一封信、同一条消息他又说了一次，就改原来那条——他改主意了，以新的为准。"""
    same = None
    if inbox_id is not None or target is not None:
        same = conn.execute("SELECT id FROM examples WHERE area = ? AND removed_at IS NULL AND rule_id IS NULL"
                            " AND inbox_id IS ? AND target IS ?", (area, inbox_id, target)).fetchone()
    with conn:
        if same:
            conn.execute("UPDATE examples SET verdict = ?, said = ?, episode_id = ?, created_at = ? WHERE id = ?",
                         (verdict, _cut(said, SAID_CHARS), episode_id, now, same["id"]))
            return same["id"], False
        cur = conn.execute("INSERT INTO examples(area, verdict, about, said, inbox_id, target, episode_id, created_at)"
                           " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (area, verdict, _cut(about, ABOUT_CHARS), _cut(said, SAID_CHARS), inbox_id, target, episode_id, now))
    return cur.lastrowid, True


def remove_example(conn: sqlite3.Connection, example_id: int, now: int) -> bool:
    with conn:
        cur = conn.execute("UPDATE examples SET removed_at = ? WHERE id = ? AND removed_at IS NULL", (now, example_id))
    return cur.rowcount > 0


def note_rating(conn: sqlite3.Connection, target: int | None, rating: int | None, said: str, now: int,
                episode_id: int | None = None) -> None:
    """他对 Alf 发的某一条点了 👎 或说了"没用"：这也是一个例子。点了 👍、或者把 👎 撤了，原来那个例子就不算了。"""
    if target is None:
        return
    row = conn.execute("SELECT kind, text FROM episodes WHERE id = ? AND role = 'alf'", (target,)).fetchone()
    if not row:
        return
    area = "chat" if row["kind"] == "chat" else "proactive"
    if rating is not None and rating < 0:
        add_example(conn, area, "bad", row["text"], said, now=now, target=target, episode_id=episode_id)
    else:
        with conn:
            conn.execute("UPDATE examples SET removed_at = ? WHERE area = ? AND target = ? AND removed_at IS NULL", (now, area, target))


def examples(conn: sqlite3.Connection, area: str | None = None, limit: int = EXAMPLES_SHOWN) -> list[sqlite3.Row]:
    """现在还算数的例子，新的在前：没被撤销，也没被一条规矩取代。"""
    return conn.execute("SELECT * FROM examples WHERE removed_at IS NULL AND rule_id IS NULL AND (? IS NULL OR area = ?)"
                        " ORDER BY id DESC LIMIT ?", (area, area, limit)).fetchall()


def example_lines(conn: sqlite3.Connection, area: str) -> list[str]:
    """给模型看的："你发的那条 → 他怎么说的"，旧的在前。"""
    return [f"你发的「{r['about']}」——他说：{r['said']}" for r in examples(conn, area)][::-1]


def mail_wishes(conn: sqlite3.Connection) -> list[dict]:
    """他表过态的以前那几封信，新的在前：例子的编号、发件人、主题、他要这类信怎么办。分拣新邮件时逐个比"是不是同一类"。

    已经归纳成规矩的例子也在里面（只要那条规矩还生效）：规矩是一句自由的话，Jev 读它难免有偏差；
    而"和那封具体的信是不是同一类"实测很稳，所以这几封信继续当锚点用。他把那条规矩忘掉了，这些例子也就不算了。
    """
    rows = conn.execute("SELECT e.* FROM examples e LEFT JOIN facts f ON f.id = e.rule_id WHERE e.area = 'mail' AND e.removed_at IS NULL"
                        " AND (e.rule_id IS NULL OR f.valid_to IS NULL) ORDER BY e.id DESC LIMIT ?", (EXAMPLES_SHOWN,)).fetchall()
    out = []
    for r in rows:
        mail = conn.execute("SELECT sender, title FROM inbox_items WHERE id = ?", (r["inbox_id"],)).fetchone() if r["inbox_id"] else None
        sender, _, subject = r["about"].partition("｜")
        if r["verdict"] in WISHES:
            out.append({"id": r["id"], "from": mail["sender"] if mail else sender, "subject": mail["title"] if mail else subject,
                        "wish": r["verdict"]})
    return out


# --- 聊天里改的设置 -----------------------------------------------------------------

def log_setting(conn: sqlite3.Connection, key: str, old, new, quote: str | None, episode_id: int | None, now: int) -> int:
    with conn:
        cur = conn.execute("INSERT INTO setting_changes(key, old, new, quote, episode_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                           (key, json.dumps(old, ensure_ascii=False), json.dumps(new, ensure_ascii=False),
                            _cut(quote, SAID_CHARS) or None, episode_id, now))
    return cur.lastrowid


def setting_changes(conn: sqlite3.Connection, limit: int = 8) -> list[dict]:
    """聊天里改过、还没改回去的设置，新的在前。"""
    rows = conn.execute("SELECT * FROM setting_changes WHERE undone_at IS NULL ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"id": r["id"], "key": r["key"], "label": prefs.CHAT_SETTINGS.get(r["key"], r["key"]),
             "old": prefs.show(r["key"], json.loads(r["old"])), "new": prefs.show(r["key"], json.loads(r["new"])),
             "quote": r["quote"], "at": r["created_at"]} for r in rows]


def undo_setting(conn: sqlite3.Connection, cfg: Settings, change_id: int, now: int) -> str | None:
    """把聊天里的一次改动改回去。成功返回 None；做不了就返回原因（给用户看）。

    那项设置之后又被改过（现在的值已经不是当时改成的值）就不动它：改回"当时的旧值"会把后来的改动也冲掉。
    """
    r = conn.execute("SELECT * FROM setting_changes WHERE id = ? AND undone_at IS NULL", (change_id,)).fetchone()
    if not r:
        return "没有这条改动"
    if prefs.load(conn, cfg).get(r["key"]) != json.loads(r["new"]):
        with conn:
            conn.execute("UPDATE setting_changes SET undone_at = ? WHERE id = ?", (now, change_id))
        return "这项设置后来又改过了，没有动它"
    prefs.save(conn, cfg, {r["key"]: json.loads(r["old"])})
    with conn:
        conn.execute("UPDATE setting_changes SET undone_at = ? WHERE id = ?", (now, change_id))
    return None


# --- 从例子到规矩的提议 --------------------------------------------------------------

def unproposed(conn: sqlite3.Connection, area: str) -> list[sqlite3.Row]:
    """这一类里还没拿去提议过的例子，旧的在前。"""
    return conn.execute("SELECT * FROM examples WHERE area = ? AND removed_at IS NULL AND rule_id IS NULL AND proposed_at IS NULL"
                        " ORDER BY id", (area,)).fetchall()


def add_proposal(conn: sqlite3.Connection, text: str, scope: str, example_ids: list[int], now: int) -> int:
    """记下一条提议，并把它引用的例子标成"提议过"——他要是不点头，这几个例子不会再拿来问第二遍。"""
    with conn:
        cur = conn.execute("INSERT INTO rule_proposals(text, scope, examples, created_at) VALUES (?, ?, ?, ?)",
                           (text, scope, json.dumps(example_ids), now))
        conn.executemany("UPDATE examples SET proposed_at = ? WHERE id = ?", [(now, i) for i in example_ids])
    return cur.lastrowid


def open_proposals(conn: sqlite3.Connection, now: int) -> list[sqlite3.Row]:
    """还在等他点头的提议。放了几天没回音的顺手作废。"""
    with conn:
        conn.execute("UPDATE rule_proposals SET status = 'expired' WHERE status = 'open' AND created_at < ?",
                     (now - PROPOSAL_DAYS * 86400,))
    return conn.execute("SELECT * FROM rule_proposals WHERE status = 'open' ORDER BY id").fetchall()


def settle_proposal(conn: sqlite3.Connection, proposal_id: int, *, rule_id: int | None = None) -> sqlite3.Row | None:
    """他对一条提议表了态。rule_id 给了 = 他同意，规矩已经记成了那条记忆，出处的例子从此由规矩取代；没给 = 他不要。"""
    r = conn.execute("SELECT * FROM rule_proposals WHERE id = ? AND status = 'open'", (proposal_id,)).fetchone()
    if not r:
        return None
    with conn:
        conn.execute("UPDATE rule_proposals SET status = ?, rule_id = ? WHERE id = ?",
                     ("accepted" if rule_id else "declined", rule_id, proposal_id))
        if rule_id:
            conn.executemany("UPDATE examples SET rule_id = ? WHERE id = ?", [(rule_id, i) for i in json.loads(r["examples"])])
    return r


# --- 给记忆页看的 -------------------------------------------------------------------

def view(conn: sqlite3.Connection, now: int) -> dict:
    """记忆页上「学到的」那一栏：要求（带范围和原话）、例子、聊天里改的设置、等他点头的提议。"""
    return {
        "rules": [{"id": f.id, "text": f.text, "scope": f.scope or "all", "scope_label": SCOPES[f.scope or "all"],
                   "quote": f.quote, "recorded_at": f.recorded_at, "source": f.source} for f in rules(conn)],
        "examples": [{"id": r["id"], "area": r["area"], "area_label": AREAS.get(r["area"], r["area"]), "about": r["about"],
                      "said": r["said"], "wish": WISHES.get(r["verdict"]) if r["area"] == "mail" else None, "at": r["created_at"]}
                     for r in examples(conn, limit=30)],
        "settings": setting_changes(conn),
        "proposals": [{"id": r["id"], "text": r["text"], "scope": r["scope"], "scope_label": SCOPES.get(r["scope"], SCOPES["all"]),
                       "count": len(json.loads(r["examples"])), "at": r["created_at"]} for r in open_proposals(conn, now)],
        "scopes": SCOPES,
    }
