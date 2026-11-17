"""印象：Alf 对他这个人的看法。

记忆存的是他说过的事实；印象是从他的话里看出来的（"做事爱拖到最后一刻，但总能赶上""心情不好的时候不喜欢被讲道理"）。
是推断，所以规矩比记忆严：

- 只在夜里从当天的话里形成，不在聊天时当场下结论；每条必须指明出自他的哪几句话，程序核对编号。
- 存下之前由 Jev 再看两眼：这几句话撑不撑得起这个印象；是不是在给他贴标签（诊断、外貌、智力、家境……）。过不了就不存。
- 只凭一天的话不算数：先是"苗头"，不进对话；另一天又印证了才算形成。和"两次同类的纠正才提议规矩"是一个道理。
- 形成的那天晚上在汇总里告诉他一声，他说不对就收回；收回的不会再形成。记忆页上每条都看得到依据，可以改、可以删。
- 很久没再印证的会淡出对话，不删。

用在哪：聊天和主动找他时放进上下文，让 Alf 把握分寸、知道他吃哪一套。提示词要求它不当面下结论，他问起才说。
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime

from .decide import DecisionError, impression_label_question, impression_supported_question
from .heartbeat import JobContext
from .llm import LLMError

log = logging.getLogger("alf.impressions")

FORM_DAYS = 2            # 有这么多个不同的日子印证过，才算形成
FADE_DAYS = 45           # 这么久没再印证：不再带进对话（页面上还在）
HUNCH_DAYS = 14          # 苗头放这么久没等到第二次印证：丢掉
IN_CONTEXT = 8           # 一次最多给模型看这么多条
MIN_MESSAGES = 5         # 当天他说的话少于这么多句，就不去想印象
MAX_MESSAGES = 80
PER_NIGHT = 3
TEXT_CHARS = 60
QUOTE_CHARS = 80
EVIDENCE_KEPT = 8        # 每条印象最多留这么多句依据（最新的）
TOLD_PER_NIGHT = 2
# 两个门槛都放在 Jev 给的两堆分数中间（题目在 evals/cases/guards.jsonl 的 impression / label 两类）。
# 依据：撑得起的 0.57–0.98（最低的是三句话里看出"对朋友很上心"）；撑不起的 0.35 以下（一句气话 → "脾气暴躁"；
# 只是让它建提醒 → "做事很有计划"）。宁可严：没留下的印象，他再说起类似的话还会再提出来。
SUPPORT_FLOOR = 0.45
# 标签：诊断、外貌、智力、家境、性取向这类 0.30–0.97（最低的是口语里的"有拖延症"）；描述他怎么做事的 0.11 以下。
LABEL_FLOOR = 0.2
_SQUASH = re.compile(r"[\W_]+")

PROMPT = """下面是用户今天和私人助理 Alf 的对话，他说的每句话前面有编号。你要替 Alf 想一想：从他今天的话里，看得出他是个什么样的人吗？输出 JSON。

什么算印象：他做事的方式、在意什么、怎么对待自己和别人、喜欢被怎么对待——朋友相处久了心里会有的那种看法。比如「做事爱拖到最后一刻，但总能赶上」「嘴上说无所谓，其实很在意成绩」「心情不好的时候不喜欢被讲道理」。

不算的：
- 他直接说出来的事实（住哪、几号考试、喜欢吃什么）——那是记忆，不是印象
- 只是今天的心情或状态（「今天好累」）
- 诊断和标签（抑郁、焦虑症、多动症），对外貌、智力、家境、性取向、信仰的评价
- 「他否认过的」里已经有的看法

怎么写：
- 每条一句话，不超过 30 个字，是对他这个人的描述，省略主语。说得具体，宁可窄一点，不要拔高成性格测试的结论。
- from 填这条印象出自哪几句话的编号，至少一个。只能根据他自己说的话，Alf 说的话只用来看上下文。
- 「已有的印象」里已经有同一个看法、今天的话又印证了它：same_as 填那条的编号，text 照抄原来的。是新的看法，same_as 填 null。
- 多数日子看不出什么新东西，那就输出空列表。最多 3 条。

输出格式：{"impressions": [{"text": "……", "from": [编号], "same_as": null}]}"""


def _same(a: str, b: str) -> bool:
    return _SQUASH.sub("", a).lower() == _SQUASH.sub("", b).lower()


def get(conn: sqlite3.Connection, impression_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM impressions WHERE id = ?", (impression_id,)).fetchone()


def current(conn: sqlite3.Connection, now: int, limit: int = IN_CONTEXT) -> list[sqlite3.Row]:
    """形成了、而且最近还印证过的印象：印证的天数多的在前。"""
    return conn.execute(
        "SELECT * FROM impressions WHERE status = 'current' AND formed_at IS NOT NULL AND seen_at >= ?"
        " ORDER BY days DESC, seen_at DESC, id LIMIT ?", (now - FADE_DAYS * 86400, limit)).fetchall()


def held(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """没被收回的全部（形成的和苗头），旧的在前。夜里形成新印象时给模型对照用。"""
    return conn.execute("SELECT * FROM impressions WHERE status = 'current' ORDER BY id").fetchall()


def denied(conn: sqlite3.Connection, limit: int = 20) -> list[str]:
    """他否认过的看法：不再形成同样的。"""
    return [r["text"] for r in conn.execute("SELECT text FROM impressions WHERE status = 'withdrawn' ORDER BY id DESC LIMIT ?", (limit,))]


def context_lines(conn: sqlite3.Connection, now: int) -> list[str]:
    """给模型看的。编号写成"印3"，和记忆的 #3、约定的"约3"分开。"""
    return [f"印{r['id']} {r['text']}" for r in current(conn, now)]


def add(conn: sqlite3.Connection, text: str, evidence: list[dict], now: int) -> int:
    """一个新的苗头。"""
    with conn:
        cur = conn.execute("INSERT INTO impressions(text, evidence, created_at, seen_at) VALUES (?, ?, ?, ?)",
                           (text, json.dumps(evidence[-EVIDENCE_KEPT:], ensure_ascii=False), now, now))
    return cur.lastrowid


def reinforce(conn: sqlite3.Connection, impression_id: int, evidence: list[dict], now: int, tz) -> bool:
    """今天的话又印证了这条。是新的一天就多算一天；攒够 FORM_DAYS 天就算形成。返回这一次是不是刚形成。"""
    r = get(conn, impression_id)
    if not r or r["status"] != "current":
        return False
    known = json.loads(r["evidence"])
    seen = {e.get("episode") for e in known}
    merged = (known + [e for e in evidence if e.get("episode") not in seen])[-EVIDENCE_KEPT:]
    new_day = datetime.fromtimestamp(r["seen_at"], tz).date() != datetime.fromtimestamp(now, tz).date()
    days = r["days"] + (1 if new_day else 0)
    formed_now = r["formed_at"] is None and days >= FORM_DAYS
    with conn:
        conn.execute("UPDATE impressions SET evidence = ?, days = ?, seen_at = ?, formed_at = COALESCE(formed_at, ?) WHERE id = ?",
                     (json.dumps(merged, ensure_ascii=False), days, now, now if formed_now else None, impression_id))
    return formed_now


def withdraw(conn: sqlite3.Connection, impression_id: int, now: int, note: str | None = None) -> sqlite3.Row | None:
    """他说这个印象不对：收回。留着这一行，为的是以后不再形成同样的看法。"""
    r = get(conn, impression_id)
    if not r or r["status"] != "current":
        return None
    with conn:
        conn.execute("UPDATE impressions SET status = 'withdrawn', note = ?, seen_at = ? WHERE id = ?",
                     (" ".join(str(note or "").split())[:QUOTE_CHARS] or None, now, impression_id))
    return r


def rewrite(conn: sqlite3.Connection, impression_id: int, text: str) -> bool:
    """他在页面上把这条印象改成了自己的说法。依据留着；既然是他亲手写的，直接算形成。"""
    text = " ".join(str(text).split())[:TEXT_CHARS]
    with conn:
        cur = conn.execute("UPDATE impressions SET text = ?, formed_at = COALESCE(formed_at, seen_at), told_at = COALESCE(told_at, seen_at)"
                           " WHERE id = ? AND status = 'current'", (text, impression_id))
    return bool(text) and cur.rowcount > 0


def prune(conn: sqlite3.Connection, now: int) -> None:
    """等不到第二次印证的苗头不留着：那只是某一天的一句话。"""
    with conn:
        conn.execute("DELETE FROM impressions WHERE status = 'current' AND formed_at IS NULL AND seen_at < ?", (now - HUNCH_DAYS * 86400,))


def untold(conn: sqlite3.Connection, limit: int = TOLD_PER_NIGHT) -> list[sqlite3.Row]:
    """形成了、还没告诉过他的。"""
    return conn.execute("SELECT * FROM impressions WHERE status = 'current' AND formed_at IS NOT NULL AND told_at IS NULL"
                        " ORDER BY formed_at, id LIMIT ?", (limit,)).fetchall()


def mark_told(conn: sqlite3.Connection, ids: list[int], now: int) -> None:
    with conn:
        conn.executemany("UPDATE impressions SET told_at = ? WHERE id = ?", [(now, i) for i in ids])


def view(conn: sqlite3.Connection, now: int) -> dict:
    """记忆页上「它眼里的你」那一栏。"""
    def row(r: sqlite3.Row) -> dict:
        return {"id": r["id"], "text": r["text"], "days": r["days"], "at": r["created_at"], "seen_at": r["seen_at"],
                "faded": r["seen_at"] < now - FADE_DAYS * 86400, "note": r["note"],
                "evidence": [{"quote": e.get("quote", ""), "at": e.get("at")} for e in json.loads(r["evidence"])][-4:]}

    rows = held(conn)
    gone = conn.execute("SELECT * FROM impressions WHERE status = 'withdrawn' ORDER BY seen_at DESC, id DESC LIMIT 10").fetchall()
    return {"formed": sorted((row(r) for r in rows if r["formed_at"] is not None), key=lambda x: (-x["days"], -x["seen_at"])),
            "hunches": [row(r) for r in rows if r["formed_at"] is None][::-1],
            "withdrawn": [row(r) for r in gone], "form_days": FORM_DAYS}


# --- 夜里：从当天的话里形成印象 ---------------------------------------------------------

def checked(ctx: JobContext, text: str, quotes: list[str]) -> bool:
    """这条印象能不能留：他的话撑得起它，而且它不是在贴标签。判断不了就不留——没人核对过的推断不存。"""
    try:
        a = ctx.decider.ask("impressions.check", {"user_messages": quotes},
                            {"supported": impression_supported_question(text), "label": impression_label_question(text)})
    except DecisionError as e:
        log.info("印象没核对成，不留：%s（%s）", text, e)
        return False
    ok = a["supported"].noul >= SUPPORT_FLOOR and a["label"].noul < LABEL_FLOOR
    if not ok:
        log.info("印象没过核对（依据 %.2f，标签 %.2f）：%s", a["supported"].noul, a["label"].noul, text)
    return ok


def impressions_job(ctx: JobContext) -> str | None:
    """每晚一次，排在归纳经历之后、汇总之前：这样当晚形成的印象赶得上在汇总里告诉他。"""
    if ctx.llm is None or not ctx.prefs.get("memory.auto", True) or not ctx.prefs.get("impressions.enabled", True):
        return None
    state = json.loads(ctx.cursor or "{}")
    midnight = ctx.now.replace(hour=0, minute=0, second=0, microsecond=0)
    since = state.get("until") or int(midnight.timestamp())
    ts = int(ctx.now.timestamp())
    ctx.cursor = json.dumps({"until": ts})
    prune(ctx.conn, ts)
    rows = ctx.conn.execute(
        "SELECT id, role, text, created_at FROM episodes WHERE created_at > ? AND created_at <= ? AND (role = 'user' OR kind = 'chat')"
        " ORDER BY id", (since, ts)).fetchall()
    his = [r for r in rows if r["role"] == "user" and len(r["text"].strip()) >= 4][-MAX_MESSAGES:]
    if len(his) < MIN_MESSAGES:
        return None
    number = {r["id"]: i for i, r in enumerate(his, 1)}
    listing = []
    for r in rows:
        if r["id"] in number:
            listing.append(f"{number[r['id']]}. [{datetime.fromtimestamp(r['created_at'], ctx.now.tzinfo):%H:%M}] 他：{r['text'][:300]}")
        elif r["role"] == "alf" and listing:
            listing.append(f"   Alf：{r['text'][:120]}")
    known = held(ctx.conn)
    try:
        data = ctx.llm.json("impressions.form", [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "已有的印象：\n" + ("\n".join(f"{r['id']}. {r['text']}" for r in known) or "（还没有）")
                                        + "\n\n他否认过的：\n" + ("\n".join(denied(ctx.conn)) or "（没有）")
                                        + "\n\n今天的对话：\n" + "\n".join(listing[-2 * MAX_MESSAGES:])},
        ], max_tokens=500, temperature=0)
    except LLMError as e:
        return f"没想成：{e}"
    by_id = {r["id"]: r for r in known}
    new = seen_again = formed = 0
    proposals = data.get("impressions") if isinstance(data.get("impressions"), list) else []
    for item in proposals[:PER_NIGHT]:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text", "")).split())
        cited = sorted({n for n in (item.get("from") if isinstance(item.get("from"), list) else [])
                        if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(his)})
        if not text or len(text) > TEXT_CHARS or not cited:
            continue                                   # 说不出出自哪句话的，不留
        if any(_same(text, d) for d in denied(ctx.conn)):
            continue
        same = item.get("same_as")
        old = by_id.get(same) if isinstance(same, int) and not isinstance(same, bool) else None
        old = old or next((r for r in known if _same(r["text"], text)), None)
        evidence = [{"episode": his[n - 1]["id"], "quote": " ".join(his[n - 1]["text"].split())[:QUOTE_CHARS], "at": his[n - 1]["created_at"]}
                    for n in cited]
        # 印证旧印象时核对的是库里那句话，不是模型这次写的：它可能悄悄改了说法。
        if not checked(ctx, old["text"] if old else text, [e["quote"] for e in evidence]):
            continue
        if old:
            formed += reinforce(ctx.conn, old["id"], evidence, ts, ctx.now.tzinfo)
            seen_again += 1
        else:
            add(ctx.conn, text, evidence, ts)
            new += 1
    done = ([f"多了 {new} 个苗头"] if new else []) + ([f"印证了 {seen_again} 条"] if seen_again else []) + ([f"其中 {formed} 条算形成了"] if formed else [])
    return "，".join(done) or None


def digest_lines(conn: sqlite3.Connection, now: int) -> str | None:
    """每晚汇总里的那一句：新形成的印象。说过就不再说。没有就是 None。"""
    rows = untold(conn)
    if not rows:
        return None
    mark_told(conn, [r["id"] for r in rows], now)
    return "对你多了一点印象：" + "".join(f"「{r['text']}」" for r in rows) + "。不对就说，我收回。"
