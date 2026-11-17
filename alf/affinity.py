"""好感度：Alf 对用户有多亲近。它是 Alf 记忆的一部分，记忆页上看得到。

规则是用户定的（2026-11-17）：从 50 开始，聊到有意思的内容加 1，满分 100；之后说话的语气跟着好感度走。
只升不降——他没说要扣分，一个会因为你今天话少就疏远你的朋友也不讨喜。

"有意思"由 Jev 判断：是不是朋友之间聊得来的那种话（讲一件事、一种感受、一个看法、一个玩笑），
而不是指令、报数字、"好的"。每一分都记着是哪句话挣来的，所以分数不是凭空的。
"""
from __future__ import annotations

import logging
import re
import sqlite3

from .decide import Decider, DecisionError, interesting_question

log = logging.getLogger("alf.affinity")

START, MAX = 50, 100
# 实测（2026-11-17，jev-1.14.0，45 句）：聊得来的话 0.58–0.93（17 句里 16 句在 0.78 以上）；
# 指令、报数、应答、问功能的 0.03–0.20。门槛放在两堆中间。
INTEREST_FLOOR = 0.5
# 加过一分之后这么久之内不再加：一段聊得来的对话算一分，不是一句话一分。
# 不设的话，一个晚上就能聊到满分，后面的语气就再也不会变了。
COOLDOWN = 3600
NOTE_CHARS = 60
_TRIVIAL = re.compile(r"[\W_]+")

# 从高到低：（起始分数，名字，这一档怎么说话）。只对"像朋友"的说话风格生效。
TIERS = (
    (90, "死党", "你们是死党：有默契，话可以很短，可以用你们之间的梗，损他两句也没关系；但他认真的时候你也认真。"),
    (75, "好朋友", "你们是好朋友：很放松，说话直接，不用客套，可以互相调侃、吐槽，会惦记他最近提过的事。"),
    (60, "朋友", "你们已经是朋友：说话随意，可以开玩笑，会关心一句他最近提过的事。"),
    (0, "刚熟悉", "你们刚熟悉起来：友好、自然，稍微客气一点；玩笑点到为止，不调侃他。"),
)


def score(conn: sqlite3.Connection) -> int:
    return min(MAX, START + conn.execute("SELECT COUNT(*) FROM affinity").fetchone()[0])


def tier(points: int) -> tuple[str, str]:
    """（这一档的名字，这一档的说话方式）"""
    return next((name, tone) for floor, name, tone in TIERS if points >= floor)


def recent(conn: sqlite3.Connection, limit: int = 5) -> list[dict]:
    """最近几次加分是因为哪句话。"""
    rows = conn.execute("SELECT note, created_at FROM affinity ORDER BY id DESC LIMIT ?", (limit,))
    return [{"note": r["note"], "at": r["created_at"]} for r in rows]


def view(conn: sqlite3.Connection, now: int | None = None) -> dict:
    from . import mood

    points = score(conn)
    out = {"score": points, "max": MAX, "start": START, "tier": tier(points)[0], "recent": recent(conn)}
    if now is not None:
        out["mood"] = mood.view(conn, now)      # 好感度是长期的；这是眼前这一阵的气氛
    return out


def worth_asking(text: str) -> bool:
    """"好""嗯嗯""ok"这种不用去判断有没有意思。"""
    return len(_TRIVIAL.sub("", text)) >= 4


def consider(conn: sqlite3.Connection, decider: Decider, text: str, previous: str, episode_id: int, now: int) -> bool:
    """用户刚说了一句话：够有意思就加一分。返回加没加。判断不了就不加，下一句再说。
    （对话里把这个问题和别的几个合成一次调用，直接用 add；这个函数留给单独使用的场合。）"""
    if score(conn) >= MAX or not worth_asking(text):
        return False
    try:
        p = decider.ask("affinity.interesting", {"assistant_previous_message": previous, "user_latest_message": text},
                        {"fun": interesting_question()})["fun"].noul
    except DecisionError as e:
        log.info("这句话有没有意思没判断成：%s", e)
        return False
    return add(conn, text, p, episode_id, now)


def add(conn: sqlite3.Connection, text: str, p: float, episode_id: int, now: int) -> bool:
    """这句话"有意思"的把握是 p：过了门槛、没满分、也不在冷却期，就加一分。"""
    if p < INTEREST_FLOOR or score(conn) >= MAX:
        return False
    last = conn.execute("SELECT MAX(created_at) FROM affinity").fetchone()[0]
    if last is not None and now - last < COOLDOWN:
        return False
    note = " ".join(text.split())
    with conn:
        conn.execute("INSERT OR IGNORE INTO affinity(episode_id, note, score, created_at) VALUES (?, ?, ?, ?)",
                     (episode_id, note[:NOTE_CHARS] + ("…" if len(note) > NOTE_CHARS else ""), p, now))
    log.info("好感度 +1（现在 %d）：%s", score(conn), note[:40])
    return True
