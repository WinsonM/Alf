"""短期情绪：好感度之外、会自己散掉的那一层。

好感度是长期的，只升不降（用户定的规则）。但朋友之间还有眼前这一阵的气氛：刚聊得开心，接下来说话更放得开；
刚被说了一句"没用"，接下来两三天会收着点。这里记两个数，都随时间自己衰减，只影响语气，不动好感度的分数：

- warm（余温）：聊到有意思的内容时加一点，12 小时衰减一半——管的是"那个晚上"。
- tension（疙瘩）：他表示不满意时加一点，72 小时衰减一半——管的是"这两三天"。

两个半衰期取自 Anima Brain 的情绪模型（心情 12 小时、紧张 72 小时）；它那套五个维度的关系模型没有搬，
Alf 只要"此刻该多随意"这一个结论。信号都是 Alf 本来就在收集的：聊得有没有意思（好感度用的那个判断）、
👍👎、"没用"，外加一个判断——这句话是不是在对 Alf 表示不满。
"""
from __future__ import annotations

import sqlite3

WARM_HALF_LIFE = 12 * 3600
TENSION_HALF_LIFE = 72 * 3600
CAP = 2.0          # 再怎么累加也到此为止：从 2 衰减到 0.5，余温要一天，疙瘩要六天
FELT = 0.5         # 高于这个数才影响语气：加一次（1.0）之后，余温管 12 小时，疙瘩管 3 天
REASON_CHARS = 40
# 实测（2026-11-17，jev-1.14.0，见 evals/cases/guards.jsonl 的 annoyed 一类）：门槛见 chat.py 调用处的说明。
CAREFUL = "他最近对你有点不满意：这两天说话收着点——先把事办准，少开玩笑，少发表情，不要辩解。"
RELAXED = "你们刚聊得挺开心：这会儿可以比平时更放得开一点。"


def _row(conn: sqlite3.Connection):
    return conn.execute("SELECT warm, tension, reason, updated_at FROM mood WHERE id = 1").fetchone()


def current(conn: sqlite3.Connection, now: int) -> tuple[float, float, str | None]:
    """此刻的（余温，疙瘩，最近一次让他不满意的那句话）——已经按过去的时间衰减过。"""
    r = _row(conn)
    if not r:
        return 0.0, 0.0, None
    age = max(0, now - r["updated_at"])
    return r["warm"] * 0.5 ** (age / WARM_HALF_LIFE), r["tension"] * 0.5 ** (age / TENSION_HALF_LIFE), r["reason"]


def _save(conn: sqlite3.Connection, warm: float, tension: float, reason: str | None, now: int) -> None:
    with conn:
        conn.execute("INSERT INTO mood(id, warm, tension, reason, updated_at) VALUES (1, ?, ?, ?, ?)"
                     " ON CONFLICT(id) DO UPDATE SET warm = excluded.warm, tension = excluded.tension,"
                     " reason = excluded.reason, updated_at = excluded.updated_at",
                     (min(CAP, max(0.0, warm)), min(CAP, max(0.0, tension)), reason, now))


def cheer(conn: sqlite3.Connection, now: int) -> None:
    """刚聊到有意思的内容。"""
    warm, tension, reason = current(conn, now)
    _save(conn, warm + 1, tension, reason, now)


def strain(conn: sqlite3.Connection, now: int, why: str) -> None:
    """他表示了不满意（👎、"没用"、或者一句埋怨）。"""
    warm, tension, _ = current(conn, now)
    why = " ".join(why.split())
    _save(conn, warm, tension + 1, why[:REASON_CHARS] + ("…" if len(why) > REASON_CHARS else ""), now)


def ease(conn: sqlite3.Connection, now: int) -> None:
    """他点了 👍 或说了"有用"：疙瘩消掉一半。"""
    warm, tension, reason = current(conn, now)
    _save(conn, warm, tension / 2, reason, now)


def tone(conn: sqlite3.Connection, now: int) -> str | None:
    """此刻要加进"说话方式"里的那一句；平常就没有。有疙瘩的时候，疙瘩说了算。"""
    warm, tension, _ = current(conn, now)
    return CAREFUL if tension >= FELT else RELAXED if warm >= FELT else None


def view(conn: sqlite3.Connection, now: int) -> dict:
    warm, tension, reason = current(conn, now)
    state = "careful" if tension >= FELT else "relaxed" if warm >= FELT else "even"
    label = {"careful": "这两天收着点", "relaxed": "刚聊得挺开心", "even": "平常"}[state]
    return {"state": state, "label": label, "warm": round(warm, 2), "tension": round(tension, 2),
            "reason": reason if state == "careful" else None}
