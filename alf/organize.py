"""整理旧记忆：给分层之前就存下的记忆补上归类——属于哪个主题、是哪一项的值、哪天之后就过去了。

只改"归在哪"，不改记的是什么：内容一个字都不动，也不删任何一条。归类是模型提议的，逐条由程序核对后才写；
拿不准的就不动，留在常驻层——多带一条进对话，比归错了找不到要好。
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime

from .config import Settings
from .decide import Decider
from .llm import LLM
from .memory import FEEDBACK, MOMENT, TOPIC_CHARS, Memory

log = logging.getLogger("alf.organize")

PROMPT = """下面是一个私人助理记下的、关于用户的事，每条有编号。请给它们归类，输出 JSON。

对每一条决定三件事，拿不准就留空：
- topic：这条是不是某个"会越攒越多的同类细节"里的一条。是的话给主题起个短名字（2 到 6 个字，比如"健身""COMP5999""回国"），同一类用同一个名字，
  优先用"已有的主题"里的名字。关于他这个人的基本情况（身份、家人朋友、长期喜好、住哪、生日）不是细节，topic 留空。只有一条、以后也不太会再添的事，留空。
- slot：填了 topic 时，这条是哪一项的值，比如动作名"上斜哑铃卧推"。照抄原文里的叫法。
- until：这件事有一个具体日期、过了那天就不再是现状（考试、行程、约了人），填那个日期，写成 2026-11-20；生日、纪念日这种每年都有的，或者没有日期的，留空。

输出格式：{"facts": [{"id": 编号, "topic": "", "slot": "", "until": ""}]}。每条都要出现在输出里。"""


def organize(conn: sqlite3.Connection, llm: LLM, decider: Decider, cfg: Settings, now: datetime) -> list[str]:
    """返回做了哪些改动，一行一条。没有要整理的就返回空列表。"""
    memory = Memory(conn, decider, cfg.decision_threshold)
    todo = [f for f in memory.current(limit=500)
            if f.topic is None and f.expires_at is None and f.category not in (MOMENT, FEEDBACK)]
    if not todo:
        return []
    listing = "\n".join(f"{f.id}. {f.text}" for f in sorted(todo, key=lambda f: f.id))
    data = llm.json("memory.organize", [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"今天：{now:%Y-%m-%d}\n已有的主题：{'、'.join(t for t, _, _ in memory.topics()) or '（还没有）'}\n\n{listing}"},
    ], max_tokens=1500, temperature=0)
    by_id = {f.id: f for f in todo}
    changes = []
    proposals = data.get("facts") if isinstance(data.get("facts"), list) else []
    # 一个主题至少要有两条才成立：只有一条的"主题"只会让它从常驻层消失，没有任何好处。
    names = [" ".join(str(p.get("topic") or "").split())[:TOPIC_CHARS] for p in proposals if isinstance(p, dict)]
    existing = {t for t, _, _ in memory.topics()}
    for p in proposals:
        fact = by_id.get(p.get("id")) if isinstance(p, dict) else None
        if fact is None:
            continue
        topic = " ".join(str(p.get("topic") or "").split())[:TOPIC_CHARS] or None
        if topic and topic not in existing and names.count(topic) < 2:
            topic = None
        slot = " ".join(str(p.get("slot") or "").split())[:40] or None if topic else None
        expires = None if fact.category == "important_date" else _end_of_day(p.get("until"), now)
        if topic is None and expires is None:
            continue
        memory.retag(fact.id, topic=topic, slot=slot, expires_at=expires)
        parts = ([f"归到「{topic}」" + (f"（{slot}）" if slot else "")] if topic else []) \
            + ([f"到 {datetime.fromtimestamp(expires, cfg.tz):%-m月%-d日} 为止"] if expires else [])
        changes.append(f"#{fact.id} {fact.text}：" + "，".join(parts))
        log.info("整理记忆 %s", changes[-1])
    return changes


def _end_of_day(said, now: datetime) -> int | None:
    """只认 2026-11-20 这种写法，而且得是今天或以后——已经过去的日子不动它。"""
    try:
        day = date.fromisoformat(str(said or "").strip())
    except ValueError:
        return None
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=now.tzinfo)
    return int(end.timestamp()) if end > now else None
