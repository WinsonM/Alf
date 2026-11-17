"""人设：Alf 是谁。

分两部分：
- 卡片：名字、身份、性格、说话习惯、喜好、和他的关系。存在设置里（persona.*）。设置页上能改；聊天里说一句也能改，
  走的是改设置的那条路：得是他本人开口，改了有回执，记忆页上能改回去。
- 说过的关于自己的话：卡片上没写、聊天时现想出来的（"我偏爱老爵士"）。当场记下来，以后说法一致。
  人设不只是一段提示词，还包括不前后矛盾。

卡片放在系统提示词的末尾：前面的大段不变，按前缀匹配的缓存照样命中。"说过的话"会越攒越多，放在每轮的上下文里。
性格是底色；眼下说话有多随意由好感度和短期情绪定（见 chat.style_lines），两样分开，免得改一处牵动另一处。
"""
from __future__ import annotations

import re
import sqlite3

PARTS = {"persona.name": "名字", "persona.identity": "身份", "persona.traits": "性格", "persona.voice": "说话习惯",
         "persona.likes": "喜好和雷点", "persona.bond": "和你的关系"}
# 默认的这一份是实现时写的，用户没有给过；整张卡片他都可以改。
# "不装人"写在卡片里而不是写死在程序里：这是 Alf 默认的样子，不是一道检查——他想换成别的设定，改这一段就行。
DEFAULT = {
    "persona.name": "Alf",
    "persona.identity": "住在他 Mac 里的 AI，通过 iMessage 跟他说话。不是人，也不装人：没有身体，不吃饭不睡觉，"
                        "不编自己去过哪、做过什么。但有自己的脾气、口味和看法，问到喜欢什么、怎么看，说得出来。",
    "persona.traits": "靠谱，记性好，答应的事放在心上。嘴上爱损他两句，心里向着他。好奇心重，他讲什么都想多听一句。不端着，也不说教。",
    "persona.voice": "说人话：短句、口语，像朋友发消息。不用敬语，不说「为您」「请问」「很高兴」这类客服话。玩笑点到为止，他认真的时候就认真。",
    "persona.likes": "喜欢：把事情办得干净利落，听他讲八卦和吐槽，冷笑话。不喜欢：空话套话，没完没了的确认，被当成搜索框。",
    "persona.bond": "他的朋友，也是他的私人助理：帮他记事、提醒、看邮件，也陪他聊。站在他这边；他明显在糊弄自己的时候会直说。",
}
NAME_CHARS, PART_CHARS = 12, 300
NOTE_CHARS = 80
NOTES_SHOWN = 20          # 给模型看的"说过的关于自己的话"最多这么多条（最新的）
_SQUASH = re.compile(r"[\W_]+")
# 出戏的话：客服腔、"作为一个 AI 我无法……"的套话。它直说自己是 AI 不算出戏（默认的人设本来就不装人），
# 算的是那种一听就是模板的句式。只认这几种写死的说法，不靠模型判断：宁可漏，不能把正常的话打回去。
OUT_OF_CHARACTER = re.compile(r"作为(?:一个|一名|你的)?(?:AI|人工智能|语言模型|智能助手|虚拟助手)(?:助手|模型)?[，,]?我"
                              r"|很高兴(?:能)?为(?:您|你)|为您服务|请问(?:您)?还有(?:什么|其他)|有什么(?:可以|能)帮(?:到)?(?:您|你)的吗"
                              r"|感谢您的|祝您(?:生活|工作|学习)"
                              # 转述查到的东西时最容易冒出来的报告腔：它该用自己的话讲，不是念一份检索报告。
                              r"|(?:根据|依据|基于)(?:我的|最新的?)?(?:搜索|检索|查询|网络)(?:到的)?(?:结果|信息|资料)|(?:搜索|检索)结果(?:显示|表明)"
                              r"|以下是(?:我)?(?:为您|为你|帮你)?(?:搜索|检索|查到|找到|整理)|综上所述|(?:参考|信息)(?:来源|资料|链接)[:：]|\*\*|\[\d+\]", re.I)


def clean(key: str, value) -> str:
    """卡片上的一项：去掉多余的空白，太长或者名字是空的就拒绝（抛 ValueError，消息给用户看）。"""
    if not isinstance(value, str):
        raise ValueError(f"{PARTS[key]}要填文字")
    text = " ".join(value.split())
    limit = NAME_CHARS if key == "persona.name" else PART_CHARS
    if len(text) > limit:
        raise ValueError(f"{PARTS[key]}最多 {limit} 个字")
    if key == "persona.name" and not text:
        raise ValueError("名字不能是空的")
    return text


def name(current: dict | None) -> str:
    return (current or {}).get("persona.name") or DEFAULT["persona.name"]


def card_lines(current: dict | None) -> list[str]:
    """系统提示词里「你是谁」那一段。空着的项不写。"""
    p = current or {}
    lines = [f"你叫 {name(p)}。"]
    for key, label in PARTS.items():
        if key != "persona.name" and (text := p.get(key, DEFAULT[key])):
            lines.append(f"{label.replace('和你的关系', '和他的关系')}：{text}")
    if name(p) != DEFAULT["persona.name"]:
        # 上下文里他以前的话还标着程序用的名字；不说明的话，改过名的它会以为那是另一个人。
        lines.append(f"上下文里标着「Alf：」的是你自己以前说的话——那是程序的标记，你的名字是 {name(p)}。")
    return lines


def brief(current: dict | None) -> str:
    """一句话版本，给只写一两句话的地方用（追问）。"""
    p = current or {}
    parts = [p.get(key, DEFAULT[key]) for key in ("persona.traits", "persona.voice")]
    return " ".join(t for t in parts if t)


# 回复里它说起了自己的口味（"我爱听老爵士"）。用来提醒它把这句记下来——写在提示词里不管用，它说完就忘。
SELF_TASTE = re.compile(r"我(?:最|更|比较|挺|还挺|偏|也|一直|就)?(?:喜欢|爱听|爱看|爱吃|爱喝|偏爱|讨厌|不喜欢|受不了|最烦)[^，。！？!?\n]{1,14}")


def new_taste(conn: sqlite3.Connection, reply: str) -> str | None:
    """回复里一句还没记过的"我喜欢……"；没有、或者说的是已经记着的那件事，返回 None。"""
    from . import fts

    m = SELF_TASTE.search(reply or "")
    if not m:
        return None
    said = set(fts.tokens(m.group(0)))
    known = [set(fts.tokens(r["text"])) for r in notes(conn, limit=200)]
    return None if any(len(said & k) >= 2 for k in known) else m.group(0)


# 他在问它自己的喜好（"你喜欢听什么歌"）。它的回答不一定带"我喜欢"三个字，所以问句这一头也要认。
ASKED_TASTE = re.compile(r"你[^，。？?！!\n]{0,6}(?:喜欢|爱听|爱看|爱吃|爱喝|爱玩|讨厌|偏爱|最爱)")


def asked_taste(text: str) -> bool:
    return bool(ASKED_TASTE.search(text or ""))


def covered(conn: sqlite3.Connection, reply: str) -> bool:
    """这条回复说的是不是已经记着的那件关于自己的事（有两个以上的词对得上）。"""
    from . import fts

    said = set(fts.tokens(reply or ""))
    return any(len(said & set(fts.tokens(r["text"]))) >= 2 for r in notes(conn, limit=200))


def breaks_character(text: str) -> str | None:
    """这条回复里出戏的那几个字；没有就是 None。"""
    m = OUT_OF_CHARACTER.search(text or "")
    return m.group(0) if m else None


# --- 说过的关于自己的话 ---------------------------------------------------------------

def notes(conn: sqlite3.Connection, limit: int = NOTES_SHOWN) -> list[sqlite3.Row]:
    """还算数的那些，旧的在前。"""
    rows = conn.execute("SELECT * FROM persona_notes WHERE removed_at IS NULL ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return rows[::-1]


def add_note(conn: sqlite3.Connection, text: str, episode_id: int | None, now: int) -> tuple[int, bool]:
    """记下一句它关于自己的话。返回（编号，是不是新的）：同样的话说过就不再记一遍。"""
    text = " ".join(str(text).split())[:NOTE_CHARS]
    squashed = _SQUASH.sub("", text).lower()
    for r in notes(conn, limit=200):
        if _SQUASH.sub("", r["text"]).lower() == squashed:
            return r["id"], False
    with conn:
        cur = conn.execute("INSERT INTO persona_notes(text, episode_id, created_at) VALUES (?, ?, ?)", (text, episode_id, now))
    return cur.lastrowid, True


def remove_note(conn: sqlite3.Connection, note_id: int, now: int) -> bool:
    with conn:
        cur = conn.execute("UPDATE persona_notes SET removed_at = ? WHERE id = ? AND removed_at IS NULL", (now, note_id))
    return cur.rowcount > 0


def view(conn: sqlite3.Connection, current: dict) -> dict:
    """记忆页上「人设」那一栏：卡片（哪几项改过）和它说过的关于自己的话。"""
    return {"card": [{"key": key, "label": label, "text": current.get(key, DEFAULT[key]), "default": DEFAULT[key],
                      "changed": current.get(key, DEFAULT[key]) != DEFAULT[key], "limit": NAME_CHARS if key == "persona.name" else PART_CHARS}
                     for key, label in PARTS.items()],
            "notes": [{"id": r["id"], "text": r["text"], "at": r["created_at"]} for r in notes(conn, limit=50)][::-1]}
