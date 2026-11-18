"""人和声纹：录音里的这句话是谁说的。

切分模型只知道"这一小段里有两个人在轮流说"，不知道他们是谁，也不知道今天的这个人和昨天的是不是同一个。
这里补的就是这一块：每个人留几条声纹，新听到的一段声音拿去和每个人比，够像就算是他。

- 一个没听过的声音先成为一个"没有名字的人"（声7）。它反复出现了，Alf 才在每晚汇总里问他一次"这是谁"；
  他说了名字，这个人过去和以后说的话就都有了主人。同一个名字说了两次，就是同一个人，并到一起。
- 他本人（is_me）的声纹先从他发给 Alf 的语音消息里取——那些一定是他说的，不用他专门录。

门槛（SAME、NEW、LEARN）是暂定的：只在合成的声音上看过分数——同一个声音不同的话 0.76–0.96，同一套合成引擎的
两个声音 0.50–0.55，不同引擎的 0.1 以下。真人、手表远场会低一截。认错人（把别人的话记到某个人名下）比认不出来
糟得多，所以先偏紧：够不上 SAME 的宁可记成"听不出是谁"。等有了真实录音，按 utterances.voice_score 的分布再定。
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

SAME = 0.6                # 和某个人的声纹像到这个数，才算是他
MARGIN = 0.05             # 而且要比第二像的人高出这么多；两个人差不多像的时候不硬猜
NEW = 0.4                 # 和谁都不到这个数，才算一个没听过的声音；夹在中间的记成"听不出是谁"
NEW_SECONDS = 6.0         # 至少听到这么多秒，才值得为它新建一个人：一两句路过的话不建
LOCAL_SAME = 0.65         # 同一场对话里，切分模型分出来的两个"人"声纹像到这个数，其实是同一个人（它分多了）
LEARN = 0.7               # 像到这个数的新片段才留作这个人的新声纹：不够像的留下来会把声纹带偏
LEARN_SECONDS = 4.0
PRINTS_KEPT = 12          # 每个人最多留这么多条从录音里学来的声纹，旧的换掉
WEIGHT_CAP = 30.0         # 一条声纹的分量按它的秒数算，封顶在这儿
ASK_DAYS = 2              # 一个没名字的声音在这么多个不同的日子出现过，才问他是谁
ASK_SECONDS = 120.0       # 或者一共说了这么久
ASK_RECENT = 3 * 86400    # 最近这几天还听到过才问：很久以前的声音他多半想不起来
ASK_AGAIN = 7 * 86400     # 问过没回答，隔这么久再问一次
MAX_ASKS = 2
NAME_CHARS = 20


@dataclass
class Match:
    person_id: int | None     # 认出来是谁；None = 听不出
    score: float | None       # 和最像的那个人有多像（库里一个人都没有时是 None）
    new: bool = False         # 这次新建的


def get(conn: sqlite3.Connection, person_id) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone() if isinstance(person_id, int) else None


def me(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM people WHERE is_me = 1 ORDER BY id LIMIT 1").fetchone()


def label(row: sqlite3.Row | None) -> str:
    """给模型和他看的称呼：他本人是"我"，有名字的用名字，没名字的是"声7"，听不出是谁的是"有人"。"""
    if row is None:
        return "有人"
    if row["is_me"]:
        return "我"
    return row["name"] or f"声{row['id']}"


def labels(conn: sqlite3.Connection) -> dict[int, str]:
    return {r["id"]: label(r) for r in conn.execute("SELECT * FROM people")}


_VOICE = re.compile(r"声(\d+)")


def render(text: str, names: dict[int, str]) -> str:
    """"每天的事"是在某个声音还没有名字的时候写下的（"和声3分了小组作业的工"）。给人看、给模型看之前，
    把现在已经知道是谁的换上名字；他本人换成"自己"（事情是从他的角度写的，写"我"会分不清是他还是 Alf）。"""
    def swap(m: re.Match) -> str:
        name = names.get(int(m.group(1)))
        return "自己" if name == "我" else name or m.group(0)
    return _VOICE.sub(swap, text)


def _centroids(conn: sqlite3.Connection, model: str) -> dict[int, "object"]:
    """每个人的声纹：几条的加权平均（按秒数），再归一化。"""
    import numpy as np

    sums: dict[int, "object"] = {}
    for r in conn.execute("SELECT person_id, vec, seconds FROM voiceprints WHERE model = ?", (model,)):
        vec = np.frombuffer(r["vec"], dtype=np.float32)
        sums[r["person_id"]] = sums.get(r["person_id"], 0) + vec * min(r["seconds"], WEIGHT_CAP)
    out = {}
    for pid, total in sums.items():
        norm = float(np.linalg.norm(total))
        if norm > 0:
            out[pid] = total / norm
    return out


def add_print(conn: sqlite3.Connection, person_id: int, vec, seconds: float, model: str, source: str, ref: str | None, now: int) -> None:
    import numpy as np

    with conn:
        conn.execute("INSERT INTO voiceprints(person_id, model, vec, seconds, source, ref, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (person_id, model, np.asarray(vec, dtype=np.float32).tobytes(), seconds, source, ref, now))
        # 从录音里学来的只留最近的几条；语音消息和专门录的不动——那些确定是本人
        old = conn.execute("SELECT id FROM voiceprints WHERE person_id = ? AND source = 'heard' ORDER BY id DESC LIMIT -1 OFFSET ?",
                           (person_id, PRINTS_KEPT)).fetchall()
        conn.executemany("DELETE FROM voiceprints WHERE id = ?", [(r["id"],) for r in old])


def identify(conn: sqlite3.Connection, vec, seconds: float, model: str, now: int, ref: str | None = None,
             present: set[int] | frozenset = frozenset()) -> Match:
    """这段声音是谁的。认出来了、而且够像，就顺手把它留作这个人的一条新声纹；谁都不像、又听得够久，新建一个没名字的人。

    present 是同一场对话里已经认出来的别人。切分模型说这是另一个人在说话，那它"有点像"在场的某个人就不算数——
    两个声音接近的人坐在一起聊天时，后一个不该因为像前一个而永远没有身份。够像（到 SAME）的仍然算同一个人：
    那是切分模型把一个人分成了两个。
    """
    centroids = _centroids(conn, model)
    scored = sorted(((float(c @ vec), pid) for pid, c in centroids.items()), reverse=True)
    best, who = scored[0] if scored else (None, None)
    second = scored[1][0] if len(scored) > 1 else -1.0
    if best is not None and best >= SAME and best - second >= MARGIN:
        if best >= LEARN and seconds >= LEARN_SECONDS:
            add_print(conn, who, vec, seconds, model, "heard", ref, now)
        return Match(who, best)
    others = [score for score, pid in scored if pid not in present]
    if (not others or others[0] < NEW) and seconds >= NEW_SECONDS:
        with conn:
            pid = conn.execute("INSERT INTO people(created_at) VALUES (?)", (now,)).lastrowid
        add_print(conn, pid, vec, seconds, model, "heard", ref, now)
        return Match(pid, best, new=True)
    return Match(None, best)


def enroll_me(conn: sqlite3.Connection, vec, seconds: float, model: str, source: str, ref: str | None, now: int) -> int:
    """把一段确定是他本人的声音记成他的声纹。还没有"他本人"这个人就建一个。"""
    row = me(conn)
    if row is None:
        with conn:
            pid = conn.execute("INSERT INTO people(is_me, created_at, named_at) VALUES (1, ?, ?)", (now, now)).lastrowid
    else:
        pid = row["id"]
    add_print(conn, pid, vec, seconds, model, source, ref, now)
    return pid


def clean_name(name) -> str:
    return " ".join(str(name or "").split())[:NAME_CHARS]


def merge(conn: sqlite3.Connection, source: int, target: int) -> None:
    """两个"人"其实是同一个：把 source 的声纹和说过的话都归到 target，source 删掉。"""
    if source == target:
        return
    with conn:
        conn.execute("UPDATE voiceprints SET person_id = ? WHERE person_id = ?", (target, source))
        conn.execute("UPDATE utterances SET person_id = ? WHERE person_id = ?", (target, source))
        for r in conn.execute("SELECT id, people FROM day_events WHERE people LIKE ?", (f"%{source}%",)).fetchall():
            ids = [target if p == source else p for p in json.loads(r["people"])]
            conn.execute("UPDATE day_events SET people = ? WHERE id = ?", (json.dumps(list(dict.fromkeys(ids))), r["id"]))
        # 以前整理出来的事里写着"声7"的，改成并过去的那个编号：不然这个编号没人了，名字就换不上去
        for r in conn.execute("SELECT id, text FROM day_events WHERE text LIKE ?", (f"%声{source}%",)).fetchall():
            conn.execute("UPDATE day_events SET text = ? WHERE id = ?", (re.sub(rf"声{source}(?!\d)", f"声{target}", r["text"]), r["id"]))
        conn.execute("DELETE FROM people WHERE id = ?", (source,))


def name(conn: sqlite3.Connection, person_id: int, new_name: str | None, now: int, *, is_me: bool = False) -> sqlite3.Row | None:
    """告诉 Alf 这个声音是谁。is_me = 这是他本人。已经有同名的人（或者已经有"他本人"）就并过去。返回最后留下的那个人。"""
    row = get(conn, person_id)
    if row is None:
        return None
    new_name = clean_name(new_name)
    if is_me:
        mine = me(conn)
        if mine is not None and mine["id"] != person_id:
            merge(conn, person_id, mine["id"])
            return get(conn, mine["id"])
        with conn:
            conn.execute("UPDATE people SET is_me = 1, name = NULL, named_at = ? WHERE id = ?", (now, person_id))
        return get(conn, person_id)
    if not new_name:
        return None
    same = conn.execute("SELECT id FROM people WHERE name = ? COLLATE NOCASE AND id != ? AND is_me = 0", (new_name, person_id)).fetchone()
    if same is not None:
        merge(conn, person_id, same["id"])
        return get(conn, same["id"])
    with conn:
        conn.execute("UPDATE people SET name = ?, named_at = ?, is_me = 0 WHERE id = ?", (new_name, now, person_id))
    return get(conn, person_id)


def ignore(conn: sqlite3.Connection, person_id: int) -> bool:
    """他说不用记这个人：以后不再问是谁。说过的话还在，只是一直没有名字。"""
    with conn:
        return conn.execute("UPDATE people SET ignored = 1 WHERE id = ? AND name IS NULL AND is_me = 0", (person_id,)).rowcount > 0


def days_of(conn: sqlite3.Connection, person_id: int) -> list[str]:
    """这个人在哪些日子说过话。改了名字、并了人之后，这些天的检索索引要重做（里面写着"谁：说了什么"）。"""
    return [r["day"] for r in conn.execute("SELECT DISTINCT day FROM utterances WHERE person_id = ?", (person_id,))]


def forget(conn: sqlite3.Connection, person_id: int) -> bool:
    """把这个人和他的声纹删掉。他说过的话留着，但不再属于谁；以后再听到这个声音，会当成没听过的。"""
    with conn:
        conn.execute("DELETE FROM voiceprints WHERE person_id = ?", (person_id,))
        conn.execute("UPDATE utterances SET person_id = NULL WHERE person_id = ?", (person_id,))
        for r in conn.execute("SELECT id, people FROM day_events WHERE people LIKE ?", (f"%{person_id}%",)).fetchall():
            conn.execute("UPDATE day_events SET people = ? WHERE id = ?",
                         (json.dumps([p for p in json.loads(r["people"]) if p != person_id]), r["id"]))
        return conn.execute("DELETE FROM people WHERE id = ?", (person_id,)).rowcount > 0


def _heard(conn: sqlite3.Connection, person_id: int) -> sqlite3.Row:
    return conn.execute("SELECT COUNT(*) AS lines, COUNT(DISTINCT day) AS days, COALESCE(SUM(ended_at - started_at), 0) AS seconds,"
                        " MAX(ended_at) AS last FROM utterances WHERE person_id = ?", (person_id,)).fetchone()


def unknown_worth_asking(conn: sqlite3.Connection, now: int, limit: int = 1) -> list[dict]:
    """值得问一句"这是谁"的声音：没名字、反复出现、最近还听到过、没问够两次。带上他说过的一句话，好让他认出来。"""
    out = []
    rows = conn.execute("SELECT * FROM people WHERE name IS NULL AND is_me = 0 AND ignored = 0 AND asks < ?"
                        " AND (asked_at IS NULL OR asked_at <= ?) ORDER BY id", (MAX_ASKS, now - ASK_AGAIN)).fetchall()
    for r in rows:
        h = _heard(conn, r["id"])
        if not h["lines"] or (h["days"] < ASK_DAYS and h["seconds"] < ASK_SECONDS) or now - (h["last"] or 0) > ASK_RECENT:
            continue
        sample = conn.execute("SELECT text, started_at, day FROM utterances WHERE person_id = ? ORDER BY day DESC,"
                              " LENGTH(text) DESC LIMIT 1", (r["id"],)).fetchone()
        out.append({"id": r["id"], "days": h["days"], "seconds": round(h["seconds"]), "last": h["last"],
                    "said": " ".join(sample["text"].split())[:40], "said_at": sample["started_at"], "day": sample["day"]})
    out.sort(key=lambda p: (-p["days"], -p["seconds"]))
    return out[:limit]


def mark_asked(conn: sqlite3.Connection, person_id: int, now: int) -> None:
    with conn:
        conn.execute("UPDATE people SET asked_at = ?, asks = asks + 1 WHERE id = ?", (now, person_id))


CONTEXT_NAMED = 8
CONTEXT_UNKNOWN = 3


def _ago(ts: float, now: datetime) -> str:
    at = datetime.fromtimestamp(ts, now.tzinfo)
    days = (now.date() - at.date()).days
    return f"今天 {at:%H:%M}" if days == 0 else f"昨天 {at:%H:%M}" if days == 1 else f"{days} 天前"


def context_lines(conn: sqlite3.Connection, now: datetime) -> list[str]:
    """给模型看的"录音里的人"：最近听到过的有名字的人，和几个还没名字、但说了不少话的声音（他随口说出是谁时，对得上号）。"""
    ts = now.timestamp()
    out = []
    for r in conn.execute("SELECT p.id, p.name, MAX(u.ended_at) AS last FROM people p JOIN utterances u ON u.person_id = p.id"
                          " WHERE p.is_me = 0 AND p.name IS NOT NULL GROUP BY p.id HAVING last >= ? ORDER BY last DESC LIMIT ?",
                          (ts - 14 * 86400, CONTEXT_NAMED)):
        out.append(f"人{r['id']} {r['name']}（上次听到：{_ago(r['last'], now)}）")
    unknown = conn.execute("SELECT p.id, COUNT(DISTINCT u.day) AS days, SUM(u.ended_at - u.started_at) AS seconds, MAX(u.ended_at) AS last"
                           " FROM people p JOIN utterances u ON u.person_id = p.id WHERE p.is_me = 0 AND p.name IS NULL AND p.ignored = 0"
                           " GROUP BY p.id HAVING last >= ? AND seconds >= 20 ORDER BY seconds DESC LIMIT ?", (ts - ASK_RECENT, CONTEXT_UNKNOWN)).fetchall()
    for r in unknown:
        said = conn.execute("SELECT text, started_at FROM utterances WHERE person_id = ? ORDER BY day DESC, LENGTH(text) DESC LIMIT 1", (r["id"],)).fetchone()
        out.append(f"声{r['id']} 还不知道是谁：{r['days']} 天里听到过，{_ago(said['started_at'], now)}说过「{' '.join(said['text'].split())[:40]}」")
    return out


def view(conn: sqlite3.Connection) -> list[dict]:
    """记忆页上的"身边的人"：他本人在最前，有名字的其次，没名字的按听到的多少排。只听到过一两句的陌生声音不列。"""
    out = []
    for r in conn.execute("SELECT * FROM people ORDER BY is_me DESC, name IS NULL, id"):
        h = _heard(conn, r["id"])
        if not r["is_me"] and not r["name"] and h["seconds"] < 20:
            continue
        prints = conn.execute("SELECT COUNT(*) FROM voiceprints WHERE person_id = ?", (r["id"],)).fetchone()[0]
        out.append({"id": r["id"], "name": r["name"], "is_me": bool(r["is_me"]), "label": label(r), "ignored": bool(r["ignored"]),
                    "lines": h["lines"], "days": h["days"], "seconds": round(h["seconds"]), "last": h["last"], "prints": prints})
    return out
