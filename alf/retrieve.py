"""检索：按字面和按意思两路找，再按名次合到一起。

几类东西各找各的：
- fact   记忆——事实和经历（他对 Alf 的要求不在这里：那些每轮都带着）
- chunk  他以前说过的话，按"一小段对话"算一条
- mail   邮件：发件人、主题、提要
- heard  手表录音里听到的话，按"一小段对话"算一条（hearing.py 切的）
- event  从听到的话里整理出来的"每天的事"（days.py）

为什么要两路：
- 按字面（全文索引，中文按二元组）找得准，但只认字。"我对象叫什么"找不到"女朋友叫小满"，"bench press"找不到"卧推"。
- 按意思（向量，见 embed.py）补的就是这一块；可它对专有名词、编号不如字面可靠。
所以两路各出一个名次表，用倒数名次相加（RRF）合成一个：两边都靠前的排最前，只有一边找到的也留着。
只用名次、不用分数——两路的分数不是一个量纲，向量的余弦还全挤在 0.7–0.9（见 embed.py）。

没有嵌入模型时只剩按字面这一路，结果和以前一样。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass

from . import fts, people
from .connectors import sender_name
from .memory import FEEDBACK

log = logging.getLogger("alf.retrieve")

KINDS = ("fact", "chunk", "mail", "heard", "event")
RRF_K = 40                # 倒数名次相加里的常数：越大，名次靠后的结果分量越重。沿用 Anima Brain 的取值
# 合的时候，按意思的名次算全份，按字面的算半份。在评测集上（evals/retrieval.py）：对自然语言的问句，字面这一路
# 噪声大，等权相加会把按意思找对的第一名挤下去；可专有名词、编号又是字面最可靠。半份是两头都不丢的取法。
WORDS_WEIGHT = 0.5
CANDIDATES = 30           # 每一路先各取这么多条，再合
CHUNK_GAP = 600           # 两句话隔了这么久，就算另起一段
CHUNK_CHARS = 400
CHUNK_LINES = 12
SYNC_BATCH = 64           # 后台每次最多算这么多条向量：一批一两秒，不拖住别的任务
_SQUASH = re.compile(r"[\W_]+")


@dataclass
class Hit:
    ref_id: int
    score: float
    by_words: int | None = None      # 按字面排第几（从 1 起）；没找到是 None
    by_meaning: int | None = None    # 按意思排第几


# --- 每一类拿什么文字去找 ---------------------------------------------------------------

def fact_text(r: sqlite3.Row) -> str:
    """和全文索引用的是同一段：主题、项目名、正文。"""
    return " ".join(p for p in (r["topic"], r["slot"], r["text"]) if p)


def mail_text(r: sqlite3.Row) -> str:
    try:
        gist = json.loads(r["summary"] or "{}").get("gist") or ""
    except ValueError:
        gist = ""
    return "｜".join(p for p in (sender_name(r["sender"] or "", 30), r["title"] or "", gist or (r["snippet"] or "")[:200]) if p)


def _hash(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\n{text}".encode()).hexdigest()[:16]


# --- 把他说过的话切成一小段一小段 ---------------------------------------------------------

def rebuild_chunks(conn: sqlite3.Connection) -> int:
    """他的消息平均十来个字，单句没法按意思找；连着说的几句合成一段才有内容。

    挨着的两句隔不到 CHUNK_GAP 就算同一段，一段太长就断开。最后一段可能还没说完，所以每次从它的第一句重新切。
    """
    last = conn.execute("SELECT id FROM chunks ORDER BY id DESC LIMIT 1").fetchone()
    start = last["id"] if last else 0
    rows = conn.execute("SELECT id, text, created_at FROM episodes WHERE role = 'user' AND id >= ? ORDER BY id", (start,)).fetchall()
    groups: list[list[sqlite3.Row]] = []
    for r in rows:
        cur = groups[-1] if groups else None
        if (cur and r["created_at"] - cur[-1]["created_at"] <= CHUNK_GAP and len(cur) < CHUNK_LINES
                and sum(len(x["text"]) for x in cur) + len(r["text"]) <= CHUNK_CHARS):
            cur.append(r)
        else:
            groups.append([r])
    with conn:
        conn.execute("DELETE FROM chunks WHERE id >= ?", (start,))
        for g in groups:
            text = "\n".join(" ".join(x["text"].split()) for x in g)
            if len(_SQUASH.sub("", text)) < 4:
                continue                    # 整段只有"好""嗯嗯"：没有可找的内容
            conn.execute("INSERT INTO chunks(id, last_episode, text, started_at, ended_at) VALUES (?, ?, ?, ?, ?)",
                         (g[0]["id"], g[-1]["id"], text, g[0]["created_at"], g[-1]["created_at"]))
    return len(groups)


def ensure_indexed(conn: sqlite3.Connection) -> None:
    """不用模型就能做的那部分索引：对话切段、邮件的全文索引。跟得上就什么都不做，所以查的时候顺手调一下也不贵。"""
    newest = conn.execute("SELECT MAX(id) FROM episodes WHERE role = 'user'").fetchone()[0] or 0
    chunked = conn.execute("SELECT MAX(last_episode) FROM chunks").fetchone()[0] or 0
    if newest > chunked:
        rebuild_chunks(conn)
    missing = conn.execute("SELECT i.* FROM inbox_items i LEFT JOIN inbox_fts f ON f.rowid = i.id"
                           " WHERE f.rowid IS NULL AND COALESCE(i.tier, '') != 'dup' LIMIT 500").fetchall()
    if missing:
        with conn:
            for r in missing:
                fts.index(conn, "inbox_fts", r["id"], mail_text(r))


# --- 算向量（后台做）--------------------------------------------------------------------

def _sources(conn: sqlite3.Connection) -> list[tuple[str, int, str]]:
    """现在该有向量的每一条：（类别，编号，文字）。"""
    out = [("fact", r["id"], fact_text(r)) for r in conn.execute(
        "SELECT id, topic, slot, text FROM facts WHERE valid_to IS NULL AND category != ? ORDER BY id", (FEEDBACK,))]
    out += [("chunk", r["id"], r["text"]) for r in conn.execute("SELECT id, text FROM chunks ORDER BY id")]
    out += [("mail", r["id"], mail_text(r)) for r in conn.execute(
        "SELECT * FROM inbox_items WHERE COALESCE(tier, '') != 'dup' ORDER BY id")]
    out += [("heard", r["id"], r["text"]) for r in conn.execute("SELECT id, text FROM heard_passages ORDER BY id")]
    names = people.labels(conn)          # 事情里的"声3"换成现在的名字：起了名之后文字变了，向量会跟着重算
    out += [("event", r["id"], people.render(r["text"], names)) for r in conn.execute("SELECT id, text FROM day_events WHERE removed_at IS NULL ORDER BY id")]
    return [(kind, ref, text) for kind, ref, text in out if text.strip()]


def sync(conn: sqlite3.Connection, embedder, now: int, limit: int = SYNC_BATCH) -> dict:
    """把还没有向量的、文字变过的补上；已经不算数的（过时的记忆、换了模型的）清掉。返回这次算了几条、还剩几条。"""
    ensure_indexed(conn)
    if embedder is None:
        return {"embedded": 0, "left": 0, "model": None}
    sources = _sources(conn)
    have = {(r["kind"], r["ref_id"]): (r["model"], r["hash"]) for r in conn.execute("SELECT kind, ref_id, model, hash FROM embeddings")}
    wanted = {(kind, ref): _hash(embedder.name, text) for kind, ref, text in sources}
    stale = [key for key in have if key not in wanted]
    todo = [(kind, ref, text) for kind, ref, text in sources if have.get((kind, ref)) != (embedder.name, wanted[(kind, ref)])]
    batch = todo[:limit]
    vectors = embedder.encode([text for _, _, text in batch]) if batch else []
    with conn:
        conn.executemany("DELETE FROM embeddings WHERE kind = ? AND ref_id = ?", stale)
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings(kind, ref_id, model, hash, vec, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [(kind, ref, embedder.name, wanted[(kind, ref)], vec.tobytes(), now) for (kind, ref, _), vec in zip(batch, vectors)])
    return {"embedded": len(batch), "left": len(todo) - len(batch), "model": embedder.name}


def coverage(conn: sqlite3.Connection, model: str | None) -> dict:
    """每一类有多少条、其中多少条已经有向量。给自检和状态页看。"""
    out = {}
    for kind in KINDS:
        total = sum(1 for k, _, _ in _sources(conn) if k == kind)
        done = conn.execute("SELECT COUNT(*) FROM embeddings WHERE kind = ? AND model = ?", (kind, model or "")).fetchone()[0]
        out[kind] = {"total": total, "embedded": min(done, total)}
    return out


# --- 找 --------------------------------------------------------------------------------

def by_words(conn: sqlite3.Connection, kind: str, query: str, limit: int = CANDIDATES) -> list[int]:
    """按字面找：全文索引排出来的名次。"""
    q = fts.query(query, content_only=True)     # 去掉"什么""怎么"这类到处都有的词，剩下的才说明问的是什么
    if not q:
        return []
    if kind == "fact":
        rows = conn.execute("SELECT f.id FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid WHERE facts_fts MATCH ?"
                            " AND f.valid_to IS NULL AND f.category != ? ORDER BY bm25(facts_fts) LIMIT ?", (q, FEEDBACK, limit))
        return [r["id"] for r in rows]
    if kind == "mail":
        rows = conn.execute("SELECT i.id FROM inbox_fts JOIN inbox_items i ON i.id = inbox_fts.rowid WHERE inbox_fts MATCH ?"
                            " ORDER BY bm25(inbox_fts) LIMIT ?", (q, limit))
        return [r["id"] for r in rows]
    if kind in ("heard", "event"):
        table = "heard_fts" if kind == "heard" else "events_fts"
        return [r["rowid"] for r in conn.execute(f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY bm25({table}) LIMIT ?", (q, limit))]
    # 对话：全文索引是按单句建的，找到之后归到它所在的那一段
    out: list[int] = []
    for r in conn.execute("SELECT rowid FROM episodes_fts WHERE episodes_fts MATCH ? ORDER BY bm25(episodes_fts) LIMIT ?", (q, limit * 3)):
        chunk = conn.execute("SELECT id FROM chunks WHERE id <= ? AND last_episode >= ?", (r["rowid"], r["rowid"])).fetchone()
        if chunk and chunk["id"] not in out:
            out.append(chunk["id"])
    return out[:limit]


def by_meaning(conn: sqlite3.Connection, kind: str, query: str, embedder, limit: int = CANDIDATES) -> list[int]:
    """按意思找：和这句话的向量最接近的那些。几千条以内直接全算一遍，用不着专门的向量库。"""
    import numpy as np

    if kind == "fact":
        rows = conn.execute("SELECT e.ref_id, e.vec FROM embeddings e JOIN facts f ON f.id = e.ref_id"
                            " WHERE e.kind = 'fact' AND e.model = ? AND f.valid_to IS NULL", (embedder.name,)).fetchall()
    else:
        rows = conn.execute("SELECT ref_id, vec FROM embeddings WHERE kind = ? AND model = ?", (kind, embedder.name)).fetchall()
    if not rows:
        return []
    matrix = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    scores = matrix @ embedder.encode([query], query=True)[0]
    return [rows[i]["ref_id"] for i in np.argsort(-scores)[:limit]]


def search(conn: sqlite3.Connection, kind: str, query: str, limit: int = 5, embedder=None, mode: str = "hybrid",
           exclude: set[int] | frozenset = frozenset()) -> list[Hit]:
    """在一类东西里找和 query 有关的，最相关的在前。mode：hybrid 两路合 | keyword 只按字面 | vector 只按意思。"""
    query = " ".join(str(query).split())
    if not query:
        return []
    ensure_indexed(conn)
    words = by_words(conn, kind, query) if mode != "vector" else []
    meaning: list[int] = []
    if embedder is not None and mode != "keyword":
        try:
            meaning = by_meaning(conn, kind, query, embedder)
        except Exception:                       # 模型出了岔子：这一次只按字面找，不让一次检索失败带垮回复
            log.exception("按意思检索出错，这次只按字面找")
    hits: dict[int, Hit] = {}
    for ranking, field, weight in ((words, "by_words", WORDS_WEIGHT if meaning else 1.0), (meaning, "by_meaning", 1.0)):
        for rank, ref in enumerate(ranking, 1):
            if ref in exclude:
                continue
            hit = hits.setdefault(ref, Hit(ref, 0.0))
            hit.score += weight / (RRF_K + rank)
            setattr(hit, field, rank)
    return sorted(hits.values(), key=lambda h: (-h.score, h.ref_id))[:limit]


STANDOUT_MIN = 8          # 经历少于这么多条时，"比别的高出多少"算不稳，不用这道门
STANDOUT_MARGIN = 0.2


def standout_moment(conn: sqlite3.Connection, query: str, embedder, skip: set[int] | frozenset = frozenset()) -> int | None:
    """他这句话有没有明显指向某一条更早的经历。有就返回那条的编号，没有（闲聊、办事的话）返回 None。

    每一轮都会问这一句，所以要一道门槛：闲聊不该想起任何事。余弦的绝对值不能用（见 embed.py）；用的是相对的——
    这句话和每条经历的相似度里，最高的那条比平均高出几个标准差。随便一句话在 n 条里的最高值大约是 sqrt(2 ln n) 个
    标准差，所以门槛定在它上面一点。评测语料上：十六句闲聊和办事的话一条都没过；十五句确实在说某件事的话过了十三句。
    """
    import math

    import numpy as np

    rows = conn.execute("SELECT e.ref_id, e.vec FROM embeddings e JOIN facts f ON f.id = e.ref_id WHERE e.kind = 'fact'"
                        " AND e.model = ? AND f.valid_to IS NULL AND f.category = 'moment'", (embedder.name,)).fetchall()
    if len(rows) < STANDOUT_MIN or len(_SQUASH.sub("", query)) < 4:
        return None
    matrix = np.frombuffer(b"".join(r["vec"] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    scores = matrix @ embedder.encode([query], query=True)[0]
    spread = float(scores.std())
    best = int(np.argmax(scores))
    if spread <= 0 or rows[best]["ref_id"] in skip:
        return None
    z = (float(scores[best]) - float(scores.mean())) / spread
    return rows[best]["ref_id"] if z >= math.sqrt(2 * math.log(len(rows))) + STANDOUT_MARGIN else None
