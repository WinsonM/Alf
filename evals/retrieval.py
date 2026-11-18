"""检索评测：一份编出来的语料（一个虚构留学生三个月里记下的事、说过的话、收到的邮件）和带标准答案的问题。

量的是"该找回的那一条，排没排进前 5"。三种找法各跑一遍：
- keyword  只按字面（改动之前的做法；邮件那时候根本搜不了）
- vector   只按意思
- hybrid   两路按名次合（现在用的）
问题分了类——字面对得上的、换了说法的、中英混着问的、问经历的、翻旧话的、找邮件的——因为平均数会把差别盖住：
按字面本来就行的问题不该变差，换了说法的问题才是加向量要解决的。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from alf import db, fts, retrieve

CASES = Path(__file__).parent / "cases"
MODES = ("keyword", "vector", "hybrid")
K = 5


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (CASES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def build(now: datetime):
    """把语料装进一个内存里的库。返回（连接，库里的编号 → 语料里的编号）；说过的话按"哪一段"算。"""
    conn = db.connect(":memory:")
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ids: dict[tuple[str, int], str] = {}
    for item in load("retrieval_corpus"):
        at = int((midnight + timedelta(days=item["day"], minutes=item.get("minute", 720))).timestamp())
        if item["kind"] in ("fact", "moment"):
            cur = conn.execute(
                "INSERT INTO facts(category, text, valid_from, recorded_at, topic, slot, importance, ours) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (item.get("category", "moment"), item["text"], at, at, item.get("topic"), item.get("slot"), item.get("importance"),
                 bool(item.get("ours"))))
            fts.index(conn, "facts_fts", cur.lastrowid, " ".join(p for p in (item.get("topic"), item.get("slot"), item["text"]) if p))
            ids[("fact", cur.lastrowid)] = item["id"]
        elif item["kind"] == "said":
            cur = conn.execute("INSERT INTO episodes(session_id, role, text, created_at) VALUES ('eval', 'user', ?, ?)", (item["text"], at))
            fts.index(conn, "episodes_fts", cur.lastrowid, item["text"])
            ids[("said", cur.lastrowid)] = item["burst"]
        else:
            cur = conn.execute(
                "INSERT INTO inbox_items(source, external_id, sender, title, snippet, notified, created_at, tier, summary)"
                " VALUES ('email.eval', ?, ?, ?, '', ?, ?, ?, ?)",
                (item["id"], item["sender"], item["title"], item["tier"] != "skip", at, item["tier"],
                 json.dumps({"gist": item["gist"]}, ensure_ascii=False)))
            ids[("mail", cur.lastrowid)] = item["id"]
    conn.commit()
    retrieve.ensure_indexed(conn)
    return conn, ids


def found(conn, ids, question: dict, mode: str, embedder, limit: int = K) -> list[str]:
    """这个问题按 mode 找回来的前几条，写成语料里的编号。"""
    kind = {"fact": "fact", "said": "chunk", "mail": "mail"}[question["in"]]
    hits = retrieve.search(conn, kind, question["q"], limit=limit, embedder=embedder, mode=mode)
    if kind == "chunk":       # 一段对话算它所属的那一段
        return [ids[("said", h.ref_id)] for h in hits]
    return [ids[(question["in"], h.ref_id)] for h in hits]


def run(args) -> int:
    from alf import embed

    now = datetime.fromisoformat("2026-11-18T12:00").replace(tzinfo=__import__("zoneinfo").ZoneInfo("Australia/Sydney"))
    conn, ids = build(now)
    embedder = embed.default()
    modes = [m for m in MODES if m == "keyword" or embedder is not None]
    if embedder is None:
        print("嵌入模型不在本机，只量按字面这一路（`uv run alf index` 会下载模型）")
    else:
        while retrieve.sync(conn, embedder, int(now.timestamp()))["left"]:
            pass
    questions = [q for q in load("retrieval") if q["want"]]
    kinds = list(dict.fromkeys(q["kind"] for q in questions))
    table = {m: {k: [] for k in kinds} for m in modes}          # 每个问题记一个"排第几"（没进前 K 是 None）
    for q in questions:
        for mode in modes:
            got = found(conn, ids, q, mode, embedder)
            rank = next((i for i, x in enumerate(got, 1) if x in q["want"]), None)
            table[mode][q["kind"]].append(rank)
            if args.verbose and rank is None:
                print(f"  ✗ [{mode}] {q['q']}  想要 {q['want']}  找到 {got}")
    label = {"literal": "字面对得上", "paraphrase": "换了说法", "crosslingual": "中英混问", "moment": "问经历", "said": "翻旧话", "mail": "找邮件"}
    print(f"\n该找回的那条排进前 {K} 的比例（括号里是排第一的比例）")
    print(f"{'':<8}" + "".join(f"{m:>18}" for m in modes) + "   题数")
    for kind in kinds:
        row = f"{label.get(kind, kind):<{8 - len(label.get(kind, kind)) // 2}}"
        for mode in modes:
            ranks = table[mode][kind]
            row += f"{sum(r is not None for r in ranks) / len(ranks):>11.0%} ({sum(r == 1 for r in ranks) / len(ranks):>4.0%})"
        print(row + f"   {len(table[modes[0]][kind])}")
    for mode in modes:
        ranks = [r for k in kinds for r in table[mode][k]]
        print(f"{mode:<8} 合计 {sum(r is not None for r in ranks)}/{len(ranks)} 进前 {K}，"
              f"平均倒数名次 {sum(1 / r for r in ranks if r) / len(ranks):.2f}")
    # 守住一条：加了向量之后，按字面本来找得到的不许丢
    if "hybrid" in modes:
        lost = [q["q"] for q in questions
                if next((1 for x in found(conn, ids, q, "keyword", embedder) if x in q["want"]), None)
                and not next((1 for x in found(conn, ids, q, "hybrid", embedder) if x in q["want"]), None)]
        if lost:
            print("按字面找得到、合起来反而丢了的：", lost)
            return 1
    return 0
