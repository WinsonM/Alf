"""Chinese-aware full-text search on SQLite FTS5.

FTS5's default tokenizer treats a run of Chinese characters as a single token, so
"我老婆的生日是3月14号" would only ever match that exact string. We index
overlapping character bigrams instead (the classic CJK approach): the query
"生日" then matches "生日" anywhere in a stored sentence. Latin words and numbers
stay whole, so "COMP5318" is one token.
"""
from __future__ import annotations

import re
import sqlite3

# 到处都有、说明不了两句话讲的是同一件事的二元组：疑问词、指代、虚词。
# 两个用处：回想更早的经历时不算数（memory.py）；按字面检索自然语言的问句时先去掉（retrieve.py）——
# 不去的话"我什么时候生过病"会先找到所有带"时候"的记忆。
COMMON = frozenset(
    "今天 昨天 明天 前天 后天 一个 我们 你们 他们 什么 怎么 可以 没有 这个 那个 时候 觉得 一下 还是 就是 不是 已经 现在 然后 "
    "因为 所以 但是 感觉 有点 真的 自己 知道 这样 那样 一起 之后 之前 以后 以前 最近 上次 那次 这次 还有 应该 可能 "
    "其实 起来 出来 一点 很多 一直 时间 事情 东西 "
    "多少 哪里 哪个 哪些 哪家 哪天 哪儿 是谁 几点 为什 么时 怎样 如何 是不 有没 那封 那家 那天 那些 这些 说过 我的 你的 他的 "
    "的时 了什 是什 么样 么回 回事 的那 过什 叫什 么名 名字".split())

_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿]+|[a-z0-9]+")


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for run in _RUN.findall(text.lower()):
        if run.isascii() or len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def index(conn: sqlite3.Connection, table: str, rowid: int, text: str) -> None:
    conn.execute(f"INSERT INTO {table}(rowid, tokens) VALUES (?, ?)", (rowid, " ".join(tokens(text))))


def query(text: str, content_only: bool = False) -> str | None:
    """OR over the text's tokens; bm25 ranks rows sharing more (and rarer) tokens first.
    With content_only, the everywhere-bigrams in COMMON and single characters are left out: what remains are the
    words that say what the sentence is about. Returns None when nothing is left."""
    unique = list(dict.fromkeys(tokens(text)))
    if content_only:
        unique = [t for t in unique if t not in COMMON and len(t) > 1]
    if not unique:
        return None
    return " OR ".join(f'"{t}"' for t in unique)
