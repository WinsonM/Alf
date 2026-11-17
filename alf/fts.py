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


def query(text: str) -> str | None:
    """OR over the text's tokens; bm25 ranks rows sharing more (and rarer) tokens first."""
    unique = list(dict.fromkeys(tokens(text)))
    if not unique:
        return None
    return " OR ".join(f'"{t}"' for t in unique)
