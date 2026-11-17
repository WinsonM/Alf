"""MCP 服务器：新闻标题。从配置的 RSS 源取最新的几条标题，只取标题和时间，不取正文。

订阅哪些源由环境变量 ALF_NEWS_FEEDS 决定（JSON：[{"topic": "...", "source": "...", "url": "..."}]），
Alf 会按设置里勾选的新闻主题来填它。
"""
from __future__ import annotations

import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

ATOM = "{http://www.w3.org/2005/Atom}"
USER_AGENT = "Mozilla/5.0 (Alf personal assistant; RSS reader)"
server = MCPServer("alf-news")


def parse_feed(xml_bytes: bytes, limit: int) -> list[dict]:
    """RSS 2.0 和 Atom 都认。标题里的换行和多余空格压成一个空格。"""
    root = ET.fromstring(xml_bytes)
    entries = root.findall(".//item") or root.findall(f".//{ATOM}entry")
    out = []
    for e in entries[:limit]:
        title = e.findtext("title") or e.findtext(f"{ATOM}title") or ""
        when = e.findtext("pubDate") or e.findtext(f"{ATOM}updated") or e.findtext(f"{ATOM}published") or ""
        title = " ".join(title.split())
        if title:
            out.append({"title": title[:160], "published": when.strip()})
    return out


def fetch(feed: dict, limit: int) -> dict:
    try:
        request = urllib.request.Request(feed["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=12) as r:
            items = parse_feed(r.read(2_000_000), limit)
        return {"topic": feed.get("topic", ""), "source": feed.get("source", feed["url"]), "items": items}
    except Exception as e:   # 某个源挂了不影响其他源
        return {"topic": feed.get("topic", ""), "source": feed.get("source", feed.get("url", "?")), "items": [],
                "error": f"{type(e).__name__}: {e}"[:120]}


def collect(feeds: list[dict], limit: int) -> dict:
    with ThreadPoolExecutor(max_workers=6) as pool:
        return {"feeds": list(pool.map(lambda f: fetch(f, limit), feeds))}


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def headlines(per_feed: int = 6) -> dict:
    """各个新闻源最新的几条标题，按主题分组。"""
    try:
        feeds = json.loads(os.getenv("ALF_NEWS_FEEDS", "[]"))
    except ValueError:
        feeds = []
    return collect([f for f in feeds if isinstance(f, dict) and f.get("url")], max(1, min(per_feed, 15)))


if __name__ == "__main__":
    server.run()
