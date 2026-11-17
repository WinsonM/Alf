"""通过 MCP 汇总"现在的情况"：今天什么日子、日历上有什么、有什么新闻。

Alf 在这里是 MCP 客户端。对每个配置的 MCP 服务器，连上去、列出工具，然后把其中"只读且不需要参数"的
工具各调用一次，结果存成一份快照。主动判断和早报读的是快照，不在做决定的那一刻去联网。

用"只读 + 无参数"这个约定，是为了不需要模型来决定调哪个工具：便宜、可预测，而且外部内容（新闻标题、
日程名）不会有机会指挥 Alf 去调用别的工具。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

from mcp import Client, StdioServerParameters

from .config import ROOT, Settings

log = logging.getLogger("alf.sources")

BUILTIN = {"days": "alf.mcp_servers.days", "calendar": "alf.mcp_servers.calendar",
           "reminders": "alf.mcp_servers.apple_reminders", "news": "alf.mcp_servers.news"}
LABELS = {"days": "特殊日子", "calendar": "日历", "reminders": "提醒事项", "news": "新闻"}
SERVER_TIMEOUT = 120      # 一个服务器从连接到取完的总时限（读日历可能要等用户点授权）
MAX_RESULT_CHARS = 6000

# 设置里的新闻主题 -> RSS 源。2026-11-16 逐个验证过能取到内容。
FEEDS = {
    "科技与 AI": [("Solidot", "https://www.solidot.org/index.rss"), ("Hacker News", "https://hnrss.org/frontpage?count=10")],
    "财经与市场": [("FT中文网", "https://www.ftchinese.com/rss/news"), ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html")],
    "国内要闻": [("中新网", "https://www.chinanews.com.cn/rss/scroll-news.xml"), ("BBC 中文", "https://feeds.bbci.co.uk/zhongwen/simp/rss.xml")],
    "澳洲本地": [("ABC News", "https://www.abc.net.au/news/feed/2942460/rss.xml")],
}


def servers_for(cfg: Settings, current: dict) -> dict[str, StdioServerParameters]:
    """按设置决定连哪些 MCP 服务器：自带的几个各有开关，另外可以在 mcp.servers 里加第三方的。"""
    feeds = [{"topic": topic, "source": name, "url": url}
             for topic in current.get("news.topics", []) for name, url in FEEDS.get(topic, [])]
    env = {**os.environ, "ALF_TZ": cfg.tz.key, "ALF_NEWS_FEEDS": json.dumps(feeds, ensure_ascii=False)}
    out = {}
    for name, module in BUILTIN.items():
        if current.get(f"sources.{name}", True) and (name != "news" or feeds):
            out[name] = StdioServerParameters(command=sys.executable, args=["-m", module], env=env, cwd=str(ROOT))
    for extra in current.get("mcp.servers", []):
        out[extra["name"]] = StdioServerParameters(command=extra["command"], args=list(extra.get("args", [])),
                                                   env={**os.environ, **extra.get("env", {})})
    return out


def is_snapshot_tool(tool: Any) -> bool:
    """只读、且一个必填参数都没有的工具，才会在汇总时被自动调用。"""
    read_only = bool(tool.annotations and tool.annotations.read_only_hint)
    return read_only and not (tool.input_schema or {}).get("required")


def result_value(result: Any) -> Any:
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    text = "\n".join(block.text for block in result.content if getattr(block, "text", None))
    try:
        return json.loads(text)
    except ValueError:
        return text


async def _ask(name: str, params: StdioServerParameters) -> dict:
    found = {}
    async with Client(params) as client:
        for tool in (await client.list_tools()).tools:
            if not is_snapshot_tool(tool):
                continue
            result = await client.call_tool(tool.name, {})
            value = result_value(result)
            if getattr(result, "is_error", False):
                found[tool.name] = {"error": str(value)[:300]}
            elif len(json.dumps(value, ensure_ascii=False)) > MAX_RESULT_CHARS:
                found[tool.name] = {"truncated": json.dumps(value, ensure_ascii=False)[:MAX_RESULT_CHARS]}
            else:
                found[tool.name] = value
    return found


async def _collect(servers: dict[str, StdioServerParameters]) -> dict:
    async def one(name: str, params: StdioServerParameters) -> tuple[str, dict]:
        try:
            return name, await asyncio.wait_for(_ask(name, params), SERVER_TIMEOUT)
        except Exception as e:   # 一个来源出问题（没装、超时、协议不对）不影响其他来源
            log.warning("来源 %s 没取到：%s: %s", name, type(e).__name__, e)
            return name, {"error": f"{type(e).__name__}: {e}"[:300] or "超时"}

    return dict(await asyncio.gather(*(one(n, p) for n, p in servers.items())))


def gather(cfg: Settings, current: dict) -> dict:
    """取一份快照：{"at": 时间, "sources": {服务器名: {工具名: 结果}}}。"""
    servers = servers_for(cfg, current)
    return {"at": int(time.time()), "sources": asyncio.run(_collect(servers)) if servers else {}}
