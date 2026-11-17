"""MCP 服务器：本机「日历」里今天和接下来几天的日程。实际读取见 alf/apple.py。"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .. import apple

server = MCPServer("alf-calendar")
NEEDS_PERMISSION = apple.NEEDS_PERMISSION["event"]


def read_events(days: int, helper=None, run=None, ask: bool = False) -> dict:
    return apple.read_events(days, ask=ask, helper=helper, run=run)


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def calendar_events(days: int = 2) -> dict:
    """本机日历里从今天起几天内的日程：标题、起止时间、是否全天、所在日历、地点。"""
    return apple.read_events(max(1, min(days, 14)))


if __name__ == "__main__":
    server.run()
