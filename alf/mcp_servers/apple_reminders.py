"""MCP 服务器：本机「提醒事项」里近几天到期、还没完成的条目。实际读取见 alf/apple.py。"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .. import apple

server = MCPServer("alf-apple-reminders")


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def apple_reminders(days: int = 7) -> dict:
    """「提醒事项」App 里从现在起几天内到期（含已经逾期）且还没完成的条目：标题、到期时间、所在列表。"""
    return apple.read_reminders(max(1, min(days, 14)))


if __name__ == "__main__":
    server.run()
