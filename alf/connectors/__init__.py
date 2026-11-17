"""连接器：把外部世界的新消息（邮件，以后是日历、课程通知……）拉进 Alf。

接一个新来源只需要写一个类，实现下面的 Connector 协议，然后加进 build()。
去重、分诊（哪些值得打扰你）、通知、进度保存都由心跳里的 inbox 任务统一处理，
连接器只管一件事：从上次的位置往后，有哪些新东西。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import Settings


@dataclass
class Item:
    external_id: str          # 来源内唯一，用来去重
    sender: str
    title: str
    snippet: str              # 正文开头一小段，分诊用
    received_at: int | None   # UTC 秒
    body: str = ""            # 正文开头较长的一段：写摘要、起草回信时用
    reply_to: str = ""        # 回信该寄到的地址
    message_id: str = ""
    references: str = ""      # 回信的 References 头：原信的 References 加上它自己的 Message-ID


def sender_name(sender: str | None, limit: int = 16) -> str:
    """发件人的名字：有名字用名字，没有就用地址 @ 前面那段。"""
    sender = sender or ""
    name = sender.split("<")[0].strip().strip('"')
    return (name or sender.strip("<>").split("@")[0])[:limit] or "（未知发件人）"


class ConnectorError(RuntimeError):
    """连接器这次没拉成功。消息是给用户看的。"""


class Connector(Protocol):
    name: str      # 任务名，也是 inbox_items.source
    kind: str      # 时间线上的消息类型
    label: str
    every: float   # 多久拉一次（秒）

    def fetch_new(self, cursor: str | None) -> tuple[list[Item], str | None]:
        """返回 (上次游标之后的新消息, 新游标)。

        cursor 是连接器自己定义的进度标记，框架只负责原样存取。第一次调用时 cursor 为
        None：应当只返回"从现在开始"的游标、不返回历史消息，免得一接入就把整个收件箱
        当成新邮件播报一遍。
        """
        ...


def build(cfg: Settings) -> list[Connector]:
    """按配置实例化已启用的连接器。没配置的来源不会出现在这里。"""
    from .email_imap import ImapEmail

    return [ImapEmail(mailbox) for mailbox in cfg.mailboxes]
