"""iMessage 渠道：借这台 Mac 的「信息」App 收发消息。

收：只读地查「信息」的本机数据库（~/Library/Messages/chat.db）里有没有新消息。
发：用 AppleScript 让「信息」App 发出去。

这不是苹果提供的机器人接口，所以有三条硬性的保护：
- 只处理配置里指定的地址（ALF_IMESSAGE_OWNER）在一对一聊天里发来的消息。别人就算知道 Alf 的
  账号也指挥不了它，他们的消息根本不会交给模型。
- 发送有频率上限，并且会丢掉"自己刚发出去又收回来"的消息，防止配置错了以后自己跟自己聊个不停。
- 读数据库用只读模式，不会改动「信息」的任何数据。

需要两项系统权限，都得用户自己在「系统设置 → 隐私与安全性」里给运行 Alf 的 python：
「完全磁盘访问权限」（读数据库）和「自动化 → 信息」（发消息）。
"""
from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import asr, db, feedback, lessons, media, mood
from .agent import Quoted, Reply, Turn
from .config import Settings, normalize_handle
from .heartbeat import JobState, load_state, save_state

log = logging.getLogger("alf.imessage")

CHANNEL = "imessage"
ONE_TO_ONE = 45                  # chat.style：45 是一对一，43 是群聊
MAX_SENDS_PER_MINUTE = 20        # 超过就暂停发送：正常聊天到不了这个数，到了多半是出了循环
PAUSE_AFTER_FLOOD = 300
STALE_AFTER = 6 * 3600           # 积压超过这么久还没发出去的消息直接放弃，免得恢复后一口气补发一堆过期提醒
ECHO_WINDOW = 120                # 这么多秒内收到和自己刚发的一模一样的内容，当作回声丢掉
OUTGOING_PREFIX = {"reminder": "⏰ 提醒：", "briefing": "☀️ ", "digest": "📝 ", "email": "📧 ", "calendar": "📅 ", "followup": ""}
UNSUPPORTED = "这种附件我还看不了，目前能看图片、能听语音。"
MAX_IMAGES = 3
QUOTED_CHARS = 300               # 被引用的那条消息最多带这么长给模型看
ATTACHMENT_WAIT = 6.0            # 消息先到、附件后到：最多等这么多秒让它下载完

SEND_SCRIPT = """
on run {targetHandle, messageText}
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        send messageText to participant targetHandle of targetService
    end tell
end run
"""
COUNT_ACCOUNTS_SCRIPT = 'tell application "Messages" to return count of (accounts whose service type = iMessage)'


class ChannelError(RuntimeError):
    """渠道这次没成功。消息是给用户看的。"""


@dataclass
class Attachment:
    path: Path | None          # None = 数据库里没给路径
    kind: str | None           # "audio" | "image" | None（视频、文件……）


@dataclass
class Incoming:
    rowid: int
    text: str
    handle: str
    has_attachments: bool
    attachments: list[Attachment] = field(default_factory=list)
    quoted: str | None = None      # 这条是引用回复（长按某条消息点「回复」）时，被回复的那条的原文


@dataclass
class Reaction:
    """主人对某条消息点了 👍👎 之类的回应。target 是被点的那条消息的原文。"""
    rowid: int
    kind: int
    target: str


def decode_attributed_body(blob: bytes | None) -> str | None:
    """从「信息」存的富文本归档里取出纯文本。

    新版 macOS 常把正文只存在 attributedBody 里（text 列是空的）。它是 NSArchiver 的 typedstream
    格式：在类名 NSString 之后有一个字节串标记 `84 01 2b`，接着是长度和 UTF-8 正文。长度不超过
    127 时占一个字节；更长时是 0x81 加 2 字节，或 0x82 加 4 字节（小端）。
    """
    if not blob:
        return None
    i = blob.find(b"NSString")
    if i < 0:
        return None
    i = blob.find(b"\x84\x01+", i)
    if i < 0:
        return None
    i += 3
    if i >= len(blob):
        return None
    n = blob[i]
    if n == 0x81:
        n, i = int.from_bytes(blob[i + 1:i + 3], "little"), i + 3
    elif n == 0x82:
        n, i = int.from_bytes(blob[i + 1:i + 5], "little"), i + 5
    else:
        i += 1
    return blob[i:i + n].decode("utf-8", "replace") or None


def _apple_time(value) -> int | None:
    """「信息」里的时间从 2001 年起算；消息表用纳秒，附件表用秒。换成通常的时间戳。"""
    if not value:
        return None
    return int((value / 1e9 if value > 1e12 else value) + 978307200)


class ChatDB:
    """「信息」数据库的只读视图。每次查询新开一个连接：它在不停地被「信息」写入，长连接容易读到旧快照。"""

    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=3)
            conn.row_factory = sqlite3.Row
            conn.execute("SELECT 1 FROM message LIMIT 1")
            return conn
        except sqlite3.Error as e:
            raise ChannelError(self._why_unreadable(e)) from e

    def _why_unreadable(self, error: sqlite3.Error) -> str:
        """SQLite 只会说"打不开"。直接去开一下文件，看系统给的真实原因，免得把所有失败都说成权限问题。"""
        try:
            with open(self.path, "rb") as f:
                f.read(16)
        except FileNotFoundError:
            return f"找不到「信息」数据库 {self.path}：这台 Mac 的「信息」App 还没登录过"
        except PermissionError:
            # macOS 的隐私保护拦下来的就是这个错误（Operation not permitted）。
            return ("读不了「信息」数据库：需要在 系统设置 → 隐私与安全性 → 完全磁盘访问权限 里加上运行 Alf 的 python"
                    f"（{os.path.realpath(sys.executable)}），加完后重启服务")
        except OSError as e:
            return f"读不了「信息」数据库：{e}"
        return f"「信息」数据库能打开，但读取失败：{error}"

    def max_rowid(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message").fetchone()[0]
        finally:
            conn.close()

    def new_messages(self, after: int, owners: tuple[str, ...], reactions: list[Reaction] | None = None) -> tuple[list[Incoming], int]:
        """ROWID 大于 after 的、主人在一对一聊天里发来的消息，以及这次看到的最大 ROWID。
        给了 reactions 列表的话，主人这段时间点的 👍👎 会放进去。"""
        conn = self._connect()
        try:
            # 不同版本的 macOS 列不完全一样；没有的列当作空值，不让整个查询失败。
            have = {r["name"] for r in conn.execute("PRAGMA table_info(message)")}
            extra = "".join(f", {'m.' + c if c in have else 'NULL'} AS {c}"
                            for c in ("is_audio_message", "associated_message_guid", "thread_originator_guid"))
            rows = conn.execute(
                "SELECT m.ROWID AS rowid, m.text, m.attributedBody, m.is_from_me, m.cache_has_attachments,"
                f"       m.associated_message_type, m.item_type, h.id AS handle, c.style{extra}"
                " FROM message m"
                " LEFT JOIN handle h ON h.ROWID = m.handle_id"
                " LEFT JOIN chat_message_join j ON j.message_id = m.ROWID"
                " LEFT JOIN chat c ON c.ROWID = j.chat_id"
                " WHERE m.ROWID > ? ORDER BY m.ROWID", (after,)).fetchall()
            out, last = [], after
            for r in rows:
                last = max(last, r["rowid"])
                if r["is_from_me"] or not r["handle"] or normalize_handle(r["handle"]) not in owners:
                    continue
                if r["style"] not in (None, ONE_TO_ONE):
                    continue                                  # 群聊里的话不是对 Alf 说的
                if r["associated_message_type"]:
                    # 点赞之类的回应，不是正文。记下他点的是哪条消息，用来统计"主动消息有没有用"。
                    if reactions is not None and (target := self._reacted_to(conn, r["associated_message_guid"])):
                        reactions.append(Reaction(r["rowid"], r["associated_message_type"], target))
                    continue
                if r["item_type"]:
                    continue                                  # 群成员变动等系统事件
                text = (r["text"] or decode_attributed_body(r["attributedBody"]) or "").replace("￼", "").strip()
                has = bool(r["cache_has_attachments"])
                # 引用回复：thread_originator_guid 指向被回复的那条消息（macOS 11 起有这一列）。
                # reply_to_guid 不能用——它在几乎每条消息上都有，指的只是"上一条"。
                out.append(Incoming(r["rowid"], text, r["handle"], has,
                                    self._attachments(conn, r["rowid"], bool(r["is_audio_message"])) if has else [],
                                    quoted=self._text_of(conn, r["thread_originator_guid"])))
        except sqlite3.Error as e:
            raise ChannelError(f"查询「信息」数据库失败：{e}") from e
        finally:
            conn.close()
        return out, last

    def recent(self, owners: tuple[str, ...], limit: int = 8) -> list[dict]:
        """最近几条消息的"形态"，用来查"这条为什么没回"：谁发的、是不是一对一、带了什么附件、会被怎么处理。
        只有形态，没有内容——不返回正文，也不返回对方的地址。"""
        conn = self._connect()
        try:
            have = {r["name"] for r in conn.execute("PRAGMA table_info(message)")}
            a_have = {r["name"] for r in conn.execute("PRAGMA table_info(attachment)")}
            cols = "".join(f", {'m.' + c if c in have else 'NULL'} AS {c}"
                           for c in ("is_audio_message", "service", "date", "balloon_bundle_id", "thread_originator_guid"))
            rows = conn.execute(
                "SELECT m.ROWID AS rowid, m.text, m.attributedBody, m.is_from_me, m.cache_has_attachments,"
                f"       m.associated_message_type, m.item_type, h.id AS handle, c.style{cols}"
                " FROM message m LEFT JOIN handle h ON h.ROWID = m.handle_id"
                " LEFT JOIN chat_message_join j ON j.message_id = m.ROWID LEFT JOIN chat c ON c.ROWID = j.chat_id"
                " ORDER BY m.ROWID DESC LIMIT ?", (limit,)).fetchall()
            out = []
            for r in rows:
                mine = bool(r["handle"]) and normalize_handle(r["handle"]) in owners
                verdict = ("跳过：是这台 Mac 自己发出的" if r["is_from_me"] else
                           "跳过：发信地址不在「主人」名单里" if not mine else
                           "跳过：不是一对一聊天" if r["style"] not in (None, ONE_TO_ONE) else
                           f"跳过：是对某条消息的回应（类型 {r['associated_message_type']}）" if r["associated_message_type"] else
                           f"跳过：系统事件（类型 {r['item_type']}）" if r["item_type"] else "会处理")
                text = (r["text"] or decode_attributed_body(r["attributedBody"]) or "").replace("￼", "").strip()
                extra = "".join(f", {'a.' + c if c in a_have else 'NULL'} AS {c}" for c in ("uti", "transfer_state", "total_bytes", "created_date", "start_date"))
                files = []
                for a in conn.execute(f"SELECT a.filename, a.mime_type{extra} FROM attachment a JOIN message_attachment_join j"
                                      " ON j.attachment_id = a.ROWID WHERE j.message_id = ? ORDER BY a.ROWID", (r["rowid"],)):
                    path = Path(a["filename"]).expanduser() if a["filename"] else None
                    files.append({"kind": media.kind_of(path, a["mime_type"] or "", bool(r["is_audio_message"])),
                                  "mime": a["mime_type"], "uti": a["uti"], "ext": path.suffix.lower() if path else None,
                                  "transfer_state": a["transfer_state"], "bytes": a["total_bytes"],
                                  # 这台 Mac 上建出这条附件记录的时间：和消息的发出时间差得多，说明是送到这台 Mac 晚了
                                  "created_here": _apple_time(a["created_date"]), "download_started": _apple_time(a["start_date"]),
                                  "on_disk": bool(path and path.is_file() and path.stat().st_size > 0)})
                out.append({"rowid": r["rowid"], "at": _apple_time(r["date"]), "from_me": bool(r["is_from_me"]), "from_owner": mine,
                            "address_kind": None if not r["handle"] else "邮箱" if "@" in r["handle"] else "电话",
                            "service": r["service"], "one_to_one": r["style"] in (None, ONE_TO_ONE),
                            "associated_message_type": r["associated_message_type"], "item_type": r["item_type"],
                            "app": r["balloon_bundle_id"], "text_chars": len(text),
                            # 是不是引用回复，以及被引用的那条在库里找不找得到（找不到的话 Alf 只当普通消息处理）
                            "is_reply": bool(r["thread_originator_guid"]),
                            "quoted_found": bool(self._text_of(conn, r["thread_originator_guid"])) if r["thread_originator_guid"] else None,
                            "has_attachments": bool(r["cache_has_attachments"]), "attachments": files, "verdict": verdict})
            return out
        except sqlite3.Error as e:
            raise ChannelError(f"查询「信息」数据库失败：{e}") from e
        finally:
            conn.close()

    @staticmethod
    def _attachments(conn: sqlite3.Connection, rowid: int, audio_message: bool) -> list[Attachment]:
        try:
            rows = conn.execute("SELECT a.filename, a.mime_type FROM attachment a JOIN message_attachment_join j ON j.attachment_id = a.ROWID"
                                " WHERE j.message_id = ? ORDER BY a.ROWID", (rowid,)).fetchall()
        except sqlite3.Error:
            return []
        out = []
        for r in rows:
            path = Path(r["filename"]).expanduser() if r["filename"] else None
            out.append(Attachment(path, media.kind_of(path, r["mime_type"] or "", audio_message)))
        return out

    @staticmethod
    def _reacted_to(conn: sqlite3.Connection, reference: str | None) -> str | None:
        """回应指向的那条消息的原文。引用的写法是 "p:0/消息标识" 或 "bp:消息标识"。"""
        if not reference:
            return None
        return ChatDB._text_of(conn, reference.rpartition("/")[2].removeprefix("bp:"))

    @staticmethod
    def _text_of(conn: sqlite3.Connection, guid: str | None) -> str | None:
        """某条消息的原文（按消息标识找）。找不到、或者那条没有文字，返回 None。"""
        if not guid:
            return None
        try:
            row = conn.execute("SELECT text, attributedBody FROM message WHERE guid = ?", (guid,)).fetchone()
        except sqlite3.Error:
            return None
        return ((row["text"] or decode_attributed_body(row["attributedBody"]) or "").replace("￼", "").strip() or None) if row else None


def send(handle: str, text: str) -> None:
    """让「信息」App 发一条 iMessage。地址和正文作为参数传给脚本，不拼进脚本文本里。"""
    try:
        r = subprocess.run(["osascript", "-e", SEND_SCRIPT, handle, text], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise ChannelError("「信息」App 没有响应") from e
    except OSError as e:
        raise ChannelError(f"运行不了 osascript：{e}") from e
    if r.returncode != 0:
        raise ChannelError(explain_osascript(r.stderr))


def explain_osascript(stderr: str) -> str:
    err = " ".join(stderr.split())
    if "-1743" in err or "Not authorized" in err:
        return "没有控制「信息」App 的权限：系统设置 → 隐私与安全性 → 自动化，给 python 勾上「信息」"
    if "-1728" in err or "Can’t get" in err or "Can't get" in err:
        return f"「信息」App 发不到这个地址，确认它已登录 Alf 的 Apple ID、地址写对了（{err[:120]}）"
    if "-600" in err or "isn’t running" in err or "isn't running" in err:
        return "「信息」App 没有运行，而且启动不了"
    return f"发送失败：{err[:200] or '未知错误'}"


def imessage_account_count() -> int:
    """「信息」App 里登录了几个 iMessage 账号。第一次调用会触发系统的"是否允许控制「信息」"弹窗。"""
    try:
        r = subprocess.run(["osascript", "-e", COUNT_ACCOUNTS_SCRIPT], capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as e:
        raise ChannelError(f"问不到「信息」App：{e}") from e
    if r.returncode != 0:
        raise ChannelError(explain_osascript(r.stderr))
    return int(r.stdout.strip() or 0)


def format_outgoing(kind: str, text: str) -> str:
    return OUTGOING_PREFIX.get(kind, "") + text


class IMessageChannel(threading.Thread):
    """一个线程，循环做两件事：把主人新发来的消息交给 Alf；把 Alf 要发给主人的消息送出去。

    要发的消息不是直接调用发送，而是 episodes 表里 channel='imessage' 且还没送达的行。这样聊天回复、
    到点的提醒、每晚的汇总都走同一条路，按顺序发，发失败了下一轮自动重试，进程重启也不会丢。
    """

    def __init__(self, cfg: Settings, handle: Callable[[Turn, Callable[[Reply], None]], Reply],
                 sender: Callable[[str, str], None] = send, chatdb: ChatDB | None = None,
                 clock: Callable[[], float] = time.time, recognizer=None):
        super().__init__(daemon=True, name="alf-imessage")
        assert cfg.imessage is not None
        self.cfg, self.im, self.handle, self.sender, self.clock = cfg, cfg.imessage, handle, sender, clock
        self.recognizer = recognizer       # 语音识别；None = 听不了语音
        self.chatdb = chatdb or ChatDB(self.im.db_path)
        self.status: dict = {"ok": None, "error": None, "last_inbound": None, "last_outbound": None, "last_poll": None}
        self._halt = threading.Event()
        self._conn: sqlite3.Connection | None = None
        self._sent_at: deque[float] = deque()
        self._recent_out: deque[tuple[float, str]] = deque(maxlen=50)
        self._retry_at = 0.0
        self._failures = 0

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                self.step()
            except Exception:
                log.exception("iMessage 这一轮出错，下一轮继续")
            self._halt.wait(self.im.poll)

    def stop(self) -> None:
        self._halt.set()

    def step(self) -> None:
        if self._conn is None:
            self._conn = db.connect(self.cfg.db_path)
        if self.clock() < self._retry_at:
            return
        self.status["last_poll"] = int(self.clock())      # 上一次去看「信息」数据库的时间：用来分清"没收到"和"没去看"
        try:
            self._inbound()
            self._outbound()
        except ChannelError as e:
            # 多半是权限没给或「信息」没登录：不是等一秒就能好的事，所以越等越久，别刷日志。
            self._failures += 1
            self._retry_at = max(self._retry_at, self.clock() + min(5 * 2 ** self._failures, 300))
            if self.status["error"] != str(e):
                log.warning("iMessage：%s", e)
            self.status.update(ok=False, error=str(e))
        else:
            self._failures = 0
            self.status.update(ok=True, error=None)

    # --- 收 -----------------------------------------------------------------------

    def _inbound(self) -> None:
        state = load_state(self._conn, CHANNEL)
        if state.cursor is None:
            # 第一次运行：从现在开始，不处理以前的聊天记录。
            self._save_cursor(state, self.chatdb.max_rowid())
            return
        reactions: list[Reaction] = []
        messages, last = self.chatdb.new_messages(int(state.cursor), self.im.owners, reactions)
        for r in reactions:
            self._react(r)
        for m in messages:
            # 先推进游标再处理：处理到一半崩了，宁可漏掉这一条，也不要重启后把"建提醒"再执行一遍。
            self._save_cursor(state, m.rowid)
            if m.text and self._is_echo(m.text):
                continue
            self.status["last_inbound"] = int(self.clock())
            turn = self._turn_for(m)
            if turn is None:
                continue
            if isinstance(turn, str):       # 这条处理不了：直接回一句说明
                self._queue(turn)
                self._outbound()
                continue
            turn.reply_to = self._quoted(m.quoted)
            failed: list[ChannelError] = []

            def deliver(reply: Reply) -> None:
                # 回复一写好就先发出去。发不出去也让 Alf 把这一轮做完（整理记忆），回头再报错重试。
                try:
                    self._outbound()
                except ChannelError as e:
                    failed.append(e)

            try:
                self.handle(turn, deliver)
            except Exception:
                log.exception("处理消息时出错")
                self._queue("我刚才出了点问题，没处理好这条消息，再发一次试试？")
            if failed:
                raise failed[0]
        if last > int(state.cursor):
            self._save_cursor(state, last)

    # --- 语音、图片、点按回应 ---------------------------------------------------------

    def _turn_for(self, m: Incoming) -> Turn | str | None:
        """把收到的一条消息变成交给 Alf 的一轮对话。处理不了时返回一句直接回给用户的话；什么都没有就返回 None。"""
        audio = [a for a in m.attachments if a.kind == "audio"]
        images = [a for a in m.attachments if a.kind == "image"]
        try:
            if audio:
                return self._voice(m, audio[0])
            if images:
                return self._pictures(m, images[:MAX_IMAGES])
        except media.MediaError as e:
            return f"{e}，换一种方式发给我试试？"
        except OSError as e:
            log.warning("读附件失败：%s", e)
            return "这个附件我读不出来，再发一次试试？"
        if m.text:
            return Turn(m.text, channel=CHANNEL)
        return UNSUPPORTED if m.has_attachments else None

    def _fetch(self, a: Attachment) -> Path | None:
        """等附件下载完。只读「信息」自己的附件目录里的文件——数据库里的路径不会被拿去读别处的东西。"""
        if a.path is None:
            return None
        root = self.im.attachments_root.resolve()
        path = a.path.resolve()
        if root not in path.parents:
            log.warning("附件不在「信息」的附件目录里，没有读：%s", a.path.name)
            return None
        deadline = time.monotonic() + ATTACHMENT_WAIT
        while not (path.is_file() and path.stat().st_size > 0):
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.25)
        return path

    def _voice(self, m: Incoming, a: Attachment) -> Turn | str:
        if self.recognizer is None:
            return "语音我现在听不了，先打字跟我说吧。"
        src = self._fetch(a)
        if src is None:
            return "这条语音还没下载到这台 Mac 上，过几秒再发一次？"
        attachment_id, stored = media.store(self._conn, src, self.cfg.attachment_dir, "audio")
        if not self.recognizer.cached():
            # 第一次用要下载识别模型，可能要几分钟。先说一声，免得他以为没收到。
            self._queue("收到语音。第一次听要先下载识别模型（几百 MB），可能要等几分钟。")
            try:
                self._outbound()
            except ChannelError:
                pass
        try:
            heard = self.recognizer.transcribe(media.to_wav(stored), asr.vocabulary(self._conn))
        except Exception as e:
            log.exception("语音识别失败")
            return f"这条语音没识别成（{type(e).__name__}），先打字跟我说吧。"
        asr.save_transcript(self._conn, attachment_id, heard)
        if not heard.text:
            return "这条语音我没听清，再说一遍或者打字？"
        return Turn(" ".join(filter(None, [heard.text, m.text])), source="voice", attachment_id=attachment_id,
                    asr_confidence=heard.confidence, channel=CHANNEL)

    def _pictures(self, m: Incoming, attachments: list[Attachment]) -> Turn | str:
        views, first = [], None
        for a in attachments:
            src = self._fetch(a)
            if src is None:
                continue
            attachment_id, stored = media.store(self._conn, src, self.cfg.attachment_dir, "image")
            views.append(media.to_jpeg(stored))
            first = first or attachment_id
        if not views:
            return "图片还没下载到这台 Mac 上，过几秒再发一次？"
        return Turn(m.text, attachment_id=first, channel=CHANNEL, images=views)

    def _sent_episode(self, text: str) -> sqlite3.Row | None:
        """「信息」里的一条消息原文 → Alf 在时间线里发的哪一条。对不上（太久以前的、不是 Alf 发的）返回 None。"""
        rows = self._conn.execute("SELECT id, kind, text FROM episodes WHERE role = 'alf' AND channel = ? AND delivered_at > 0"
                                  " ORDER BY id DESC LIMIT 200", (CHANNEL,)).fetchall()
        return next((row for row in rows if format_outgoing(row["kind"], row["text"]).strip() == text.strip()), None)

    def _quoted(self, text: str | None) -> Quoted | None:
        """他引用回复的是哪一条。是 Alf 发的就对回时间线里的编号（这样才知道说的是哪封邮件、哪条提醒）；
        对不上的，多半是他引用了自己的话，或者太久以前的消息——那就只带原文。"""
        if not text:
            return None
        row = self._sent_episode(text)
        if row:
            return Quoted(text[:QUOTED_CHARS], row["id"], row["kind"], mine=True)
        own = self._conn.execute("SELECT 1 FROM episodes WHERE role = 'user' AND channel = ? AND text = ? ORDER BY id DESC LIMIT 1",
                                 (CHANNEL, text.strip())).fetchone()
        return Quoted(text[:QUOTED_CHARS], mine=False if own else None)

    def _react(self, r: Reaction) -> None:
        """他点的是 Alf 发的哪一条：按原文对回去，记下评价。对不上（点的是他自己的话、太久以前的）就算了。"""
        row = self._sent_episode(r.target)
        episode = row["id"] if row else None
        outcome = feedback.from_tapback(self._conn, r.kind, episode, int(self.clock()))
        if outcome:
            log.info("收到评价：%s（消息 #%s）", outcome, episode)
            # 👎 也是一个"这种别再发"的例子；👍 或者把 👎 撤了，那个例子就不算了。
            lessons.note_rating(self._conn, episode, -1 if outcome == "down" else None, "👎", int(self.clock()))
        if outcome == "down":      # 👎 留个疙瘩，接下来两三天说话收着点；👍 消掉一半
            mood.strain(self._conn, int(self.clock()), "对一条消息点了 👎")
        elif outcome == "up":
            mood.ease(self._conn, int(self.clock()))

    def _save_cursor(self, state: JobState, rowid: int) -> None:
        state.cursor = str(rowid)
        save_state(self._conn, state)

    def _is_echo(self, text: str) -> bool:
        now = self.clock()
        return any(now - at < ECHO_WINDOW and sent == text for at, sent in self._recent_out)

    def _queue(self, text: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO episodes(session_id, role, text, created_at, channel) VALUES ('system', 'alf', ?, ?, ?)",
                (text, int(self.clock()), CHANNEL))

    # --- 发 -----------------------------------------------------------------------

    def _outbound(self) -> None:
        now = self.clock()
        rows = self._conn.execute(
            "SELECT id, kind, text, created_at FROM episodes WHERE role = 'alf' AND channel = ? AND delivered_at IS NULL"
            " ORDER BY id LIMIT 20", (CHANNEL,)).fetchall()
        for r in rows:
            if now - r["created_at"] > STALE_AFTER:
                self._mark(r["id"], 0)
                log.warning("消息 #%d 积压太久，放弃发送", r["id"])
                continue
            while self._sent_at and now - self._sent_at[0] > 60:
                self._sent_at.popleft()
            if len(self._sent_at) >= MAX_SENDS_PER_MINUTE:
                self._retry_at = now + PAUSE_AFTER_FLOOD
                raise ChannelError(f"一分钟内要发的消息超过 {MAX_SENDS_PER_MINUTE} 条，暂停 5 分钟（怕是出了循环）")
            text = format_outgoing(r["kind"], r["text"])
            try:
                self.sender(self.im.owners[0], text)
            except ChannelError:
                with self._conn:
                    self._conn.execute("UPDATE episodes SET send_attempts = send_attempts + 1 WHERE id = ?", (r["id"],))
                raise   # 保持顺序：这条没发出去，后面的也先不发
            self._mark(r["id"], int(now))
            self._sent_at.append(now)
            self._recent_out.append((now, text))
            self.status["last_outbound"] = int(now)

    def _mark(self, episode_id: int, delivered_at: int) -> None:
        with self._conn:
            self._conn.execute("UPDATE episodes SET delivered_at = ? WHERE id = ?", (delivered_at, episode_id))
