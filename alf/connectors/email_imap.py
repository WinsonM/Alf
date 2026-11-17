"""邮箱连接器（IMAP）。

只读：用 EXAMINE 打开邮箱、用 BODY.PEEK 取信，所以不会把邮件标成已读，也不会
移动或删除任何东西。回信走另一条路（email_smtp.py），而且要用户确认。

适用于给 IMAP 提供"应用专用密码 / 授权码"的邮箱：Gmail、iCloud、QQ、163 等。
微软 365（多数学校和公司邮箱）已经停用密码登录 IMAP，这个连接器连不上，见 README。
"""
from __future__ import annotations

import email
import email.policy
import html
import imaplib
import re
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parseaddr, parsedate_to_datetime

from ..config import EmailConfig
from . import ConnectorError, Item

FETCH_BYTES = 65536   # 每封只取开头 64KB：够拿到信头和正文开头，又不会把几十 MB 的附件拉下来
SNIPPET_CHARS = 500
BODY_CHARS = 4000
# 网易（163/126）要求客户端先用 ID 命令报上名字，否则打开收件箱时报 "Unsafe Login"。别家收到这条命令也无妨。
CLIENT_ID = '("name" "Alf" "version" "1.0" "vendor" "personal assistant")'
# "编号比上次大"不等于"新邮件"。QQ 邮箱第一次被 IMAP 访问时才开始给邮件编号，而且是边访问边补：
# 2026-11-17 接入后五分钟，它又给一千多封 2020–2022 年的旧信排了新编号，Alf 把其中 30 封当新邮件报了。
# 所以再加一条：服务器收到这封信的时间（INTERNALDATE）得在 Alf 接入之后。差这么多秒以内算时钟误差。
OLD_MAIL_SLACK = 600
LEGACY_LOOKBACK = 86400   # 早先存下的游标里没记接入时间：只把最近一天内收到的当新邮件
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")   # 不用 strftime：它跟着系统语言走
# QQ 的日期不补位（"1-Jul-2022"），标准库的 Internaldate2tuple 认不出来，所以自己解析。
INTERNALDATE = re.compile(rb'INTERNALDATE "\s*(\d{1,2})-([A-Za-z]{3})-(\d{4}) (\d{2}):(\d{2}):(\d{2}) ([+-])(\d{2})(\d{2})"')


class ImapEmail:
    kind = "email"

    def __init__(self, cfg: EmailConfig, clock=time.time):
        self.cfg, self.clock = cfg, clock
        self.name, self.label, self.every = cfg.name, cfg.label, cfg.every

    def fetch_new(self, cursor: str | None) -> tuple[list[Item], str | None]:
        """游标是"编号有效期:看到的最后一封的编号:接入时间"。新邮件要同时满足：编号更大，而且是接入之后才到的。"""
        imap = self._login()
        try:
            validity, uidnext = self._status(imap)
            now = int(self.clock())
            baseline = f"{validity}:{uidnext - 1}:{now}"
            if cursor is None:
                return [], baseline
            seen_validity, last, since = (cursor.split(":") + [""])[:3]
            if seen_validity != str(validity):
                # UIDVALIDITY 变了说明服务器重排了编号，旧游标作废，只能从现在重新开始。
                return [], baseline
            last_uid, since = int(last), int(since) if since else now - LEGACY_LOOKBACK
            if uidnext - 1 <= last_uid:
                return [], f"{validity}:{last_uid}:{since}"
            self._ok(imap.select(self._mailbox(), readonly=True), "打开邮箱")
            # 先让服务器按日期粗筛，免得把一千封旧信挨个下载回来。SINCE 只精确到天、还不管时区，所以往前多放两天，
            # 准确的判断在下面用每封信自己的收信时间做。
            day = datetime.fromtimestamp(since, timezone.utc) - timedelta(days=2)
            data = self._ok(imap.uid("search", None, f"UID {last_uid + 1}:* SINCE {day.day}-{MONTHS[day.month - 1]}-{day.year}"),
                            "查找新邮件")
            # "n:*" 在没有更新的邮件时也会返回最后一封，所以要再过滤一遍。
            found = sorted(u for u in map(int, (data[0] or b"").split()) if u > last_uid)
            uids = found[: self.cfg.max_per_run]
            items = []
            for uid in uids:
                item, arrived = self._fetch(imap, uid, validity)
                if item is None:
                    continue
                slack = OLD_MAIL_SLACK
                if arrived is None:
                    # 服务器没给收信时间：退一步看信头里的日期。那是发件人写的，可能差出几个小时，所以放宽到一天。
                    arrived, slack = item.received_at, 86400
                if arrived is not None and arrived < since - slack:
                    continue   # 接入之前就在邮箱里的旧信，只是刚被排上编号
                items.append(item)
            # 这一批没拿完就停在最后一封上，下一轮接着来；拿完了就直接跳到邮箱末尾，把中间的旧信一次跳过。
            reached = uids[-1] if len(found) > len(uids) else max([uidnext - 1, *uids])
            return items, f"{validity}:{reached}:{since}"
        except (imaplib.IMAP4.error, OSError) as e:
            raise ConnectorError(f"读取邮箱失败：{_brief(e)}") from e
        finally:
            _quiet_logout(imap)

    def preview(self, n: int = 3) -> list[Item]:
        """最新的 n 封，用来自检账号能不能连、中文能不能正常解析。不改任何状态。"""
        imap = self._login()
        try:
            validity, uidnext = self._status(imap)
            self._ok(imap.select(self._mailbox(), readonly=True), "打开邮箱")
            data = self._ok(imap.uid("search", None, f"UID {max(uidnext - n, 1)}:*"), "查找邮件")
            uids = sorted(map(int, (data[0] or b"").split()))[-n:]
            return [item for uid in uids if (item := self._fetch(imap, uid, validity)[0])]
        except (imaplib.IMAP4.error, OSError) as e:
            raise ConnectorError(f"读取邮箱失败：{_brief(e)}") from e
        finally:
            _quiet_logout(imap)

    def _login(self) -> imaplib.IMAP4:
        cls = imaplib.IMAP4_SSL if self.cfg.ssl else imaplib.IMAP4
        try:
            imap = cls(self.cfg.host, self.cfg.port, timeout=30)
        except (OSError, imaplib.IMAP4.error) as e:
            raise ConnectorError(f"连不上邮件服务器 {self.cfg.host}:{self.cfg.port}（{_brief(e)}）") from e
        try:
            imap.login(self.cfg.user, self.cfg.password)
        except imaplib.IMAP4.error as e:
            _quiet_logout(imap)
            raise ConnectorError(f"邮箱登录失败，检查账号和授权码（{_brief(e)}）") from e
        except OSError as e:
            _quiet_logout(imap)
            raise ConnectorError(f"邮箱登录超时或断开（{_brief(e)}）") from e
        if "ID" in imap.capabilities:
            try:
                imap.xatom("ID", CLIENT_ID)
            except (imaplib.IMAP4.error, OSError):
                pass   # 报不上名字不是致命的：真的被拒会在下一步打开收件箱时说明原因
        return imap

    def _mailbox(self) -> str:
        return '"' + self.cfg.mailbox.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def _status(self, imap: imaplib.IMAP4) -> tuple[int, int]:
        data = self._ok(imap.status(self._mailbox(), "(UIDNEXT UIDVALIDITY)"), "读取邮箱状态")
        text = b" ".join(d for d in data if isinstance(d, bytes)).decode("ascii", "replace")
        validity, uidnext = re.search(r"UIDVALIDITY (\d+)", text), re.search(r"UIDNEXT (\d+)", text)
        if validity and uidnext:
            return int(validity.group(1)), int(uidnext.group(1))
        return self._status_by_opening(imap, text)

    def _status_by_opening(self, imap: imaplib.IMAP4, said: str) -> tuple[int, int]:
        """服务器不肯说"下一封的编号"时，自己算出来。

        网易（163/126，Coremail）就是这样：STATUS 里问 UIDNEXT 它直接略过，打开收件箱时也不给
        （2026-11-17 在真实的 163 邮箱上看到的应答是 `"INBOX" (UIDVALIDITY 1)`）。
        标准允许服务器这么做，所以换个问法：打开收件箱，问最后一封的编号，下一封就是它加一。
        """
        count = self._ok(imap.select(self._mailbox(), readonly=True), "打开邮箱")
        validity = imap.response("UIDVALIDITY")[1][0]
        if not validity or not validity.strip().isdigit():
            raise ConnectorError(f"邮件服务器返回了看不懂的状态：{said[:80]}")
        if not int(count[0] or 0):
            return int(validity), 1   # 空邮箱：没有"最后一封"可问
        data = self._ok(imap.uid("search", None, "*"), "查找最新的邮件")
        uids = [int(u) for u in (data[0] or b"").split()]
        return int(validity), max(uids, default=0) + 1

    @staticmethod
    def _ok(response: tuple, what: str) -> list:
        typ, data = response
        if typ != "OK":
            raise ConnectorError(f"{what}失败：{_brief(data)}")
        return data

    def _fetch(self, imap: imaplib.IMAP4, uid: int, validity: int) -> tuple[Item | None, int | None]:
        """取一封信，连同服务器收到它的时间（没给或认不出来就是 None）。"""
        data = self._ok(imap.uid("fetch", str(uid), f"(INTERNALDATE BODY.PEEK[]<0.{FETCH_BYTES}>)"), "读取邮件")
        raw = next((part[1] for part in data if isinstance(part, tuple)), None)
        if raw is None:
            return None, None   # 查到之后、取信之前被删掉了
        # 收信时间在正文之外的那几段应答里；有的服务器放在正文前，有的放在后面，都找一遍。
        around = b" ".join(part[0] if isinstance(part, tuple) else part for part in data if isinstance(part, (tuple, bytes)))
        return parse(raw, fallback_id=f"uid:{validity}:{uid}"), internal_date(around)


def internal_date(response: bytes) -> int | None:
    m = INTERNALDATE.search(response)
    if not m or m.group(2).decode().title() not in MONTHS:
        return None
    day, year, hour, minute, second = (int(m.group(i)) for i in (1, 3, 4, 5, 6))
    offset = timedelta(hours=int(m.group(8)), minutes=int(m.group(9))) * (-1 if m.group(7) == b"-" else 1)
    try:
        when = datetime(year, MONTHS.index(m.group(2).decode().title()) + 1, day, hour, minute, second, tzinfo=timezone(offset))
    except ValueError:
        return None
    return int(when.timestamp())


def parse(raw: bytes, fallback_id: str) -> Item:
    """把（可能被截断的）原始邮件变成 Item。邮件格式千奇百怪，每一步都容错，宁可字段留空也不抛异常。"""
    msg: EmailMessage = email.message_from_bytes(raw, policy=email.policy.default)
    name, address = parseaddr(_header(msg, "from"))
    sender = f"{name} <{address}>" if name and address else (address or name or "（未知发件人）")
    received = None
    try:
        received = int(parsedate_to_datetime(_header(msg, "date")).timestamp())
    except (TypeError, ValueError):
        pass
    message_id = _header(msg, "message-id").strip()
    body = _text(msg)
    return Item(
        external_id=message_id or fallback_id,
        sender=sender,
        title=" ".join(_header(msg, "subject").split()) or "（无主题）",
        snippet=body[:SNIPPET_CHARS],
        received_at=received,
        body=body[:BODY_CHARS],
        reply_to=parseaddr(_header(msg, "reply-to"))[1] or address,
        message_id=message_id,
        references=" ".join(f"{_header(msg, 'references')} {message_id}".split()),
    )


def _header(msg: EmailMessage, name: str) -> str:
    try:
        return str(msg.get(name, ""))
    except Exception:   # 畸形信头会让标准库的解析器抛各种异常
        return ""


def _text(msg: EmailMessage) -> str:
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        if part is None:
            return ""
        text = part.get_content()
        if part.get_content_type() == "text/html":
            text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", text)
            text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", text))
    except Exception:   # 截断在编码中间、声明了不存在的字符集……
        return ""
    return " ".join(text.split())


def _brief(e: object) -> str:
    if isinstance(e, (list, tuple)):
        e = b" ".join(x for x in e if isinstance(x, bytes))
    if isinstance(e, BaseException) and e.args and isinstance(e.args[0], bytes):
        e = e.args[0]
    text = e.decode("utf-8", "replace") if isinstance(e, bytes) else str(e)
    return text[:160] or type(e).__name__


def _quiet_logout(imap: imaplib.IMAP4) -> None:
    try:
        imap.logout()
    except Exception:
        pass
