"""邮箱连接器：用标准库 imaplib 连一个本地的小型 IMAP 服务器，协议是真的走了一遍，只是不出本机。"""
import re
import socketserver
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import pytest

from alf.config import EmailConfig
from alf.connectors import ConnectorError
from alf.connectors.email_imap import MONTHS, ImapEmail, internal_date, parse

NOW = 1_800_000_000   # 测试里连接器眼中的"现在"（2027-01-15）；假服务器上的新邮件默认比它晚一分钟到


class Mailbox:
    def __init__(self):
        self.validity = 777
        self.mails: dict[int, bytes] = {}   # uid -> 原始邮件
        self.password = "app-password"
        self.commands: list[str] = []
        self.wants_id = self.identified = False   # 像网易邮箱那样：不先用 ID 报上名字，就不让打开收件箱
        self.hides_uidnext = False                # 也像网易：问"下一封的编号"时不回答
        self.arrived: dict[int, int] = {}         # uid -> 服务器收到这封信的时间（INTERNALDATE）
        self.ignores_since = False                # 不按日期筛，问什么都把编号全给回来
        self.pads_day = True                      # QQ 不补位："1-Jul-2022"
        self.date_after_body = False              # 有的服务器把收信时间放在正文后面
        self.hides_date = False                   # 干脆不给收信时间

    def add(self, raw: bytes, arrived: int = NOW + 60) -> int:
        uid = max(self.mails, default=0) + 1
        self.mails[uid], self.arrived[uid] = raw, arrived
        return uid

    def internal(self, uid: int) -> str:
        t = datetime.fromtimestamp(self.arrived[uid], timezone.utc)
        day = f"{t.day:02d}" if self.pads_day else str(t.day)
        return f'INTERNALDATE "{day}-{MONTHS[t.month - 1]}-{t.year} {t:%H:%M:%S} +0000"'

    @property
    def uidnext(self) -> int:
        return max(self.mails, default=0) + 1


class Handler(socketserver.StreamRequestHandler):
    def send(self, line: str) -> None:
        self.wfile.write(line.encode() + b"\r\n")

    def handle(self) -> None:
        box: Mailbox = self.server.mailbox
        self.send("* OK fake imap ready")
        for raw in self.rfile:
            tag, _, rest = raw.decode().strip().partition(" ")
            box.commands.append(rest)
            cmd = rest.upper()
            if cmd.startswith("CAPABILITY"):
                self.send("* CAPABILITY IMAP4rev1" + (" ID" if box.wants_id else ""))
            elif cmd.startswith("ID "):
                box.identified = True
                self.send('* ID ("name" "fake")')
            elif cmd.startswith("LOGIN"):
                if box.password not in rest:
                    self.send(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials")
                    continue
            elif cmd.startswith("STATUS"):
                if box.hides_uidnext:
                    self.send(f'* STATUS "INBOX" (UIDVALIDITY {box.validity})')   # 真实的 163 就是这么回的
                else:
                    self.send(f'* STATUS "INBOX" (UIDNEXT {box.uidnext} UIDVALIDITY {box.validity})')
            elif cmd.startswith("EXAMINE"):
                if box.wants_id and not box.identified:
                    self.send(f"{tag} NO EXAMINE Unsafe Login. Please contact kefu@188.com for help")
                    continue
                self.send(f"* {len(box.mails)} EXISTS")
                self.send(f"* OK [UIDVALIDITY {box.validity}] UIDs valid")
            elif cmd == "UID SEARCH *":
                self.send("* SEARCH " + " ".join(map(str, sorted(box.mails)[-1:])))   # 最后一封的编号
            elif cmd.startswith("UID SEARCH"):
                low = int(re.search(r"UID (\d+):\*", cmd).group(1))
                hits = [u for u in sorted(box.mails) if u >= low] or sorted(box.mails)[-1:]   # n:* 至少返回最后一封
                since = re.search(r"SINCE (\d+)-(\w+)-(\d+)", rest)
                if since and not box.ignores_since:
                    day = datetime(int(since.group(3)), MONTHS.index(since.group(2)) + 1, int(since.group(1)), tzinfo=timezone.utc)
                    hits = [u for u in hits if box.arrived[u] >= day.timestamp()]
                self.send("* SEARCH " + " ".join(map(str, hits)))
            elif cmd.startswith("UID FETCH"):
                uid, size = int(cmd.split()[2]), int(re.search(r"<0\.(\d+)>", cmd).group(1))
                if uid in box.mails:
                    body = box.mails[uid][:size]
                    date = "" if box.hides_date else box.internal(uid)
                    before, after = ("", " " + date) if box.date_after_body else (date + " ", "")
                    self.wfile.write(f"* {uid} FETCH (UID {uid} {before}BODY[]<0> {{{len(body)}}}\r\n".encode() + body
                                     + f"{after})\r\n".encode())
            elif cmd.startswith("LOGOUT"):
                self.send("* BYE")
                self.send(f"{tag} OK done")
                return
            else:
                self.send(f"{tag} BAD unsupported")
                continue
            self.send(f"{tag} OK done")


@pytest.fixture
def box():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.mailbox = Mailbox()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.mailbox.port = server.server_address[1]
    yield server.mailbox
    server.shutdown()
    server.server_close()


def connector(box, password="app-password", **kw):
    return ImapEmail(EmailConfig("127.0.0.1", "me@example.com", password, port=box.port, ssl=False, **kw), clock=lambda: NOW)


def at(uid, validity=777, since=NOW):
    """游标：编号有效期、看到的最后一封、接入时间。"""
    return f"{validity}:{uid}:{since}"


def message(subject, sender, body=None, html=None, attachment_kb=0, message_id=None, date="Tue, 22 Sep 2026 10:00:00 +1000"):
    m = EmailMessage()
    m["From"], m["Subject"], m["Date"] = sender, subject, date
    if message_id:
        m["Message-ID"] = message_id
    if body is not None:
        m.set_content(body)
    if html is not None:
        m.add_alternative(html, subtype="html") if body is not None else m.set_content(html, subtype="html")
    if attachment_kb:
        m.add_attachment(b"\x00" * 1024 * attachment_kb, maintype="application", subtype="octet-stream", filename="big.bin")
    return m.as_bytes()


def test_first_run_reports_nothing_and_later_runs_return_only_new_mail(box):
    box.add(message("旧邮件", "old@example.com", "很久以前"))
    c = connector(box)
    items, cursor = c.fetch_new(None)
    assert items == [] and cursor == at(1)

    assert c.fetch_new(cursor) == ([], at(1))            # 没有新邮件：游标不动

    box.add(message("COMP5318 作业二截止时间调整", "李老师 <li@uni.edu>", "各位同学：\n截止时间改到周五。", message_id="<a1@uni.edu>"))
    box.add(message("双十一大促", "Shop <ad@shop.com>", html="<style>p{}</style><p>全场 <b>五折</b>&nbsp;起</p>"))
    items, cursor = c.fetch_new(cursor)
    assert cursor == at(3)
    assert [(i.title, i.sender, i.snippet) for i in items] == [
        ("COMP5318 作业二截止时间调整", "李老师 <li@uni.edu>", "各位同学： 截止时间改到周五。"),
        ("双十一大促", "Shop <ad@shop.com>", "全场 五折 起"),
    ]
    assert items[0].external_id == "<a1@uni.edu>" and items[1].external_id == "uid:777:3"
    assert items[0].received_at == int(datetime(2026, 9, 22, 0, 0, tzinfo=timezone.utc).timestamp())   # 10:00 +10:00


def test_mailbox_is_opened_read_only_and_mail_is_peeked(box):
    c = connector(box)
    _, cursor = c.fetch_new(None)
    box.add(message("hi", "a@b.c", "x"))
    c.fetch_new(cursor)
    sent = " | ".join(box.commands).upper()
    assert "EXAMINE" in sent and "BODY.PEEK[]" in sent
    assert not re.search(r"\bSELECT\b|STORE|EXPUNGE|DELETE|APPEND|COPY|MOVE", sent)


def test_large_attachment_is_not_downloaded_but_text_is_still_read(box):
    c = connector(box)
    _, cursor = c.fetch_new(None)
    box.add(message("合同", "hr@corp.com", "请查收附件里的合同。", attachment_kb=300))
    [item], _ = c.fetch_new(cursor)
    assert (item.title, item.snippet) == ("合同", "请查收附件里的合同。")
    fetch = next(cmd for cmd in box.commands if cmd.upper().startswith("UID FETCH"))
    assert "<0.65536>" in fetch


def test_backlog_is_processed_in_batches(box):
    c = connector(box, max_per_run=2)
    _, cursor = c.fetch_new(None)
    for n in range(5):
        box.add(message(f"第{n}封", "a@b.c", "x"))
    seen = []
    for _ in range(3):
        items, cursor = c.fetch_new(cursor)
        seen.append([i.title for i in items])
    assert seen == [["第0封", "第1封"], ["第2封", "第3封"], ["第4封"]] and cursor == at(5)


def test_renumbered_mailbox_starts_over_instead_of_replaying_everything(box):
    box.add(message("a", "a@b.c", "x"))
    c = connector(box)
    _, cursor = c.fetch_new(None)
    box.validity = 778
    assert c.fetch_new(cursor) == ([], at(1, validity=778))


def test_wrong_password_and_unreachable_server_give_readable_errors(box):
    with pytest.raises(ConnectorError, match="邮箱登录失败，检查账号和授权码"):
        connector(box, password="wrong").fetch_new(None)
    box.port = 1   # 没有服务在听
    with pytest.raises(ConnectorError, match="连不上邮件服务器 127.0.0.1:1"):
        connector(box).fetch_new(None)


def test_preview_returns_latest_mail_without_touching_state(box):
    for n in range(5):
        box.add(message(f"第{n}封", "a@b.c", "x"))
    assert [i.title for i in connector(box).preview(3)] == ["第2封", "第3封", "第4封"]


def test_parse_survives_garbage():
    item = parse(b"\xff\xfe not an email at all", fallback_id="uid:1:9")
    assert (item.external_id, item.title, item.sender) == ("uid:1:9", "（无主题）", "（未知发件人）")
    assert item.snippet.endswith("not an email at all") and item.received_at is None
    encoded = parse("Subject: =?UTF-8?B?5L2g5aW9?=\r\nFrom: =?UTF-8?B?5byg5LiJ?= <z@x.cn>\r\n\r\nhi".encode(), "id")
    assert (encoded.title, encoded.sender) == ("你好", "张三 <z@x.cn>")


def test_netease_style_server_gets_the_client_id_before_the_inbox_is_opened(box):
    """163/126 邮箱不报 ID 就拒绝："NO EXAMINE Unsafe Login"。"""
    box.wants_id = True
    c = connector(box)
    _, cursor = c.fetch_new(None)
    box.add(message("hi", "a@b.c", "x"))
    [item], _ = c.fetch_new(cursor)
    assert item.title == "hi" and any(cmd.startswith('ID ("name" "Alf"') for cmd in box.commands)


def test_server_that_will_not_say_the_next_mail_number_still_works(box):
    """163 邮箱不回答"下一封的编号"（STATUS 只回 UIDVALIDITY）。2026-11-17 真实遇到：登录成功了，Alf 却报"看不懂的状态"。"""
    box.wants_id = box.hides_uidnext = True
    c = connector(box)
    assert c.fetch_new(None) == ([], at(0))               # 空邮箱也要能定下起点

    box.add(message("旧邮件", "old@example.com", "很久以前"))
    _, cursor = c.fetch_new(None)
    assert cursor == at(1)
    assert c.fetch_new(cursor) == ([], at(1))            # 没有新邮件：游标不动，也不重报旧的

    box.add(message("面试时间确认", "hr@corp.cn", "周四下午三点"))
    items, cursor = c.fetch_new(cursor)
    assert [i.title for i in items] == ["面试时间确认"] and cursor == at(2)
    assert [i.title for i in c.preview(3)] == ["旧邮件", "面试时间确认"]

    sent = " | ".join(box.commands).upper()
    assert not re.search(r"\bSELECT\b|STORE|EXPUNGE|DELETE|APPEND|COPY|MOVE", sent)   # 换了问法，仍然只读


def test_status_without_any_numbers_is_still_reported_as_unreadable(box):
    box.hides_uidnext = True
    box.validity = ""   # 连编号有效期都不给：没法判断哪些是新邮件，宁可报错也不乱猜
    with pytest.raises(ConnectorError, match="看不懂的状态"):
        connector(box).fetch_new(None)


def years_ago(n):
    return NOW - n * 365 * 86400


def test_old_mail_that_the_server_numbers_late_is_not_reported_as_new(box):
    """QQ 邮箱 2026-11-17 的真实情况：接入之后，服务器才给一千多封 2020–2022 年的旧信排上（更大的）编号。
    当时 Alf 把其中 30 封当成新邮件，通知了 3 封。编号更大不算数，得是接入之后才到的。"""
    box.pads_day = False                                   # QQ 的日期写法："1-Jul-2022"
    box.add(message("接入前就有的", "a@b.c", "x"), arrived=NOW - 3600)
    c = connector(box)
    _, cursor = c.fetch_new(None)
    assert cursor == at(1)

    for n in range(40):                                    # 比一批（30 封）还多：一轮就得全部跳过
        box.add(message(f"{2020 + n % 3} 年的账单 {n}", "bank@example.com", "旧信"), arrived=years_ago(5) + n * 86400)
    box.add(message("今天的面试通知", "hr@corp.cn", "周四下午三点"))
    items, cursor = c.fetch_new(cursor)
    assert [i.title for i in items] == ["今天的面试通知"]
    assert cursor == at(42)                                # 直接跳到邮箱末尾，旧信不会在后面几轮里再冒出来
    assert sum(cmd.upper().startswith("UID FETCH") for cmd in box.commands) == 1   # 旧信连下载都没下载
    assert c.fetch_new(cursor) == ([], at(42))


@pytest.mark.parametrize("quirk", ["ignores_since", "date_after_body", "hides_date"])
def test_old_mail_is_still_skipped_when_the_server_is_less_helpful(box, quirk):
    """服务器不按日期筛、把收信时间放在正文后面、或者干脆不给（这时看信头里的日期）——旧信都不该被当成新的。"""
    c = connector(box)
    _, cursor = c.fetch_new(None)
    setattr(box, quirk, True)
    box.pads_day = False
    box.ignores_since = True                               # 都让旧信走到"逐封判断"那一步
    old_date = "Fri, 1 Jul 2022 10:29:27 +0800"
    box.add(message("2022 年的旧信", "old@example.com", "x", date=old_date), arrived=years_ago(4))
    box.add(message("刚到的", "new@example.com", "y", date="Fri, 15 Jan 2027 08:01:00 +0000"))
    items, cursor = c.fetch_new(cursor)
    assert [i.title for i in items] == ["刚到的"] and cursor == at(2)


def test_mail_that_arrived_while_alf_was_off_is_not_lost(box):
    """旧信的界线是"接入的那一刻"，不是"最近多少小时"：电脑关了三天，这三天里到的信回来还得报。"""
    c = connector(box)
    _, cursor = c.fetch_new(None)
    box.add(message("周一到的", "a@b.c", "x"), arrived=NOW + 86400)
    later = ImapEmail(c.cfg, clock=lambda: NOW + 4 * 86400)
    items, cursor = later.fetch_new(cursor)
    assert [i.title for i in items] == ["周一到的"] and cursor == at(1)   # 接入时间不跟着往后挪


def test_cursor_saved_before_the_connect_time_was_recorded_still_works(box):
    """早先存下的游标只有两段。升级后：最近一天内到的照常报，更早的不报，游标补上第三段。"""
    box.add(message("已经看过的", "a@b.c", "x"), arrived=NOW - 7200)
    box.add(message("三年前的", "old@example.com", "x"), arrived=years_ago(3))
    box.add(message("一小时前到的", "new@example.com", "y"), arrived=NOW - 3600)
    items, cursor = connector(box).fetch_new("777:1")
    assert [i.title for i in items] == ["一小时前到的"] and cursor == at(3, since=NOW - 86400)
    assert connector(box).fetch_new("777:3") == ([], at(3, since=NOW - 86400))


def test_internal_date_is_read_in_the_spellings_real_servers_use():
    expected = int(datetime(2022, 7, 1, 10, 29, 27, tzinfo=timezone(timedelta(hours=8))).timestamp())
    for spelling in ("1-Jul-2022", " 1-Jul-2022", "01-Jul-2022", "01-JUL-2022"):
        assert internal_date(f'5 (UID 9 INTERNALDATE "{spelling} 10:29:27 +0800" BODY[]<0> {{12}}'.encode()) == expected
    assert internal_date(b'INTERNALDATE "31-Dec-2026 23:00:00 -0500"') == int(datetime(2027, 1, 1, 4, tzinfo=timezone.utc).timestamp())
    for garbage in (b"", b"5 (UID 9)", b'INTERNALDATE "31-Foo-2026 23:00:00 +0000"', b'INTERNALDATE "31-Feb-2026 23:00:00 +0000"'):
        assert internal_date(garbage) is None


def test_reply_details_and_a_longer_body_are_kept_for_drafting_a_reply(box):
    m = EmailMessage()
    m["From"], m["Reply-To"], m["Subject"] = "张老师 <zhang@uni.edu>", "course-admin@uni.edu", "关于周五的面谈"
    m["Message-ID"], m["References"] = "<m2@uni.edu>", "<m0@uni.edu> <m1@uni.edu>"
    m.set_content("同学你好：\n" + "周五下午两点可以吗？" * 100)
    c = connector(box)
    _, cursor = c.fetch_new(None)
    box.add(m.as_bytes())
    box.add(message("没有回复地址的", "noreply@shop.com", "x"))
    [item, plain], _ = c.fetch_new(cursor)
    assert (item.reply_to, item.message_id, item.references) == ("course-admin@uni.edu", "<m2@uni.edu>", "<m0@uni.edu> <m1@uni.edu> <m2@uni.edu>")
    assert len(item.snippet) == 500 and len(item.body) == 1006 and item.body.startswith("同学你好： 周五下午两点可以吗？")
    assert (plain.reply_to, plain.message_id, plain.references) == ("noreply@shop.com", "", "")


def test_three_mailboxes_only_need_an_address_and_an_app_password(monkeypatch):
    from alf import config, connectors

    for key in ("ALF_EMAIL_HOST", "ALF_EMAIL_USER", "ALF_EMAIL_PASSWORD", "ALF_GMAIL_USER", "ALF_GMAIL_PASSWORD",
                "ALF_163_USER", "ALF_163_PASSWORD", "ALF_QQ_USER", "ALF_QQ_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    assert config.settings().mailboxes == () and connectors.build(config.settings()) == []
    monkeypatch.setenv("ALF_GMAIL_USER", "me@gmail.com")
    monkeypatch.setenv("ALF_GMAIL_PASSWORD", "abcd efgh ijkl mnop")        # Gmail 给的是四组带空格的
    monkeypatch.setenv("ALF_163_USER", "me@163.com")
    monkeypatch.setenv("ALF_163_PASSWORD", "AUTHCODE")
    monkeypatch.setenv("ALF_QQ_USER", "10001@qq.com")                       # 只填了地址，没填授权码：不启用
    cfg = config.settings()
    assert [(m.name, m.label, m.host, m.smtp_host, m.smtp_port, m.user) for m in cfg.mailboxes] == [
        ("email.gmail", "Gmail", "imap.gmail.com", "smtp.gmail.com", 465, "me@gmail.com"),
        ("email.163", "163 邮箱", "imap.163.com", "smtp.163.com", 465, "me@163.com")]
    assert cfg.mailboxes[0].password == "abcdefghijklmnop" and "abcdefghijklmnop" not in repr(cfg)
    assert [(c.name, c.kind, c.label) for c in connectors.build(cfg)] == [("email.gmail", "email", "Gmail"), ("email.163", "email", "163 邮箱")]
