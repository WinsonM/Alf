"""发信（SMTP）。整个项目里只有这一个函数会往外发邮件，而调用它之前要过"用户确认了这一份草稿"的检查
（见 chat.py 的 send_email_draft）。它自己不做任何判断，只管把一封已经定稿的回信发出去。
"""
from __future__ import annotations

import re
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from ..config import EmailConfig
from . import ConnectorError

ADDRESS = re.compile(r"[^@\s<>,;\"']+@[^@\s<>,;\"']+\.[^@\s<>,;\"']+")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def build(cfg: EmailConfig, to: str, subject: str, body: str, *, in_reply_to: str = "", references: str = "") -> EmailMessage:
    if not ADDRESS.fullmatch(to):
        raise ConnectorError(f"收件地址不像一个邮箱地址：{to[:60]}")
    if "\n" in subject or "\r" in subject:
        raise ConnectorError("主题里不能有换行")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = cfg.user, to, subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=cfg.user.rpartition("@")[2] or "localhost")
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = references or in_reply_to
    msg.set_content(body)
    return msg


def send(cfg: EmailConfig, to: str, subject: str, body: str, *, in_reply_to: str = "", references: str = "") -> str:
    """发出一封信，返回它的 Message-ID。失败抛 ConnectorError，消息是给用户看的。"""
    msg = build(cfg, to, subject, body, in_reply_to=in_reply_to, references=references)
    _session(cfg, lambda server: server.send_message(msg))
    return msg["Message-ID"]


def check(cfg: EmailConfig) -> None:
    """只登录、不发信：用来在设置页确认授权码能不能发信。"""
    _session(cfg, lambda server: server.noop())


def _session(cfg: EmailConfig, work) -> None:
    if not cfg.smtp_host:
        raise ConnectorError(f"{cfg.label}没有配置发信服务器，只能收不能发")
    try:
        if cfg.smtp_ssl:
            server = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=30, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30)
        with server:
            if not cfg.smtp_ssl:
                server.ehlo()
                if server.has_extn("starttls"):
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                elif cfg.smtp_host not in LOCAL_HOSTS:
                    # 不加密就登录等于把授权码明文发出去。只有连本机（测试）时才允许。
                    raise ConnectorError(f"{cfg.smtp_host} 不支持加密连接，没有发")
            server.login(cfg.user, cfg.password)
            work(server)
    except smtplib.SMTPAuthenticationError as e:
        raise ConnectorError(f"{cfg.label}发信登录失败，检查授权码（{_brief(e)}）") from e
    except smtplib.SMTPRecipientsRefused as e:
        raise ConnectorError(f"收件地址被服务器拒收：{'、'.join(e.recipients)}") from e
    except (smtplib.SMTPException, OSError) as e:
        raise ConnectorError(f"发信失败：{_brief(e)}") from e


def _brief(e: BaseException) -> str:
    detail = getattr(e, "smtp_error", None)
    text = detail.decode("utf-8", "replace") if isinstance(detail, bytes) else str(e)
    return " ".join(text.split())[:160] or type(e).__name__
