"""运行配置，全部来自环境变量：同一份代码既能当终端程序跑，也能当常驻服务跑。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent


def env_file() -> Path:
    """配置文件的位置。默认是项目里的 .env；ALF_ENV_FILE 可以指到别处（比如起一个隔离的演示实例）。"""
    return Path(os.getenv("ALF_ENV_FILE") or ROOT / ".env").expanduser()


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 读取（KEY=VALUE）。环境里已有的变量优先；空值当作没写。"""
    path = path or env_file()
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def update_dotenv(path: Path, changes: dict[str, str]) -> None:
    """把几项配置写进 .env：已有的行原地改，没有的追加在末尾，注释和其他行不动。

    写完把文件权限收紧到只有本人可读写——里面有 API key。
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = dict(changes)
    for i, line in enumerate(lines):
        key = line.partition("=")[0].strip()
        if key in pending and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={pending.pop(key)}"
    lines += [f"{key}={value}" for key, value in pending.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)


@dataclass(frozen=True)
class EmailConfig:
    host: str
    user: str
    password: str = field(repr=False)   # 不进 repr，免得被日志或报错信息带出去
    port: int = 993
    ssl: bool = True
    mailbox: str = "INBOX"
    every: float = 300                  # 多久查一次新邮件（秒）
    triage: bool = True                 # False = 不把邮件发给 Jev，新邮件一律通知
    notify_threshold: float = 0.6       # "需要你处理"的概率达到多少才通知
    max_per_run: int = 30               # 一次最多处理几封，防止积压时一口气打几百次 Jev
    name: str = "email"                 # 任务名，也记在每封信上（回信从收到它的邮箱发出）。多个邮箱时是 email.gmail 这种
    label: str = "邮箱"
    smtp_host: str = ""                 # 发信服务器；空 = 这个邮箱只收不发
    smtp_port: int = 465
    smtp_ssl: bool = True               # 465 端口一上来就加密；False 时先连再升级加密（587）


# 三家常用邮箱的服务器地址。用户只需要填邮箱地址和"授权码 / 应用专用密码"。
MAIL_PROVIDERS = {
    "gmail": ("Gmail", "imap.gmail.com", "smtp.gmail.com"),
    "163": ("163 邮箱", "imap.163.com", "smtp.163.com"),
    "qq": ("QQ 邮箱", "imap.qq.com", "smtp.qq.com"),
}


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    timeout: float = 40.0


@dataclass(frozen=True)
class IMessageConfig:
    owners: tuple[str, ...]             # 只处理这些地址发来的消息；第一个也是 Alf 发消息的目标
    db_path: Path = Path.home() / "Library" / "Messages" / "chat.db"
    poll: float = 1.5                   # 多久看一次有没有新消息（秒）
    attachments_root: Path = Path.home() / "Library" / "Messages" / "Attachments"   # 只读这个目录里的附件


@dataclass(frozen=True)
class Settings:
    tz: ZoneInfo
    db_path: Path
    attachment_dir: Path
    decision_threshold: float   # Jev 置信度低于它，Alf 反问而不是执行
    fact_side_threshold: float  # 设提醒时顺带记下事实所需的 P(含长期事实)
    default_hour: int           # "明天提醒我…" 没说几点时用它
    asr_model: str
    asr_language: str | None    # None = 自动检测，中英混说时更准
    asr_low_confidence: float   # 低于它，语音来的事实标为"未确认"
    web_port: int = 8765
    heartbeat_tick: float = 5.0     # 心跳多久醒一次（秒）；空转只查本地库，不调模型
    reminder_every: float = 10.0    # 多久检查一次到期提醒，也就是提醒最多晚这么久
    briefing_at: time | None = None  # 每天几点播报当天安排；None = 不播报
    email: EmailConfig | None = None       # 手填服务器地址的那一个邮箱（ALF_EMAIL_HOST…）
    emails: tuple[EmailConfig, ...] = ()   # Gmail / 163 / QQ：只填地址和授权码的
    llm: LLMConfig | None = None           # 生成式模型（DeepSeek）；None = 退回只用 Jev 的模板模式
    imessage: IMessageConfig | None = None
    digest_at: time | None = None          # 每晚几点发"今天记了这些"；None = 不发

    @property
    def mailboxes(self) -> tuple[EmailConfig, ...]:
        """所有接入的邮箱。"""
        return (*self.emails, *((self.email,) if self.email else ()))

    @property
    def push_channel(self) -> str | None:
        """Alf 主动发的消息（提醒、播报……）要送去的渠道。"""
        return "imessage" if self.imessage else None


def _hhmm(value: str) -> time | None:
    value = value.strip().lower()
    if value in ("", "off", "none", "no"):
        return None
    hour, _, minute = value.partition(":")
    return time(int(hour), int(minute or 0))


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() not in ("0", "false", "no", "off")


def _mail_common() -> dict:
    return {"every": float(os.getenv("ALF_EMAIL_EVERY", "300")), "triage": _flag("ALF_EMAIL_TRIAGE", True),
            "notify_threshold": float(os.getenv("ALF_EMAIL_NOTIFY_THRESHOLD", "0.6"))}


def _email() -> EmailConfig | None:
    host, user, password = (os.getenv(f"ALF_EMAIL_{k}") for k in ("HOST", "USER", "PASSWORD"))
    if not (host and user and password):
        return None
    return EmailConfig(
        host=host, user=user, password=password,
        port=int(os.getenv("ALF_EMAIL_PORT", "993")),
        ssl=_flag("ALF_EMAIL_SSL", True),
        mailbox=os.getenv("ALF_EMAIL_MAILBOX", "INBOX"),
        smtp_host=os.getenv("ALF_EMAIL_SMTP_HOST", ""),
        smtp_port=int(os.getenv("ALF_EMAIL_SMTP_PORT", "465")),
        smtp_ssl=_flag("ALF_EMAIL_SMTP_SSL", True),
        **_mail_common(),
    )


def _emails() -> tuple[EmailConfig, ...]:
    out = []
    for key, (label, imap_host, smtp_host) in MAIL_PROVIDERS.items():
        user, password = os.getenv(f"ALF_{key.upper()}_USER", "").strip(), os.getenv(f"ALF_{key.upper()}_PASSWORD", "")
        if user and password:
            # Gmail 的应用专用密码显示成四组、中间带空格，粘贴过来时去掉。
            out.append(EmailConfig(host=imap_host, user=user, password=password.replace(" ", "") if key == "gmail" else password.strip(),
                                   name=f"email.{key}", label=label, smtp_host=smtp_host, **_mail_common()))
    return tuple(out)


def normalize_handle(handle: str) -> str:
    """iMessage 地址的统一写法：邮箱小写；电话去掉空格、横线和括号（数据库里是 +61412345678 这种）。"""
    handle = handle.strip()
    if "@" in handle:
        return handle.lower()
    return "".join(ch for ch in handle if ch.isdigit() or ch == "+")


def _llm() -> LLMConfig | None:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return None
    return LLMConfig(api_key=key,
                     base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
                     model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))


def _imessage() -> IMessageConfig | None:
    owners = tuple(h for h in (normalize_handle(x) for x in os.getenv("ALF_IMESSAGE_OWNER", "").split(",")) if h)
    if not owners:
        return None
    db_path = os.getenv("ALF_IMESSAGE_DB")
    return IMessageConfig(owners=owners, **({"db_path": Path(db_path).expanduser()} if db_path else {}))


def settings() -> Settings:
    data = Path(os.getenv("ALF_DATA_DIR", ROOT / "data"))
    return Settings(
        tz=ZoneInfo(os.getenv("ALF_TZ", "Australia/Sydney")),
        db_path=data / "alf.db",
        attachment_dir=data / "attachments",
        decision_threshold=float(os.getenv("ALF_DECISION_THRESHOLD", "0.6")),
        fact_side_threshold=float(os.getenv("ALF_FACT_SIDE_THRESHOLD", "0.8")),
        default_hour=int(os.getenv("ALF_DEFAULT_HOUR", "9")),
        asr_model=os.getenv("ALF_ASR_MODEL", "small"),
        asr_language=os.getenv("ALF_ASR_LANGUAGE") or None,
        asr_low_confidence=float(os.getenv("ALF_ASR_LOW_CONFIDENCE", "0.6")),
        web_port=int(os.getenv("ALF_WEB_PORT", "8765")),
        heartbeat_tick=float(os.getenv("ALF_HEARTBEAT_TICK", "5")),
        reminder_every=float(os.getenv("ALF_REMINDER_EVERY", "10")),
        briefing_at=_hhmm(os.getenv("ALF_BRIEFING_AT", "08:00")),
        email=_email(),
        emails=_emails(),
        llm=_llm(),
        imessage=_imessage(),
        digest_at=_hhmm(os.getenv("ALF_DIGEST_AT", "21:30")),
    )
