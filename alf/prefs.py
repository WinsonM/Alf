"""运行时可调的设置：说话风格、主动消息的开关和时间、记忆、提醒的默认值。

存在数据库的 settings 表里。对话每一轮、心跳每一次都会重新读，所以在界面上改了立刻生效，不用重启。
没改过的项沿用 .env 给的默认值。密钥和 iMessage 地址不在这里——那些在 .env 里，改了需要重启。
"""
from __future__ import annotations

import json
import re
import sqlite3
import time

from . import persona
from .config import Settings

STYLE_PRESETS = ("friend", "brief", "secretary")
LEVEL_KEYS = ("quiet", "normal", "active", "chatty")
NEWS_TOPICS = ("科技与 AI", "财经与市场", "国内要闻", "澳洲本地")
_HHMM = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)")
# 聊天里说一句就能改的设置（change_setting 工具只认这些），以及回执和记忆页上怎么称呼它们。
# 不在这里的只能在设置页上改：第三方 MCP 服务器（会在本机起进程）、邮件"立刻说"的门槛和轮询间隔（数字，说不清）、
# 要不要自动记忆、读哪些信息来源。「说话方式的额外要求」也不在：那类话记成"对 Alf 的要求"，带着原话和范围。
CHAT_SETTINGS = {
    "style.preset": "说话风格", "style.user_name": "怎么称呼你", "style.emoji": "表情符号",
    "briefing.enabled": "早上的今日安排播报", "briefing.at": "今日安排播报的时间",
    "digest.enabled": "每晚的记忆汇总", "digest.at": "记忆汇总的时间",
    "email.report": "每天的邮件汇总", "email.report_at": "邮件汇总的时间",
    "quiet.start": "安静时段从几点开始", "quiet.end": "安静时段到几点结束",
    "proactive.enabled": "主动找你说话", "proactive.level": "主动找你的程度",
    "followup.enabled": "提醒之后的追问", "followup.after_minutes": "提醒之后过多久追问（分钟）",
    "calendar.alerts": "日程开始前的提醒", "calendar.lead_minutes": "日程提前多久提醒（分钟）",
    "reminders.default_hour": "没说几点的提醒默认定在几点", "notify.desktop": "Mac 上的系统通知",
    "sources.news": "新闻", "news.topics": "关心的新闻主题",
    "impressions.enabled": "对你形成印象", "search.enabled": "上网查资料",
    "hearing.enabled": "听手表录音", "hearing.keep_days": "录音的原声留几天",
    # 人设卡片的六项也走这条路：他说"你性格再损一点""以后你叫小艾"，改的是这里。
    **{key: f"人设里的{label}" for key, label in persona.PARTS.items()},
}
TIME_KEYS = ("briefing.at", "digest.at", "quiet.start", "quiet.end", "email.report_at")
VALUE_LABELS = {"style.preset": {"friend": "像朋友", "brief": "简短直接", "secretary": "像秘书"},
                "proactive.level": {"quiet": "安静", "normal": "适中", "active": "活跃", "chatty": "很活跃"}}


def show(key: str, value) -> str:
    """一项设置的值写成给人看的样子：开/关、「像朋友」、20:00。"""
    if isinstance(value, bool):
        return "开" if value else "关"
    if isinstance(value, list):
        return "、".join(str(v) for v in value) or "（无）"
    if key in VALUE_LABELS:
        return VALUE_LABELS[key].get(value, str(value))
    return str(value) if value != "" else "（无）"


def defaults(cfg: Settings) -> dict:
    return {
        "style.preset": "friend",            # friend 像朋友 | brief 简短直接 | secretary 像秘书
        "style.user_name": "",               # 怎么称呼用户；空 = 不特意称呼
        "style.emoji": True,
        "style.extra": "",                   # 用户自己写的额外要求，原样放进提示词
        "briefing.enabled": cfg.briefing_at is not None,
        "briefing.at": f"{cfg.briefing_at:%H:%M}" if cfg.briefing_at else "08:00",
        "digest.enabled": cfg.digest_at is not None,
        "digest.at": f"{cfg.digest_at:%H:%M}" if cfg.digest_at else "21:30",
        "notify.desktop": True,              # 提醒到点时是否同时弹 Mac 的系统通知
        "memory.auto": True,                 # 是否自动从聊天里记事
        "memory.semantic": True,             # 是否按意思检索（要本机有嵌入模型；没有就自动只按字面找）
        "reminders.default_hour": cfg.default_hour,
        "reminders.check_every": int(cfg.reminder_every),
        "email.every": int(cfg.mailboxes[0].every) if cfg.mailboxes else 300,
        "email.threshold": cfg.mailboxes[0].notify_threshold if cfg.mailboxes else 0.6,   # 多大把握才算"得立刻告诉他"
        "email.report": True,                # 没立刻说的邮件，每天固定时间汇总一次
        "email.report_at": "18:00",
        "calendar.alerts": True,             # 日历日程开始前提醒
        "calendar.lead_minutes": 15,
        "followup.enabled": True,            # 提醒发出后没回应，过一阵追问一次
        "followup.after_minutes": 30,
        "proactive.enabled": True,           # 没人说话时，是否主动找用户
        "proactive.level": "active",         # quiet 安静 | normal 适中 | active 活跃 | chatty 很活跃
        "quiet.start": "22:00",              # 安静时段：这段时间不主动说话（用户自己设的提醒照常）
        "quiet.end": "08:00",
        "sources.days": True,                # 通过 MCP 汇总哪些信息
        "sources.calendar": True,
        "sources.reminders": True,           # Apple「提醒事项」里快到期的条目
        "sources.news": True,
        "news.topics": ["科技与 AI", "财经与市场", "国内要闻"],
        "mcp.servers": [],                   # 额外的第三方 MCP 服务器：[{"name", "command", "args", "env"}]
        "impressions.enabled": True,         # 是否从聊天里慢慢形成对他的印象（关掉后已有的印象留着，不再新增）
        "search.enabled": True,              # 需要最新信息时让 DeepSeek 上网查（按搜索条数计费）
        "hearing.enabled": True,             # 听手表传来的录音（要本机有语音模型；没有就不启动）。关掉后已经记下的留着
        "hearing.keep_days": 7,              # 原声留几天；转出来的文字不受这个影响
        # 在局域网上开一个只收录音的端口（见 intake.py）。默认关；只能在设置页上开，聊天里一句话开不了——
        # 这是 Alf 唯一对外的口子，不该因为一句（可能是听来的、读来的）话就打开。
        "hearing.intake": False,
        **persona.DEFAULT,                   # 人设卡片：名字、身份、性格、说话习惯、喜好、和他的关系
    }


def load(conn: sqlite3.Connection, cfg: Settings) -> dict:
    prefs = defaults(cfg)
    for row in conn.execute("SELECT key, value FROM settings"):
        if row["key"] in prefs:
            try:
                prefs[row["key"]] = json.loads(row["value"])
            except ValueError:
                pass   # 坏掉的一项就用默认值
    return prefs


def save(conn: sqlite3.Connection, cfg: Settings, changes: dict) -> dict:
    """校验并保存。任何一项不合法就整体拒绝（抛 ValueError，消息给用户看），不会存一半。"""
    base = defaults(cfg)
    clean = {}
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f"没有 {key} 这项设置")
        clean[key] = validate(key, value)
    with conn:
        for key, value in clean.items():
            conn.execute("INSERT INTO settings(key, value, updated_at) VALUES (?, ?, ?)"
                         " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                         (key, json.dumps(value, ensure_ascii=False), int(time.time())))
    return load(conn, cfg)


def validate(key: str, value):
    """一项设置的值合不合法；合法就返回规整后的值，不合法抛 ValueError（消息给用户看）。"""
    def fail(why: str):
        raise ValueError(f"{key}：{why}")

    if key in ("style.emoji", "briefing.enabled", "digest.enabled", "notify.desktop", "memory.auto", "memory.semantic", "calendar.alerts", "followup.enabled",
               "email.report", "impressions.enabled", "search.enabled", "hearing.enabled", "hearing.intake",
               "proactive.enabled", "sources.days", "sources.calendar", "sources.reminders", "sources.news"):
        if not isinstance(value, bool):
            fail("要填开或关")
        return value
    if key in persona.PARTS:
        try:
            return persona.clean(key, value)
        except ValueError as e:
            fail(str(e))
    if key == "style.preset":
        if value not in STYLE_PRESETS:
            fail("只能是 friend、brief 或 secretary")
        return value
    if key in ("style.user_name", "style.extra"):
        limit = 20 if key == "style.user_name" else 500
        if not isinstance(value, str) or len(value.strip()) > limit:
            fail(f"最多 {limit} 个字")
        return value.strip()
    if key == "proactive.level":
        if value not in LEVEL_KEYS:
            fail("只能是 quiet、normal、active 或 chatty")
        return value
    if key == "news.topics":
        if not isinstance(value, list) or any(v not in NEWS_TOPICS for v in value):
            fail("只能从 " + "、".join(NEWS_TOPICS) + " 里选")
        return [t for t in NEWS_TOPICS if t in value]
    if key == "mcp.servers":
        if not isinstance(value, list):
            fail("要是一个列表")
        for item in value:
            ok = (isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"].strip()
                  and isinstance(item.get("command"), str) and item["command"].strip()
                  and isinstance(item.get("args", []), list) and all(isinstance(a, str) for a in item.get("args", []))
                  and isinstance(item.get("env", {}), dict))
            if not ok:
                fail("每一项都要有 name 和 command；args 是字符串列表，env 是键值对")
            if item["name"] in ("days", "calendar", "reminders", "news"):
                fail(f"{item['name']} 是自带服务器的名字，换一个")
        return value
    if key in TIME_KEYS:
        m = _HHMM.fullmatch(str(value).strip())
        if not m:
            fail("时间写成 08:00 这种格式")
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    if key == "email.threshold":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            fail("要在 0 和 1 之间")
        return float(value)
    low, high = {"reminders.default_hour": (0, 23), "reminders.check_every": (5, 300), "email.every": (60, 86400),
                 "calendar.lead_minutes": (0, 240), "followup.after_minutes": (5, 720), "hearing.keep_days": (1, 90)}[key]
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        fail(f"要在 {low} 和 {high} 之间")
    return value


def in_quiet_hours(now, current: dict) -> bool:
    """现在是不是安静时段（跨午夜的也算对）。安静时段里 Alf 不主动开口；用户自己设的提醒照常。"""
    start, end = current.get("quiet.start", "22:00"), current.get("quiet.end", "08:00")
    t = f"{now:%H:%M}"
    return (start <= t < end) if start <= end else (t >= start or t < end)


def hhmm(value: str):
    """"08:00" -> datetime.time"""
    from datetime import time as clock

    hour, minute = value.split(":")
    return clock(int(hour), int(minute))
