"""主动找用户说话：要不要说、说什么。

每隔一阵（由活跃度决定）想一次，分四步，前一步不过就停：
1. 程序的硬规则：安静时段、今天说够了没有、离上一条够不够久、用户是不是正在聊天。不花钱。
2. DeepSeek 看一遍"现在的情况"（日程、特殊日子、新闻、记忆、最近对话、它今天说过什么），按人设决定
   有没有值得说的，有就写出那句话和理由。这一步不带任何工具：情况里有新闻标题、日程名这些外部内容，
   它们最多影响一句话怎么说，指挥不了 Alf 做别的事。
3. Jev 给"用户现在愿不愿意收到这条"打分，过了活跃度对应的门槛才发。
4. 发出，并把这次判断记进 proactive_log——发了、忍住了、还是没话说，理由是什么。

"聊天习惯"体现在两处：用户平时这个钟点活不活跃，会告诉模型和 Jev；最近几条主动消息如果都没人理，
间隔自动加倍。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from . import affinity, impressions, lessons, mood, pacts, persona, timeparse
from .chat import DETAIL_ALWAYS, PROACTIVE_LABELS, RULE_FLOOR, reminder_line, strip_fake_receipts, style_lines
from .decide import DecisionError, broke_rule_question, noul
from .heartbeat import JobContext, load_state
from .llm import LLMError
from .memory import FEEDBACK, LOOP_MAX_ASKS, LOOP_QUIET, Memory
from .prefs import in_quiet_hours
from .reminders import Reminders


@dataclass(frozen=True)
class Level:
    label: str
    per_day: int        # 一天最多主动几条
    gap: int            # 两条之间至少隔多少分钟
    threshold: float    # Jev 打分要到多少才发
    wake: int           # 多少分钟想一次
    guide: str          # 写进提示词里的尺度


# 门槛是按 2026-11-16 的一次实测定的（14 条样本）：不该发的消息 Jev 给 0.07–0.36（无关新闻、凌晨、
# 复述已有提醒、最近几条都没人理），该发的给 0.36–0.73（明天的日程、相关的重大新闻在 0.66 以上）。
# 样本很少，应该隔一阵对着 proactive_log 里的真实分数再校一次。
LEVELS = {
    "quiet": Level("安静", 2, 240, 0.70, 30, "很克制：只在有明确要紧的事时开口，宁可不说。"),
    "normal": Level("适中", 5, 120, 0.55, 20, "适中：只说对他比较有用的事。"),
    "active": Level("活跃", 10, 60, 0.40, 15, "活跃：只要有点实际内容、对他有用或他会感兴趣，就可以说。"),
    "chatty": Level("很活跃", 20, 30, 0.30, 10, "很活跃：想到就说，除了有用的事，也可以单纯关心一句或聊两句。"),
}
KINDS = {"schedule": "日程", "special_day": "特殊日子", "news": "新闻", "follow_up": "跟进", "memory": "记忆", "care": "关心"}
BUSY_MINUTES = 5          # 用户这么久之内说过话，说明正聊着，不插嘴
ANSWER_WINDOW = 2 * 3600  # 主动消息发出后这么久内用户说过话，算"有回应"
STALE_SNAPSHOT = 3 * 3600

THINK_PROMPT = """你是 {name}，用户的私人助理，也是他的朋友。现在没有人在跟你说话，你在考虑要不要主动给他发一条 iMessage。

## 你的尺度
{guide}

## 值得主动说的
- 接下来的日程需要准备或该出发了
- 今天或近几天是特别的日子（节日、节气、他记挂的人的生日），顺带一个具体的小建议
- 和他关心的主题有关的重大新闻：只挑真正的大事，一次最多一条，一句话说清是什么、为什么他可能在意
- 他提过要做、看起来还没做的事（刚发出的提醒不算：那个程序会自己追问，不用你管）
- 根据你记得的事想到的、现在提正合适的话
- [刚过去的事]或[惦记的事]里的某一件：问一句怎么样、准备得如何（一次只问一件；[最近对话]里刚聊过的不要再问）。问了就在输出里用 about 写上那件事的编号
- [你们的约定]里的一条：你答应过的事该兑现了，或者他答应的事该问一句进展（一次只提一条，别催）。提了就在输出里用 pact 写上那条约定的编号（「约」后面的数字）

## 不要
- 重复你今天已经主动说过的内容，同一件事不说第二遍
- 把他自己设的提醒再念一遍（提醒到点会单独发）
- 没话找话。没有值得说的就不发，这很正常
- 把两件不相干的事揉进一条消息（比如拿节气去引出生日）。一条只说一件事；说到"还有几天"时照抄给你的天数
- 编造日程、日期或新闻。只用下面给你的信息；提到时间和日期时照抄给你的写法，不要自己换算

## 怎么说
{style}
只用纯文本，不用 Markdown。一条消息说一件事，通常一两句话。

## 输出
只输出一个 JSON 对象。
要发：{{"send": true, "kind": "schedule|special_day|news|follow_up|memory|care", "message": "要发给他的话", "why": "为什么现在说这个，一句话", "about": 提到的那件事的编号（没有就写 null）, "pact": 提到的那条约定的编号（没有就写 null）}}
不发：{{"send": false, "why": "为什么不发，一句话"}}"""


def gate_question() -> dict:
    return noul(
        "Right now is a good moment for the assistant to send this message unprompted: the user would be glad to "
        "get it, rather than feel interrupted, nagged, or told something they already know.",
        true="Welcome and well timed.",
        false="Better left unsent right now.",
    )


@dataclass
class Habits:
    nudges_today: int
    minutes_since_nudge: float | None
    minutes_since_user: float | None
    minutes_since_alf: float | None      # Alf 上一次主动发任何东西（含提醒、早报）
    ignored_streak: int                  # 最近连续几条主动消息没有得到回应
    last_nudge_answered: bool | None
    active_now: bool | None              # 用户平时这个钟点活不活跃；None = 数据还不够

    @classmethod
    def load(cls, conn: sqlite3.Connection, now: datetime) -> Habits:
        ts = int(now.timestamp())
        midnight = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())

        def since(where: str) -> float | None:
            last = conn.execute(f"SELECT MAX(created_at) FROM episodes WHERE {where}").fetchone()[0]
            return None if last is None else (ts - last) / 60

        nudges = [r[0] for r in conn.execute(
            "SELECT created_at FROM episodes WHERE kind = 'nudge' ORDER BY id DESC LIMIT 6")]
        streak, last_answered = 0, None
        for sent in nudges:
            if ts - sent < ANSWER_WINDOW:
                continue                                  # 还没到下结论的时候
            answered = conn.execute("SELECT 1 FROM episodes WHERE role = 'user' AND created_at BETWEEN ? AND ?",
                                    (sent, sent + ANSWER_WINDOW)).fetchone() is not None
            if last_answered is None:
                last_answered = answered
            if answered:
                break
            streak += 1
        hours = [r[0] for r in conn.execute(
            "SELECT CAST(strftime('%H', created_at, 'unixepoch', 'localtime') AS INTEGER) FROM episodes"
            " WHERE role = 'user' AND created_at >= ?", (ts - 14 * 86400,))]
        active_now = None if len(hours) < 20 else hours.count(now.hour) >= 2
        return cls(
            nudges_today=conn.execute("SELECT COUNT(*) FROM episodes WHERE kind = 'nudge' AND created_at >= ?", (midnight,)).fetchone()[0],
            minutes_since_nudge=since("kind = 'nudge'"), minutes_since_user=since("role = 'user'"),
            minutes_since_alf=since("role = 'alf' AND kind != 'chat'"),
            ignored_streak=streak, last_nudge_answered=last_answered, active_now=active_now)


def blocked(habits: Habits, level: Level, now: datetime, current: dict) -> str | None:
    """硬规则。返回不该开口的原因；None 表示可以去问模型了。"""
    if in_quiet_hours(now, current):
        return "安静时段"
    if habits.minutes_since_user is not None and habits.minutes_since_user < BUSY_MINUTES:
        return "你正在跟它聊天"
    if habits.nudges_today >= level.per_day:
        return f"今天已经主动说了 {habits.nudges_today} 条，到上限了"
    gap = level.gap * (2 if habits.ignored_streak >= 3 else 1)
    if habits.minutes_since_nudge is not None and habits.minutes_since_nudge < gap:
        return f"离上一条主动消息还不到 {gap} 分钟" + ("（最近几条都没回应，间隔加倍了）" if gap != level.gap else "")
    if habits.minutes_since_alf is not None and habits.minutes_since_alf < 10:
        return "刚发过提醒或播报"
    return None


def describe_snapshot(snapshot: dict, now: datetime) -> list[str]:
    """把 MCP 汇总来的快照写成给模型看的几段文字。时间都在这里换成本地时间，模型照抄即可。"""
    lines: list[str] = []
    sources = snapshot.get("sources", {})
    if at := snapshot.get("at"):
        age = now.timestamp() - at
        if age > STALE_SNAPSHOT:
            lines.append(f"（下面的外部信息是 {age / 3600:.0f} 小时前取的，可能不是最新）")

    days = sources.get("days", {}).get("special_days")
    if isinstance(days, dict):
        today = "、".join(days.get("today", [])) or "平常的一天"
        lines += ["[今天是什么日子]", f"{days.get('lunar', '')}。今天：{today}"]
        for d in days.get("upcoming", []):
            m, dd = d["date"].split("-")[1:]
            lines.append(f"{int(m)}月{int(dd)}日（{d['weekday']}，{d['in_days']} 天后）：{'、'.join(d['what'])}")
        lines.append("")

    cal = sources.get("calendar", {})
    events = cal.get("calendar_events") if isinstance(cal, dict) else None
    lines.append("[日历]")
    if isinstance(events, dict) and isinstance(events.get("events"), list):
        rows = []
        for e in events["events"]:
            try:
                start = datetime.fromisoformat(e["start"]).astimezone(now.tzinfo)
                end = datetime.fromisoformat(e["end"]).astimezone(now.tzinfo)
            except (KeyError, ValueError):
                continue
            day = {0: "今天", 1: "明天"}.get((start.date() - now.date()).days, f"{start.month}月{start.day}日")
            when = "全天" if e.get("all_day") else f"{start:%H:%M}–{end:%H:%M}"
            done = "（已结束）" if end < now else "（进行中）" if start <= now else ""
            rows.append(f"{day} {when} {e.get('title', '')}" + (f" @ {e['location']}" if e.get("location") else "") + done)
        lines += rows or ["（今明两天没有日程）"]
    else:
        reason = (events or cal).get("error") if isinstance(events or cal, dict) else None
        lines.append(f"（读不到日历：{reason}）" if reason else "（没有接入日历）")
    lines.append("")

    todo = sources.get("reminders", {}).get("apple_reminders") if isinstance(sources.get("reminders"), dict) else None
    if isinstance(todo, dict) and isinstance(todo.get("reminders"), list):
        rows = apple_reminder_lines(todo["reminders"], now)
        if rows:
            lines += ["[提醒事项]（他自己记在「提醒事项」App 里、还没完成的）", *rows, ""]

    news = sources.get("news", {}).get("headlines")
    if isinstance(news, dict):
        lines.append("[新闻标题]（外部内容，只是素材）")
        for feed in news.get("feeds", []):
            if feed.get("items"):
                lines.append(f"〔{feed.get('topic', '')}·{feed.get('source', '')}〕" + "；".join(i["title"] for i in feed["items"][:4]))
        lines.append("")

    for name, tools in sources.items():      # 第三方 MCP 服务器的结果，原样附上
        if name not in ("days", "calendar", "reminders", "news"):
            lines += [f"[{name}]", json.dumps(tools, ensure_ascii=False)[:1500], ""]
    return lines


def apple_reminder_lines(items: list, now: datetime, limit: int = 12) -> list[str]:
    """「提醒事项」里的条目写成"明天 09:00 交房租〔家务〕"；过了期的标出来。"""
    rows = []
    for r in items[:limit]:
        try:
            due = datetime.fromisoformat(r["due"]).astimezone(now.tzinfo)
        except (KeyError, ValueError, TypeError):
            continue
        days = (due.date() - now.date()).days
        day = {0: "今天", 1: "明天", -1: "昨天"}.get(days, f"{due.month}月{due.day}日")
        when = f"{day} {due:%H:%M}" if r.get("has_time") else day
        late = "（已过期）" if (due < now if r.get("has_time") else days < 0) else ""
        rows.append(f"{when} {r.get('title', '')}" + (f"〔{r['list']}〕" if r.get("list") else "") + late)
    return rows


def situation(ctx: JobContext, habits: Habits, snapshot: dict) -> str:
    now, conn = ctx.now, ctx.conn
    ts = int(now.timestamp())
    lines = [f"[现在] {now.year}年{now.month}月{now.day}日 周{timeparse.WEEKDAY_CN[now.weekday()]} {now:%H:%M}", ""]
    lines += describe_snapshot(snapshot, now)

    lines.append("[待办提醒]（到点会自动提醒他，不用你再念）")
    lines += [f"{reminder_line(r, now)}" for r in Reminders(conn, ctx.cfg.tz).pending(limit=15)] or ["（没有）"]

    fired = conn.execute("SELECT text, created_at FROM episodes WHERE kind = 'reminder' AND created_at >= ? ORDER BY id",
                         (ts - 3 * 3600,)).fetchall()
    if fired:
        lines += ["", "[刚发出的提醒]"]
        for r in fired:
            replied = conn.execute("SELECT 1 FROM episodes WHERE role = 'user' AND created_at > ?", (r["created_at"],)).fetchone()
            lines.append(f"{datetime.fromtimestamp(r['created_at'], now.tzinfo):%H:%M} {r['text']}——{'之后他说过话' if replied else '之后他还没有任何回应'}")

    lines += ["", "[你记得的关于他的事]"]
    memory = Memory(conn, ctx.decider, ctx.cfg.decision_threshold)
    memory.expire(ts)                                    # 带日期的事过了那天，就挪到[刚过去的事]
    # 常驻的全带；主题里的细节不多时也带上，多了就只给目录——主动找他说话用不着翻细账。
    topics = memory.topics()
    detail = sum(n for _, n, _ in topics) <= DETAIL_ALWAYS
    facts = [f for f in memory.core(limit=60) if f.category != FEEDBACK] + [f for name, _, _ in topics if detail for f in memory.in_topic(name)]
    for f in sorted(facts, key=lambda f: f.id):
        note = ""
        for t in timeparse.parse(f.text, now).times:      # 记忆里带日期的（生日……），由程序算好还有几天
            if t.date_kind == "md":
                left = (t.when.date() - now.date()).days
                if 0 <= left <= 30:
                    note = f"（{'就是今天' if left == 0 else f'还有 {left} 天'}，周{timeparse.WEEKDAY_CN[t.when.weekday()]}）"
        lines.append(f"{f.text}{note}")
    if not facts:
        lines.append("（还没有）")
    if topics and not detail:
        lines.append("另外按主题记着这些细节（这里没列）：" + "、".join(f"{name} {n} 条" for name, n, _ in topics))
    # 只带管得到这里的要求：哪里都管的，和专管"主动找他"的。专管聊天、专管邮件的不带，免得它拿来对号入座。
    if rules := lessons.rules(conn, "proactive"):
        lines += ["", "[他对你的要求]（照做）"] + [f.text for f in rules]
    # 和聊天时同一个分寸：今天聊过的、提了两次他都没接话的，不再摆出来。
    past = [f for f in memory.just_past(ts) if f.asks < LOOP_MAX_ASKS and not (f.touched_at and ts - f.touched_at < LOOP_QUIET)]
    if past:
        lines += ["", "[刚过去的事]"] + [f"#{f.id} {f.text}" for f in past]
    if loops := memory.open_loops(ts, limit=4):
        lines += ["", "[惦记的事]（还没了结，按要紧程度排）"] + [f"#{f.id} {f.text}" for f, _ in loops]
    pacts.lapse(conn, ts)
    if agreed := pacts.situation_lines(conn, now):
        lines += ["", "[你们的约定]（说好的事。你答应的到时候要兑现；他答应的可以问一句进展）"] + agreed
    moments = memory.moments(limit=5, since=ts - 30 * 86400)[::-1]
    if his := [f.text for f in moments if not f.ours]:
        lines += ["", "[最近的经历]"] + his
    if ours := [f.text for f in moments if f.ours]:
        lines += ["", "[你们之间的事]"] + ours
    if views := impressions.context_lines(conn, ts):
        lines += ["", "[你对他的印象]（你自己的看法，不是他说的事实：用来把握分寸，不要当面下结论）"] + views

    history = conn.execute("SELECT role, kind, text, created_at FROM episodes WHERE created_at >= ? ORDER BY id DESC LIMIT 12",
                           (ts - 24 * 3600,)).fetchall()[::-1]
    lines += ["", "[最近对话]"]
    for r in history:
        who = "他" if r["role"] == "user" else "你"
        tag = f"[{PROACTIVE_LABELS[r['kind']]}] " if r["kind"] in PROACTIVE_LABELS else ""
        lines.append(f"{datetime.fromtimestamp(r['created_at'], now.tzinfo):%H:%M} {who}：{tag}{r['text'][:200]}")
    if not history:
        lines.append("（过去一天没有聊过）")

    midnight = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    said = conn.execute("SELECT text, created_at FROM episodes WHERE kind = 'nudge' AND created_at >= ? ORDER BY id", (midnight,)).fetchall()
    lines += ["", "[你今天已经主动说过的]"]
    lines += [f"{datetime.fromtimestamp(r['created_at'], now.tzinfo):%H:%M} {r['text'][:200]}" for r in said] or ["（还没有）"]

    # 他点过 👎、回过"没用"、或者专门挑过毛病的那几条主动消息，连同他当时怎么说的。
    if faults := lessons.example_lines(conn, "proactive"):
        lines += ["", "[他挑过毛病的主动消息]（同类的别再发）"] + faults

    lines += ["", "[他的习惯]"]
    if habits.minutes_since_user is not None:
        lines.append(f"他上一次说话是 {_ago(habits.minutes_since_user)}。")
    if habits.active_now is not None:
        lines.append("他平时这个钟点" + ("比较活跃。" if habits.active_now else "很少说话。"))
    if habits.ignored_streak:
        lines.append(f"你最近 {habits.ignored_streak} 条主动消息他都没有回应。")
    elif habits.last_nudge_answered:
        lines.append("你上一条主动消息他回应了。")
    return "\n".join(lines)


def _ago(minutes: float) -> str:
    return f"{minutes:.0f} 分钟前" if minutes < 90 else f"{minutes / 60:.0f} 小时前" if minutes < 48 * 60 else f"{minutes / 1440:.0f} 天前"


def record(conn: sqlite3.Connection, now: datetime, outcome: str, *, kind: str | None = None, message: str | None = None,
           why: str | None = None, score: float | None = None) -> None:
    with conn:
        conn.execute("INSERT INTO proactive_log(ts, outcome, kind, message, why, score) VALUES (?, ?, ?, ?, ?, ?)",
                     (int(now.timestamp()), outcome, kind, message, why, score))


def recent(conn: sqlite3.Connection, limit: int = 30) -> list[dict]:
    rows = conn.execute("SELECT ts, outcome, kind, message, why, score FROM proactive_log ORDER BY id DESC LIMIT ?", (limit,))
    return [{**dict(r), "kind_label": KINDS.get(r["kind"] or "", r["kind"])} for r in rows]


def think_prompt(current: dict, level: Level, points: int = affinity.START, feeling: str | None = None) -> str:
    # 和聊天时同一个人：先是人设（底色），再是同一套说话方式——亲近程度跟着好感度走，再叠上眼前的气氛。
    style = [*persona.card_lines(current), "上面是你的底色；眼下说话有多随意，以下面几句为准。", *style_lines(current, points, feeling)]
    if name := current.get("style.user_name"):
        style.append(f"称呼他为「{name}」。")
    if extra := current.get("style.extra"):
        style.append(f"他对你说话方式的额外要求：{extra}")
    return THINK_PROMPT.format(name=persona.name(current), guide=level.guide, style="\n".join(style))


def proactive_job(ctx: JobContext) -> str | None:
    current = ctx.prefs
    if ctx.llm is None or not current.get("proactive.enabled", True):
        return None
    level = LEVELS.get(current.get("proactive.level", "active"), LEVELS["active"])
    habits = Habits.load(ctx.conn, ctx.now)
    if reason := blocked(habits, level, ctx.now, current):
        return f"没有开口：{reason}"

    try:
        snapshot = json.loads(load_state(ctx.conn, "sources").cursor or "{}")
    except ValueError:
        snapshot = {}
    try:
        data = ctx.llm.json("proactive.think", [
            {"role": "system", "content": think_prompt(current, level, affinity.score(ctx.conn),
                                                      mood.tone(ctx.conn, int(ctx.now.timestamp())))},
            {"role": "user", "content": situation(ctx, habits, snapshot)},
        ], max_tokens=400)
    except LLMError as e:
        return f"没想成：{e}"

    why = str(data.get("why") or "")[:200]
    message = strip_fake_receipts(str(data.get("message") or ""))[:400]
    if not data.get("send") or not message:
        record(ctx.conn, ctx.now, "silent", why=why or "没有值得说的")
        return f"没有开口：{why or '没有值得说的'}"
    kind = data.get("kind") if data.get("kind") in KINDS else "care"

    # 发之前核对：这句话有没有违反他提过的要求（和聊天回复是同一道核对）。违反了就不发——主动消息不是非发不可的。
    if broken := broken_rules(ctx, message):
        record(ctx.conn, ctx.now, "held", kind=kind, message=message, why="不合他的要求：" + "；".join(broken))
        return f"想说一句但没发：不合他提过的要求（{'；'.join(broken)}）"

    score = None
    if ctx.decider.backend.name == "jev":       # 只有 Jev 的分数是校准过的概率，才拿来卡门槛
        try:
            score = ctx.decider.ask("proactive.gate", {
                "now": f"{now_text(ctx.now)}", "message": message, "why_now": why,
                "minutes_since_user_last_spoke": None if habits.minutes_since_user is None else round(habits.minutes_since_user),
                "unprompted_messages_already_sent_today": habits.nudges_today,
                "recent_unprompted_messages_ignored": habits.ignored_streak,
                "user_usually_chats_at_this_hour": habits.active_now,
            }, {"welcome": gate_question()})["welcome"].noul
        except DecisionError:
            score = None                         # Jev 一时不可用：不因此把话憋回去
        if score is not None and score < level.threshold:
            record(ctx.conn, ctx.now, "held", kind=kind, message=message, why=why, score=score)
            return f"想说一句但忍住了（把握 {score:.0%}，低于 {level.threshold:.0%}）"

    ctx.post("nudge", message)
    record(ctx.conn, ctx.now, "sent", kind=kind, message=message, why=why, score=score)
    about = data.get("about")
    if isinstance(about, int) and not isinstance(about, bool):
        # 主动问起了一件惦记的事：记一次。他接话了会清零；连问两次都没接话，就不再问。
        memory = Memory(ctx.conn, ctx.decider, ctx.cfg.decision_threshold)
        ts = int(ctx.now.timestamp())
        if about in {f.id for f in memory.just_past(ts)} | {f.id for f, _ in memory.open_loops(ts, limit=50)}:
            memory.touch(about, ts, by_user=False)
    pact = data.get("pact")
    if isinstance(pact, int) and not isinstance(pact, bool) and (row := pacts.get(ctx.conn, pact)) and row["status"] == "active":
        # 主动提了一条约定：和惦记的事一样记一次，免得天天催。
        pacts.touch(ctx.conn, pact, int(ctx.now.timestamp()), by_user=False)
    return f"主动说了一句（{KINDS[kind]}）"


def broken_rules(ctx: JobContext, message: str) -> list[str]:
    """这条主动消息违反了他提过的哪几条要求。没有要求就不问；判断不了就当没违反。"""
    rules = lessons.rules(ctx.conn, "proactive")
    if not rules:
        return []
    try:
        a = ctx.decider.ask("proactive.rules", {"user_latest_message": "", "assistant_reply": message},
                            {f"rule_{f.id}": broke_rule_question(f.text) for f in rules})
    except DecisionError:
        return []
    return [f.text for f in rules if a[f"rule_{f.id}"].noul >= RULE_FLOOR]


def now_text(now: datetime) -> str:
    return f"{now:%Y-%m-%d} {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][now.weekday()]} {now:%H:%M}"


def next_quiet_end(now: datetime, current: dict) -> datetime:
    """安静时段什么时候结束（给界面显示用）。"""
    hour, minute = map(int, current.get("quiet.end", "08:00").split(":"))
    end = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return end if end > now else end + timedelta(days=1)
