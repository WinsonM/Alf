"""One turn: route (a single Jev call, speculative fan-out) -> code handler -> templated reply.

Jev decides; code acts and speaks. A decision below the confidence threshold turns
into a question for the user instead of an action (confidence-gated routing).
"""
from __future__ import annotations

import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from . import fts, timeparse
from .config import Settings
from .decide import INTENT_LABELS, NONE, Answer, Decider, DecisionError, pick, route_questions
from .memory import Memory
from .reminders import Reminder, Reminders


@dataclass
class Quoted:
    """他在 iMessage 里长按某条消息点「回复」时，被回复的那一条。"""
    text: str
    episode_id: int | None = None      # 对得上时间线里的哪一条（对不上就只有原文）
    kind: str | None = None            # 那一条的类型：chat | email | nudge | reminder……
    mine: bool | None = None           # True = 是 Alf 发的；False = 他引用的是自己说过的话；None = 对不上是谁的


@dataclass
class Turn:
    text: str
    source: str = "text"               # "text" | "voice"
    attachment_id: str | None = None
    asr_confidence: float | None = None
    channel: str | None = None         # 从哪个渠道来的（"imessage"）；回复也会送回同一个渠道
    images: list[Path] = field(default_factory=list)   # 随消息发来的图片（已经转成 JPEG）
    reply_to: Quoted | None = None     # 这句话是引用回复时，回的是哪一条


@dataclass
class Reply:
    text: str
    intent: str | None = None
    confidence: float | None = None


HELP = ("我是 Alf，现在能做这些：\n"
        "  · 设提醒：「明天下午三点提醒我交作业」「每天早上8点提醒我吃药」\n"
        "  · 查提醒：「我明天有什么安排」\n"
        "  · 取消提醒：「取消交作业的提醒」\n"
        "  · 记住事情：「我对花生过敏」\n"
        "  · 回忆：「我对什么过敏？」")

_MEMORY_TRIGGER = re.compile(r"^(?:请|麻烦)?(?:帮我)?(?:记住|记一下|记下|remember(?:\s+that)?)[:：，,\s]*", re.I)


def duration(minutes: int) -> str:
    if minutes % 1440 == 0:
        return f"{minutes // 1440}天"
    if minutes == 30:
        return "半小时"
    if minutes % 60 == 0:
        return f"{minutes // 60}小时"
    if minutes > 60 and minutes % 60 == 30:
        return f"{minutes // 60}个半小时"
    return f"{minutes}分钟"


def period_word(hour: int) -> str:
    return "凌晨" if hour < 6 else "上午" if hour < 12 else "中午" if hour == 12 else "下午" if hour < 18 else "晚上"


def format_reminders(items: list[Reminder], now: datetime) -> str:
    lines = []
    for r in items:
        line = f"  #{r.id}  {timeparse.describe(r.due, now)}  {r.title}"
        if r.recurrence:
            line += f"  [{timeparse.describe_recurrence(r.recurrence)}]"
        if r.lead_minutes:
            line += f"  (提前{duration(r.lead_minutes)})"
        lines.append(line)
    return "\n".join(lines)


class Agent:
    def __init__(self, conn: sqlite3.Connection, decider: Decider, cfg: Settings,
                 clock: Callable[[], datetime] | None = None, session_id: str | None = None):
        self.conn = conn
        self.decider = decider
        self.cfg = cfg
        self.clock = clock or (lambda: datetime.now(cfg.tz))
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.memory = Memory(conn, decider, cfg.decision_threshold)
        self.reminders = Reminders(conn, cfg.tz)
        self._pending: dict | None = None   # a question Alf asked and is waiting on

    def handle(self, turn: Turn, on_reply: Callable[[Reply], None] | None = None) -> Reply:
        now = self.clock()
        episode = self._log("user", turn, now)
        try:
            # 这个只用 Jev 的模式没有能看图的模型。
            reply = Reply("图片我现在看不了：要先在设置里填 DeepSeek 的 key。") if turn.images else self._handle(turn, now, episode)
        except DecisionError as e:
            reply = Reply(f"抱歉，现在做不了判断：{e}")
        self._log("alf", Turn(reply.text, channel=turn.channel), now)
        if on_reply:
            on_reply(reply)
        return reply

    # --- routing ---------------------------------------------------------------

    def _handle(self, turn: Turn, now: datetime, episode: int) -> Reply:
        pending, self._pending = self._pending, None
        text = turn.text.strip()
        if pending and pending["kind"] == "intent" and text in ("1", "2"):
            p = pending
            return self._dispatch(p["options"][int(text) - 1], p["turn"], now, p["episode"], p["answers"])
        if pending and pending["kind"] == "time" and timeparse.parse(text, now, self.cfg.default_hour).times:
            return self._create_reminder(pending["turn"], now, pending["episode"], pending["answers"], when_text=text)

        answers = self.decider.ask("route", {"message": turn.text}, route_questions())
        intent = answers["intent"]
        if intent.confidence < self.cfg.decision_threshold:
            options = sorted(intent.probabilities, key=intent.probabilities.get, reverse=True)[:2]
            self._pending = {"kind": "intent", "options": options, "turn": turn, "episode": episode, "answers": answers}
            return Reply(f"我不太确定你的意思，你是想：\n  1. {INTENT_LABELS[options[0]]}\n"
                         f"  2. {INTENT_LABELS[options[1]]}\n回复 1 或 2，或者换个说法。", None, intent.confidence)
        return self._dispatch(intent.choice, turn, now, episode, answers)

    def _dispatch(self, intent: str, turn: Turn, now: datetime, episode: int, answers: dict[str, Answer]) -> Reply:
        handler = {
            "create_reminder": self._create_reminder,
            "list_reminders": self._list,
            "cancel_reminder": self._cancel,
            "remember_fact": self._remember,
            "recall": self._recall,
        }.get(intent, self._other)
        reply = handler(turn, now, episode, answers)
        # A reminder request can carry a lasting fact ("明天是我妈生日，提醒我买花"): memorise it on the way.
        if intent == "create_reminder" and answers["states_fact"].noul >= self.cfg.fact_side_threshold:
            if note := self._memorise(turn, now, episode, answers, quiet=True):
                reply.text += "\n" + note
        reply.intent, reply.confidence = intent, answers["intent"].confidence
        return reply

    # --- reminders ---------------------------------------------------------------

    def _create_reminder(self, turn: Turn, now: datetime, episode: int, answers: dict[str, Answer],
                         when_text: str | None = None) -> Reply:
        own = timeparse.parse(turn.text, now, self.cfg.default_hour)
        parsed = timeparse.parse(when_text, now, self.cfg.default_hour) if when_text else own
        what = timeparse.title(turn.text, own)
        waiting = {"kind": "time", "turn": turn, "episode": episode, "answers": answers}
        if not parsed.times:
            self._pending = waiting
            return Reply(f"要在什么时候提醒你{what}？可以说「明天下午3点」或「半小时后」。")
        chosen = parsed.times[0]
        if len(parsed.times) > 1:
            a = self.decider.ask("reminder.when", {"message": when_text or turn.text}, {
                "when": pick("Which time expression in the message is when the user wants to be reminded, "
                             "or when the event happens?",
                             {f"t{i}": t.span for i, t in enumerate(parsed.times)},
                             "None of these is the reminder time."),
            })["when"]
            if a.choice == NONE or a.confidence < self.cfg.decision_threshold:
                self._pending = waiting
                spans = "、".join(f"「{t.span}」" for t in parsed.times)
                return Reply(f"你提到了几个时间：{spans}，要在哪个时间提醒你？")
            chosen = parsed.times[int(a.choice[1:])]
        if chosen.when <= now:
            self._pending = waiting
            return Reply(f"{timeparse.describe(chosen.when, now)} 已经过去了，换个时间？")
        r = self.reminders.add(what or "提醒", chosen.when, now=int(now.timestamp()),
                               lead_minutes=parsed.lead_minutes, recurrence=parsed.recurrence,
                               source=turn.source, attachment_id=turn.attachment_id, episode_id=episode)
        return Reply(self._confirmation(r, chosen, now, turn))

    def _confirmation(self, r: Reminder, chosen: timeparse.TimeExpr, now: datetime, turn: Turn) -> str:
        when = timeparse.describe(r.due, now)
        if r.recurrence:
            text = f"好的，{timeparse.describe_recurrence(r.recurrence)} {r.due:%H:%M} 提醒你：{r.title}（第一次是 {when}）"
        elif r.lead_minutes:
            text = f"好的，「{r.title}」在 {when}，我会提前{duration(r.lead_minutes)}（{r.notify:%H:%M}）提醒你"
        else:
            text = f"好的，{when} 提醒你：{r.title}"
        notes = []
        if chosen.time_defaulted:
            notes.append(f"你没说几点，先定在 {r.due:%H:%M}")
        if chosen.ampm_guessed:
            notes.append(f"「{chosen.span}」我理解成{period_word(r.due.hour)} {r.due:%H:%M}")
        if turn.source == "voice" and (turn.asr_confidence or 0) < self.cfg.asr_low_confidence:
            notes.append(f"语音可能没听准，我听到的是「{turn.text}」")
        if notes:
            text += "\n（" + "；".join(notes) + "，不对的话告诉我）"
        return text

    def _list(self, turn: Turn, now: datetime, episode: int, answers: dict[str, Answer]) -> Reply:
        dated = next((t for t in timeparse.parse(turn.text, now, self.cfg.default_hour).times if t.has_date), None)
        if dated:
            start = dated.when.replace(hour=0, minute=0, second=0, microsecond=0)
            label = timeparse.describe(start, now)[:-5]
            items = self.reminders.pending(start, start + timedelta(days=1))
            return Reply(f"{label}的提醒：\n{format_reminders(items, now)}" if items else f"{label}没有安排。")
        items = self.reminders.pending(limit=20)
        return Reply(f"待办提醒：\n{format_reminders(items, now)}" if items else "现在没有待办提醒。")

    def _cancel(self, turn: Turn, now: datetime, episode: int, answers: dict[str, Answer]) -> Reply:
        items = self.reminders.pending(limit=20)
        if not items:
            return Reply("现在没有待办提醒。")
        a = self.decider.ask("reminder.cancel", {"message": turn.text}, {
            "target": pick("Which existing reminder does the user want to cancel, delete or mark as done?",
                           {f"r{r.id}": f"{r.title} ({timeparse.describe(r.due, now)})" for r in items},
                           "None of these reminders."),
        })["target"]
        if a.choice != NONE and a.confidence >= self.cfg.decision_threshold:
            if r := self.reminders.cancel(int(a.choice[1:])):
                return Reply(f"已取消：{r.title}（{timeparse.describe(r.due, now)}）")
        return Reply(f"不确定你要取消哪一个：\n{format_reminders(items, now)}\n用 /cancel <编号> 取消。")

    # --- memory ------------------------------------------------------------------

    def _remember(self, turn: Turn, now: datetime, episode: int, answers: dict[str, Answer]) -> Reply:
        return Reply(self._memorise(turn, now, episode, answers, quiet=False) or "我没找到需要记住的内容。")

    def _memorise(self, turn: Turn, now: datetime, episode: int, answers: dict[str, Answer], quiet: bool) -> str | None:
        parts = timeparse.clauses(turn.text)
        if quiet and len(parts) < 2:
            return None  # the only clause is the reminder request itself
        if len(parts) > 1:
            a = self.decider.ask("memory.clause", {"message": turn.text}, {
                "clause": pick("Which part of the message states a lasting fact about the user or their life?",
                               {f"c{i}": c for i, c in enumerate(parts)}, "No part states a lasting fact."),
            })["clause"]
            if a.choice == NONE or a.confidence < self.cfg.decision_threshold:
                return None
            statement = parts[int(a.choice[1:])]
        else:
            statement = turn.text.strip()
        statement = timeparse.absolutize(_MEMORY_TRIGGER.sub("", statement).strip(), now)
        if not statement:
            return None
        unsure = turn.source == "voice" and (turn.asr_confidence or 0) < self.cfg.asr_low_confidence
        res = self.memory.write(statement, answers["fact_category"].choice, now=int(now.timestamp()),
                                source=turn.source, attachment_id=turn.attachment_id, episode_id=episode,
                                confidence=turn.asr_confidence, needs_confirmation=unsure)
        if res.status == "duplicate":
            msg = f"这个我已经记着了：{res.fact.text}"
        elif res.status == "updated":
            msg = "更新了记忆：「" + "」「".join(f.text for f in res.superseded) + f"」→「{res.fact.text}」"
        else:
            msg = f"记住了：{res.fact.text}"
        if res.maybe_same:
            msg += "\n它可能取代了原来的「" + "」「".join(f.text for f in res.maybe_same) + "」，我拿不准，两条都留着了。"
        if unsure:
            msg += "（语音识别置信度较低，如果我听错了请纠正我）"
        return msg

    def _recall(self, turn: Turn, now: datetime, episode: int, answers: dict[str, Answer]) -> Reply:
        fact, _ = self.memory.recall(turn.text)
        if not fact:
            return Reply("我没有这方面的记录。")
        note = "，来自语音、还没确认" if fact.needs_confirmation else ""
        return Reply(f"我记得：{fact.text}（{fact.recorded(self.cfg.tz)}记下的{note}）")

    def _other(self, turn: Turn, now: datetime, episode: int, answers: dict[str, Answer]) -> Reply:
        return Reply(HELP)

    def _log(self, role: str, turn: Turn, now: datetime) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO episodes(session_id, role, text, source, attachment_id, created_at, channel)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, role, turn.text, turn.source, turn.attachment_id, int(now.timestamp()), turn.channel))
            if role == "user":
                fts.index(self.conn, "episodes_fts", cur.lastrowid, turn.text)
        return cur.lastrowid
