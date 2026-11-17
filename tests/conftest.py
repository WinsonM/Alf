from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import json
import sqlite3

from alf import db
from alf.agent import Agent
from alf.chat import ChatAgent
from alf.config import Settings
from alf.decide import Answer, Decider
from alf.llm import LLM, LLMReply, ToolCall

TZ = ZoneInfo("Australia/Sydney")
NOW = datetime(2026, 9, 22, 10, 0, tzinfo=TZ)  # Tuesday


class Scripted:
    """Stands in for Jev. Rules map a question name (or its prefix before the last '_', so 'rel'
    covers 'rel_12') to a label, a (label, confidence, probabilities) tuple, a float for Noul,
    or a callable(question, state) returning one of those."""

    name = "scripted"

    def __init__(self, **rules):
        self.rules = rules
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        out = {}
        for name, q in questions.items():
            rule = self.rules.get(name, self.rules.get(name.rsplit("_", 1)[0]))
            if callable(rule):
                rule = rule(q, state)
            if q["type"] == "noul":
                p = 0.1 if rule is None else rule
                out[name] = Answer(noul=p, confidence=max(p, 1 - p))
                continue
            if rule is None:
                raise KeyError(f"no scripted answer for {name}")
            label, conf, probs = rule if isinstance(rule, tuple) else (rule, 0.95, None)
            out[name] = Answer(choice=label, confidence=conf, probabilities=probs or {label: conf})
        return out, {"model": "scripted", "input_tokens": 10, "output_tokens": 0}


def option_containing(text):
    """Rule helper: pick the option whose description contains `text`."""
    return lambda q, state: next(k for k, v in q["criteria"].items() if v and text in v)


@pytest.fixture
def conn():
    return db.connect(":memory:")


@pytest.fixture
def cfg(tmp_path):
    return Settings(tz=TZ, db_path=tmp_path / "alf.db", attachment_dir=tmp_path / "attachments",
                    decision_threshold=0.6, fact_side_threshold=0.8, default_hour=9,
                    asr_model="tiny", asr_language=None, asr_low_confidence=0.6)


@pytest.fixture
def make_agent(conn, cfg):
    def make(backend, now=NOW):
        return Agent(conn, Decider(backend, conn), cfg, clock=lambda: now)
    return make


class FakeLLM:
    """照剧本回答的生成式模型。

    对话调用按顺序取 chat 里的一项：字符串 = 最终回复；列表 = 工具调用 [(工具名, 参数)]；异常 = 抛出。
    JSON 调用（记忆抽取）按顺序取 facts 里的一项，取完了就回答"没有要记的"。
    """

    name, model = "fake", "fake-model"

    def __init__(self, *chat, facts=()):
        self.script, self.extractions, self.requests, self.events = list(chat), list(facts), [], []

    def complete(self, messages, *, tools=None, json_mode=False, max_tokens=1024, temperature=None):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools, "json_mode": json_mode})
        usage = {"model": self.model, "input_tokens": 100, "output_tokens": 10, "cached_tokens": 60}
        if json_mode:
            self.events.append("extract")
            item = self.extractions.pop(0) if self.extractions else {"facts": []}
            if isinstance(item, Exception):
                raise item
            return LLMReply(content=item if isinstance(item, str) else json.dumps(item, ensure_ascii=False), usage=usage)
        self.events.append("chat")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return LLMReply(content=item, message={"role": "assistant", "content": item}, usage=usage)
        calls = [ToolCall(f"call_{len(self.requests)}_{i}", name, args) for i, (name, args) in enumerate(item)]
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in calls]}
        return LLMReply(content="", tool_calls=calls, message=message, usage=usage)

    @property
    def chat_requests(self):
        return [r for r in self.requests if not r["json_mode"]]

    @property
    def extract_requests(self):
        return [r for r in self.requests if r["json_mode"]]

    def context(self, n=-1):
        """第 n 次对话调用里给模型看的上下文（第二条消息）。"""
        return self.chat_requests[n]["messages"][1]["content"]


@pytest.fixture
def make_chat(conn, cfg):
    def make(fake, backend=None, now=NOW, **kw):
        # 默认的判断替身：新事实和已有记忆"无关"；改正记忆时"是同一条"、删除时"就是这一条"（守门放行）；
        # 回复里出现"记下了/设好了"这类字眼时，认定它是在声称刚办了事；提醒的时间程序对不上时，认定是用户说过的。
        backend = backend or Scripted(rel="unrelated", same=0.9, target=0.9, claim=0.9, said=0.9)
        return ChatAgent(conn, LLM(fake, conn), Decider(backend, conn), cfg, clock=lambda: now, **kw)
    return make


class FakeApple:
    """「日历」和「提醒事项」的替身：记下 Alf 写了什么，接口和 alf.apple 里的同名函数一样。"""

    def __init__(self):
        self.events, self.todos, self.fail = {}, {}, None

    def add_event(self, title, start, end, *, all_day=False, location=""):
        if self.fail:
            return {"error": self.fail}
        key = f"E{len(self.events) + 1}"
        self.events[key] = (title, start, end, all_day, location)
        return {"id": key, "calendar": "个人"}

    def delete_event(self, event_id):
        if self.fail:
            return {"error": self.fail}
        return {"deleted": event_id} if self.events.pop(event_id, None) else {"error": "在日历里找不到这一条，可能已经被删掉了"}

    def add_reminder(self, title, due=None, *, has_time=True):
        if self.fail:
            return {"error": self.fail}
        key = f"R{len(self.todos) + 1}"
        self.todos[key] = (title, due, has_time)
        return {"id": key, "list": "提醒"}

    def delete_reminder(self, item_id):
        return {"deleted": item_id} if self.todos.pop(item_id, None) else {"error": "在提醒事项里找不到这一条，可能已经被删掉了"}


@pytest.fixture(autouse=True)
def never_touch_the_real_env_file(tmp_path, monkeypatch):
    """服务会读写 .env。测试里一律指到临时目录，绝不碰项目里那份带着真实 key 的文件。"""
    monkeypatch.setenv("ALF_ENV_FILE", str(tmp_path / ".env"))


@pytest.fixture(autouse=True)
def never_launch_the_calendar_helper(monkeypatch, tmp_path):
    """读写日历和提醒事项要启动一个真的小 App，还可能弹授权框。测试里一律换成"还没授权"；
    专门测这部分的用例自己传 run=… 或换掉 apple 的函数。"""
    from alf import apple

    helper = tmp_path / "AlfCalendar.app"
    helper.mkdir()
    monkeypatch.setenv("ALF_CALENDAR_HELPER", str(helper))
    monkeypatch.setattr(apple, "launch", lambda app, out, command, *args: out.write_text('{"error": "not_determined"}'))
    monkeypatch.setattr(apple, "SETTLE", 0)


@pytest.fixture(autouse=True)
def no_real_sources(monkeypatch):
    """汇总信息会启动 MCP 服务器、上网取新闻、打开读日历的小 App。测试里默认换成空快照；
    专门测汇总的用例自己用 real_gather 取回真的函数。"""
    from alf import sources

    real = sources.gather
    monkeypatch.setattr(sources, "gather", lambda cfg, current: {"at": 0, "sources": {}})
    return real


# --- 假的「信息」数据库：只有查询用到的那几张表和列 ---------------------------------------

@pytest.fixture
def chatdb(tmp_path):
    path = tmp_path / "chat.db"
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, style INTEGER);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, attributedBody BLOB, handle_id INTEGER,
            is_from_me INTEGER DEFAULT 0, cache_has_attachments INTEGER DEFAULT 0,
            associated_message_type INTEGER DEFAULT 0, item_type INTEGER DEFAULT 0);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO handle VALUES (1, '+61412345678'), (2, 'stranger@example.com');
        INSERT INTO chat VALUES (1, 45), (2, 43), (3, 45);   -- 1: 和主人一对一  2: 群聊  3: 和陌生人一对一
    """)
    c.commit()
    c.close()
    return path
