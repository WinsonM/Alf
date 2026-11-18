from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from alf import asr, db
from alf.decide import OfflineBackend, UnconfiguredBackend
from alf.server import create_app


class StubRecognizer:
    def __init__(self, text="我对花生过敏", confidence=0.91):
        self.text, self.confidence, self.seen = text, confidence, []

    def transcribe(self, path, vocabulary):
        self.seen.append((path.suffix, path.read_bytes(), vocabulary))
        return asr.Transcript(self.text, self.confidence, "zh", [])


@pytest.fixture
def recognizer():
    return StubRecognizer()


@pytest.fixture
def client(cfg, recognizer):
    app = create_app(cfg, OfflineBackend(), recognizer=recognizer, start_heartbeat=False)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        c.heartbeat = app.state.heartbeat
        yield c


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>Alf</title>" in r.text and r.headers["cache-control"] == "no-store"


def test_chat_replies_and_both_turns_land_on_the_timeline(client):
    r = client.post("/api/chat", json={"text": "我对花生过敏"})
    assert r.status_code == 200 and r.json()["reply"] == "记住了：我对花生过敏" and r.json()["intent"] == "remember_fact"
    items = client.get("/api/timeline").json()["items"]
    assert [(i["role"], i["kind"], i["text"]) for i in items] == [
        ("user", "chat", "我对花生过敏"), ("alf", "chat", "记住了：我对花生过敏")]
    assert client.get(f"/api/timeline?after={items[-1]['id']}").json() == {"items": []}
    assert [i["id"] for i in client.get(f"/api/timeline?after={items[0]['id']}").json()["items"]] == [items[1]["id"]]


def test_blank_or_malformed_messages_are_rejected(client):
    assert client.post("/api/chat", json={"text": "   "}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422
    assert client.post("/api/chat", content="text=hi", headers={"content-type": "text/plain"}).status_code == 422
    assert client.get("/api/timeline").json() == {"items": []}


def test_voice_is_transcribed_stored_and_answered(client, recognizer, cfg):
    r = client.post("/api/voice", content=b"fake-webm-bytes", headers={"content-type": "audio/webm;codecs=opus"})
    assert r.status_code == 200
    assert r.json() == {"transcript": "我对花生过敏", "confidence": 0.91, "reply": "记住了：我对花生过敏"}
    assert [(ext, data) for ext, data, _ in recognizer.seen] == [(".webm", b"fake-webm-bytes")]
    user_turn = client.get("/api/timeline").json()["items"][0]
    assert (user_turn["text"], user_turn["source"]) == ("我对花生过敏", "voice")
    row = db.connect(cfg.db_path).execute("SELECT kind, transcript, confidence FROM attachments").fetchone()
    assert tuple(row) == ("audio", "我对花生过敏", 0.91)


def test_voice_edge_cases(client, recognizer):
    assert client.post("/api/voice", content=b"x", headers={"content-type": "text/plain"}).status_code == 415
    assert client.post("/api/voice", content=b"", headers={"content-type": "audio/mp4"}).status_code == 422
    recognizer.text = ""
    r = client.post("/api/voice", content=b"silence", headers={"content-type": "audio/mp4"})
    assert r.json()["transcript"] == "" and r.json()["reply"] is None
    assert client.get("/api/timeline").json() == {"items": []}


def test_status_reports_jobs_counts_and_connectors(client):
    client.post("/api/chat", json={"text": "明天下午三点提醒我交作业"})
    s = client.get("/api/status").json()
    assert s["backend"] == "offline" and s["tz"] == "Australia/Sydney"
    assert [(j["name"], j["schedule"], j["last_run"]) for j in s["jobs"]] == [
        ("reminders", "每 10 秒", None), ("followups", "每 1 分钟", None), ("calendar", "每 5 分钟", None),
        ("calendar_alerts", "每 30 秒", None), ("index", "每 2 分钟", None), ("sources", "每 30 分钟", None)]
    assert s["counts"] == {"reminders": 1, "facts": 0, "inbox": 0}
    assert s["connectors"] == [{"name": "email", "label": "邮箱", "configured": False}]
    assert s["heartbeat"]["alive"] is False   # 这个测试没启动心跳线程


def test_job_can_be_triggered_from_the_api(client):
    assert client.post("/api/jobs/reminders/run").status_code == 202
    assert client.post("/api/jobs/nope/run").status_code == 404
    assert client.heartbeat._requested == {"reminders"}


def test_requests_for_other_hosts_are_refused(client):
    assert client.get("/api/timeline", headers={"host": "evil.example"}).status_code == 400
    assert client.get("/api/timeline", headers={"host": "localhost:8765"}).status_code == 200


def test_missing_key_degrades_instead_of_crashing(cfg):
    app = create_app(cfg, UnconfiguredBackend(), recognizer=StubRecognizer(), start_heartbeat=False)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        r = c.post("/api/chat", json={"text": "你好"})
        assert r.status_code == 200 and "还没配置 Jev" in r.json()["reply"]
        assert c.get("/api/status").json()["backend"] == "unconfigured"


def test_heartbeat_thread_delivers_a_due_reminder_to_the_timeline(cfg, monkeypatch):
    import time
    from datetime import datetime, timedelta

    from alf import jobs
    from alf.reminders import Reminders

    monkeypatch.setattr(jobs, "notify_desktop", lambda *a: None)
    fast = replace(cfg, heartbeat_tick=0.05, reminder_every=0.05)
    Reminders(db.connect(fast.db_path), fast.tz).add("喝水", datetime.now(fast.tz) - timedelta(seconds=5), now=0)
    with TestClient(create_app(fast, OfflineBackend(), recognizer=StubRecognizer()), base_url="http://127.0.0.1") as c:
        deadline = time.monotonic() + 3
        items = []
        while not items and time.monotonic() < deadline:
            items = c.get("/api/timeline").json()["items"]
            time.sleep(0.02)
        assert [(i["kind"], i["text"].split("（")[0]) for i in items] == [("reminder", "喝水")]
        assert c.get("/api/status").json()["heartbeat"]["alive"] is True


# --- 第 ① 期：DeepSeek 对话 + iMessage ------------------------------------------------

def wait_until(predicate, timeout=4.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_chat_goes_through_the_generative_model_when_one_is_configured(cfg):
    from .conftest import FakeLLM, Scripted

    fake = FakeLLM([("create_reminder", {"title": "交作业", "time_text": "明天下午三点"})], "好嘞～")
    app = create_app(cfg, Scripted(), llm_backend=fake, recognizer=StubRecognizer(), start_heartbeat=False)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        reply = c.post("/api/chat", json={"text": "明天下午三点提醒我交作业"}).json()["reply"]
        assert reply.startswith("好嘞～\n⏰ 交作业｜明天 ") and reply.endswith("15:00")
        s = c.get("/api/status").json()
        assert (s["brain"], s["model"], s["backend"]) == ("fake", "fake-model", "scripted")
        assert s["channels"] == [{"name": "imessage", "label": "iMessage", "configured": False}]
        assert s["counts"]["reminders"] == 1


def test_imessage_end_to_end_with_real_threads(cfg, tmp_path, monkeypatch):
    """假的「信息」数据库 + 记录型的发送器，其余全是真的：渠道线程、心跳线程、对话、数据库。"""
    from datetime import datetime, timedelta

    from alf import jobs
    from alf.config import IMessageConfig
    from alf.imessage import ChatDB, IMessageChannel
    from alf.reminders import Reminders

    from .conftest import FakeLLM, Scripted
    from .test_imessage import OWNER, Outbox, receive

    monkeypatch.setattr(jobs, "notify_desktop", lambda *a: None)
    chatdb = tmp_path / "chat.db"
    import sqlite3
    sqlite3.connect(chatdb).executescript("""
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, style INTEGER);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, attributedBody BLOB, handle_id INTEGER,
            is_from_me INTEGER DEFAULT 0, cache_has_attachments INTEGER DEFAULT 0,
            associated_message_type INTEGER DEFAULT 0, item_type INTEGER DEFAULT 0);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO handle VALUES (1, '+61412345678');
        INSERT INTO chat VALUES (1, 45);
    """)
    live = replace(cfg, heartbeat_tick=0.05, reminder_every=0.05,
                   imessage=IMessageConfig(owners=(OWNER,), db_path=chatdb, poll=0.02))
    fake = FakeLLM([("create_reminder", {"title": "喝水", "time_text": "明天上午十点"})], "好嘞，明天叫你")
    out = Outbox()
    channel = IMessageChannel(live, None, sender=out, chatdb=ChatDB(chatdb))
    app = create_app(live, Scripted(rel="unrelated"), llm_backend=fake, recognizer=StubRecognizer(), imessage=channel)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert wait_until(lambda: channel.status["ok"] is True)            # 起点已定
        receive(chatdb, "明天上午十点提醒我喝水")
        assert wait_until(lambda: len(out.sent) == 1)
        handle, text = out.sent[0]
        assert handle == OWNER and text.startswith("好嘞，明天叫你\n⏰ 喝水｜明天 ") and text.endswith("10:00")

        # 一条已经到点的提醒：心跳认领 → 写进时间线 → 渠道线程推到 iMessage
        Reminders(db.connect(live.db_path), live.tz).add("交作业", datetime.now(live.tz) - timedelta(seconds=3), now=0)
        assert wait_until(lambda: len(out.sent) == 2)
        assert out.sent[1][1].startswith("⏰ 提醒：交作业（")

        status = c.get("/api/status").json()
        imsg = status["channels"][0]
        assert (imsg["configured"], imsg["ok"], imsg["error"]) == (True, True, None) and imsg["last_outbound"]
        kinds = [(i["role"], i["kind"]) for i in c.get("/api/timeline").json()["items"]]
        assert kinds == [("user", "chat"), ("alf", "chat"), ("alf", "reminder")]      # 网页上也看得到这段对话


# --- 设置页用的接口 -------------------------------------------------------------------

@pytest.fixture
def app_env(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# 说明\nDEEPSEEK_API_KEY=sk-secret-value\nTYPESAFE_API_KEY=\nALF_IMESSAGE_OWNER=+61412345678\n")
    return path


def test_prefs_can_be_read_and_changed(client):
    assert client.get("/api/prefs").json()["prefs"]["style.preset"] == "friend"
    r = client.patch("/api/prefs", json={"style.preset": "brief", "digest.at": "22:00"})
    assert r.status_code == 200 and (r.json()["prefs"]["style.preset"], r.json()["prefs"]["digest.at"]) == ("brief", "22:00")
    bad = client.patch("/api/prefs", json={"style.preset": "pirate"})
    assert bad.status_code == 422 and bad.json()["detail"] == "style.preset：只能是 friend、brief 或 secretary"
    assert client.get("/api/prefs").json()["prefs"]["style.preset"] == "brief"


def test_setup_never_returns_secrets_and_tracks_unapplied_changes(cfg, app_env):
    restarts = []
    app = create_app(cfg, OfflineBackend(), recognizer=StubRecognizer(), start_heartbeat=False, env_path=app_env,
                     restart=lambda: restarts.append(1))
    with TestClient(app, base_url="http://127.0.0.1") as c:
        view = c.get("/api/setup").json()
        mail = {f"ALF_{name}_USER": {"set": False, "value": ""} for name in ("GMAIL", "163", "QQ")}
        mail |= {f"ALF_{name}_PASSWORD": {"set": False, "length": 0} for name in ("GMAIL", "163", "QQ")}
        assert view["env"] == {"DEEPSEEK_API_KEY": {"set": True, "length": 15}, "TYPESAFE_API_KEY": {"set": False, "length": 0},
                               "ALF_IMESSAGE_OWNER": {"set": True, "value": "+61412345678"}, **mail}
        assert "sk-secret-value" not in c.get("/api/setup").text and view["pending_restart"] is False

        saved = c.post("/api/setup", json={"TYPESAFE_API_KEY": " apikey_abc ", "ALF_IMESSAGE_OWNER": "+61 400 111 222, Me@Example.com"}).json()
        assert saved["env"]["TYPESAFE_API_KEY"] == {"set": True, "length": 10}
        assert saved["env"]["ALF_IMESSAGE_OWNER"]["value"] == "+61400111222,me@example.com" and saved["pending_restart"] is True
        assert app_env.read_text() == "# 说明\nDEEPSEEK_API_KEY=sk-secret-value\nTYPESAFE_API_KEY=apikey_abc\nALF_IMESSAGE_OWNER=+61400111222,me@example.com\n"

        assert c.post("/api/setup", json={"ALF_IMESSAGE_OWNER": "0412 345 678"}).status_code == 422      # 没带国家码
        assert c.post("/api/setup", json={"DEEPSEEK_API_KEY": "a\nALF_IMESSAGE_OWNER=evil"}).status_code == 422
        assert "evil" not in app_env.read_text()

        assert c.post("/api/restart").status_code == 202 and restarts == [1]

        # 邮箱：地址原样给回去核对，授权码只说填没填
        boxed = c.post("/api/setup", json={"ALF_GMAIL_USER": " me@gmail.com ", "ALF_GMAIL_PASSWORD": "abcd efgh ijkl mnop"})
        assert boxed.json()["env"]["ALF_GMAIL_USER"] == {"set": True, "value": "me@gmail.com"}
        assert boxed.json()["env"]["ALF_GMAIL_PASSWORD"] == {"set": True, "length": 19} and "abcd" not in boxed.text
        assert "ALF_GMAIL_PASSWORD=abcd efgh ijkl mnop\n" in app_env.read_text() and oct(app_env.stat().st_mode)[-3:] == "600"
        assert c.post("/api/setup", json={"ALF_163_USER": "不是邮箱"}).status_code == 422


def test_mailboxes_show_up_in_status_and_can_be_checked_without_reading_or_sending_anything(cfg):
    from alf.config import EmailConfig
    from alf.connectors import ConnectorError

    gmail = EmailConfig("imap.gmail.com", "me@gmail.com", "pw", name="email.gmail", label="Gmail", smtp_host="smtp.gmail.com")
    netease = EmailConfig("imap.163.com", "me@163.com", "pw", name="email.163", label="163 邮箱", smtp_host="smtp.163.com")
    tried = []

    def inbox(box):
        tried.append(("收", box.name))
        if box.name == "email.163":
            raise ConnectorError("邮箱登录失败，检查账号和授权码（LOGIN failed）")

    app = create_app(replace(cfg, emails=(gmail, netease)), OfflineBackend(), recognizer=StubRecognizer(), start_heartbeat=False,
                     check_inbox=inbox, check_outbox=lambda box: tried.append(("发", box.name)))
    with TestClient(app, base_url="http://127.0.0.1") as c:
        status = c.get("/api/status").json()
        assert status["connectors"] == [
            {"name": "email.gmail", "label": "Gmail", "configured": True, "user": "me@gmail.com", "can_send": True},
            {"name": "email.163", "label": "163 邮箱", "configured": True, "user": "me@163.com", "can_send": True}]
        assert [j["name"] for j in status["jobs"]][-3:] == ["email.gmail", "email.163", "mail_report"]   # 每个邮箱一个任务，外加每天的汇总
        result = c.post("/api/email/test").json()["mailboxes"]
        assert result == [{"name": "email.gmail", "label": "Gmail", "user": "me@gmail.com", "receive": "ok", "send": "ok"},
                          {"name": "email.163", "label": "163 邮箱", "user": "me@163.com",
                           "receive": "邮箱登录失败，检查账号和授权码（LOGIN failed）", "send": "ok"}]
        assert tried == [("收", "email.gmail"), ("发", "email.gmail"), ("收", "email.163"), ("发", "email.163")] and "pw" not in str(result)


def test_other_sites_cannot_trigger_actions(client):
    """别的网页能让浏览器向 127.0.0.1 发不带请求体的 POST；靠 Origin 把它们挡在外面。"""
    for origin in ("https://evil.example", "http://127.0.0.1:3000", "null"):
        assert client.post("/api/jobs/reminders/run", headers={"origin": origin}).status_code == 403
        assert client.post("/api/chat", json={"text": "你好"}, headers={"origin": origin}).status_code == 403
    assert client.post("/api/jobs/reminders/run", headers={"origin": "http://127.0.0.1"}).status_code == 202   # 自己的页面
    assert client.post("/api/jobs/reminders/run").status_code == 202                                         # curl、脚本
    assert client.get("/api/timeline", headers={"origin": "https://evil.example"}).status_code == 200         # 只读的不拦（浏览器自己不让跨站读）
    assert client.get("/api/timeline").json() == {"items": []}                                               # 被拦的那句"你好"没有进来


def test_memory_can_be_listed_edited_and_forgotten(client, cfg):
    client.post("/api/chat", json={"text": "我对花生过敏"})
    [fact] = client.get("/api/memory").json()["facts"]
    assert (fact["text"], fact["label"], fact["current"], fact["layer"]) == ("我对花生过敏", "健康", True, "core")
    edited = client.patch(f"/api/memory/{fact['id']}", json={"text": "对花生和虾都过敏"}).json()["fact"]
    assert edited["text"] == "对花生和虾都过敏" and edited["id"] != fact["id"]
    history = client.get("/api/memory?all=true").json()["facts"]
    assert [(f["text"], f["current"], f["superseded_by"]) for f in sorted(history, key=lambda f: f["id"])] == [
        ("我对花生过敏", False, edited["id"]), ("对花生和虾都过敏", True, None)]
    assert client.delete(f"/api/memory/{edited['id']}").json() == {"forgot": "对花生和虾都过敏"}
    empty = client.get("/api/memory").json()
    # 人设、印象、约定各有一栏；一开始人设是默认的那张卡片，另外两样是空的。
    assert [(c["label"], c["changed"]) for c in empty.pop("persona")["card"]][:3] == [("名字", False), ("身份", False), ("性格", False)]
    assert empty.pop("impressions") == {"formed": [], "hunches": [], "withdrawn": [], "form_days": 2}
    assert empty.pop("pacts") == {"active": [], "settled": []}
    assert (empty.pop("people"), empty.pop("days")) == ([], [])       # 没听过录音：没有人，也没有"每天的事"
    assert empty == {"facts": [], "affinity": {
        "score": 50, "max": 100, "start": 50, "tier": "刚熟悉", "recent": [],
        "mood": {"state": "even", "label": "平常", "warm": 0.0, "tension": 0.0, "reason": None}},
        "lessons": {"rules": [], "examples": [], "settings": [], "proposals": [],
                    "scopes": {"all": "哪里都管", "chat": "聊天时", "mail": "说邮件时", "proactive": "主动找你时"}}}
    assert client.patch("/api/memory/999", json={"text": "x"}).status_code == 404
    assert client.delete("/api/memory/999").status_code == 404
    assert client.patch(f"/api/memory/{fact['id']}", json={"text": ""}).status_code == 422


def test_setup_helpers_test_message_settings_and_doctor(cfg, tmp_path):
    from alf.config import IMessageConfig
    from alf.imessage import ChatDB, IMessageChannel

    from .test_imessage import OWNER, Outbox

    opened = []
    plain = create_app(cfg, OfflineBackend(), recognizer=StubRecognizer(), start_heartbeat=False, open_settings=lambda: opened.append(1))
    with TestClient(plain, base_url="http://127.0.0.1") as c:
        assert c.post("/api/test-message").status_code == 409                       # 没接 iMessage
        assert c.post("/api/open-privacy-settings").status_code == 202 and opened == [1]
        checks = {x["key"]: x for x in c.get("/api/doctor").json()["checks"]}
        assert checks["deepseek"]["ok"] is False and checks["owner"]["ok"] is False and checks["jev"]["ok"] is None
        assert checks["service"]["ok"] is False                                     # 这个测试没启动心跳

    live = replace(cfg, imessage=IMessageConfig(owners=(OWNER,), db_path=tmp_path / "chat.db"))
    channel = IMessageChannel(live, None, sender=Outbox(), chatdb=ChatDB(tmp_path / "chat.db"))
    with TestClient(create_app(live, OfflineBackend(), recognizer=StubRecognizer(), start_heartbeat=False, imessage=channel),
                    base_url="http://127.0.0.1") as c:
        assert c.post("/api/test-message").status_code == 202
        row = db.connect(live.db_path).execute("SELECT text, channel, delivered_at FROM episodes").fetchone()
        assert tuple(row) == ("这是 Alf 的测试消息，收到说明发送正常。", "imessage", None)


def test_entrance_animation_never_hides_content():
    """实际遇到过：窗口被挡住时 WebKit 暂停动画，从"透明"开始的入场动画让消息和设置卡片一直看不见。"""
    import re

    from alf.server import INDEX

    css = INDEX.read_text(encoding="utf-8")
    for name, body in re.findall(r"@keyframes (\w+) \{(.*?\})\s*\}", css, flags=re.S):
        if name == "pop":
            assert "opacity" not in body
            break
    else:
        raise AssertionError("没找到入场动画 pop")


def test_proactive_endpoint_reports_level_sources_and_recent_decisions(client, cfg):
    import json

    from alf import proactive
    from alf.heartbeat import JobState, save_state

    conn = db.connect(cfg.db_path)
    save_state(conn, JobState("sources", cursor=json.dumps({"at": 123, "sources": {
        "days": {"special_days": {"today": []}}, "calendar": {"calendar_events": {"error": "没有访问日历的权限"}}}})))
    from datetime import datetime
    proactive.record(conn, datetime.now(cfg.tz), "held", kind="news", message="某条新闻", why="相关", score=0.31)
    view = client.get("/api/proactive").json()
    assert view["level"] == {"key": "active", "label": "活跃", "per_day": 10, "gap_minutes": 60, "wake_minutes": 15, "threshold": 0.4}
    assert set(view["levels"]) == {"quiet", "normal", "active", "chatty"} and view["gated_by_jev"] is False
    assert view["sources"] == {"at": 123, "state": {"days": "ok", "calendar": "没有访问日历的权限"}}
    assert [(e["outcome"], e["kind_label"], e["message"], e["score"]) for e in view["log"]] == [("held", "新闻", "某条新闻", 0.31)]
    assert client.patch("/api/prefs", json={"proactive.level": "chatty"}).status_code == 200
    assert client.get("/api/proactive").json()["level"]["label"] == "很活跃"
    assert client.post("/api/jobs/sources/run").status_code == 202 and client.post("/api/jobs/reminders/run").status_code == 202


def test_calendar_permission_is_only_requested_when_the_user_asks(cfg):
    calls = []

    def fake_calendar(days, ask=False):
        calls.append(ask)
        return {"events": [{"title": "组会"}]} if len(calls) > 1 else {"error": "没有访问日历的权限"}

    def fake_reminders(days, ask=False):
        calls.append(("提醒事项", ask))
        return {"reminders": []}

    app = create_app(cfg, OfflineBackend(), recognizer=StubRecognizer(), start_heartbeat=False, read_calendar=fake_calendar,
                     read_reminders=fake_reminders)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert c.post("/api/calendar/connect").json() == {"ok": False, "error": "没有访问日历的权限", "events": 0,
                                                          "reminders_ok": True, "reminders_error": None}
        assert c.post("/api/calendar/connect").json() == {"ok": True, "error": None, "events": 1, "reminders_ok": True, "reminders_error": None}
        # 日历和提醒事项是两项权限，各申请一次；授权完马上重新同步，不等下一个周期
        assert calls == [True, ("提醒事项", True), True, ("提醒事项", True)] and app.state.background._requested == {"sources", "calendar"}


def test_what_alf_learned_is_listed_on_the_memory_page_and_each_item_can_be_taken_back(client, cfg):
    from alf import lessons, prefs
    from alf.decide import Decider
    from alf.memory import FEEDBACK, Memory

    conn = db.connect(cfg.db_path)
    rule = Memory(conn, Decider(OfflineBackend(), conn), 0.6).write("邮件汇报再短点", FEEDBACK, now=1_700_000_000, quote="邮件汇报再短点").fact
    example, _ = lessons.add_example(conn, "mail", "later", "CommBank｜对账单", "这种不用马上告诉我", now=1_700_000_100, inbox_id=7)
    change = lessons.log_setting(conn, "email.report_at", "18:00", "20:00", "改到晚上八点", None, 1_700_000_200)
    prefs.save(conn, cfg, {"email.report_at": "20:00"})
    proposal = lessons.add_proposal(conn, "银行的对账单放到汇总里", "mail", [example], 2_000_000_000)
    data = client.get("/api/memory").json()
    learned = data["lessons"]
    assert [(r["text"], r["scope"], r["scope_label"], r["quote"]) for r in learned["rules"]] == [("邮件汇报再短点", "all", "哪里都管", "邮件汇报再短点")]
    assert [(x["area_label"], x["about"], x["said"], x["wish"]) for x in learned["examples"]] == [("邮件", "CommBank｜对账单", "这种不用马上告诉我", "留到每天的汇总")]
    assert [(c["label"], c["old"], c["new"], c["quote"]) for c in learned["settings"]] == [("邮件汇总的时间", "18:00", "20:00", "改到晚上八点")]
    assert [(p["text"], p["scope_label"], p["count"]) for p in learned["proposals"]] == [("银行的对账单放到汇总里", "说邮件时", 1)]
    assert [f["layer"] for f in data["facts"]] == ["rule"]                    # 页面上不再把它混在"一直记着"里

    # 范围记错了可以直接改；改回"哪里都管"也行
    assert client.patch(f"/api/memory/{rule.id}", json={"scope": "mail"}).status_code == 200
    assert client.get("/api/memory").json()["lessons"]["rules"][0]["scope"] == "mail"
    assert client.patch("/api/memory/999", json={"scope": "mail"}).status_code == 404
    assert client.patch(f"/api/memory/{rule.id}", json={}).status_code == 422
    # 聊天里改的设置改回去
    assert client.post(f"/api/lessons/settings/{change}/undo").json() == {"ok": True}
    assert client.get("/api/prefs").json()["prefs"]["email.report_at"] == "18:00"
    assert client.post(f"/api/lessons/settings/{change}/undo").status_code == 409
    # 提议点头：变成一条专管邮件的要求，出处的例子由它取代
    accepted = client.post(f"/api/lessons/proposals/{proposal}", json={"accept": True}).json()
    learned = client.get("/api/memory").json()["lessons"]
    assert [r["text"] for r in learned["rules"]] == ["邮件汇报再短点", "银行的对账单放到汇总里"] and learned["rules"][1]["id"] == accepted["rule_id"]
    assert learned["proposals"] == [] and learned["examples"] == []
    assert client.post(f"/api/lessons/proposals/{proposal}", json={"accept": True}).status_code == 404
    # 例子可以单独撤销
    other, _ = lessons.add_example(conn, "proactive", "bad", "今天金价又涨了", "没用", now=1_700_000_300, target=3)
    assert client.delete(f"/api/lessons/examples/{other}").json() == {"ok": True}
    assert client.delete(f"/api/lessons/examples/{other}").status_code == 404
    assert client.get("/api/memory").json()["lessons"]["examples"] == []
