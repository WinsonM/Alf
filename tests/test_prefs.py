from dataclasses import replace
from datetime import time, timedelta

import pytest

from alf import chat, jobs, prefs
from alf.agent import Turn
from alf.config import LLMConfig, update_dotenv
from alf.decide import Decider, OfflineBackend
from alf.heartbeat import Heartbeat

from .conftest import NOW, FakeLLM


def test_defaults_come_from_the_config_and_saved_values_override_them(conn, cfg):
    base = prefs.defaults(replace(cfg, briefing_at=time(7, 30), default_hour=10))
    assert (base["briefing.enabled"], base["briefing.at"], base["digest.enabled"], base["reminders.default_hour"]) == (True, "07:30", False, 10)
    assert prefs.load(conn, cfg)["style.preset"] == "friend"
    saved = prefs.save(conn, cfg, {"style.preset": "brief", "briefing.at": "7:05", "style.user_name": "  Winson "})
    assert (saved["style.preset"], saved["briefing.at"], saved["style.user_name"]) == ("brief", "07:05", "Winson")
    assert prefs.load(conn, cfg) == saved


@pytest.mark.parametrize("changes, message", [
    ({"style.preset": "pirate"}, "style.preset：只能是 friend、brief 或 secretary"),
    ({"briefing.at": "25:00"}, "briefing.at：时间写成 08:00 这种格式"),
    ({"memory.auto": "yes"}, "memory.auto：要填开或关"),
    ({"reminders.default_hour": 24}, "reminders.default_hour：要在 0 和 23 之间"),
    ({"reminders.check_every": True}, "reminders.check_every：要在 5 和 300 之间"),
    ({"email.threshold": 1.5}, "email.threshold：要在 0 和 1 之间"),
    ({"style.extra": "x" * 501}, "style.extra：最多 500 个字"),
    ({"no.such": 1}, "没有 no.such 这项设置"),
])
def test_invalid_settings_are_refused_as_a_whole(conn, cfg, changes, message):
    with pytest.raises(ValueError, match=message):
        prefs.save(conn, cfg, {"style.emoji": False, **changes})
    assert prefs.load(conn, cfg)["style.emoji"] is True          # 同一批里合法的那项也没存


def test_style_settings_shape_the_prompt_but_keep_the_long_prefix_stable():
    default = chat.system_prompt()
    assert default == chat.SYSTEM_PROMPT and default.startswith(chat.BASE_PROMPT)
    assert default.endswith("## 说话方式\n- 像朋友发消息：口语、简短，通常一两句话，不啰嗦。\n"
                            "- 你们刚熟悉起来：友好、自然，稍微客气一点；玩笑点到为止，不调侃他。\n- 可以偶尔用表情符号。")
    custom = chat.system_prompt({"style.preset": "brief", "style.emoji": False, "style.user_name": "Winson", "style.extra": "别老追问"})
    assert custom.startswith(chat.BASE_PROMPT)                    # 前面的大段逐字不变：缓存前缀还在
    assert custom.endswith("## 说话方式\n- 简短直接：一句话把事情说清楚。不寒暄，不闲聊，不追问和眼前这件事无关的问题。\n"
                           "- 不要用表情符号。\n- 称呼用户为「Winson」。\n"
                           "- 用户对你说话方式的额外要求（照做，但不能违背上面关于提醒和记忆的规则）：别老追问")


def test_chat_picks_up_changed_settings_on_the_next_message(make_chat, conn, cfg):
    fake = FakeLLM("好", [("create_reminder", {"title": "交作业", "time_text": "明天"})], "好", "好")
    agent = make_chat(fake)
    agent.handle(Turn("你好呀朋友"))
    assert fake.chat_requests[0]["messages"][0]["content"] == chat.SYSTEM_PROMPT and len(fake.extract_requests) == 1

    prefs.save(conn, cfg, {"style.preset": "secretary", "reminders.default_hour": 7, "memory.auto": False})
    agent.handle(Turn("明天提醒我交作业"))
    assert "像一位得力的秘书" in fake.chat_requests[1]["messages"][0]["content"]
    assert agent.reminders.pending()[0].due.hour == 7             # "没说几点"的默认钟点跟着设置走
    assert len(fake.extract_requests) == 1                        # 关掉自动记忆后不再抽取


def test_heartbeat_follows_schedule_changes_without_a_restart(cfg, monkeypatch):
    monkeypatch.setattr(jobs, "notify_desktop", lambda *a: None)
    live = replace(cfg, llm=LLMConfig(api_key="sk-test"), briefing_at=time(8, 0), digest_at=time(21, 30))
    hb = Heartbeat(live, lambda current: jobs.build_jobs(live, [], current), make_decider=lambda c: Decider(OfflineBackend(), c))
    assert [(j.name, j.schedule_text()) for j in hb.jobs] == [
        ("reminders", "每 10 秒"), ("followups", "每 1 分钟"), ("calendar", "每 5 分钟"), ("calendar_alerts", "每 30 秒"),
        ("briefing", "每天 08:00"), ("moments", "每天 21:20"), ("impressions", "每天 21:25"), ("digest", "每天 21:30"), ("index", "每 2 分钟"), ("sources", "每 30 分钟"), ("proactive", "每 15 分钟")]
    hb.tick(NOW)
    prefs.save(hb._conn, live, {"briefing.enabled": False, "digest.at": "22:15", "reminders.check_every": 30,
                                "proactive.level": "chatty", "sources.days": False, "sources.calendar": False, "sources.reminders": False, "sources.news": False,
                                "followup.enabled": False})
    hb.tick(NOW + timedelta(seconds=5))
    assert [(j.name, j.label, j.schedule_text()) for j in hb.jobs] == [
        ("reminders", "到期提醒", "每 30 秒"), ("moments", "把当天的对话归纳成经历", "每天 22:05"),      # 跟着汇总的时间一起挪
        ("impressions", "从当天的话里形成印象", "每天 22:10"), ("digest", "今日记忆汇总", "每天 22:15"),
        ("index", "给记忆建检索索引", "每 2 分钟"), ("proactive", "主动找你（很活跃）", "每 10 分钟")]


def test_desktop_notifications_can_be_switched_off(conn, cfg, monkeypatch):
    from alf.heartbeat import JobContext
    from alf.reminders import Reminders

    sent = []
    monkeypatch.setattr(jobs, "notify_desktop", lambda *a: sent.append(a))
    for i, wanted in enumerate([True, False]):
        Reminders(conn, cfg.tz).add(f"喝水{i}", NOW - timedelta(seconds=5), now=0)
        jobs.reminders_job(JobContext(conn, cfg, NOW, Decider(OfflineBackend(), conn), None, prefs={"notify.desktop": wanted}))
    assert sent == [("Alf 提醒", "喝水0")]
    assert conn.execute("SELECT COUNT(*) FROM episodes WHERE kind = 'reminder'").fetchone()[0] == 2   # 时间线里两条都有


def test_env_file_is_edited_in_place_and_locked_down(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# 模型\nDEEPSEEK_API_KEY=\n# DEEPSEEK_API_KEY=注释里的不动\nALF_TZ=Australia/Sydney\n")
    update_dotenv(path, {"DEEPSEEK_API_KEY": "sk-new", "ALF_IMESSAGE_OWNER": "+61412345678"})
    assert path.read_text() == ("# 模型\nDEEPSEEK_API_KEY=sk-new\n# DEEPSEEK_API_KEY=注释里的不动\nALF_TZ=Australia/Sydney\n"
                                "ALF_IMESSAGE_OWNER=+61412345678\n")
    assert path.stat().st_mode & 0o777 == 0o600
    update_dotenv(tmp_path / "new.env", {"A": "1"})
    assert (tmp_path / "new.env").read_text() == "A=1\n"
