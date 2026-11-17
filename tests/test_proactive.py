import json
from datetime import date, timedelta

import pytest

from alf import jobs, proactive, sources, timeline
from alf.decide import Decider
from alf.heartbeat import JobContext, JobState, save_state
from alf.llm import LLM, LLMError
from alf.mcp_servers import calendar as calendar_server
from alf.mcp_servers import days, news
from alf.proactive import LEVELS, Habits, blocked, in_quiet_hours
from alf.reminders import Reminders

from .conftest import NOW, TZ, FakeLLM, Scripted

PREFS = {"proactive.enabled": True, "proactive.level": "active", "quiet.start": "22:00", "quiet.end": "08:00",
         "style.preset": "friend", "style.emoji": True}


class Jev(Scripted):
    name = "jev"          # 只有判断后端是 Jev 时才会用它的分数卡门槛


def say(conn, role, text, minutes_ago, kind="chat", now=NOW):
    conn.execute("INSERT INTO episodes(session_id, role, kind, text, created_at) VALUES ('t', ?, ?, ?, ?)",
                 (role, kind, text, int((now - timedelta(minutes=minutes_ago)).timestamp())))


def ctx_for(conn, cfg, fake=None, backend=None, prefs=PREFS, now=NOW):
    return JobContext(conn, cfg, now, Decider(backend or Scripted(), conn), None, prefs=dict(prefs),
                      llm=LLM(fake, conn) if fake else None)


def idle(**changes):
    base = dict(nudges_today=0, minutes_since_nudge=None, minutes_since_user=600, minutes_since_alf=None,
                ignored_streak=0, last_nudge_answered=None, active_now=None)
    return Habits(**{**base, **changes})


# --- 硬规则 ---------------------------------------------------------------------------

def test_quiet_hours_wrap_around_midnight():
    at = lambda h, m=0: NOW.replace(hour=h, minute=m)   # noqa: E731
    assert [in_quiet_hours(at(h), PREFS) for h in (21, 22, 23, 0, 7, 8, 12)] == [False, True, True, True, True, False, False]
    daytime = {"quiet.start": "13:00", "quiet.end": "14:30"}
    assert [in_quiet_hours(at(*t), daytime) for t in ((12, 59), (13, 0), (14, 29), (14, 30))] == [False, True, True, False]


@pytest.mark.parametrize("habits, now, reason", [
    (idle(), NOW.replace(hour=23), "安静时段"),
    (idle(minutes_since_user=3), NOW, "你正在跟它聊天"),
    (idle(nudges_today=10), NOW, "今天已经主动说了 10 条，到上限了"),
    (idle(minutes_since_nudge=45), NOW, "离上一条主动消息还不到 60 分钟"),
    (idle(minutes_since_nudge=90, ignored_streak=3), NOW, "离上一条主动消息还不到 120 分钟（最近几条都没回应，间隔加倍了）"),
    (idle(minutes_since_alf=4), NOW, "刚发过提醒或播报"),
    (idle(minutes_since_nudge=61, minutes_since_user=6, nudges_today=9, minutes_since_alf=11), NOW, None),
])
def test_hard_rules_decide_before_any_model_is_asked(habits, now, reason):
    assert blocked(habits, LEVELS["active"], now, PREFS) == reason


def test_levels_get_stricter_from_chatty_to_quiet():
    order = [LEVELS[k] for k in ("chatty", "active", "normal", "quiet")]
    assert [l.per_day for l in order] == [20, 10, 5, 2] and [l.gap for l in order] == [30, 60, 120, 240]
    assert [l.threshold for l in order] == sorted(l.threshold for l in order)


def test_habits_are_read_from_the_conversation_history(conn):
    say(conn, "alf", "第一条没人理", 60 * 9, kind="nudge")
    say(conn, "alf", "第二条没人理", 60 * 6, kind="nudge")
    say(conn, "alf", "第三条刚发，还不能算", 30, kind="nudge")
    say(conn, "alf", "喝水（09:50）", 10, kind="reminder")
    say(conn, "user", "很久以前说的话", 60 * 30)
    h = Habits.load(conn, NOW)
    assert (h.nudges_today, round(h.minutes_since_nudge), round(h.minutes_since_alf), round(h.minutes_since_user)) == (3, 30, 10, 1800)
    assert (h.ignored_streak, h.last_nudge_answered, h.active_now) == (2, False, None)

    say(conn, "user", "回了第二条", 60 * 5)
    assert (Habits.load(conn, NOW).ignored_streak, Habits.load(conn, NOW).last_nudge_answered) == (0, True)
    for day in range(1, 11):                                   # 过去十天每天上午十点都说两句话
        say(conn, "user", "早", 60 * 24 * day)
        say(conn, "user", "在忙", 60 * 24 * day + 5)
    assert Habits.load(conn, NOW).active_now is True
    assert Habits.load(conn, NOW.replace(hour=15)).active_now is False


# --- 完整的一次判断 ---------------------------------------------------------------------

def test_a_worthwhile_message_is_sent_and_the_reasoning_is_recorded(conn, cfg):
    fake = FakeLLM(facts=[{"send": True, "kind": "special_day", "message": "今天冬至，记得吃顿饺子～", "why": "今天是冬至"}])
    ctx = ctx_for(conn, cfg, fake, backend=Jev(welcome=0.82))
    assert proactive.proactive_job(ctx) == "主动说了一句（特殊日子）"
    row = conn.execute("SELECT role, kind, text FROM episodes").fetchone()
    assert tuple(row) == ("alf", "nudge", "今天冬至，记得吃顿饺子～")
    [entry] = proactive.recent(conn)
    assert (entry["outcome"], entry["kind_label"], entry["message"], entry["why"], entry["score"]) == (
        "sent", "特殊日子", "今天冬至，记得吃顿饺子～", "今天是冬至", 0.82)
    system, user = fake.requests[0]["messages"]
    assert "活跃：只要有点实际内容" in system["content"] and "像朋友发消息" in system["content"]
    assert user["content"].startswith("[现在] 2026年9月22日 周二 10:00") and fake.requests[0]["tools"] is None


def test_jev_holds_back_messages_below_the_level_threshold(conn, cfg):
    reply = {"send": True, "kind": "news", "message": "英伟达又发新卡了", "why": "科技新闻"}
    assert proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(facts=[reply]), backend=Jev(welcome=0.35))) == "想说一句但忍住了（把握 35%，低于 40%）"
    assert conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
    assert (proactive.recent(conn)[0]["outcome"], proactive.recent(conn)[0]["score"]) == ("held", 0.35)
    # 同样的分数，"很活跃"的门槛更低，就会发
    chatty = {**PREFS, "proactive.level": "chatty"}
    assert proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(facts=[reply]), backend=Jev(welcome=0.35), prefs=chatty)) == "主动说了一句（新闻）"


def test_jev_sees_the_message_and_the_users_habits(conn, cfg):
    say(conn, "user", "嗨", 200)
    backend = Jev(welcome=0.9)
    proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(facts=[{"send": True, "kind": "care", "message": "忙完了吗", "why": "下午没动静"}]), backend=backend))
    state, questions = backend.calls[0]
    assert state == {"now": "2026-09-22 Tue 10:00", "message": "忙完了吗", "why_now": "下午没动静", "minutes_since_user_last_spoke": 200,
                     "unprompted_messages_already_sent_today": 0, "recent_unprompted_messages_ignored": 0, "user_usually_chats_at_this_hour": None}
    assert list(questions) == ["welcome"]


def test_without_jev_the_models_own_decision_stands_and_jev_outages_do_not_mute_alf(conn, cfg):
    from alf.decide import DecisionError

    reply = {"send": True, "kind": "care", "message": "在忙吗", "why": "关心一下"}
    assert proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(facts=[reply]), backend=Scripted())) == "主动说了一句（关心）"
    assert proactive.recent(conn)[0]["score"] is None

    class JevDown(Jev):
        def system_one(self, state, questions):
            raise DecisionError("Jev 限流了，稍后再试")

    later = NOW + timedelta(hours=2)
    assert proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(facts=[reply]), backend=JevDown(), now=later)) == "主动说了一句（关心）"


@pytest.mark.parametrize("reply, result, outcome", [
    ({"send": False, "why": "没什么新鲜事"}, "没有开口：没什么新鲜事", "silent"),
    ({"send": True, "message": "   ", "why": "x"}, "没有开口：x", "silent"),
    ({"send": True, "message": "⏰ 交作业｜明天 15:00"}, "没有开口：没有值得说的", "silent"),   # 冒充回执的内容不发
])
def test_nothing_worth_saying_means_silence(conn, cfg, reply, result, outcome):
    assert proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(facts=[reply]))) == result
    assert conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0 and proactive.recent(conn)[0]["outcome"] == outcome


def test_guards_model_failures_and_switches(conn, cfg):
    assert proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(facts=[LLMError("DeepSeek 响应超时")]))) == "没想成：DeepSeek 响应超时"
    assert proactive.recent(conn) == []                                                                     # 没想成不算一次判断
    assert proactive.proactive_job(ctx_for(conn, cfg, None)) is None                                       # 没配生成式模型
    assert proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(), prefs={**PREFS, "proactive.enabled": False})) is None
    quiet = FakeLLM()
    assert proactive.proactive_job(ctx_for(conn, cfg, quiet, now=NOW.replace(hour=23))) == "没有开口：安静时段"
    assert quiet.requests == []                                                                             # 硬规则挡住时不花钱


def test_second_evaluation_respects_the_gap_after_sending(conn, cfg):
    reply = {"send": True, "kind": "care", "message": "在忙吗", "why": "关心"}
    proactive.proactive_job(ctx_for(conn, cfg, FakeLLM(facts=[reply])))
    again = FakeLLM(facts=[reply])
    assert proactive.proactive_job(ctx_for(conn, cfg, again, now=NOW + timedelta(minutes=30))) == "没有开口：离上一条主动消息还不到 60 分钟"
    assert again.requests == []


# --- 给模型看的"现在的情况" --------------------------------------------------------------

SNAPSHOT = {"at": int(NOW.timestamp()) - 600, "sources": {
    "days": {"special_days": {"date": "2026-09-22", "weekday": "周二", "lunar": "农历八月十二", "today": [],
                              "upcoming": [{"date": "2026-09-25", "weekday": "周五", "in_days": 3, "what": ["中秋节"]}]}},
    "calendar": {"calendar_events": {"events": [
        {"title": "COMP5999 Lecture", "start": "2026-09-22T05:00:00Z", "end": "2026-09-22T07:00:00Z", "all_day": False, "calendar": "课表", "location": "ABS 1110"},
        {"title": "晨跑", "start": "2026-09-22T07:00:00+10:00", "end": "2026-09-22T08:00:00+10:00", "all_day": False, "calendar": "个人", "location": ""},
        {"title": "交房租", "start": "2026-09-23T00:00:00+10:00", "end": "2026-09-24T00:00:00+10:00", "all_day": True, "calendar": "个人", "location": ""}]}},
    "news": {"headlines": {"feeds": [{"topic": "科技与 AI", "source": "Solidot", "items": [{"title": "某大模型发布", "published": ""}]},
                                     {"topic": "国内要闻", "source": "坏了的源", "items": [], "error": "URLError"}]}},
    "weather": {"today_weather": {"summary": "多云，18 度"}},
}}


def test_situation_gives_the_model_local_times_and_precomputed_date_distances(conn, cfg):
    save_state(conn, JobState("sources", cursor=json.dumps(SNAPSHOT)))
    Reminders(conn, TZ).add("健身", NOW.replace(hour=19), now=0, recurrence="WEEKLY:0,2,4")
    conn.execute("INSERT INTO facts(category, text, valid_from, recorded_at) VALUES ('important_date', '妈妈的生日是9月30日', 0, 0),"
                 " ('health', '对花生过敏', 0, 0), ('important_date', '爸爸的生日是3月2日', 0, 0)")
    say(conn, "alf", "喝水（09:30）", 30, kind="reminder")
    say(conn, "alf", "早上那条大模型新闻你看了吗", 100, kind="nudge")
    say(conn, "user", "昨晚说的话", 60 * 14)
    fake = FakeLLM(facts=[{"send": False, "why": "没有"}])
    proactive.proactive_job(ctx_for(conn, cfg, fake))
    text = fake.requests[0]["messages"][1]["content"]
    for expected in [
        "[今天是什么日子]\n农历八月十二。今天：平常的一天\n9月25日（周五，3 天后）：中秋节",
        "[日历]\n今天 15:00–17:00 COMP5999 Lecture @ ABS 1110\n今天 07:00–08:00 晨跑（已结束）\n明天 全天 交房租",   # UTC 时间换成了本地
        "〔科技与 AI·Solidot〕某大模型发布",
        '[weather]\n{"today_weather": {"summary": "多云，18 度"}}',                                                # 第三方 MCP 服务器的结果原样附上
        "健身｜今天 9月22日（周二）19:00｜每周一、三、五",
        "[刚发出的提醒]\n09:30 喝水（09:30）——之后他还没有任何回应",
        "妈妈的生日是9月30日（还有 8 天，周三）", "对花生过敏\n", "爸爸的生日是3月2日\n",                                # 远的不标
        "[你今天已经主动说过的]\n08:20 早上那条大模型新闻你看了吗",
        "他上一次说话是 14 小时前。",
    ]:
        assert expected in text, expected
    assert "坏了的源" not in text


def test_snapshot_problems_are_stated_plainly_not_hidden():
    lines = "\n".join(proactive.describe_snapshot({"at": int(NOW.timestamp()) - 5 * 3600, "sources": {
        "calendar": {"calendar_events": {"error": "没有访问日历的权限"}}}}, NOW))
    assert "（下面的外部信息是 5 小时前取的，可能不是最新）" in lines and "[日历]\n（读不到日历：没有访问日历的权限）" in lines
    assert "[日历]\n（没有接入日历）" in "\n".join(proactive.describe_snapshot({}, NOW))


# --- 信息来源（MCP）---------------------------------------------------------------------

def test_special_days_come_from_calendar_libraries():
    assert days.describe(date(2026, 12, 20), 7)["upcoming"] == [
        {"date": "2026-12-22", "weekday": "周二", "in_days": 2, "what": ["节气：冬至"]},
        {"date": "2026-12-24", "weekday": "周四", "in_days": 4, "what": ["平安夜"]},
        {"date": "2026-12-25", "weekday": "周五", "in_days": 5, "what": ["圣诞节", "澳大利亚（新州）公共假期：Christmas Day"]},
        {"date": "2026-12-26", "weekday": "周六", "in_days": 6, "what": ["澳大利亚（新州）公共假期：Boxing Day"]}]
    eve = days.describe(date(2027, 2, 5), 1)
    assert (eve["lunar"], eve["today"], eve["upcoming"][0]["what"]) == ("农历腊月廿九", ["除夕（中国公共假期）"], ["春节（中国公共假期）"])
    assert days.describe(date(2026, 11, 16), 0) == {"date": "2026-11-16", "weekday": "周一", "lunar": "农历十月初八", "today": [], "upcoming": []}


def test_rss_and_atom_titles_are_parsed_and_a_dead_feed_does_not_break_the_rest():
    rss = "<rss><channel><item><title> 第一条\n 新闻 </title><pubDate>Mon, 16 Nov 2026 15:09:17 +0800</pubDate></item><item><title>第二条</title></item><item><title></title></item></channel></rss>"
    assert news.parse_feed(rss.encode(), 5) == [{"title": "第一条 新闻", "published": "Mon, 16 Nov 2026 15:09:17 +0800"}, {"title": "第二条", "published": ""}]
    atom = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom 条目</title><updated>2026-11-16T04:00:00Z</updated></entry></feed>'
    assert news.parse_feed(atom.encode(), 5) == [{"title": "Atom 条目", "published": "2026-11-16T04:00:00Z"}]
    [dead] = news.collect([{"topic": "x", "source": "坏的", "url": "http://127.0.0.1:1/rss"}], 3)["feeds"]
    assert dead["items"] == [] and "error" in dead


def test_calendar_reader_reports_each_failure_in_plain_words(tmp_path):
    helper = tmp_path / "elsewhere" / "AlfCalendar.app"
    assert "还没编译" in calendar_server.read_events(2, helper)["error"]
    helper.mkdir(parents=True)

    asked = []

    def writes(text):
        return lambda app, out, command, *args: (asked.append("ask" in args), out.write_text(text))

    good = '{"events": [{"title": "组会", "start": "2026-09-22T15:00:00+10:00", "end": "2026-09-22T16:00:00+10:00", "all_day": false, "calendar": "工作", "location": ""}]}'
    assert calendar_server.read_events(2, helper, run=writes(good))["events"][0]["title"] == "组会"
    assert "隐私与安全性 → 日历" in calendar_server.read_events(2, helper, run=writes('{"error": "denied"}'))["error"]
    assert calendar_server.read_events(2, helper, run=lambda *a: None) == {"error": "读日历失败：小 App 没有给出结果"}
    # 后台读取时不申请授权（不弹窗），只有明确要求时才申请
    assert calendar_server.read_events(2, helper, run=writes('{"error": "not_determined"}'))["error"] == calendar_server.NEEDS_PERMISSION
    calendar_server.read_events(2, helper, run=writes(good), ask=True)
    assert asked == [False, False, False, True]

    def hangs(app, out, command, *args):
        import subprocess
        raise subprocess.TimeoutExpired("open", 100)

    assert "看看有没有弹窗" in calendar_server.read_events(2, helper, run=hangs)["error"]


def test_only_read_only_tools_without_required_arguments_are_called_automatically():
    from types import SimpleNamespace as NS

    tool = lambda ro, required: NS(annotations=NS(read_only_hint=ro) if ro is not None else None, input_schema={"required": required})   # noqa: E731
    assert [sources.is_snapshot_tool(t) for t in (tool(True, []), tool(True, ["city"]), tool(False, []), tool(None, []))] == [True, False, False, False]


def test_real_mcp_round_trip_with_the_builtin_days_server(cfg, no_real_sources):
    """真的走一遍 MCP：起一个自带的服务器进程，作为客户端连上去、列工具、调用。不联网。"""
    snapshot = no_real_sources(cfg, {"sources.days": True, "sources.calendar": False, "sources.reminders": False, "sources.news": False, "news.topics": []})
    result = snapshot["sources"]["days"]["special_days"]
    assert set(result) == {"date", "weekday", "lunar", "today", "upcoming"} and result["lunar"].startswith("农历")
    assert list(snapshot["sources"]) == ["days"]


def test_server_selection_follows_settings(cfg):
    on = sources.servers_for(cfg, {"sources.days": True, "sources.calendar": True, "sources.news": True, "news.topics": ["科技与 AI"],
                                   "mcp.servers": [{"name": "weather", "command": "uvx", "args": ["some-weather-mcp"], "env": {"CITY": "Sydney"}}]})
    assert list(on) == ["days", "calendar", "reminders", "news", "weather"]
    assert json.loads(on["news"].env["ALF_NEWS_FEEDS"])[0] == {"topic": "科技与 AI", "source": "Solidot", "url": "https://www.solidot.org/index.rss"}
    assert (on["weather"].command, on["weather"].args, on["weather"].env["CITY"]) == ("uvx", ["some-weather-mcp"], "Sydney")
    off = sources.servers_for(cfg, {"sources.days": False, "sources.calendar": False, "sources.reminders": False, "sources.news": True, "news.topics": []})
    assert list(off) == []                                                    # 开着新闻但没选主题，等于没有


def test_sources_job_keeps_good_results_and_names_the_broken_source(conn, cfg, monkeypatch):
    monkeypatch.setattr(sources, "gather", lambda c, p: {"at": 1, "sources": {
        "days": {"special_days": {"today": []}}, "calendar": {"calendar_events": {"error": "没有访问日历的权限"}}, "news": {"error": "TimeoutError: "}}})
    ctx = ctx_for(conn, cfg)
    assert jobs.sources_job(ctx) == "汇总了 1 个来源；日历：没有访问日历的权限；新闻：TimeoutError: "
    assert json.loads(ctx.cursor)["sources"]["days"] == {"special_days": {"today": []}}


def test_briefing_opens_with_a_written_note_and_lists_calendar_events(conn, cfg, monkeypatch):
    monkeypatch.setattr(jobs, "notify_desktop", lambda *a: None)
    save_state(conn, JobState("sources", cursor=json.dumps(SNAPSHOT)))
    Reminders(conn, TZ).add("交作业", NOW.replace(hour=20), now=0)
    fake = FakeLLM(facts=[{"note": "周五就是中秋了，记得给家里打个电话。下午有课，别迟到。"}])
    morning = NOW.replace(hour=8, minute=0)                      # 晨跑 07:00–08:00 刚结束，不再列出
    assert jobs.briefing_job(ctx_for(conn, cfg, fake, now=morning)) == "播报了 2 个安排，带一段梳理"
    assert timeline.fetch(conn)[0]["text"] == (
        "周五就是中秋了，记得给家里打个电话。下午有课，别迟到。\n"
        "今天还有 1 个安排：\n  #1  今天 9月22日（周二）20:00  交作业\n"
        "日历上：\n  15:00–17:00 COMP5999 Lecture @ ABS 1110")
    # 开场白写不出来时，安排清单照发
    broken = FakeLLM(facts=[LLMError("DeepSeek 响应超时")])
    assert jobs.briefing_job(ctx_for(conn, cfg, broken, now=morning)) == "播报了 2 个安排"
    assert timeline.fetch(conn)[-1]["text"].startswith("今天还有 1 个安排：")


def test_proactive_settings_are_validated(conn, cfg):
    from alf import prefs

    saved = prefs.save(conn, cfg, {"proactive.level": "chatty", "quiet.start": "23:30", "news.topics": ["国内要闻", "科技与 AI"],
                                   "mcp.servers": [{"name": "weather", "command": "uvx", "args": ["w"]}]})
    assert (saved["proactive.level"], saved["quiet.start"], saved["news.topics"]) == ("chatty", "23:30", ["科技与 AI", "国内要闻"])
    for bad, message in [({"proactive.level": "loud"}, "只能是 quiet、normal、active 或 chatty"), ({"news.topics": ["体育"]}, "只能从"),
                         ({"mcp.servers": [{"name": "days", "command": "x"}]}, "是自带服务器的名字"), ({"mcp.servers": [{"name": "w"}]}, "每一项都要有 name 和 command")]:
        with pytest.raises(ValueError, match=message):
            prefs.save(conn, cfg, bad)
