"""日历和提醒事项：同步、发现变动、开始前提醒，以及对话里的写入和撤销。全程用替身，不启动真的小 App。"""
import json
from datetime import datetime, timedelta

import pytest

from alf import apple, calendar_sync, timeline
from alf.agent import Turn
from alf.connectors import ConnectorError
from alf.decide import Decider
from alf.heartbeat import JobContext

from .conftest import NOW, TZ, FakeApple, FakeLLM, Scripted


def event(title, day, hour, minute=0, *, hours=1, key=None, **more):
    start = datetime(2026, 9, day, hour, minute, tzinfo=TZ)
    return {"id": key or title, "title": title, "start": start.isoformat(), "end": (start + timedelta(hours=hours)).isoformat(),
            "all_day": False, "calendar": "个人", "location": "", **more}


def ctx_for(conn, cfg, now=NOW, cursor=None, prefs=None):
    return JobContext(conn, cfg, now, Decider(Scripted(), conn), cursor, prefs=prefs or {})


def posted(conn):
    return [(r["kind"], r["text"]) for r in timeline.fetch(conn)]


def until(now=NOW):
    return int((now + timedelta(days=7)).replace(hour=0, minute=0).timestamp())


# --- 同步和变动 -----------------------------------------------------------------------

def test_first_sync_announces_nothing_and_later_changes_are_named(conn):
    lecture, gym = event("COMP5999 Lecture", 22, 15, hours=2, location="ABS 1110"), event("健身", 23, 19)
    assert calendar_sync.apply(conn, [lecture, gym], NOW, None) == []            # 第一次：全是"新的"，一条都不播报
    assert [calendar_sync.row_line(r, NOW) for r in calendar_sync.upcoming(conn, NOW)] == [
        "今天 15:00–17:00 COMP5999 Lecture @ ABS 1110", "明天 19:00–20:00 健身"]
    # 用户在日历里加了一个、把健身挪到了后天、删了今天的课
    changes = calendar_sync.apply(conn, [event("组会", 25, 14), event("健身", 24, 19)], NOW, until())
    assert changes == ["改期：健身 从 明天 19:00–20:00 改到 后天 19:00–20:00", "新增：9月25日（周五） 14:00–15:00 组会",
                       "取消：今天 15:00–17:00 COMP5999 Lecture @ ABS 1110"]
    assert calendar_sync.apply(conn, [event("组会", 25, 14), event("健身", 24, 19)], NOW, until()) == []   # 没变就不说


def test_events_that_merely_slid_into_the_window_or_ended_are_not_news(conn):
    calendar_sync.apply(conn, [event("晨跑", 22, 7), event("健身", 23, 19)], NOW, None)
    next_day = NOW + timedelta(days=1)
    # 第二天：晨跑已经过去（从结果里消失），29 号的日程是随着窗口滑进来的——都不是用户刚做的改动
    assert calendar_sync.apply(conn, [event("健身", 23, 19), event("月会", 29, 9, minute=30)], next_day, until()) == []
    assert calendar_sync.apply(conn, [event("健身", 23, 19), event("月会", 29, 9, minute=30), event("牙医", 24, 11)], next_day, until(next_day)) == [
        "新增：明天 11:00–12:00 牙医"]


def test_each_occurrence_of_a_repeating_event_is_tracked(conn):
    weekly = [event("课", 22, 15, key="W"), event("课", 24, 15, key="W")]
    calendar_sync.apply(conn, weekly, NOW, None)
    assert calendar_sync.apply(conn, weekly[:1], NOW, until()) == ["取消：后天 15:00–16:00 课"]


def test_job_keeps_night_changes_for_the_morning(conn, cfg):
    read = lambda days: {"events": feed}   # noqa: E731
    feed = [event("健身", 23, 19)]
    first = ctx_for(conn, cfg)
    assert calendar_sync.calendar_job(first, read) == "同步了 1 条日程" and posted(conn) == []
    feed = [event("健身", 23, 19), event("组会", 25, 14)]
    night = ctx_for(conn, cfg, now=NOW.replace(hour=23), cursor=first.cursor)
    assert calendar_sync.calendar_job(night, read) == "同步了 2 条日程，1 处变动等早上再说" and posted(conn) == []
    morning = ctx_for(conn, cfg, now=NOW + timedelta(days=1), cursor=night.cursor)
    assert calendar_sync.calendar_job(morning, read) == "同步了 2 条日程，告诉了你 1 处变动"
    assert posted(conn) == [("calendar", "日历有变动：\n新增：9月25日（周五） 14:00–15:00 组会")]
    assert json.loads(morning.cursor)["pending"] == []


def test_missing_permission_is_a_status_not_a_failure(conn, cfg):
    assert calendar_sync.calendar_job(ctx_for(conn, cfg)) == "还没同步：还没授权读日历：在 Alf 的设置里点「授权日历和提醒事项」"
    with pytest.raises(ConnectorError, match="隐私与安全性 → 日历"):
        calendar_sync.calendar_job(ctx_for(conn, cfg), lambda days: {"error": "没有访问日历的权限：系统设置 → 隐私与安全性 → 日历，给 Alf（或 python）打开"})


def test_alert_goes_out_once_shortly_before_a_timed_event(conn, cfg):
    calendar_sync.apply(conn, [event("组会", 22, 10, minute=15, location="ABS 1110"), event("午饭", 22, 12),
                              {**event("交房租", 22, 0), "all_day": True}, event("早就开始的", 22, 9, minute=50)], NOW, None)
    assert calendar_sync.calendar_alerts_job(ctx_for(conn, cfg)) == "发出 1 条日程提醒"
    assert posted(conn) == [("calendar", "15 分钟后：组会（10:15–11:15 @ ABS 1110）")]
    assert calendar_sync.calendar_alerts_job(ctx_for(conn, cfg)) is None                      # 不会响第二遍
    calendar_sync.apply(conn, [event("组会", 22, 10, minute=15, location="ABS 1110"), event("午饭", 22, 12)], NOW, until())
    assert calendar_sync.calendar_alerts_job(ctx_for(conn, cfg, now=NOW + timedelta(minutes=5))) is None   # 再同步一次也不会重置
    assert calendar_sync.calendar_alerts_job(ctx_for(conn, cfg, now=NOW.replace(hour=11, minute=55), prefs={"calendar.lead_minutes": 5})) == "发出 1 条日程提醒"
    assert posted(conn)[-1] == ("calendar", "5 分钟后：午饭（12:00–13:00）")
    assert calendar_sync.calendar_alerts_job(ctx_for(conn, cfg, prefs={"calendar.alerts": False})) is None


# --- 小 App 的调用 --------------------------------------------------------------------

def test_helper_gets_a_payload_file_and_errors_come_back_in_plain_words(tmp_path):
    helper = tmp_path / "Helper.app"
    helper.mkdir()
    seen = {}

    def run(app, out, command, *args):
        seen.update(command=command, payload=json.loads(open(args[0]).read()) if command.startswith("add") else None, args=args)
        out.write_text(json.dumps(seen.get("answer", {"id": "EV-1", "calendar": "个人"})))

    start = datetime(2026, 9, 25, 14, 0, tzinfo=TZ)
    assert apple.add_event("组会", start, start + timedelta(hours=1), location="ABS", helper=helper, run=run) == {"id": "EV-1", "calendar": "个人"}
    assert seen["command"] == "add-event" and seen["payload"] == {
        "title": "组会", "start": "2026-09-25T14:00:00+10:00", "end": "2026-09-25T15:00:00+10:00", "all_day": False, "location": "ABS"}
    apple.add_reminder("买牛奶", helper=helper, run=run)
    assert seen["command"] == "add-reminder" and seen["payload"] == {"title": "买牛奶"}
    for answer, said in [("not_by_alf", "这一条不是 Alf 建的，不能由 Alf 删除"), ("not_found", "在日历里找不到这一条，可能已经被删掉了"),
                         ("denied", "没有访问日历的权限：系统设置 → 隐私与安全性 → 日历，给 Alf（或 python）打开"),
                         ("save_failed: 磁盘满了", "写日历失败：save_failed: 磁盘满了")]:
        seen["answer"] = {"error": answer}
        assert apple.delete_event("EV-1", helper=helper, run=run) == {"error": said}
    assert seen["args"] == ("EV-1",)
    seen["answer"] = {"error": "not_determined"}
    assert apple.read_reminders(7, helper=helper, run=run) == {"error": "还没授权读提醒事项：在 Alf 的设置里点「授权日历和提醒事项」"}
    assert apple.delete_reminder("x", helper=tmp_path / "nowhere.app") == {"error": "写提醒事项的小 App 还没编译：在项目目录运行 ./app/build.sh"}


# --- 对话里写日历 ---------------------------------------------------------------------

def test_adding_to_the_calendar_writes_it_and_says_exactly_what(make_chat, conn):
    cal = FakeApple()
    fake = FakeLLM([("add_calendar_event", {"title": "组会", "time_text": "周五下午两点", "location": "ABS 1110"})], "加到日历了",
                   "周五两点有组会")
    agent = make_chat(fake, apple_api=cal)
    assert agent.handle(Turn("把周五下午两点的组会加到日历，在 ABS 1110")).text == "加到日历了\n📅 已加到日历：组会｜9月25日（周五）14:00–15:00｜ABS 1110"
    start = datetime(2026, 9, 25, 14, 0, tzinfo=TZ)
    assert cal.events == {"E1": ("组会", start, start + timedelta(hours=1), False, "ABS 1110")}
    assert fake.chat_requests[1]["messages"][-1]["content"] == json.dumps(
        {"ok": True, "id": 1, "added": "组会｜9月25日（周五）14:00–15:00｜ABS 1110", "calendar": "个人"}, ensure_ascii=False)
    # 下一次同步读到这条日程时，不会把它当成"用户新加的"再播报一遍
    synced = {**event("组会", 25, 14, key="E1", location="ABS 1110"), "by_alf": True}
    assert calendar_sync.apply(conn, [synced], NOW, until()) == []
    agent.handle(Turn("我周五有什么安排"))
    context = fake.context()
    assert "[日历]（未来 7 天）\n9月25日（周五） 14:00–15:00 组会 @ ABS 1110" in context
    assert "[你写进日历和提醒事项的]\n#1（日历）组会｜9月25日（周五）14:00–15:00" in context


def test_a_date_without_a_time_becomes_an_all_day_event_and_bad_requests_change_nothing(make_chat):
    cal = FakeApple()
    fake = FakeLLM([("add_calendar_event", {"title": "交房租", "time_text": "下个月5号"}),
                    ("add_calendar_event", {"title": "课", "time_text": "每周一上午九点"}),
                    ("add_calendar_event", {"title": "早上的事", "time_text": "今天早上八点"}),
                    ("add_calendar_event", {"title": "长会", "time_text": "明天下午两点", "duration_minutes": 120})], "好")
    reply = make_chat(fake, apple_api=cal).handle(Turn("…"))
    assert reply.text == "好\n📅 已加到日历：交房租｜10月5日（周一）全天\n📅 已加到日历：长会｜明天 9月23日（周三）14:00–16:00"
    results = [json.loads(m["content"]) for m in fake.chat_requests[1]["messages"] if m["role"] == "tool"]
    assert results[0]["notes"] == ["用户没说几点，建成了全天日程"]
    assert results[1]["error"] == "重复的日程现在还加不了，只能加单次的" and "已经过去了" in results[2]["error"]
    assert [e[3] for e in cal.events.values()] == [True, False]


def test_saying_it_is_on_the_calendar_without_writing_it_is_sent_back(make_chat):
    """没授权时写不进去。回复不能说"加好了"。"""
    cal = FakeApple()
    cal.fail = "还没授权读日历：在 Alf 的设置里点「授权日历和提醒事项」"
    fake = FakeLLM([("add_calendar_event", {"title": "组会", "time_text": "周五下午两点"})], "好，已经加到日历了",
                   "抱歉，还没加成：日历还没授权。到 Alf 的设置里点一下「授权日历和提醒事项」，我再加。")
    # 判断替身学真的 Jev：只有"加到日历了"这种话算声称，"没加成""没写成"不算
    judge = Scripted(rel="unrelated", claim=lambda q, state: 0.9 if "加到日历了" in state["assistant_reply"] else 0.03)
    agent = make_chat(fake, backend=judge, apple_api=cal)
    assert agent.handle(Turn("把周五下午两点的组会加到日历")).text.startswith("抱歉，还没加成")
    assert "会让他以为已经改动了日历" in fake.chat_requests[2]["messages"][-1]["content"]
    assert agent.conn.execute("SELECT COUNT(*) FROM apple_items").fetchone()[0] == 0
    # 用了别的工具（建了一条 Alf 自己的提醒）也不算数：说的是日历，回执却是 ⏰
    wrong = FakeLLM([("create_reminder", {"title": "组会", "time_text": "周五下午两点"})], "加到日历了", "日历我没写成，先给你建了条提醒")
    assert make_chat(wrong, backend=judge, apple_api=cal).handle(Turn("把周五下午两点的组会加到日历")).text == "日历我没写成，先给你建了条提醒\n⏰ 组会｜9月25日（周五）14:00"
    assert "会让他以为已经改动了日历" in wrong.chat_requests[2]["messages"][-1]["content"]


def test_undo_only_reaches_what_alf_wrote_and_checks_the_target(make_chat, conn):
    cal = FakeApple()
    fake = FakeLLM([("add_calendar_event", {"title": "组会", "time_text": "周五下午两点"}),
                    ("add_apple_reminder", {"title": "买牛奶"}), ("add_apple_reminder", {"title": "交论文", "time_text": "周五下午五点"})], "都加好了",
                   [("undo_apple_item", {"id": 1})], "你要删哪个？",
                   [("undo_apple_item", {"id": 1}), ("undo_apple_item", {"id": 99})], "删掉了",
                   [("undo_apple_item", {"id": 2})], "也删了")
    judge = Scripted(rel="unrelated", claim=0.9, target=0.05)
    agent = make_chat(fake, backend=judge, apple_api=cal)
    assert agent.handle(Turn("…")).text == ("都加好了\n📅 已加到日历：组会｜9月25日（周五）14:00–15:00"
                                            "\n☑️ 已加到提醒事项：买牛奶\n☑️ 已加到提醒事项：交论文｜9月25日（周五）17:00")
    assert cal.todos == {"R1": ("买牛奶", None, False), "R2": ("交论文", datetime(2026, 9, 25, 17, 0, tzinfo=TZ), True)}
    assert agent.handle(Turn("今天好累")).text == "你要删哪个？" and "E1" in cal.events           # 用户没说要删：拦下
    assert json.loads(fake.chat_requests[3]["messages"][-1]["content"])["error"] == "用户这句话看起来不是要删掉「组会」，没有删"
    judge.rules["target"] = 0.9
    assert agent.handle(Turn("撤销刚才加的组会")).text == "删掉了\n🗑 已从日历删除：组会｜9月25日（周五）14:00–15:00"
    assert cal.events == {} and conn.execute("SELECT COUNT(*) FROM calendar_events").fetchone()[0] == 0
    assert "只能撤销[你写进日历和提醒事项的]里列出的" in json.loads(fake.chat_requests[5]["messages"][-1]["content"])["hint"]
    cal.todos.clear()                                                                           # 用户自己在手机上删了
    assert agent.handle(Turn("牛奶那条也删了")).text == "也删了\n🗑 已从提醒事项删除：买牛奶（它之前已经不在了）"
    assert [tuple(r) for r in conn.execute("SELECT id, kind, status FROM apple_items")] == [(1, "event", "removed"), (2, "reminder", "removed"), (3, "reminder", "active")]
