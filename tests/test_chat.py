import json
from datetime import datetime, timedelta

import pytest

from alf import chat, timeline
from alf.agent import Agent, Turn
from alf.chat import ChatAgent, make_agent
from alf.heartbeat import JobState, save_state
from alf.decide import DecisionError
from alf.llm import LLMError

from .conftest import NOW, TZ, FakeLLM, Scripted


def tool_results(fake, n=-1):
    """第 n 次对话调用时，模型看到的工具返回。"""
    return [json.loads(m["content"]) for m in fake.chat_requests[n]["messages"] if m["role"] == "tool"]


# --- 提醒：场景 1、2、3、6 -----------------------------------------------------------

def test_creating_a_reminder_appends_a_code_computed_receipt(make_chat):
    fake = FakeLLM([("create_reminder", {"title": "交作业", "time_text": "明天下午三点"})], "好嘞，到时候叫你～")
    agent = make_chat(fake)
    reply = agent.handle(Turn("明天下午三点提醒我交作业"))
    assert reply.text == "好嘞，到时候叫你～\n⏰ 交作业｜明天 9月23日（周三）15:00"
    [r] = agent.reminders.pending()
    assert (r.title, r.due) == ("交作业", datetime(2026, 9, 23, 15, 0, tzinfo=TZ))
    assert tool_results(fake) == [{"ok": True, "id": 1, "title": "交作业", "when": "明天 9月23日（周三）15:00",
                                   "will_notify_at": "明天 9月23日（周三）15:00（准点）"}]


def test_model_sees_time_reminders_memory_and_the_message(make_chat):
    fake = FakeLLM("在呢")
    agent = make_chat(fake)
    agent.reminders.add("开会", NOW + timedelta(hours=5), now=0, lead_minutes=30)
    agent.memory.write("对花生过敏", "health", now=int(NOW.timestamp()))
    agent.handle(Turn("在吗"))
    request = fake.chat_requests[0]
    assert request["messages"][0] == {"role": "system", "content": chat.SYSTEM_PROMPT}
    assert [t["function"]["name"] for t in request["tools"]] == [
        "create_reminder", "update_reminder", "cancel_reminder", "add_calendar_event", "add_apple_reminder", "undo_apple_item",
        "read_email", "draft_email_reply", "send_email_draft", "discard_email_draft", "remember",
            "change_setting", "note_correction", "settle_rule_proposal",
            "note_about_self", "withdraw_impression", "make_pact", "settle_pact", "web_search", "recall", "update_memory", "forget_memory"]
    assert fake.context() == "\n".join([
        "[现在] 2026年9月22日 周二 10:00（Australia/Sydney）", "",
        "[待办提醒]", "#1 开会｜今天 9月22日（周二）15:00｜提前半小时（14:30）提醒", "",
        "[日历]（未来 7 天）", "（还没接上他的日历，所以这里是空的；被问到日历上有什么就直说还没接上）", "",
        "[你记得的事]", "#1（健康）对花生过敏", "",
        "[你对他的好感度] 50/100（刚熟悉）", "",
        "[最近对话]", "（这是第一句话）", "",
        "[用户刚发来的消息]", "在吗"])


def test_moving_keeps_the_day_and_cancelling_removes_it(make_chat):
    fake = FakeLLM([("create_reminder", {"title": "交作业", "time_text": "明天下午三点"})], "好",
                   [("update_reminder", {"id": 1, "time_text": "四点"})], "改好啦",
                   [("cancel_reminder", {"id": 1})], "行，不提醒了")
    agent = make_chat(fake)
    agent.handle(Turn("明天下午三点提醒我交作业"))
    assert agent.handle(Turn("改到四点")).text == "改好啦\n⏰ 已改：交作业｜明天 9月23日（周三）16:00"
    assert tool_results(fake)[0]["was"] == {"when": "明天 9月23日（周三）15:00", "will_notify_at": "明天 9月23日（周三）15:00（准点）"}
    assert agent.handle(Turn("取消吧")).text == "行，不提醒了\n🗑 已取消：交作业（明天 9月23日（周三）16:00）"
    assert "#1 交作业｜明天 9月23日（周三）16:00" in fake.context(4)         # 第三轮开头看到的是改后的时间
    assert agent.reminders.pending() == []


def test_shift_title_recurrence_and_lead(make_chat):
    fake = FakeLLM([("create_reminder", {"title": "健身", "time_text": "每周一三五晚上7点"}),
                    ("create_reminder", {"title": "看牙医", "time_text": "后天上午十点，提前一天"})], "都设好了",
                   [("update_reminder", {"id": 2, "shift_minutes": 90, "title": "看牙"})], "好")
    agent = make_chat(fake)
    assert agent.handle(Turn("…")).text == ("都设好了\n⏰ 健身｜明天 9月23日（周三）19:00｜每周一、三、五"
                                            "\n⏰ 看牙医｜后天 9月24日（周四）10:00｜提前1天（9月23日 10:00）提醒")
    assert agent.handle(Turn("牙医推迟一个半小时")).text == "好\n⏰ 已改：看牙｜后天 9月24日（周四）11:30｜提前1天（9月23日 11:30）提醒"


def test_guesses_are_passed_back_to_the_model_as_notes(make_chat):
    fake = FakeLLM([("create_reminder", {"title": "开会", "time_text": "3点"}),
                    ("create_reminder", {"title": "交房租", "time_text": "下个月5号"})], "好")
    make_chat(fake).handle(Turn("…"))
    assert [r["notes"] for r in tool_results(fake)] == [["用户没说上午还是下午，按 15:00 算的"], ["用户没说几点，先定在了 09:00"]]


@pytest.mark.parametrize("args, error", [
    ({"title": "交作业", "time_text": "有空的时候"}, "没看懂这个时间"),
    ({"title": "交作业", "time_text": "冬至那天下午三点"}, "时间里有一部分程序没看懂：冬 那天"),
    ({"title": "交作业", "time_text": "明天下午三点提醒我交作业"}, "时间里有一部分程序没看懂：交作业"),
    ({"title": "准备面试", "time_text": "周五上午10点面试，周四晚上8点"}, "time_text 里有好几个时间：周五上午10点、周四晚上8点"),
    ({"title": "交作业", "time_text": "今天上午9点"}, "今天 9月22日（周二）09:00 已经过去了"),
    ({"title": " ", "time_text": "明天"}, "没说要提醒什么"),
    ({"title": "交作业"}, "参数不对：KeyError('time_text')"),
])
def test_a_time_that_is_not_fully_understood_creates_nothing(make_chat, args, error):
    fake = FakeLLM([("create_reminder", args)], "我没弄明白时间，能说具体点吗？")
    agent = make_chat(fake)
    reply = agent.handle(Turn("…"))
    assert tool_results(fake)[0]["error"] == error
    assert agent.reminders.pending() == [] and reply.text == "我没弄明白时间，能说具体点吗？"


def test_iso_dates_from_the_model_are_read_as_dates_not_dropped(make_chat):
    fake = FakeLLM([("create_reminder", {"title": "交作业", "time_text": "2026-09-25 15:00"})], "好")
    agent = make_chat(fake)
    agent.handle(Turn("…"))
    assert agent.reminders.pending()[0].due == datetime(2026, 9, 25, 15, 0, tzinfo=TZ)


def test_unknown_ids_tools_and_bad_arguments_are_reported_to_the_model(make_chat):
    from alf.llm import LLMReply, ToolCall

    fake = FakeLLM([("update_reminder", {"id": 99, "time_text": "四点"}), ("cancel_reminder", {"id": 99}),
                    ("update_memory", {"id": 99, "text": "x"}), ("forget_memory", {"id": 99}),
                    ("send_email", {"to": "x"}), ("update_reminder", {"id": 1})], "没找到")
    agent = make_chat(fake)
    agent.reminders.add("交作业", NOW + timedelta(days=1), now=0)
    assert agent.handle(Turn("…")).text == "没找到"
    assert [r["error"] for r in tool_results(fake)] == [
        "没有编号 99 的待办提醒", "没有编号 99 的待办提醒", "没有编号 99 的记忆", "没有编号 99 的记忆",
        "没有 send_email 这个工具", "没说要改成什么"]

    class BadJson(FakeLLM):
        def complete(self, messages, **kw):
            if len(self.requests) == 0:
                self.requests.append({"messages": messages, "tools": kw.get("tools"), "json_mode": False})
                call = ToolCall("c1", "cancel_reminder", None)
                return LLMReply("", [call], {"role": "assistant", "content": None, "tool_calls": []}, {})
            return super().complete(messages, **kw)

    bad = BadJson("再说一遍？")
    make_chat(bad).handle(Turn("…"))
    assert tool_results(bad)[0]["error"] == "参数不是合法的 JSON 对象"


def test_vague_request_is_answered_with_a_question_and_changes_nothing(make_chat):
    agent = make_chat(FakeLLM("哪件事呀？"))
    assert agent.handle(Turn("提醒我那件事")).text == "哪件事呀？"
    assert agent.reminders.pending() == []


# --- 出错 ---------------------------------------------------------------------------

def test_model_outage_is_admitted_and_finished_work_is_still_reported(make_chat):
    agent = make_chat(FakeLLM(LLMError("DeepSeek 响应超时")))
    assert agent.handle(Turn("在吗")).text == "我这会儿连不上脑子（DeepSeek 响应超时），等下再跟我说一遍？"
    fake = FakeLLM([("create_reminder", {"title": "交作业", "time_text": "明天下午三点"})], LLMError("DeepSeek 限流了"))
    reply = make_chat(fake).handle(Turn("明天下午三点提醒我交作业"))
    assert reply.text == "我这边刚才卡了一下（DeepSeek 限流了），不过下面这些已经办好了：\n⏰ 交作业｜明天 9月23日（周三）15:00"


def test_tool_loop_is_bounded_and_empty_replies_get_a_fallback(make_chat):
    looping = FakeLLM(*[[("cancel_reminder", {"id": 99})]] * chat.MAX_STEPS)
    assert make_chat(looping).handle(Turn("…")).text == "我好像绕进去了，你再说一遍？"
    assert len(looping.chat_requests) == chat.MAX_STEPS
    assert make_chat(FakeLLM("  ")).handle(Turn("…")).text == "嗯？我没太明白，再说一遍？"
    silent = FakeLLM([("create_reminder", {"title": "喝水", "time_text": "半小时后"})], "")
    assert make_chat(silent).handle(Turn("…")).text == "好了～\n⏰ 喝水｜今天 9月22日（周二）10:30"


# --- 自动记忆：场景 4、5 ---------------------------------------------------------------

def test_facts_mentioned_in_passing_are_stored_after_the_reply_is_out(make_chat, conn):
    fake = FakeLLM("哇，要搬家啦", facts=[{"facts": [{"text": "2026年10月搬到 Glebe", "category": "location", "quote": "搬到 Glebe 了"},
                                                    {"text": "养了一只猫叫年糕", "category": "pets", "quote": "我的猫年糕"}]}])
    agent = make_chat(fake)
    sent = []
    agent.handle(Turn("我下个月搬到 Glebe 了，带着我的猫年糕"), on_reply=lambda reply: (fake.events.append("reply sent"), sent.append(reply.text)))
    # 回复先发出去，后台再补记。新记的事不出声：不再追发一条"顺手记下了"（晚上的汇总和记忆页上看得到）。
    assert fake.events == ["chat", "reply sent", "extract"]
    assert sent == ["哇，要搬家啦"]
    assert [r["text"] for r in conn.execute("SELECT text FROM episodes WHERE role = 'alf' ORDER BY id")] == sent
    assert [(f.category, f.text) for f in agent.memory.current()] == [("other", "养了一只猫叫年糕"), ("location", "2026年10月搬到 Glebe")]
    extract = fake.extract_requests[0]["messages"]
    assert extract[0]["content"] == chat.EXTRACT_PROMPT
    assert extract[1]["content"].startswith("今天：2026年9月22日 周二\n")
    assert extract[1]["content"].endswith("要分析的消息：\n我下个月搬到 Glebe 了，带着我的猫年糕")
    calls = conn.execute("SELECT purpose, model, input_tokens, cached_tokens, ok FROM model_calls ORDER BY id").fetchall()
    assert [tuple(c) for c in calls] == [("chat", "fake-model", 100, 60, 1), ("memory.extract", "fake-model", 100, 60, 1),
                                         ("chat.afterthoughts", "scripted", 10, None, 1)]     # 每轮多一次快速判断：聊得有没有意思、是不是不满意……


def test_relative_days_reach_the_extractor_as_dates(make_chat):
    fake = FakeLLM("那得好好准备一下")
    make_chat(fake).handle(Turn("明天是我妈生日"))
    assert fake.extract_requests[0]["messages"][1]["content"].endswith("要分析的消息：\n9月23日是我妈生日")


def test_small_talk_and_failures_do_not_touch_memory_or_the_reply(make_chat):
    fake = FakeLLM("嗯嗯", "在的")
    agent = make_chat(fake)
    agent.handle(Turn("好"))
    agent.handle(Turn("ok!"))
    assert fake.extract_requests == []
    for broken in (LLMError("DeepSeek 响应超时"), "这不是 JSON", {"facts": "不是列表"}, {"facts": [{"text": ""}, "x", {"text": "长" * 201}]}):
        fake = FakeLLM("收到", facts=[broken, broken])
        agent = make_chat(fake)
        assert agent.handle(Turn("我对花生过敏")).text == "收到"
        assert agent.memory.current() == []


def test_restating_a_known_fact_does_not_duplicate_it(make_chat):
    fake = FakeLLM("知道啦", "嗯嗯记得", facts=[{"facts": [{"text": "对花生过敏", "category": "health", "quote": "对花生过敏"}]}] * 2)
    agent = make_chat(fake, backend=Scripted(rel="duplicate"))
    agent.handle(Turn("我对花生过敏"))
    agent.handle(Turn("别忘了我对花生过敏"))
    assert [f.text for f in agent.memory.history()] == ["对花生过敏"]


def test_correcting_a_memory_closes_the_old_one(make_chat):
    fake = FakeLLM([("update_memory", {"id": 1, "text": "住在 Newtown"})], "改过来啦", "你住 Newtown 呀")
    agent = make_chat(fake)
    agent.memory.write("住在 Glebe", "location", now=int(NOW.timestamp()))
    reply = agent.handle(Turn("不对，是 Newtown"))
    assert reply.text == "改过来啦\n📝 把原来记的「住在 Glebe」改成了：住在 Newtown"   # 改掉了旧的：这种要让他看到
    old, new = sorted(agent.memory.history(), key=lambda f: f.id)
    assert (old.text, old.superseded_by, new.text, new.category, new.valid_to) == ("住在 Glebe", new.id, "住在 Newtown", "location", None)
    agent.handle(Turn("我住哪来着"))
    assert "#2（地点）住在 Newtown" in fake.context() and "Glebe" not in fake.context().split("[最近对话]")[0]


def test_extraction_is_told_what_was_already_done_this_turn(make_chat):
    """抽取每轮都跑；为了不把刚建的提醒、刚改掉的内容又记回去，它能看到本轮的回执。"""
    fake = FakeLLM([("create_reminder", {"title": "交作业", "time_text": "明天下午三点"})], "好",
                   [("update_memory", {"id": 1, "text": "住在 Newtown"})], "改了")
    agent = make_chat(fake)
    agent.memory.write("住在 Glebe", "location", now=int(NOW.timestamp()))
    agent.handle(Turn("明天下午三点提醒我交作业"))
    agent.handle(Turn("不对，我住 Newtown"))
    first, second = (r["messages"][1]["content"] for r in fake.extract_requests)
    assert "助理本轮已经办的事：\n⏰ 交作业｜明天 9月23日（周三）15:00\n\n要分析的消息：" in first
    assert "助理本轮已经办的事：\n📝 把原来记的「住在 Glebe」改成了：住在 Newtown\n\n要分析的消息：" in second
    fake.script.append("辛苦啦")
    agent.handle(Turn("今天好累啊"))
    assert "助理本轮已经办的事：\n（无）\n\n" in fake.extract_requests[-1]["messages"][1]["content"]


def test_a_new_fact_cannot_overwrite_an_unrelated_memory(make_chat):
    """实测遇到过：用户说"我对花生过敏，另外我女朋友叫小雨"，模型拿"改正记忆"去改了一条看牙医的记录。"""
    fake = FakeLLM([("update_memory", {"id": 1, "text": "对花生过敏"})],
                   [("remember", {"text": "对花生过敏", "category": "health"})], "记下啦",
                   facts=[{"facts": [{"text": "对花生过敏", "category": "health", "quote": "我对花生过敏"},
                                     {"text": "女朋友叫小雨", "category": "relationship", "quote": "我女朋友叫小雨"}]}])
    agent = make_chat(fake, backend=Scripted(rel="unrelated", same=0.03))
    agent.memory.write("妈妈的生日是11月25日", "important_date", now=int(NOW.timestamp()))
    sent = []
    reply = agent.handle(Turn("我对花生过敏，另外我女朋友叫小雨"), on_reply=lambda r: sent.append(r.text))
    assert tool_results(fake)[0] == {"error": "#1 记的是另一件事，不能用这条内容覆盖它",
                                     "hint": "update_memory 只用来改正同一件事。这是一件新的事的话，用 remember 存。"}
    assert reply.text == "记下啦" and sent[1:] == []                     # 新记的两条都不附回执；模型漏掉的那件由后台补上
    assert sorted(f.text for f in agent.memory.current()) == ["女朋友叫小雨", "妈妈的生日是11月25日", "对花生过敏"]
    assert all(f.valid_to is None for f in agent.memory.history())       # 没有任何一条被冲掉


# --- 说到做到：2026-11-16 真机上出过的事 ------------------------------------------------

def test_remember_stores_each_fact_and_the_receipt_says_exactly_what(make_chat):
    fake = FakeLLM([("remember", {"text": "蝴蝶机夹胸用 30kg", "category": "health"}),
                    ("remember", {"text": "绳索后拉用  20kg", "category": "不存在的类别"})], "两条都记下了",
                   [("remember", {"text": "蝴蝶机夹胸用30kg", "category": "health"}), ("remember", {"text": " ", "category": "other"})], "这条本来就有")
    agent = make_chat(fake)
    assert agent.handle(Turn("蝴蝶机夹胸30kg，绳索后拉20kg，记一下")).text == "两条都记下了"     # 办成了就不出声
    assert tool_results(fake) == [{"ok": True, "status": "added", "id": 1, "stored": "蝴蝶机夹胸用 30kg"},
                                  {"ok": True, "status": "added", "id": 2, "stored": "绳索后拉用 20kg"}]
    assert [(f.category, f.text) for f in agent.memory.current()] == [("other", "绳索后拉用 20kg"), ("health", "蝴蝶机夹胸用 30kg")]
    # 换个写法再说一遍：不会存成两条
    assert agent.handle(Turn("夹胸30kg记了吗，再记一次")).text == "这条本来就有"
    assert tool_results(fake)[1] == {"error": "要记的内容是空的或者太长", "hint": "一句简短的陈述，一件事一条"}
    assert len(agent.memory.history()) == 2


def test_saying_noted_without_storing_anything_is_sent_back_and_then_really_done(make_chat):
    """截图里的事：用户报了训练重量，回复说"记下了"，记忆里其实什么都没多。"""
    fake = FakeLLM("好，20kg 记下了～", [("remember", {"text": "绳索后拉用 20kg", "category": "health"})], "记好啦")
    agent = make_chat(fake)
    assert agent.handle(Turn("绳索后拉20kg")).text == "记好啦"
    sent_back = fake.chat_requests[1]["messages"][-2:]
    assert sent_back[0] == {"role": "assistant", "content": "好，20kg 记下了～"}
    assert sent_back[1]["role"] == "user" and sent_back[1]["content"].startswith(
        "[系统检查] 你刚写的这份草稿还没有发给用户，他没看到。草稿会让他以为已经改动了记忆，但这一轮没有任何相应的工具执行成功")
    assert [f.text for f in agent.memory.current()] == ["绳索后拉用 20kg"]


def test_a_claim_that_survives_the_check_is_corrected_by_the_program(make_chat):
    fake = FakeLLM("记下了", "嗯嗯，记下了", "提醒给你设好了", "提醒设好了，放心")
    agent = make_chat(fake)
    assert agent.handle(Turn("绳索后拉20kg")).text == "嗯嗯，记下了\n⚠️ 核对：到这条回复为止，记忆没有任何改动。"
    assert len(fake.chat_requests) == 2                      # 只打回一次，不会没完没了
    assert agent.handle(Turn("明天叫我起床")).text == "提醒设好了，放心\n⚠️ 核对：到这条回复为止，提醒没有任何改动。"
    assert agent.memory.current() == [] and agent.reminders.pending() == []
    # "改了""设好了""删了"哪一类都可能：一条回执都没有时，直说什么都没改
    vague = FakeLLM("改好了", "已经改好了")
    assert make_chat(vague).handle(Turn("把那个改一下")).text == "已经改好了\n⚠️ 核对：到这条回复为止，我实际上什么都没有改动。"
    assert "会让他以为事情已经办好了" in vague.chat_requests[1]["messages"][-1]["content"]


def test_a_receipt_of_the_other_kind_does_not_back_the_claim(make_chat):
    fake = FakeLLM([("remember", {"text": "周五要考 COMP5999", "category": "study_work"})], "记下了，提醒也设好了",
                   [("create_reminder", {"title": "COMP5999 考试", "time_text": "周五下午五点"})], "都好了")
    agent = make_chat(fake)
    reply = agent.handle(Turn("周五下午五点考 COMP5999，提醒我"))
    assert reply.text == "都好了\n⏰ COMP5999 考试｜9月25日（周五）17:00"      # 回复里没说时间，所以提醒那行照附；记忆不出声
    assert [f.text for f in agent.memory.current()] == ["9月25日要考 COMP5999"]   # "周五"存成具体日期
    assert "已经改动了提醒" in fake.chat_requests[2]["messages"][-1]["content"]


def test_mentioning_something_done_earlier_is_not_a_claim(make_chat):
    fake = FakeLLM("这个我之前就记下了呀", "记下了")
    judge = Scripted(rel="unrelated", claim=0.03)
    agent = make_chat(fake, backend=judge)
    assert agent.handle(Turn("我对花生过敏你知道吧")).text == "这个我之前就记下了呀"
    assert len(fake.chat_requests) == 1
    state, questions = judge.calls[0]
    assert state == {"user_latest_message": "我对花生过敏你知道吧", "assistant_reply": "这个我之前就记下了呀"}
    assert list(questions) == ["claim", "accepted"]

    class Down(Scripted):
        def system_one(self, state, questions):
            raise DecisionError("Jev 限流了，稍后再试")

    fake = FakeLLM("记下了", "其实还没记，你再说一遍具体数字？")      # 判断不了时按字面当作声称，宁可多核对一次
    assert make_chat(fake, backend=Down()).handle(Turn("帮我记个东西")).text == "其实还没记，你再说一遍具体数字？"


def test_agreeing_to_remind_without_creating_a_reminder_is_sent_back(make_chat):
    """没说"设好了"，只说"到时候叫你"：用户照样会以为提醒已经有了。"""
    judge = Scripted(rel="unrelated", claim=0.08, accepted=0.98)
    fake = FakeLLM("好嘞，到时候叫你～", [("create_reminder", {"title": "起床", "time_text": "明天早上八点"})], "好嘞，到时候叫你～")
    agent = make_chat(fake, backend=judge)
    assert agent.handle(Turn("明天八点叫我起床")).text == "好嘞，到时候叫你～\n⏰ 起床｜明天 9月23日（周三）08:00"
    assert "已经改动了提醒" in fake.chat_requests[1]["messages"][-1]["content"]
    assert sum("claim" in questions for _, questions in judge.calls) == 1                       # 提醒建成之后就不再核对
    # 说的是已经有的提醒：不算应承新的，不打回
    fake.script.append("在的，明天八点会叫你")
    judge.rules["accepted"] = 0.16
    assert agent.handle(Turn("明天的提醒还在吗")).text == "在的，明天八点会叫你"
    assert len(agent.reminders.pending()) == 1


def test_forgetting_a_memory_the_user_did_not_point_at_is_refused(make_chat):
    """实测遇到过：用户说"不对！是30kg！"，模型去删了一条课程的记忆。"""
    fake = FakeLLM([("forget_memory", {"id": 1})], [("update_memory", {"id": 2, "text": "蝴蝶机夹胸用 30kg"})], "改成 30kg 了")
    judge = Scripted(rel="unrelated", same=0.9, claim=0.9, target=0.05)
    agent = make_chat(fake, backend=judge)
    agent.memory.write("这学期在上 COMP5999", "study_work", now=int(NOW.timestamp()))
    agent.memory.write("蝴蝶机夹胸用 25kg", "health", now=int(NOW.timestamp()))
    judge.calls.clear()
    reply = agent.handle(Turn("不对！是30kg！"))
    assert tool_results(fake, 1)[0] == {"error": "用户这句话看起来不是要忘掉「这学期在上 COMP5999」，没有删",
                                        "hint": "他如果是在纠正，用 update_memory 或 remember；确实要删别的，就先问清楚是哪一条"}
    assert reply.text == "改成 30kg 了\n📝 把原来记的「蝴蝶机夹胸用 25kg」改成了：蝴蝶机夹胸用 30kg"
    assert sorted(f.text for f in agent.memory.current()) == ["蝴蝶机夹胸用 30kg", "这学期在上 COMP5999"]
    state, questions = judge.calls[0]
    assert (state["user_latest_message"], state["item"], list(questions)) == ("不对！是30kg！", "记忆：这学期在上 COMP5999", ["target"])


def test_cancelling_a_reminder_the_user_did_not_point_at_is_refused(make_chat):
    fake = FakeLLM([("cancel_reminder", {"id": 1})], "你是想取消哪一条？")
    agent = make_chat(fake, backend=Scripted(rel="unrelated", target=0.05))
    agent.reminders.add("交作业", NOW + timedelta(days=1), now=0)
    assert agent.handle(Turn("好烦啊不想写了")).text == "你是想取消哪一条？"
    assert tool_results(fake)[0]["error"] == "用户这句话看起来不是要取消「交作业」这条提醒，没有取消"
    assert [r.title for r in agent.reminders.pending()] == ["交作业"]


def test_a_possible_duplicate_keeps_both_and_asks(make_chat):
    """实测遇到过："绳索后拉 20kg"被当成"绳索下拉 22.5kg"的更新，差点把后者冲掉。拿不准是不是同一件事时两条都留，问用户。"""
    fake = FakeLLM([("remember", {"text": "绳索后拉用 20kg", "category": "health"})], "记下了",
                   [("forget_memory", {"id": 1})], "好，只留新的")
    agent = make_chat(fake, backend=Scripted(rel="updates", same=0.63, target=0.9, claim=0.9))
    agent.memory.write("绳索下拉用 22.5kg", "health", now=int(NOW.timestamp()))
    reply = agent.handle(Turn("绳索后拉20kg"))
    assert reply.text == ("记下了\n❓ 「绳索后拉用 20kg」和原来记的「绳索下拉用 22.5kg」是同一件事吗？"
                          "是的话回我「替换」，我就只留新的；不是就不用管，两条都留着。")
    assert tool_results(fake)[0] == {"ok": True, "status": "added", "id": 2, "stored": "绳索后拉用 20kg",
                                     "possible_same_as": [{"id": 1, "text": "绳索下拉用 22.5kg"}],
                                     "note": "两条都留着了。回执已经问了用户它们是不是同一件事，你不要再问一遍，也不要自己删旧的。"}
    assert sorted(f.text for f in agent.memory.current()) == ["绳索下拉用 22.5kg", "绳索后拉用 20kg"]
    assert agent.handle(Turn("替换")).text == "好，只留新的\n📝 已忘掉：绳索下拉用 22.5kg"
    assert [f.text for f in agent.memory.current()] == ["绳索后拉用 20kg"]


def test_a_clear_update_still_replaces_the_old_fact(make_chat):
    fake = FakeLLM([("remember", {"text": "住在 Newtown", "category": "location"})], "记下了")
    agent = make_chat(fake, backend=Scripted(rel="updates", same=0.92, claim=0.9))
    agent.memory.write("住在 Glebe", "location", now=int(NOW.timestamp()))
    assert agent.handle(Turn("我搬到 Newtown 了")).text == "记下了\n📝 把原来记的「住在 Glebe」换成了：住在 Newtown"
    assert tool_results(fake)[0] == {"ok": True, "status": "updated", "id": 2, "stored": "住在 Newtown", "replaced": ["住在 Glebe"],
                                     "note_replaced": "换掉了旧的那条。程序会在你的回复后面附一行写明换了什么，你自然地应一句就行，不用复述。"}
    assert [f.text for f in agent.memory.current()] == ["住在 Newtown"]


def test_changing_only_how_early_keeps_the_time_of_the_event(make_chat):
    """实测遇到过：考试 17:00，用户说"提前一个半小时提醒"，结果考试时间被改成了 15:30。"""
    fake = FakeLLM([("create_reminder", {"title": "COMP5999 考试", "time_text": "周五下午五点，提前一小时"})], "好",
                   [("update_reminder", {"id": 1, "lead_minutes": 90})], "改成提前一个半小时了",
                   [("update_reminder", {"id": 1, "time_text": "提前两小时"})], "行",
                   [("update_reminder", {"id": 1, "lead_minutes": 0})], "到点再叫你")
    agent = make_chat(fake)
    assert agent.handle(Turn("周五下午五点考试，提前一小时提醒")).text == "好\n⏰ COMP5999 考试｜9月25日（周五）17:00｜提前1小时（16:00）提醒"
    assert agent.handle(Turn("提前一个半小时提醒我")).text == "改成提前一个半小时了\n⏰ 已改：COMP5999 考试｜9月25日（周五）17:00｜提前1个半小时（15:30）提醒"
    assert tool_results(fake)[0] == {"ok": True, "id": 1, "title": "COMP5999 考试", "when": "9月25日（周五）17:00",
                                     "will_notify_at": "9月25日（周五）15:30（提前1个半小时）",
                                     "was": {"when": "9月25日（周五）17:00", "will_notify_at": "9月25日（周五）16:00（提前1小时）"}}
    assert agent.handle(Turn("还是提前两小时吧")).text == "行\n⏰ 已改：COMP5999 考试｜9月25日（周五）17:00｜提前2小时（15:00）提醒"
    [r] = agent.reminders.pending()
    assert (r.due, r.lead_minutes, r.notify) == (datetime(2026, 9, 25, 17, 0, tzinfo=TZ), 120, datetime(2026, 9, 25, 15, 0, tzinfo=TZ))
    assert agent.handle(Turn("不用提前了")).text == "到点再叫你\n⏰ 已改：COMP5999 考试｜9月25日（周五）17:00"


def test_a_lead_that_puts_the_alert_in_the_past_is_refused(make_chat):
    fake = FakeLLM([("update_reminder", {"id": 1, "lead_minutes": 60}), ("update_reminder", {"id": 1, "lead_minutes": 99999})], "这个提前量不行")
    agent = make_chat(fake)
    agent.reminders.add("开会", NOW + timedelta(minutes=30), now=0)
    assert agent.handle(Turn("提前一小时提醒我")).text == "这个提前量不行"
    assert [r["error"] for r in tool_results(fake)] == ["按这个提前量，提醒时间 今天 9月22日（周二）09:30 已经过去了", "提前量要在 0 分钟到 7 天之间"]
    assert agent.reminders.pending()[0].lead_minutes == 0


def test_guard_falls_back_to_shared_wording_when_the_judge_is_down(make_chat, conn):
    from alf.decide import DecisionError

    class Down:
        name = "down"

        def system_one(self, state, questions):
            raise DecisionError("Jev 限流了，稍后再试")

    agent = make_chat(FakeLLM(), backend=Down())
    conn.execute("INSERT INTO facts(category, text, valid_from, recorded_at) VALUES ('location', '住在 Glebe', 0, 0), ('other', '周三去看牙医', 0, 0)")
    assert agent.memory.same_note(1, "住在 Newtown") is True
    assert agent.memory.same_note(2, "对花生过敏") is False
    assert agent.memory.same_note(99, "x") is None


def test_forgetting_stops_using_a_fact_but_keeps_the_record(make_chat):
    fake = FakeLLM([("forget_memory", {"id": 1})], "好，忘了")
    agent = make_chat(fake)
    agent.memory.write("前女友叫小雨", "relationship", now=int(NOW.timestamp()))
    assert agent.handle(Turn("把小雨那条忘掉")).text == "好，忘了\n📝 已忘掉：前女友叫小雨"
    assert agent.memory.current() == [] and len(agent.memory.history()) == 1


def test_voice_heard_with_low_confidence_is_marked_unconfirmed(make_chat):
    fake = FakeLLM("好", "嗯", facts=[{"facts": [{"text": "对花生过敏", "category": "health", "quote": "对花生过敏"}]}])
    agent = make_chat(fake)
    agent.handle(Turn("我对花生过敏", source="voice", asr_confidence=0.4))
    assert "[用户刚发来的消息]（语音转写，可能有错字）\n我对花生过敏" in fake.context()
    assert agent.memory.current()[0].needs_confirmation
    agent.handle(Turn("我对什么过敏"))
    assert "#1（健康）对花生过敏（语音听来的，还没确认）" in fake.context()


# --- 上下文 --------------------------------------------------------------------------

def test_history_shows_both_sides_and_marks_proactive_messages(make_chat, conn):
    fake = FakeLLM("哈喽", "交了就好")
    agent = make_chat(fake)
    agent.handle(Turn("嗨"))
    timeline.post(conn, "reminder", "交作业（10:00）", int(NOW.timestamp()))
    conn.execute("INSERT INTO episodes(session_id, role, text, created_at) VALUES ('old', 'user', '两天前说的话', ?)",
                 (int((NOW - timedelta(days=2)).timestamp()),))
    agent.handle(Turn("交了"))
    history = fake.context().split("[最近对话]\n")[1].split("\n\n")[0]
    assert history == "用户：嗨\nAlf：哈喽\nAlf：[提醒] 交作业（10:00）"


def test_digest_numbering_is_shown_for_a_day(make_chat, conn):
    save_state(conn, JobState("digest", cursor=json.dumps({"until": int(NOW.timestamp()) - 3600, "ids": [5, 7]})))
    fake = FakeLLM("好", "好")
    make_chat(fake).handle(Turn("第2条不对"))
    assert ("[最近一次记忆汇总] 一共 2 条：第1条是 #5，第2条是 #7。"
            "用户说「第几条」指的是这里的顺序，不是 # 后面的编号；没有的条数要告诉他，不要猜。") in fake.context()
    make_chat(fake, now=NOW + timedelta(days=2)).handle(Turn("第2条不对"))
    assert "记忆汇总" not in fake.context()


def test_replies_go_back_to_the_channel_the_message_came_from(make_chat, conn):
    make_chat(FakeLLM("在")).handle(Turn("在吗", channel="imessage"))
    rows = conn.execute("SELECT role, channel, delivered_at FROM episodes ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [("user", "imessage", None), ("alf", "imessage", None)]


def test_agent_choice_depends_on_whether_a_generative_model_is_configured(conn, cfg):
    assert isinstance(make_agent(conn, cfg, Scripted(), FakeLLM()), ChatAgent)
    assert isinstance(make_agent(conn, cfg, Scripted()), Agent)


def test_receipt_lines_written_by_the_model_are_replaced_by_the_real_one(make_chat):
    """实测遇到过：模型学着历史里的样子自己写了一行"回执"，而且内容和程序算的不完全一样。"""
    fake = FakeLLM([("create_reminder", {"title": "看牙医", "time_text": "后天上午十点，提前一天"})],
                   "好，看牙医那条设好了，提前一天提醒你。\n\n⏰ 看牙医｜后天 9月24日（周四）10:00｜提前一天")
    reply = make_chat(fake).handle(Turn("后天上午十点去看牙医，提前一天提醒我"))
    assert reply.text == "好，看牙医那条设好了，提前一天提醒你。\n⏰ 看牙医｜后天 9月24日（周四）10:00｜提前1天（9月23日 10:00）提醒"
    assert chat.strip_fake_receipts("⚠️ 明天有雨，带伞\n❓ 这条是不是取代了原来的「x」？\n⚠️ 核对：到这条回复为止，记忆没有任何改动。") == "⚠️ 明天有雨，带伞"
    assert chat.strip_fake_receipts("📝 已忘掉：x\n  🗑 已取消：y") == ""
    assert chat.strip_fake_receipts("今天 ⏰ 这个表情不在行首，留着") == "今天 ⏰ 这个表情不在行首，留着"
