"""记忆分层、回执什么时候出现、好感度（2026-11-17）。

三件事的出发点都是用户的原话：记忆该按类别整组取；确认信息太像机器；聊到有意思的内容好感度加一分、语气跟着变。
"""
import json
from datetime import datetime, timedelta

import pytest

from alf import affinity, chat, followup, jobs, proactive, timeline
from alf.agent import Turn
from alf.decide import Decider, DecisionError
from alf.heartbeat import JobContext
from alf.llm import LLM, LLMError
from alf.memory import Memory
from alf.organize import organize

from .conftest import NOW, TZ, FakeLLM, Scripted

TS = int(NOW.timestamp())


def tool_results(fake, n=-1):
    return [json.loads(m["content"]) for m in fake.chat_requests[n]["messages"] if m["role"] == "tool"]


def memory_of(conn, **rules):
    return Memory(conn, Decider(Scripted(**{"rel": "unrelated", **rules}), conn), 0.6)


# --- 记忆：主题和"哪一项" ---------------------------------------------------------------

def test_same_item_in_a_topic_is_replaced_without_asking_a_model(conn):
    """同一个主题、同一项报了新数值：直接替换。判断替身被设成"会拦下替换"，也不该被问到。"""
    judge = Scripted(rel="unrelated", same=0.05)
    m = Memory(conn, Decider(judge, conn), 0.6)
    old = m.write("上斜哑铃卧推单边 27.5kg", "health", now=TS, topic="健身", slot="上斜哑铃卧推").fact
    judge.calls.clear()
    res = m.write("上斜哑铃卧推单边 30kg", "health", now=TS + 60, topic=" 健身 ", slot="上斜 哑铃卧推")
    assert (res.status, [f.id for f in res.superseded], res.maybe_same) == ("updated", [old.id], [])
    assert judge.calls == []                                     # 一个模型都没问
    assert [f.text for f in m.in_topic("健身")] == ["上斜哑铃卧推单边 30kg"]
    assert m.get(old.id).superseded_by == res.fact.id            # 旧的数值还查得到：以后能看进步


def test_a_different_item_in_the_same_topic_is_never_a_replacement(conn):
    """2026-11-16 的事故："绳索后拉 20kg"被两个模型都判成"绳索下拉 22.5kg"的更新。填了是哪一项，这种事就不会发生。"""
    m = memory_of(conn, rel="updates", same=0.95)                # 判断替身一口咬定"是同一件事的更新"
    m.write("绳索下拉三头 22.5kg", "health", now=TS, topic="健身", slot="绳索下拉")
    res = m.write("绳索后拉 20kg", "health", now=TS + 60, topic="健身", slot="绳索后拉")
    assert (res.status, res.superseded, res.maybe_same) == ("added", [], [])
    assert sorted(f.text for f in m.in_topic("健身")) == ["绳索下拉三头 22.5kg", "绳索后拉 20kg"]


def test_topics_are_listed_reuse_their_spelling_and_are_found_by_name(conn):
    m = memory_of(conn)
    m.write("蝴蝶机夹胸 30kg", "health", now=TS, topic="健身", slot="蝴蝶机夹胸")
    m.write("绳索后拉 20kg", "health", now=TS + 1, topic="健 身", slot="绳索后拉")       # 写法略有不同：还是同一个主题
    m.write("周五交 A2", "study_work", now=TS + 2, topic="COMP5999")
    m.write("对花生过敏", "health", now=TS + 3)
    assert m.topics() == [("COMP5999", 1, TS + 2), ("健身", 2, TS + 1)]
    assert [f.text for f in m.core()] == ["对花生过敏"]                                # 没有主题的才是常驻层
    # "健身"两个字不在任何一条的正文里；按主题名能搜到，是因为主题和项目名一起进了索引
    assert sorted(f.text for f in m.search("健身记录")) == ["绳索后拉 20kg", "蝴蝶机夹胸 30kg"]


def test_a_dated_fact_expires_and_waits_to_be_asked_about(conn):
    m = memory_of(conn)
    exam = m.write("11月20日 COMP5999 考试", "study_work", now=TS, expires_at=TS + 3600).fact
    lasting = m.write("这学期在上 COMP5999", "study_work", now=TS).fact
    assert m.expire(TS + 1800) == 0 and len(m.current()) == 2
    assert m.expire(TS + 3601) == 1
    assert [f.id for f in m.current()] == [lasting.id]                                # 过了那天就不再是现状
    assert [f.id for f in m.just_past(TS + 7200)] == [exam.id]
    assert m.just_past(TS + 3600 + 4 * 86400) == []                                   # 过去好几天了，不用再提
    m.settle(exam.id)
    assert m.just_past(TS + 7200) == [] and m.get(exam.id).aftermath == "done"


def test_a_corrected_fact_stays_filed_where_it_was(conn):
    m = memory_of(conn)
    old = m.write("蝴蝶机夹胸 25kg", "health", now=TS, topic="健身", slot="蝴蝶机夹胸").fact
    _, new = m.revise(old.id, "蝴蝶机夹胸 30kg", now=TS + 5)
    assert (new.topic, new.slot) == ("健身", "蝴蝶机夹胸") and m.search("健身")[0].id == new.id


# --- 对话：三层怎么进上下文 --------------------------------------------------------------

def fill(agent, topic, n, start=0):
    for i in range(n):
        agent.memory.write(f"{topic}第{i}项 {start + i}kg", "health", now=TS + start + i, topic=topic, slot=f"第{i}项")


def test_a_small_memory_is_given_whole_in_layers(make_chat):
    fake = FakeLLM("在")
    agent = make_chat(fake)
    agent.memory.write("对花生过敏", "health", now=TS)
    agent.memory.write("别老问我要不要记", "feedback", now=TS + 1)
    agent.memory.write("蝴蝶机夹胸 30kg", "health", now=TS + 2, topic="健身", slot="蝴蝶机夹胸")
    agent.memory.write("9月20日和 leo 去了蓝山徒步，腿快断了", "moment", now=TS + 3)
    agent.memory.write("9月21日约了牙医", "other", now=TS - 86400, expires_at=TS - 5)      # 日子已经过了
    agent.handle(Turn("在吗"))
    middle = fake.context().split("[你记得的事]\n")[1].split("\n\n[最近对话]")[0]
    assert middle == "\n".join([
        "#1（健康）对花生过敏", "",
        "[他对你的要求]（照做）", "#2 别老问我要不要记", "",
        "[主题]（成组的细节；标着「没列出」的需要时用 recall 查，不要凭印象说）", "健身（1 条）：", "  #3 蝴蝶机夹胸 30kg", "",
        "[刚过去的事]（可以自然地问一句怎么样；他讲了就记成经历）", "#5 9月21日约了牙医", "",
        "[最近的经历]", "#4 9月20日和 leo 去了蓝山徒步，腿快断了", "",
        "[你对他的好感度] 50/100（刚熟悉）"])


def test_with_a_lot_of_detail_only_the_relevant_topic_is_opened(make_chat):
    """细节多了以后：只带目录，和这句话有关的那组才整组展开。"""
    fake = FakeLLM("a", "b", "c", "d")
    agent = make_chat(fake)
    fill(agent, "健身", 20)
    fill(agent, "读书", 20, start=100)
    agent.memory.write("对花生过敏", "health", now=TS)

    def topics_block():
        return fake.context().split("[主题]")[1].split("\n\n")[0]

    agent.handle(Turn("今天天气不错"))                       # 和哪个主题都无关：都只给目录
    assert "健身（20 条，没列出）" in topics_block() and "读书（20 条，没列出）" in topics_block()
    assert "#41（健康）对花生过敏" in fake.context()         # 常驻层照常在

    agent.handle(Turn("我的健身记录"))                       # 说到了主题名：整组展开，另一组不动
    block = topics_block()
    assert "健身（20 条）：" in block and block.count("  #") == 20 and "读书（20 条，没列出）" in block

    agent.handle(Turn("那另一个呢"))                         # 接着问的话没有关键词：上一句提到的主题还开着
    assert "健身（20 条）：" in topics_block()

    agent.handle(Turn("读书第3项是多少来着"))                # 没说主题名，但搜到了这个主题里的条目
    assert "读书（20 条）：" in topics_block()


def test_recall_finds_unlisted_detail_and_what_he_said_long_ago(make_chat, conn):
    fake = FakeLLM("嗯", [("recall", {"query": "读书"})], "读书第3项是 103kg", [("recall", {"query": "蓝山 徒步"})], "你上次说腿快断了")
    agent = make_chat(fake)
    fill(agent, "健身", 20)
    fill(agent, "读书", 20, start=100)
    agent.handle(Turn("上周和 leo 去蓝山徒步，腿快断了"))
    conn.execute("UPDATE episodes SET created_at = created_at - 30 * 86400")          # 一个月前说的，早就不在最近对话里了
    assert agent.handle(Turn("帮我看看那个记录")).text == "读书第3项是 103kg"         # 只读，不附任何回执
    found = tool_results(fake)[0]
    assert found["ok"] and len(found["memories"]) == 20 and {m["topic"] for m in found["memories"]} == {"读书"}
    agent.handle(Turn("我上次去哪玩来着"))
    assert tool_results(fake)[0]["said_before"] == [{"date": "2026年8月23日", "he_said": "上周和 leo 去蓝山徒步，腿快断了"}]


def test_remember_files_detail_under_a_topic_and_dated_things_get_an_end(make_chat):
    fake = FakeLLM([("remember", {"text": "上斜哑铃卧推单边 27.5kg", "category": "health", "topic": "健身", "slot": "上斜哑铃卧推"}),
                    ("remember", {"text": "周五考 COMP5999", "category": "study_work", "until": "周五"}),
                    ("remember", {"text": "别老问我要不要记", "category": "feedback"}),
                    ("remember", {"text": "去年去过日本", "category": "other", "until": "去年某天"})], "好",
                   [("remember", {"text": "上斜哑铃卧推单边 30kg", "category": "health", "topic": "健身", "slot": "上斜哑铃卧推"})], "30 了，可以啊")
    agent = make_chat(fake)
    # 四条里只有"对我的要求"那条附了回执：它以后会一直照做，得让他看到
    assert agent.handle(Turn("……")).text == "好\n📝 记成了你对我的要求：别老问我要不要记（不是这个意思就跟我说）"
    gym, exam, rule, trip = sorted(agent.memory.current(), key=lambda f: f.id)
    assert (gym.topic, gym.slot, gym.expires_at) == ("健身", "上斜哑铃卧推", None)
    assert exam.text == "9月25日考 COMP5999" and exam.expires_at == int(datetime(2026, 9, 25, 23, 59, 59, tzinfo=TZ).timestamp())
    assert (rule.category, rule.topic) == ("feedback", None) and trip.expires_at is None
    assert tool_results(fake)[0]["topic"] == "健身" and tool_results(fake)[3]["note_until"] == "until 里的日期没看懂，这条没有设到期"
    # 同一项报了新数值：换掉旧的，而这一种要让他看到
    assert agent.handle(Turn("上斜现在 30 了")).text == "30 了，可以啊\n📝 把原来记的「上斜哑铃卧推单边 27.5kg」换成了：上斜哑铃卧推单边 30kg"


def test_telling_how_it_went_becomes_a_moment_and_the_question_is_not_asked_again(make_chat):
    fake = FakeLLM("徒步怎么样？", [("remember", {"text": "昨天和 leo 徒步很累但风景好", "category": "moment", "about": 1,
                                                 "topic": "不该有主题", "until": "明天"})], "听着就爽")
    agent = make_chat(fake)
    agent.memory.write("9月21日约了 leo 去徒步", "other", now=TS - 86400, expires_at=TS - 3600)
    agent.handle(Turn("早"))
    assert "[刚过去的事]" in fake.context() and "#1 9月21日约了 leo 去徒步" in fake.context()
    assert agent.handle(Turn("很累但风景好")).text == "听着就爽"
    [moment] = agent.memory.moments()
    assert (moment.text, moment.topic, moment.expires_at) == ("9月21日和 leo 徒步很累但风景好", None, None)   # "昨天"存成日期；经历不归主题、不过期
    fake.script.append("嗯")
    agent.handle(Turn("对了"))
    assert "[刚过去的事]" not in fake.context() and "[最近的经历]\n#2 9月21日和 leo 徒步很累但风景好" in fake.context()


# --- 回执：办成了不出声，需要他留意才说 -------------------------------------------------

@pytest.mark.parametrize("said, receipt", [
    ("好，明天下午三点叫你", ""),                                       # 时间说对了：不用再附
    ("行，明天 15:00 喊你交作业", ""),
    ("好，9月23日下午3点提醒你", ""),
    ("好嘞，到时候叫你", "\n⏰ 交作业｜明天 9月23日（周三）15:00"),       # 没说时间
    ("好，明天下午四点叫你", "\n⏰ 交作业｜明天 9月23日（周三）15:00"),   # 说错了：回执上的才是对的
    ("好，明天叫你", "\n⏰ 交作业｜明天 9月23日（周三）15:00"),           # 没说几点
])
def test_a_reminder_receipt_is_dropped_only_when_the_reply_states_the_right_time(make_chat, said, receipt):
    fake = FakeLLM([("create_reminder", {"title": "交作业", "time_text": "明天下午三点"})], said)
    assert make_chat(fake).handle(Turn("明天下午三点提醒我交作业")).text == said + receipt


def test_reminders_with_a_lead_or_no_stated_hour_always_keep_their_receipt(make_chat):
    fake = FakeLLM([("create_reminder", {"title": "考试", "time_text": "周五下午五点，提前一个半小时"})], "好，周五下午五点考试，我提前叫你",
                   [("create_reminder", {"title": "交房租", "time_text": "明天"})], "好，明天上午九点提醒你",
                   [("update_reminder", {"id": 2, "time_text": "下午四点"})], "改到明天下午四点了")
    agent = make_chat(fake)
    assert agent.handle(Turn("…")).text == "好，周五下午五点考试，我提前叫你\n⏰ 考试｜9月25日（周五）17:00｜提前1个半小时（15:30）提醒"
    assert agent.handle(Turn("明天提醒我交房租")).text == "好，明天上午九点提醒你"     # 钟点是程序定的，但回复里说出来了
    assert agent.handle(Turn("改到下午四点")).text == "改到明天下午四点了"              # 改时间也一样


def test_a_memory_heard_unclearly_by_voice_is_shown(make_chat, cfg):
    fake = FakeLLM([("remember", {"text": "对花生过敏", "category": "health"})], "记了")
    reply = make_chat(fake).handle(Turn("我对花生过敏", source="voice", asr_confidence=0.3))
    assert reply.text == "记了\n📝 记下了：对花生过敏（语音听来的，可能有错字）"


def test_when_the_model_drops_out_everything_done_is_listed(make_chat):
    fake = FakeLLM([("remember", {"text": "对花生过敏", "category": "health"})], LLMError("超时"))
    reply = make_chat(fake).handle(Turn("我对花生过敏"))
    assert reply.text == "我这边刚才卡了一下（超时），不过下面这些已经办好了：\n📝 记下了：对花生过敏"


def test_answering_a_follow_up_is_flagged_so_the_model_does_not_go_checking_the_list(make_chat, conn, cfg):
    """2026-11-17 的真实对话：追问"做了吗"，他回"做了"，模型回了一句"那条其实已经不在待办列表里了"。"""
    fake = FakeLLM("好嘞", "嗯")
    agent = make_chat(fake)
    timeline.post(conn, "followup", "起床了吗？", TS, None)
    agent.handle(Turn("做了"))
    assert "（你上一条是提醒或追问。那条提醒发出后就不在[待办提醒]里了" in fake.context()
    agent.handle(Turn("今天好忙"))                                # 再往后就是普通聊天了
    assert "你上一条是提醒或追问" not in fake.context()


def test_the_sent_back_draft_is_described_as_unseen(make_chat):
    """被打回的草稿用户没看到。以前模型不知道这一点，重写时会说"说错了，我其实没记下啥"。"""
    assert "还没有发给用户，他没看到" in chat.SELF_CHECK and "不要说「说错了」" in chat.SELF_CHECK
    assert "那份草稿用户没有看到" in chat.BASE_PROMPT


# --- 好感度 -----------------------------------------------------------------------------

def test_an_interesting_chat_adds_one_point_and_a_whole_evening_does_not_max_it_out(conn, cfg):
    judge = Scripted(rel="unrelated", fun=lambda q, state: 0.9 if "徒步" in state["user_latest_message"] else 0.05)
    fake = FakeLLM("哈哈", "厉害", "好", "又去啦", "行")
    clock = [NOW]
    agent = chat.ChatAgent(conn, LLM(fake, conn), Decider(judge, conn), cfg, clock=lambda: clock[0])   # 时钟要能往前拨
    agent.handle(Turn("今天去蓝山徒步了，腿快断了"))
    assert affinity.score(conn) == 51
    clock[0] += timedelta(minutes=5)
    agent.handle(Turn("徒步的时候 leo 还摔了一跤"))            # 同一段对话：一小时内不再加
    assert affinity.score(conn) == 51
    clock[0] += timedelta(minutes=5)
    agent.handle(Turn("提醒我明天九点交作业呀"))                # 指令：不加
    clock[0] += timedelta(hours=2)
    agent.handle(Turn("周末又去徒步了"))
    assert affinity.score(conn) == 52
    assert [r["note"] for r in affinity.recent(conn)] == ["周末又去徒步了", "今天去蓝山徒步了，腿快断了"]
    agent.handle(Turn("ok"))                                    # 太短的话连问都不问
    asked = [s["user_latest_message"] for s, q in judge.calls if "fun" in q]
    # 冷却期间照样会问（聊得开心要留一点余温），只是不加分；"ok"不问
    assert asked == ["今天去蓝山徒步了，腿快断了", "徒步的时候 leo 还摔了一跤", "提醒我明天九点交作业呀", "周末又去徒步了"]
    assert "[你对他的好感度] 52/100（刚熟悉）" in fake.context()


def test_points_stop_at_a_hundred_and_a_judge_outage_changes_nothing(conn):
    class Down:
        name = "down"

        def system_one(self, state, questions):
            raise DecisionError("unreachable")

    assert affinity.consider(conn, Decider(Down(), conn), "今天去蓝山徒步了", "", 1, TS) is False and affinity.score(conn) == 50
    conn.execute("INSERT INTO episodes(id, session_id, role, text, created_at) VALUES (1, 's', 'user', 'x', 0)")
    with conn:
        for i in range(60):
            conn.execute("INSERT INTO episodes(session_id, role, text, created_at) VALUES ('s', 'user', 'x', 0)")
            conn.execute("INSERT INTO affinity(episode_id, note, created_at) VALUES (?, 'x', ?)", (i + 2, i))
    assert affinity.score(conn) == 100                          # 记了 60 次也只到 100
    happy = Decider(Scripted(fun=0.95), conn)
    assert affinity.consider(conn, happy, "今天去蓝山徒步了", "", 1, TS + 10 * 86400) is False


@pytest.mark.parametrize("points, name", [(50, "刚熟悉"), (59, "刚熟悉"), (60, "朋友"), (74, "朋友"), (75, "好朋友"), (89, "好朋友"),
                                          (90, "死党"), (100, "死党")])
def test_tone_follows_affinity_in_four_steps(points, name):
    tier, tone = affinity.tier(points)
    assert tier == name
    prompt = chat.system_prompt({}, points)
    assert prompt.startswith(chat.BASE_PROMPT) and f"- {tone}\n" in prompt     # 前面的大段不变，只有末尾的说话方式跟着变
    # 他选了别的说话风格时，好感度不去改它
    assert tone not in chat.system_prompt({"style.preset": "brief"}, points)
    assert tone in proactive.think_prompt({}, proactive.LEVELS["active"], points)   # 主动找他说话时也是这个语气


# --- 整理旧记忆 -------------------------------------------------------------------------

def test_organizing_files_old_facts_without_touching_what_they_say(conn, cfg):
    m = memory_of(conn)
    texts = ["蝴蝶机夹胸 30kg", "绳索后拉 20kg", "9月26日飞回中国", "妈妈的生日是11月25日", "对花生过敏", "去年 3 月去过日本", "养了一只猫"]
    ids = [m.write(t, "important_date" if "生日" in t else "other", now=TS + i).fact.id for i, t in enumerate(texts)]
    fake = FakeLLM(facts=[{"facts": [
        {"id": ids[0], "topic": "健身", "slot": "蝴蝶机夹胸", "until": ""},
        {"id": ids[1], "topic": "健身", "slot": "绳索后拉"},
        {"id": ids[2], "topic": "", "until": "2026-09-26"},
        {"id": ids[3], "until": "2026-11-25"},                   # 生日每年都有：不给它设到期
        {"id": ids[4], "topic": "", "slot": "没有主题就不该有项", "until": "不是日期"},
        {"id": ids[5], "until": "2025-03-01"},                   # 已经过去的日子：不动
        {"id": ids[6], "topic": "宠物"},                          # 只有一条的"主题"：不成立，留在常驻层
        {"id": 999, "topic": "健身"}, "乱七八糟"]}])
    changes = organize(conn, LLM(fake, conn), m.decider, cfg, NOW)
    assert changes == [f"#{ids[0]} 蝴蝶机夹胸 30kg：归到「健身」（蝴蝶机夹胸）", f"#{ids[1]} 绳索后拉 20kg：归到「健身」（绳索后拉）",
                       f"#{ids[2]} 9月26日飞回中国：到 9月26日 为止"]
    assert sorted(f.text for f in m.current()) == sorted(texts)                         # 一条没少，一个字没改
    assert m.topics() == [("健身", 2, TS + 1)] and len(m.core()) == 5
    assert m.get(ids[2]).expires_at == int(datetime(2026, 9, 26, 23, 59, 59, tzinfo=TZ).timestamp())
    assert organize(conn, LLM(FakeLLM(facts=[{"facts": []}]), conn), m.decider, cfg, NOW) == []   # 再跑一遍：没有可改的


# --- 追问的措辞 -------------------------------------------------------------------------

@pytest.mark.parametrize("said, expected", [
    ("起床了吗？", "起床了吗？"),
    ("“作业交了没”", "作业交了没"),
    ("好的！我来问问他：\n起床了吗？", "「叫我起床」做了吗？"),            # 多说了话：不用
    ("你今天早上有没有按时起床呀我有点担心你是不是又睡过头了呢？", "「叫我起床」做了吗？"),   # 太长
    ("起床", "「叫我起床」做了吗？"),                                    # 不像一句问话
    (LLMError("down"), "「叫我起床」做了吗？"),
])
def test_the_follow_up_question_is_worded_by_the_model_but_falls_back_to_a_fixed_sentence(conn, cfg, said, expected):
    ctx = JobContext(conn, cfg, NOW, Decider(Scripted(), conn), None, llm=LLM(FakeLLM(said), conn))
    assert followup.wording(ctx, "叫我起床") == expected
    assert followup.wording(JobContext(conn, cfg, NOW, Decider(Scripted(), conn), None), "叫我起床") == "「叫我起床」做了吗？"


def test_the_nightly_digest_is_where_quietly_stored_facts_get_checked(conn, cfg, monkeypatch):
    """记下新事当场不出声，所以晚上这一条得把今天记的都列出来。"""
    monkeypatch.setattr(jobs, "notify_desktop", lambda *a: None)
    m = memory_of(conn)
    m.write("蝴蝶机夹胸 30kg", "health", now=TS + 60, topic="健身", slot="蝴蝶机夹胸")
    m.write("9月21日和 leo 去了蓝山", "moment", now=TS + 120)
    ctx = JobContext(conn, cfg, NOW.replace(hour=21, minute=30), Decider(Scripted(), conn), None)
    jobs.digest_job(ctx)
    assert timeline.fetch(conn)[-1]["text"] == "今天记了这些，有不对的跟我说：\n1. 蝴蝶机夹胸 30kg\n2. 9月21日和 leo 去了蓝山"


# --- 提醒的时间得有出处 -----------------------------------------------------------------

def refuses(judge_says):
    """判断替身：程序按字面对不上时，Jev 认为"这个时间用户说过"的把握。"""
    return Scripted(rel="unrelated", claim=0.05, accepted=0.05, said=judge_says)


def test_a_reminder_at_a_time_nobody_said_is_not_created(make_chat):
    """真实回放里的事（2026-11-17）：他说"今天看完➕做题"，模型自己建了一条"今天下午五点"的提醒。"""
    fake = FakeLLM("好，周五下午五点考试", [("create_reminder", {"title": "看讲义做题", "time_text": "今天下午五点"})], "今天看完再做题，行")
    judge = refuses(0.28)
    agent = make_chat(fake, backend=judge)
    agent.handle(Turn("周五下午五点考COMP5999"))
    assert agent.handle(Turn("今天看完➕做题")).text == "今天看完再做题，行"
    assert agent.reminders.pending() == []
    assert tool_results(fake)[0] == {
        "error": "今天 9月22日（周二）17:00 这个时间用户没有说过，也没有答应过，所以没有建提醒",
        "hint": "不要替他定时间。他只是在说自己的打算，就正常聊；想帮他，就问一句要不要提醒、几点，等他回答了再建"}
    state = next(s for s, q in judge.calls if "said" in q)
    assert state["reminder"] == {"what": "看讲义做题", "when": "今天 9月22日（周二）17:00"} and state["now"] == "今天 9月22日（周二）10:00"


@pytest.mark.parametrize("before, said, time_text", [
    (None, "明早8点叫我起床", "明天早上八点"),                                 # 他自己说的，写法不同也对得上
    (None, "半小时后叫我", "半小时后"),
    ("要不要今晚七点提醒你复习？", "好", "今晚七点"),                          # 他答应了 Alf 提的时间
    ("海报上是周五晚上七点，要建个提醒吗？", "好，建一个", "周五晚上七点"),
    ("要不要今晚七点提醒你复习？", "可以设", "今天晚上七点"),
])
def test_a_time_he_said_or_agreed_to_needs_no_second_opinion(make_chat, conn, before, said, time_text):
    fake = FakeLLM([("create_reminder", {"title": "事情", "time_text": time_text})], "好")
    judge = refuses(0.01)                       # Jev 就算说"不是"也没关系：程序按字面已经对上了，根本不会去问
    agent = make_chat(fake, backend=judge)
    if before:
        timeline.post(conn, "chat", before, TS - 60, None)
    agent.handle(Turn(said))
    assert len(agent.reminders.pending()) == 1 and not any("said" in q for _, q in judge.calls)


def test_alfs_own_earlier_words_are_not_a_source_unless_he_agreed(make_chat, conn):
    """Alf 上一句里提到过 17:00，不等于他要一个 17:00 的提醒。"""
    fake = FakeLLM([("create_reminder", {"title": "健身", "time_text": "今天下午五点"})], "去吧")
    agent = make_chat(fake, backend=refuses(0.07))
    timeline.post(conn, "chat", "周五 17:00 考试，记住了", TS - 60, None)
    agent.handle(Turn("我准备去健身了"))
    assert agent.reminders.pending() == []


def test_a_time_the_program_cannot_read_is_settled_by_the_judge_and_an_outage_lets_it_through(make_chat):
    fake = FakeLLM([("create_reminder", {"title": "submit report", "time_text": "明天下午三点"})], "ok, 3pm tomorrow")
    agent = make_chat(fake, backend=refuses(0.98))                             # "tmr 3pm" 程序读不懂，Jev 认得
    assert agent.handle(Turn("tmr 3pm remind me to submit the report")).text == "ok, 3pm tomorrow\n⏰ submit report｜明天 9月23日（周三）15:00"

    class Down:
        name = "down"

        def system_one(self, state, questions):
            raise DecisionError("unreachable")

    fake = FakeLLM([("create_reminder", {"title": "submit report", "time_text": "明天下午三点"})], "ok")
    agent = make_chat(fake, backend=Down())
    agent.handle(Turn("tmr 3pm remind me to submit the report"))
    assert len(agent.reminders.pending()) == 2                                 # 又建成一条。判断不了时放行：回执上写着时间，他看得到


def test_times_from_a_picture_or_the_calendar_count_as_sources(make_chat, conn, tmp_path):
    picture = tmp_path / "poster.jpg"
    picture.write_bytes(b"\xff\xd8\xff\xe0 not a real picture")
    fake = FakeLLM([("create_reminder", {"title": "迎新晚会", "time_text": "周五晚上七点"})], "好")
    agent = make_chat(fake, backend=refuses(0.01))
    agent.handle(Turn("帮我建个提醒", images=[picture]))                        # 时间在图里，程序看不到：不查
    assert len(agent.reminders.pending()) == 1
    start = int((NOW + timedelta(days=1)).replace(hour=14, minute=0).timestamp())
    conn.execute("INSERT INTO calendar_events(key, event_id, title, start_at, end_at, all_day) VALUES ('k', 'e', '组会', ?, ?, 0)", (start, start + 3600))
    fake.script += [[("create_reminder", {"title": "组会", "time_text": "明天下午两点"})], "好"]
    agent.handle(Turn("日历上那个组会也提醒我一下"))
    assert len(agent.reminders.pending()) == 2


def test_something_that_happened_mentioned_in_passing_is_filed_as_a_moment_not_a_standing_fact(make_chat):
    """实测（2026-11-17）：他说"今天去蓝山徒步了"，后台把它当成普通事实放进了常驻层——那样它会永远跟着每一轮对话。"""
    fake = FakeLLM("蓝山确实值得", facts=[{"facts": [{"text": "9月22日去蓝山徒步了，腿快断了", "category": "moment", "topic": "徒步", "until": "2026-12-01",
                                                     "quote": "去蓝山徒步了，腿快断了"}]}])
    agent = make_chat(fake)
    assert agent.handle(Turn("今天去蓝山徒步了，腿快断了")).text == "蓝山确实值得"
    [moment] = agent.memory.moments()
    assert (moment.text, moment.topic, moment.expires_at) == ("9月22日去蓝山徒步了，腿快断了", None, None)
    assert agent.memory.core() == []


# --- 语音：不复述原话，但办了什么要看得见（2026-11-17 用户实测后定的） --------------------

def test_a_voice_message_is_not_echoed_back_but_what_it_stored_is_shown(make_chat):
    fake = FakeLLM("喂，听得见", [("remember", {"text": "上斜哑铃卧推单边 30kg", "category": "health", "topic": "健身", "slot": "上斜哑铃卧推"})], "30 了，可以啊")
    agent = make_chat(fake)
    assert agent.handle(Turn("喂喂喂,能听见我说话吗?", source="voice", asr_confidence=0.73, channel="imessage")).text == "喂，听得见"
    # 转写他看不到，所以语音记下的事要附一行：听错了从这里能看出来
    reply = agent.handle(Turn("上斜哑铃卧推现在是单边30公斤。", source="voice", asr_confidence=0.87, channel="imessage"))
    assert reply.text == "30 了，可以啊\n📝 记下了：上斜哑铃卧推单边 30kg"
    assert "他看不到转写的结果" in chat.BASE_PROMPT


def test_standing_requests_are_only_taken_in_conversation_never_by_the_background_pass(make_chat):
    """真实发生过两次（2026-11-17）：他用语音问"每句话都要转成文字发给我吗"，后台悄悄把它存成了一条长期要求；
    聊天时已经拒绝的"以后回邮件直接发"，后台又把它存成了要求。所以要求只在对话里当场记，记了一定给他看。"""
    fake = FakeLLM("嗯嗯", [("remember", {"text": "少发表情", "category": "feedback"})], "好",
                   facts=[{"facts": [{"text": "希望助理把他说的每句话都转成文字发给他", "category": "feedback",
                                     "quote": "你都要把他翻译成文字再发给我"}]}])
    agent = make_chat(fake)
    sent = []
    agent.handle(Turn("我每说一句话,你都要把他翻译成文字再发给我嘛。", source="voice", asr_confidence=0.7), on_reply=lambda r: sent.append(r.text))
    assert sent == ["嗯嗯"] and agent.memory.current() == []
    assert "feedback" not in chat.EXTRACT_PROMPT and "这些由对话当场处理" in chat.EXTRACT_PROMPT
    assert agent.handle(Turn("以后少发表情")).text == "好\n📝 记成了你对我的要求：少发表情（不是这个意思就跟我说）"
