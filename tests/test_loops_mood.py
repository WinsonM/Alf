"""2026-11-17 参考 Anima Brain 加的四样：记忆要有原话出处、惦记的事、每晚把对话归纳成经历、短期情绪。"""
import json

import pytest

from alf import affinity, chat, jobs, mood, proactive, timeline
from alf.agent import Turn
from alf.decide import Decider
from alf.heartbeat import JobContext
from alf.llm import LLM
from alf.memory import Memory, loop_priority, was_said

from .conftest import NOW, FakeLLM, Scripted

TS = int(NOW.timestamp())
DAY = 86400


def memory_of(conn, **rules):
    return Memory(conn, Decider(Scripted(**{"rel": "unrelated", **rules}), conn), 0.6)


def system_of(fake, n=-1):
    return fake.chat_requests[n]["messages"][0]["content"]


# --- 记忆要有出处 -----------------------------------------------------------------------

@pytest.mark.parametrize("quote, ok", [
    ("对花生过敏", True),
    ("我 对花生，过敏", True),                 # 空格、标点不同不算改写
    ("女朋友叫小雨", True),
    ("对海鲜过敏", False),                     # 他没说过
    ("过", False),                             # 一个字不算出处
    ("", False), (None, False),
])
def test_a_quote_counts_only_if_he_really_said_it(quote, ok):
    assert was_said(quote, "我对花生过敏，另外我女朋友叫小雨") is ok


def test_facts_learned_in_the_background_need_a_quote_from_what_he_said(make_chat):
    """后台记的事没人当场把关。指不出出自哪几个字的，多半是模型推断的而不是他说的——不记。"""
    fake = FakeLLM("哇", facts=[{"facts": [
        {"text": "对花生过敏", "category": "health", "quote": "我对花生过敏"},
        {"text": "不能吃坚果类零食", "category": "health", "quote": "不能吃坚果"},      # 推断出来的，原话里没有
        {"text": "女朋友叫小雨", "category": "relationship"},                         # 没给出处
        {"text": "9月23日要去看牙医", "category": "other", "quote": "9月23日要去看牙医"}]}])   # 出处在"明天"换成日期之后的那句话里
    agent = make_chat(fake)
    agent.handle(Turn("我对花生过敏，另外我女朋友叫小雨，明天要去看牙医"))
    stored = {f.text: f.quote for f in agent.memory.current()}
    assert stored == {"对花生过敏": "我对花生过敏", "9月23日要去看牙医": "9月23日要去看牙医"}


def test_remember_keeps_a_quote_only_when_it_checks_out(make_chat):
    fake = FakeLLM([("remember", {"text": "对花生过敏", "category": "health", "quote": "我对花生过敏"}),
                    ("remember", {"text": "女朋友叫小雨", "category": "relationship", "quote": "他的女友名叫小雨"})], "好")
    agent = make_chat(fake)
    agent.handle(Turn("我对花生过敏，另外我女朋友叫小雨"))
    # 引用对不上，只是不存那段引用；这条记忆是他当面让记的，照存
    assert {f.text: f.quote for f in agent.memory.current()} == {"对花生过敏": "我对花生过敏", "女朋友叫小雨": None}


# --- 惦记的事 ---------------------------------------------------------------------------

def test_open_loops_are_ranked_by_how_much_they_matter_and_how_recently_they_came_up(conn):
    m = memory_of(conn)
    exam = m.write("9月25日考 COMP5999", "study_work", now=TS, expires_at=TS + 3 * DAY, importance=8).fact
    worry = m.write("有点焦虑，不知道回国之后做什么", "other", now=TS - 10 * DAY, follow_up=True, importance=6).fact
    faded = m.write("想换个健身计划", "other", now=TS - 60 * DAY, follow_up=True, importance=6).fact
    m.write("对花生过敏", "health", now=TS)                                    # 不是没了结的事
    m.write("9月20日去了蓝山", "moment", now=TS, follow_up=True)                # 经历不会被惦记
    assert [(f.id, score) for f, score in m.open_loops(TS)] == [(exam.id, 8.5), (worry.id, 3.73)]
    assert round(loop_priority(m.get(faded.id), TS), 2) == 0.34               # 两个月没再提：自己淡出，没人去关它
    # 说明里举的数字就是这么算出来的
    assert (round(8 * 1 + 2 / (1 + 3), 2), round(6 * 2.718281828 ** (-10 / 21), 2)) == (8.5, 3.73)


def test_a_loop_goes_quiet_once_it_came_up_and_stops_after_two_unanswered_asks(conn):
    m = memory_of(conn)
    loop = m.write("有点焦虑，不知道回国之后做什么", "other", now=TS - DAY, follow_up=True, importance=7).fact
    m.touch(loop.id, TS, by_user=False)                                        # Alf 提了一次，他没接话
    assert m.open_loops(TS + 3600) == []                                       # 今天提过了，不再提
    assert [f.id for f, _ in m.open_loops(TS + DAY)] == [loop.id]              # 明天还可以再问一次
    m.touch(loop.id, TS + DAY, by_user=False)
    assert m.open_loops(TS + 3 * DAY) == [] and m.get(loop.id).asks == 2       # 问了两次都没接话：不再问
    assert [f.id for f in m.open_loops(TS + 3 * DAY, quiet=False)] == [loop.id]   # 但还留意着：他要是回答了，得认得出来
    m.touch(loop.id, TS + 4 * DAY, by_user=True)                               # 他自己又说起来了
    assert m.get(loop.id).asks == 0 and [f.id for f, _ in m.open_loops(TS + 6 * DAY)] == [loop.id]


def loop_judge(spoke=(), raised=(), **more):
    """判断替身：问题里的那件事包含某个词，就认为"聊到了"。"""
    def about(words):
        return lambda q, state: 0.9 if any(w in q["instructions"] for w in words) else 0.03
    return Scripted(rel="unrelated", same=0.9, target=0.9, claim=0.05, said=0.9, spoke=about(spoke), raised=about(raised), **more)


def test_the_model_sees_what_is_unsettled_and_the_program_notes_whether_it_came_up(make_chat):
    fake = FakeLLM("在", "考试准备得怎么样了？", "嗯嗯", "那就好")
    agent = make_chat(fake, backend=loop_judge(raised=["COMP5999"]))
    exam = agent.memory.write("9月25日考 COMP5999", "study_work", now=TS, expires_at=TS + 3 * DAY, importance=8).fact
    worry = agent.memory.write("有点焦虑，不知道回国之后做什么", "other", now=TS - 5 * DAY, follow_up=True, importance=6).fact
    agent.handle(Turn("在吗"))
    block = fake.context().split("[惦记的事]")[1].split("\n\n")[0]
    assert block == ("（还没了结，按要紧程度排；合适的时候自然地问一句，一次最多一件）\n"
                     f"#{exam.id} 9月25日考 COMP5999（还有 3 天）\n#{worry.id} 有点焦虑，不知道回国之后做什么（5 天前提的）")
    assert "9月25日考 COMP5999" in fake.context().split("[你记得的事]")[1].split("\n\n")[0]      # 它同时还是一条记忆
    # 那一轮的回复"在"没提考试——替身却按"事情里有 COMP5999"判成提了：记了一次，今天不再摆出来
    assert agent.memory.get(exam.id).asks == 1 and agent.memory.get(worry.id).asks == 0
    agent.handle(Turn("好"))
    assert f"#{exam.id} 9月25日考" not in fake.context().split("[惦记的事]")[1] and f"#{worry.id}" in fake.context().split("[惦记的事]")[1]


def test_when_he_brings_it_up_himself_the_unanswered_count_starts_over(make_chat):
    fake = FakeLLM("别慌，还有三天")
    agent = make_chat(fake, backend=loop_judge(spoke=["COMP5999"], raised=["COMP5999"]))
    exam = agent.memory.write("9月25日考 COMP5999", "study_work", now=TS - DAY, expires_at=TS + 3 * DAY).fact
    agent.memory.touch(exam.id, TS - DAY, by_user=False)
    agent.handle(Turn("5999 那门我感觉要挂"))
    assert (agent.memory.get(exam.id).asks, agent.memory.get(exam.id).touched_at) == (0, TS)      # 他说起的优先于"Alf 提起的"


def test_telling_the_outcome_closes_the_loop_and_new_loops_can_be_opened(make_chat):
    fake = FakeLLM([("remember", {"text": "回国先休息一个月再找工作", "category": "moment", "about": 1}),
                    ("remember", {"text": "在等 Atlassian 的面试结果", "category": "study_work", "follow_up": True, "importance": 9,
                                  "quote": "在等 Atlassian 的结果"})], "挺好的，先缓一缓")
    agent = make_chat(fake)
    agent.memory.write("有点焦虑，不知道回国之后做什么", "other", now=TS - 5 * DAY, follow_up=True, importance=6)
    agent.handle(Turn("想好了，回国先休息一个月再找工作。另外我在等 Atlassian 的结果"))
    [waiting] = [f for f, _ in agent.memory.open_loops(TS + DAY)]
    assert (waiting.text, waiting.importance, waiting.quote) == ("在等 Atlassian 的面试结果", 9, "在等 Atlassian 的结果")
    assert agent.memory.get(1).follow_up is False                              # 原来那件了结了


def test_reaching_out_about_a_loop_counts_as_one_ask(conn, cfg):
    from dataclasses import replace

    from alf.config import LLMConfig

    live = replace(cfg, llm=LLMConfig(api_key="sk-test"))
    m = memory_of(conn)
    exam = m.write("9月25日考 COMP5999", "study_work", now=TS - DAY, expires_at=TS + 3 * DAY, importance=8).fact
    fake = FakeLLM(facts=[{"send": True, "kind": "follow_up", "message": "周五的 5999 准备得怎么样了？", "why": "还有三天", "about": exam.id}])
    ctx = JobContext(conn, live, NOW.replace(hour=15), Decider(Scripted(welcome=0.9), conn), None, llm=LLM(fake, conn),
                     prefs={"proactive.enabled": True, "proactive.level": "chatty"})
    assert proactive.proactive_job(ctx) == "主动说了一句（跟进）"
    situation = fake.requests[0]["messages"][1]["content"]
    assert f"[惦记的事]（还没了结，按要紧程度排）\n#{exam.id} 9月25日考 COMP5999" in situation
    assert (m.get(exam.id).asks, m.get(exam.id).touched_at) == (1, int(NOW.replace(hour=15).timestamp()))


# --- 每晚把对话归纳成经历 ---------------------------------------------------------------

def say(conn, text, minutes):
    return timeline_user(conn, text, TS + minutes * 60)


def timeline_user(conn, text, at):
    with conn:
        return conn.execute("INSERT INTO episodes(session_id, role, text, source, created_at) VALUES ('s', 'user', ?, 'text', ?)",
                            (text, at)).lastrowid


def evening(conn, cfg, fake, backend=None, **prefs):
    return JobContext(conn, cfg, NOW.replace(hour=21, minute=20), Decider(backend or Scripted(rel="unrelated"), conn), None,
                      llm=LLM(fake, conn), prefs=prefs)


def test_the_days_conversation_is_turned_into_moments_that_cite_what_he_said(conn, cfg):
    first = say(conn, "今天去蓝山徒步了，风景绝了", 60)
    say(conn, "ok", 61)                                        # 太短，不给模型看
    say(conn, "就是腿快断了", 62)
    say(conn, "提醒我明天九点交作业", 90)
    fake = FakeLLM(facts=[{"moments": [
        {"text": "今天去蓝山徒步，风景很好，腿快断了", "from": [1, 2]},
        {"text": "作业写得很顺利", "from": []},                  # 说不出出自哪句话
        {"text": "和 leo 吃了饭", "from": [9]},                  # 编号不存在
        {"text": "x" * 201, "from": [1]}, "乱七八糟"]}])
    ctx = evening(conn, cfg, fake)
    assert jobs.moments_job(ctx) == "归纳了 1 条经历"
    [moment] = memory_of(conn).moments()
    assert (moment.text, moment.quote) == ("9月22日去蓝山徒步，风景很好，腿快断了", "今天去蓝山徒步了，风景绝了")   # "今天"存成日期；出处是他的原话
    assert conn.execute("SELECT source_episode FROM facts WHERE id = ?", (moment.id,)).fetchone()[0] == first
    listing = fake.requests[0]["messages"][1]["content"]
    assert "1. [11:00] 他：今天去蓝山徒步了，风景绝了\n2. [11:02] 他：就是腿快断了\n3. [11:30] 他：提醒我明天九点交作业" in listing and "ok" not in listing
    assert json.loads(ctx.cursor) == {"until": int(NOW.replace(hour=21, minute=20).timestamp())}


def test_quiet_days_and_things_already_recorded_cost_nothing(conn, cfg):
    say(conn, "今天去蓝山徒步了", 60)
    say(conn, "腿快断了", 61)
    fake = FakeLLM(facts=[{"moments": [{"text": "去了蓝山", "from": [1]}]}])
    assert jobs.moments_job(evening(conn, cfg, fake)) is None and fake.requests == []      # 今天只说了两句：不去归纳
    say(conn, "提醒我明天九点交作业", 62)
    assert jobs.moments_job(evening(conn, cfg, fake, **{"memory.auto": False})) is None and fake.requests == []   # 关了自动记忆
    memory_of(conn).write("9月22日去蓝山徒步了", "moment", now=TS)
    assert jobs.moments_job(evening(conn, cfg, fake, backend=Scripted(rel="duplicate"))) is None    # 聊天时已经记过：不重复
    assert "已经记过的经历（包括今天聊天时记下的事，这些不要再写）：\n9月22日去蓝山徒步了" in fake.requests[0]["messages"][1]["content"]
    assert len(memory_of(conn).moments()) == 1


# --- 短期情绪 ---------------------------------------------------------------------------

def test_warmth_fades_in_hours_and_a_grudge_in_days(conn):
    assert mood.tone(conn, TS) is None and mood.view(conn, TS)["state"] == "even"
    mood.cheer(conn, TS)
    assert mood.tone(conn, TS) == mood.RELAXED and mood.tone(conn, TS + 11 * 3600) == mood.RELAXED
    assert mood.tone(conn, TS + 13 * 3600) is None                             # 余温过半天就散了
    mood.strain(conn, TS, "你怎么又没记住")
    assert mood.tone(conn, TS) == mood.CAREFUL                                 # 有疙瘩的时候，疙瘩说了算
    assert mood.tone(conn, TS + 71 * 3600) == mood.CAREFUL and mood.tone(conn, TS + 73 * 3600) is None   # 三天
    assert mood.view(conn, TS + 3600) == {"state": "careful", "label": "这两天收着点", "warm": 0.94, "tension": 0.99, "reason": "你怎么又没记住"}
    for _ in range(5):
        mood.strain(conn, TS, "x")
    assert mood.current(conn, TS)[1] == 2.0                                    # 再怎么累加也到此为止
    mood.ease(conn, TS)
    assert mood.current(conn, TS)[1] == 1.0                                    # 👍 消掉一半


def test_tone_follows_the_moment_on_top_of_affinity(make_chat, conn):
    judge = Scripted(rel="unrelated", claim=0.05, fun=lambda q, s: 0.9 if "徒步" in s["user_latest_message"] else 0.05,
                     annoyed=lambda q, s: 0.9 if "又没记住" in s["user_latest_message"] else 0.03)
    fake = FakeLLM("哈哈", "嗯", "抱歉", "好的", "行")
    agent = make_chat(fake, backend=judge)
    agent.handle(Turn("今天去蓝山徒步了，腿快断了"))
    assert mood.RELAXED not in system_of(fake)                                 # 这一轮开始时还没有余温
    agent.handle(Turn("周末还想再去"))
    assert system_of(fake).endswith(f"- {affinity.tier(51)[1]}\n- {mood.RELAXED}\n- 可以偶尔用表情符号。")
    agent.handle(Turn("你怎么又没记住我说的"))
    agent.handle(Turn("算了你继续"))
    assert f"- {mood.CAREFUL}\n" in system_of(fake) and mood.RELAXED not in system_of(fake)
    assert system_of(fake).startswith(chat.BASE_PROMPT)                        # 前面的大段不变，只有末尾跟着变
    assert affinity.score(conn) == 51                                          # 好感度的分数不受影响：只升不降
    assert mood.view(conn, TS)["reason"] == "你怎么又没记住我说的"
    # 他选了别的说话风格时，情绪不去改它
    assert mood.CAREFUL not in chat.system_prompt({"style.preset": "brief"}, 51, mood.CAREFUL)
    assert mood.CAREFUL in proactive.think_prompt({}, proactive.LEVELS["active"], 51, mood.CAREFUL)


def test_useless_leaves_a_grudge_once_and_useful_eases_it(make_chat, conn):
    judge = Scripted(rel="unrelated", claim=0.05, annoyed=0.95)                # 判断替身也认为他不满：不能因此算两次
    fake = FakeLLM("好，以后少发这种", "嗯")
    agent = make_chat(fake, backend=judge)
    timeline.post(conn, "nudge", "今天是立冬，记得吃饺子", TS - 60, None)
    agent.handle(Turn("这条消息完全没用"))
    assert mood.current(conn, TS)[1] == 1.0
    timeline.post(conn, "nudge", "周五考试，今天要不要先把复习计划排一下？", TS - 30, None)
    agent.handle(Turn("有用"))
    assert mood.current(conn, TS)[1] == 0.5


def test_memory_page_shows_the_quote_the_loop_and_the_mood(cfg):
    import time

    from fastapi.testclient import TestClient

    from alf import db
    from alf.decide import OfflineBackend
    from alf.server import create_app

    with TestClient(create_app(cfg, OfflineBackend(), start_heartbeat=False), base_url="http://127.0.0.1") as client:   # 服务只认本机的来访
        own = db.connect(cfg.db_path)                      # 和服务用同一个库文件
        memory_of(own).write("在等 Atlassian 的面试结果", "study_work", now=TS, follow_up=True, importance=9, quote="在等 Atlassian 的结果")
        mood.strain(own, int(time.time()), "你怎么又没记住")
        data = client.get("/api/memory").json()
    [fact] = data["facts"]
    assert (fact["quote"], fact["open_loop"]) == ("在等 Atlassian 的结果", True)
    assert data["affinity"]["mood"]["state"] == "careful" and data["affinity"]["mood"]["reason"] == "你怎么又没记住"


def test_the_nightly_job_runs_ten_minutes_before_the_digest(cfg):
    from dataclasses import replace
    from datetime import time

    from alf.config import LLMConfig

    built = jobs.build_jobs(replace(cfg, llm=LLMConfig(api_key="sk-test"), digest_at=time(21, 30)), [])
    at = {j.name: j.schedule_text() for j in built}
    assert (at["moments"], at["digest"]) == ("每天 21:20", "每天 21:30")    # 新归纳的经历赶得上出现在当晚的汇总里


# --- 真实回放里暴露出来、随后修掉的三件事（2026-11-17） -----------------------------------

def test_answering_what_alf_just_asked_settles_it_even_though_it_is_no_longer_listed(make_chat):
    """回放里：Alf 问"昨天考得怎么样"，他答了；可那件事因为"今天已经提过"不在上下文里，程序没认出他在回答，
    第二天还会再问一遍。"""
    fake = FakeLLM("早，昨天考得怎么样？", "一道大题不碍事")
    judge = loop_judge(spoke=["COMP5999"], raised=["COMP5999"])
    judge.rules["spoke"] = lambda q, s: 0.9 if "COMP5999" in q["instructions"] and "考得" in s["user_latest_message"] else 0.03
    agent = make_chat(fake, backend=judge)
    exam = agent.memory.write("9月21日考 COMP5999", "study_work", now=TS - 3 * DAY, expires_at=TS - 3600).fact
    agent.handle(Turn("早"))
    assert "[刚过去的事]" in fake.context() and agent.memory.get(exam.id).asks == 1          # Alf 问了一次
    agent.handle(Turn("考得还行 就是最后一道大题没写完"))
    assert "[刚过去的事]" not in fake.context()                                              # 今天问过了，不再摆出来
    assert agent.memory.get(exam.id).aftermath == "done"                                     # 但他答了：这件事了结，明天不会再问
    assert agent.memory.just_past(TS + DAY) == []


def test_offering_to_save_something_is_sent_back_once(make_chat):
    """他说过不要问"要不要记"。提示词里写了，真实回放里模型还是说了"要不要给你记一笔"。"""
    fake = FakeLLM("哈哈腿快断了但值了。要不要给你记一笔，以后能想起今天？", "哈哈腿快断了但值了，蓝山确实费腿")
    agent = make_chat(fake)
    assert agent.handle(Turn("今天去蓝山徒步了，腿快断了")).text == "哈哈腿快断了但值了，蓝山确实费腿"
    sent_back = fake.chat_requests[1]["messages"][-1]["content"]
    assert sent_back.startswith("[系统检查] 你刚写的这份草稿还没有发给用户") and "值得记就直接调用 remember" in sent_back
    for fine in ("记得带伞", "要不要我提醒你？", "你要不要记一下考试时间", "记下了", "需要我做什么吗"):
        assert not chat.OFFER_TO_SAVE.search(fine), fine
    for offer in ("要不要我给你记下来？", "要不要给你记一笔", "需要我帮你存一下吗", "要不要我把这个数记上", "这个要不要记下来"):
        assert chat.OFFER_TO_SAVE.search(offer), offer
    stubborn = FakeLLM("要不要我记下来？", "要不要我给你记下来呢")           # 只打回一次，不会没完没了
    quiet_judge = Scripted(rel="unrelated", claim=0.03)                     # 这是在问，不是在声称记了
    assert make_chat(stubborn, backend=quiet_judge).handle(Turn("绳索后拉20kg")).text == "要不要我给你记下来呢"
    assert len(stubborn.chat_requests) == 2


def test_a_date_written_twice_is_tidied():
    from alf import timeparse

    assert timeparse.absolutize("昨天（2026年9月21日）有一场考试，考得还行", NOW) == "9月21日有一场考试，考得还行"
    assert timeparse.absolutize("9月21日（周一）考试", NOW) == "9月21日（周一）考试"          # 括号里不是同一个日期：不动
