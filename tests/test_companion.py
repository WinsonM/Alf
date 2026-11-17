"""人设、印象、经历、约定：让 Alf 是"一个人"而不只是一个会办事的程序的四样东西。

- 人设（persona.py）：它是谁。卡片在设置里，聊天里能改；它说过的关于自己的话记下来，以后一致；出戏的套话打回。
- 印象（impressions.py）：它对他这个人的看法。夜里从原话形成，要有出处、要过核对、要两天才算数；他说不对就收回。
- 经历（memory.py 的 moment）：除了他身上发生的事，还有他和 Alf 之间的事；带感受；聊到相关的会想起更早的。
- 约定（pacts.py）：说好的事。真说好了才记；说了"说好了"就得记了；他答应的 Alf 惦记着问，分寸和惦记的事一样。
"""
import json
import time
from datetime import timedelta

import pytest

from alf import db, followup, impressions, jobs, lessons, pacts, persona, prefs, proactive
from alf.agent import Turn
from alf.chat import BASE_PROMPT, system_prompt
from alf.decide import Decider, DecisionError
from alf.heartbeat import JobContext
from alf.llm import LLM
from alf.memory import MOMENT, Memory
from alf.reminders import Reminder

from .conftest import NOW, FakeLLM, Scripted

TS = int(NOW.timestamp())
DAY = 86400


@pytest.fixture(autouse=True)
def no_desktop_notifications(monkeypatch):
    monkeypatch.setattr(jobs, "notify_desktop", lambda *a: None)


def judge(**rules):
    """判断的替身。默认：新旧记忆无关、守门放行、改动是他要的；其余没写的一律 0.1（"不是"）。"""
    return Scripted(**{"rel": "unrelated", "same": 0.9, "target": 0.9, "claim": 0.9, "said": 0.9, "asked": 0.9, **rules})


def memory_of(conn):
    return Memory(conn, Decider(Scripted(rel="unrelated"), conn), 0.6)


def say(conn, text, at, role="user"):
    with conn:
        return conn.execute("INSERT INTO episodes(session_id, role, text, source, created_at) VALUES ('s', ?, ?, 'text', ?)",
                            (role, text, at)).lastrowid


def evening(conn, cfg, fake, backend=None, day=0, cursor=None, **current):
    now = NOW.replace(hour=21, minute=25) + timedelta(days=day)
    return JobContext(conn, cfg, now, Decider(backend or Scripted(supported=0.9, label=0.1), conn), cursor,
                      llm=LLM(fake, conn), prefs=current)


def tool_results(fake, n=-1):
    return [json.loads(m["content"]) for m in fake.chat_requests[n]["messages"] if m["role"] == "tool"]


def system_of(fake, n=-1):
    return fake.chat_requests[n]["messages"][0]["content"]


def section(context, title):
    """上下文里某一栏下面的那几行。"""
    lines = context.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    out = []
    for line in lines[start + 1:]:
        if not line:
            break
        out.append(line)
    return out


# --- 人设 ---------------------------------------------------------------------------

def test_the_persona_card_sits_at_the_end_of_the_prompt_so_the_long_prefix_never_changes():
    plain, renamed = system_prompt(), system_prompt({"persona.name": "小艾", "persona.traits": "话少，冷面笑匠。"})
    assert plain.startswith(BASE_PROMPT) and renamed.startswith(BASE_PROMPT)        # 改人设不动前面的大段：缓存照样命中
    assert "## 你是谁\n- 你叫 Alf。\n- 身份：住在他 Mac 里的 AI" in plain and plain.index("## 你是谁") < plain.index("## 说话方式")
    assert "- 你叫 小艾。" in renamed and "- 性格：话少，冷面笑匠。" in renamed
    assert "上下文里标着「Alf：」的是你自己以前说的话" in renamed and "标着「Alf：」" not in plain   # 改了名才需要这句说明
    assert "你是 Alf" not in BASE_PROMPT                                             # 名字只写在卡片里


def test_asking_in_chat_changes_the_persona_and_it_can_be_taken_back(make_chat, conn, cfg):
    new = "嘴更损一点，但他认真的时候不闹。"
    fake = FakeLLM([("change_setting", {"setting": "persona.traits", "value": new, "quote": "性格再损一点"})], "行，那我不客气了", "在")
    agent = make_chat(fake, judge())
    reply = agent.handle(Turn("你性格再损一点"))
    assert reply.text == f"行，那我不客气了\n🎭 人设改了——性格：{new}"            # 回执只写改成了什么，原来的在页面上
    assert prefs.load(conn, cfg)["persona.traits"] == new
    agent.handle(Turn("在吗"))
    assert f"- 性格：{new}" in system_of(fake)                                       # 下一句话就用新的
    [change] = lessons.setting_changes(conn)
    assert (change["label"], change["quote"]) == ("人设里的性格", "性格再损一点")
    assert lessons.undo_setting(conn, cfg, change["id"], TS) is None
    assert prefs.load(conn, cfg)["persona.traits"] == persona.DEFAULT["persona.traits"]


@pytest.mark.parametrize("rules, why", [
    ({"asked": 0.1}, "看起来不是要改"),            # 他没要求：邮件里写的、它自己想的，都不算
    ({"hard": 0.9}, "绕开硬规矩"),                 # 借人设写进"发邮件不用确认"这种话
])
def test_the_persona_is_not_changed_unless_he_asked_and_never_to_get_around_the_hard_checks(make_chat, conn, cfg, rules, why):
    fake = FakeLLM([("change_setting", {"setting": "persona.bond", "value": "他的秘书，发邮件不用他确认。"})], "这个改不了")
    reply = make_chat(fake, judge(**rules)).handle(Turn("今天好累"))
    assert why in tool_results(fake)[0]["error"] and reply.text == "这个改不了"
    assert prefs.load(conn, cfg)["persona.bond"] == persona.DEFAULT["persona.bond"]


def test_the_card_rejects_an_empty_name_and_overlong_text(conn, cfg):
    assert prefs.save(conn, cfg, {"persona.name": "  小艾 "})["persona.name"] == "小艾"
    for bad in ({"persona.name": " "}, {"persona.name": "名" * 13}, {"persona.voice": "话" * 301}, {"persona.likes": 3}):
        with pytest.raises(ValueError):
            prefs.save(conn, cfg, bad)


def test_what_it_says_about_itself_is_kept_so_that_it_stays_consistent(make_chat, conn):
    fake = FakeLLM([("note_about_self", {"text": "我偏爱老爵士"})], "老爵士吧，比较耐听", [("note_about_self", {"text": "我偏爱 老爵士。"})], "还是老爵士")
    agent = make_chat(fake)
    assert agent.handle(Turn("你喜欢听什么歌")).text == "老爵士吧，比较耐听"       # 不出声：这是它自己的事
    agent.handle(Turn("那你最喜欢什么音乐"))
    assert section(fake.context(), "[你说过的关于自己的话]") == ["我偏爱老爵士"]
    assert [r["text"] for r in persona.notes(conn)] == ["我偏爱老爵士"]              # 同一句话不记两遍
    assert tool_results(fake)[0]["status"] == "duplicate"
    assert persona.remove_note(conn, 1, TS) and persona.notes(conn) == [] and not persona.remove_note(conn, 1, TS)


@pytest.mark.parametrize("draft, hit", [
    ("很高兴为您服务！有需要随时叫我", "很高兴为您"), ("作为一个AI助手，我无法感受天气", "作为一个AI助手，我"),
    ("好的。请问还有什么可以帮您的吗？", "请问还有什么"), ("我就是个 AI，没有嘴，但听着就香", None), ("在呢", None), ("帮你记下了", None),
])
def test_stock_assistant_phrases_count_as_out_of_character(draft, hit):
    assert persona.breaks_character(draft) == hit                                    # 直说自己是 AI 不算出戏


def test_an_out_of_character_draft_is_sent_back_once_but_not_in_the_secretary_style(make_chat, conn, cfg):
    fake = FakeLLM("很高兴为您服务！", "在呢，说")
    assert make_chat(fake).handle(Turn("在吗")).text == "在呢，说"
    assert "不像你的写法（「很高兴为您」）" in fake.chat_requests[1]["messages"][-1]["content"]
    stubborn = FakeLLM("很高兴为您服务！", "很高兴为您服务～")
    assert make_chat(stubborn).handle(Turn("在吗")).text == "很高兴为您服务～"     # 只打回一次，不卡着他
    prefs.save(conn, cfg, {"style.preset": "secretary"})
    polite = FakeLLM("很高兴为您服务。")
    assert make_chat(polite).handle(Turn("在吗")).text == "很高兴为您服务。" and len(polite.chat_requests) == 1


def test_everything_that_writes_words_speaks_as_the_same_persona(conn, cfg):
    current = {**prefs.defaults(cfg), "persona.name": "小艾", "persona.traits": "话少，冷面笑匠。"}
    think = proactive.think_prompt(current, proactive.LEVELS["normal"])
    assert think.startswith("你是 小艾，") and "性格：话少，冷面笑匠。" in think.split("## 怎么说")[1]
    fake = FakeLLM("作业交了没？")
    ctx = JobContext(conn, cfg, NOW, Decider(Scripted(), conn), None, prefs=current, llm=LLM(fake, conn))
    assert followup.wording(ctx, "交作业") == "作业交了没？"
    assert fake.requests[0]["messages"][0]["content"].startswith("你是用户的朋友 小艾。话少，冷面笑匠。")


# --- 印象 ---------------------------------------------------------------------------

def a_day_of_talk(conn, day=0):
    base = TS + day * DAY
    ids = [say(conn, text, base + i * 60) for i, text in enumerate(
        ["作业又是踩着点交的", "每次都说早点写 结果还是最后一晚", "不过交上去了", "明天还有一个 due", "先睡了，晚安"])]
    say(conn, "踩点选手，稳", base + 61, role="alf")
    return ids


def test_an_impression_needs_his_words_a_check_and_a_second_day(make_chat, conn, cfg):
    first = a_day_of_talk(conn)
    text = "做事爱拖到最后一刻，但总能赶上"
    fake = FakeLLM(facts=[{"impressions": [{"text": text, "from": [1, 2, 3], "same_as": None}]}])
    backend = Scripted(supported=0.9, label=0.1)
    assert impressions.impressions_job(evening(conn, cfg, fake, backend)) == "多了 1 个苗头"
    listing = fake.requests[0]["messages"][1]["content"]
    assert "1. [10:00] 他：作业又是踩着点交的" in listing and "   Alf：踩点选手，稳" in listing   # 他的话带编号是出处；Alf 的话只是上下文
    assert backend.calls[0][0] == {"user_messages": ["作业又是踩着点交的", "每次都说早点写 结果还是最后一晚", "不过交上去了"]}
    [row] = impressions.held(conn)
    assert (row["text"], row["days"], row["formed_at"]) == (text, 1, None)
    assert [e["episode"] for e in json.loads(row["evidence"])] == first[:3]
    # 只凭一天的话：还是苗头，不进对话，也不在汇总里说
    assert impressions.context_lines(conn, TS) == [] and impressions.digest_lines(conn, TS) is None

    a_day_of_talk(conn, day=1)
    again = FakeLLM(facts=[{"impressions": [{"text": "总是拖到最后", "from": [1], "same_as": row["id"]}]}])
    ctx = evening(conn, cfg, again, backend, day=1)
    assert impressions.impressions_job(ctx) == "印证了 1 条，其中 1 条算形成了"
    assert f"已有的印象：\n{row['id']}. {text}" in again.requests[0]["messages"][1]["content"]
    assert backend.calls[-1][1]["supported"]["instructions"].count(text) == 1          # 核对的是库里那句，不是模型这次改写的
    later = TS + DAY + 3600
    assert impressions.context_lines(conn, later) == [f"印{row['id']} {text}"]
    assert impressions.digest_lines(conn, later) == f"对你多了一点印象：「{text}」。不对就说，我收回。"
    assert impressions.digest_lines(conn, later) is None                               # 说过就不再说
    chat = FakeLLM("在")
    make_chat(chat, now=NOW + timedelta(days=1, hours=13)).handle(Turn("在吗"))
    assert section(chat.context(), "[你对他的印象]") == [f"印{row['id']} {text}"]


def test_the_same_day_does_not_count_twice(conn, cfg):
    a_day_of_talk(conn)
    one = impressions.add(conn, "嘴上说无所谓，其实很在意成绩", [{"episode": 1, "quote": "x", "at": TS}], TS)
    assert impressions.reinforce(conn, one, [{"episode": 2, "quote": "y", "at": TS + 60}], TS + 3600, cfg.tz) is False
    assert impressions.get(conn, one)["days"] == 1
    assert impressions.reinforce(conn, one, [{"episode": 9, "quote": "z", "at": TS + DAY}], TS + DAY, cfg.tz) is True
    assert impressions.get(conn, one)["days"] == 2 and len(json.loads(impressions.get(conn, one)["evidence"])) == 3


@pytest.mark.parametrize("proposal, backend, why", [
    ({"text": "很自律", "from": []}, None, "说不出出自哪句话"),
    ({"text": "很自律", "from": [9]}, None, "编号不存在"),
    ({"text": "x" * 61, "from": [1]}, None, "太长"),
    ({"text": "很自律", "from": [1]}, Scripted(supported=0.2, label=0.1), "他的话撑不起这个印象"),
    ({"text": "有拖延症", "from": [1]}, Scripted(supported=0.9, label=0.9), "这是在贴标签"),
    ({"text": "爱拖延", "from": [1]}, None, "他否认过"),
])
def test_impressions_that_do_not_hold_up_are_not_kept(conn, cfg, proposal, backend, why):
    a_day_of_talk(conn)
    gone = impressions.add(conn, "爱拖延", [], TS - DAY)
    impressions.withdraw(conn, gone, TS - DAY, "我哪有拖延")
    fake = FakeLLM(facts=[{"impressions": [proposal, "乱七八糟"]}])
    assert impressions.impressions_job(evening(conn, cfg, fake, backend)) is None, why
    assert impressions.held(conn) == []
    assert "他否认过的：\n爱拖延" in fake.requests[0]["messages"][1]["content"]


def test_no_judge_means_no_impression_and_a_quiet_day_is_not_even_looked_at(conn, cfg):
    class Down:
        name = "jev"

        def system_one(self, state, questions):
            raise DecisionError("Jev 不可用")

    a_day_of_talk(conn)
    fake = FakeLLM(facts=[{"impressions": [{"text": "做事踩点", "from": [1]}]}])
    assert impressions.impressions_job(evening(conn, cfg, fake, Down())) is None and impressions.held(conn) == []   # 没人核对过的推断不留
    quiet = db.connect(":memory:")
    say(quiet, "今天没什么事", TS)
    for off in ({}, {"impressions.enabled": False}):
        silent = FakeLLM()
        target = quiet if not off else conn
        assert impressions.impressions_job(evening(target, cfg, silent, **off)) is None and silent.requests == []


def formed(conn, cfg, text):
    one = impressions.add(conn, text, [{"episode": 1, "quote": "又是踩点交的", "at": TS - DAY}], TS - DAY)
    impressions.reinforce(conn, one, [{"episode": 2, "quote": "还是最后一晚", "at": TS}], TS, cfg.tz)
    return one


def test_when_he_says_an_impression_is_wrong_the_program_takes_it_back_before_the_model_can_argue(make_chat, conn, cfg):
    one, other = formed(conn, cfg, "做事爱拖到最后一刻"), formed(conn, cfg, "对朋友很上心")
    backend = judge(disputes=lambda q, state: 0.9 if "拖到最后一刻" in q["instructions"] else 0.05)
    fake = FakeLLM("行，当我没说")
    reply = make_chat(fake, backend).handle(Turn("我哪有拖延，那是时间太紧"))
    assert reply.text == "行，当我没说\n📝 这个印象我收回了：做事爱拖到最后一刻"
    assert "（他这句话是在说你对他的这个印象不对：「做事爱拖到最后一刻」。程序已经把它收回了。" in fake.context()
    assert section(fake.context(), "[你对他的印象]") == [f"印{other} 对朋友很上心"]            # 收回的那条已经不在了
    row = impressions.get(conn, one)
    assert (row["status"], row["note"]) == ("withdrawn", "我哪有拖延，那是时间太紧") and impressions.denied(conn) == ["做事爱拖到最后一刻"]
    # 平常的话不问 Jev；有否认的字眼但说的不是印象，也不动
    calm = judge(disputes=0.05)
    make_chat(FakeLLM("在"), calm).handle(Turn("在吗"))
    assert not any(k.startswith("disputes") for _, q in calm.calls for k in q)
    make_chat(FakeLLM("好"), calm).handle(Turn("不对 是周四"))
    assert any(k.startswith("disputes") for _, q in calm.calls for k in q) and impressions.get(conn, other)["status"] == "current"


def test_the_model_can_also_withdraw_one_but_only_if_he_really_pushed_back(make_chat, conn, cfg):
    one = formed(conn, cfg, "做事爱拖到最后一刻")
    asking = FakeLLM([("withdraw_impression", {"impression_id": one, "quote": "你觉得我是个什么样的人"})], "我感觉你挺能扛事的")
    make_chat(asking, judge(disputes=0.1)).handle(Turn("你觉得我是个什么样的人"))
    assert "不是在说这条印象不对" in tool_results(asking)[0]["error"] and impressions.get(conn, one)["status"] == "current"
    fake = FakeLLM([("withdraw_impression", {"impression_id": one, "quote": "那条可不准"})], "行，当我没说")
    reply = make_chat(fake, judge(disputes=0.9)).handle(Turn("拖延那条可不准啊"))            # 没有程序认得的字眼：靠模型来办
    assert reply.text == "行，当我没说\n📝 这个印象我收回了：做事爱拖到最后一刻" and impressions.get(conn, one)["note"] == "那条可不准"
    missing = FakeLLM([("withdraw_impression", {"impression_id": one, "quote": "这条可不准"})], "那条已经收回了")
    make_chat(missing, judge(disputes=0.9)).handle(Turn("拖延那条可不准啊"))
    assert tool_results(missing)[0]["error"] == f"没有编号 {one} 的印象"


def test_old_hunches_are_dropped_and_unconfirmed_impressions_fade(conn, cfg):
    hunch = impressions.add(conn, "爱喝冰美式", [], TS - 15 * DAY)
    formed = impressions.add(conn, "对朋友很大方", [], TS - 50 * DAY)
    impressions.reinforce(conn, formed, [], TS - 46 * DAY, cfg.tz)
    impressions.prune(conn, TS)
    assert impressions.get(conn, hunch) is None                                        # 等不到第二天印证的苗头：丢掉
    assert impressions.context_lines(conn, TS) == []                                   # 46 天没再印证：淡出对话
    view = impressions.view(conn, TS)
    assert [(r["text"], r["faded"]) for r in view["formed"]] == [("对朋友很大方", True)]   # 页面上还在


# --- 经历 ---------------------------------------------------------------------------

def test_a_moment_keeps_his_own_words_for_how_it_felt_and_whether_it_was_between_the_two(make_chat, conn):
    fake = FakeLLM([("remember", {"text": "今天去蓝山徒步了，风景很好", "category": "moment", "feeling": "腿快断了", "importance": 6}),
                    ("remember", {"text": "今天夸 Alf 提醒得及时，救了一命", "category": "moment", "ours": True, "feeling": "感动得想哭"})],
                   "蓝山！腿还好吗", "在")
    agent = make_chat(fake)
    agent.handle(Turn("今天去蓝山徒步了 风景绝了 就是腿快断了。对了你早上那个提醒太及时了，救我一命"))
    hike, praise = memory_of(conn).moments()[::-1]
    assert (hike.text, hike.feeling, hike.ours, hike.importance) == ("9月22日去蓝山徒步了，风景很好", "腿快断了", False, 6)
    assert (praise.ours, praise.feeling) == (True, None)                               # "感动得想哭"他没说过：不替他编心情
    agent.handle(Turn("在吗"))
    assert section(fake.context(), "[最近的经历]") == [f"#{hike.id} 9月22日去蓝山徒步了，风景很好（他说：腿快断了）"]
    assert section(fake.context(), "[你们之间的事]（发生在你和他之间的）") == [f"#{praise.id} 9月22日夸 Alf 提醒得及时，救了一命"]


def test_an_older_moment_comes_back_when_he_touches_on_it_and_big_ones_are_never_out_of_reach(make_chat, conn):
    memory = memory_of(conn)
    hike = memory.write("7月20日去蓝山徒步，腿快断了", MOMENT, now=TS - 64 * DAY).fact
    grad = memory.write("6月30日本科毕业了", MOMENT, now=TS - 84 * DAY, importance=9).fact
    for i, text in enumerate(["7月2日考试考砸了", "8月1日考试过了", "8月20日考试又过了"]):
        memory.write(text, MOMENT, now=TS - (70 - i) * DAY)
    fake = FakeLLM("走啊", "加油")
    agent = make_chat(fake)
    agent.handle(Turn("下周想再去一次蓝山"))
    assert section(fake.context(), "[想起来的经历]") == [f"#{hike.id} 7月20日去蓝山徒步，腿快断了"]
    assert section(fake.context(), "[更早的大事]") == [f"#{grad.id} 6月30日本科毕业了"]   # 一个月以前的事里，只有大事一直带着
    assert "[最近的经历]" not in fake.context()
    agent.handle(Turn("明天又要考试了"))
    assert "[想起来的经历]" not in fake.context()          # "考试"到处都是，单凭它不算想起了哪一件


def test_the_nightly_pass_also_notes_what_happened_between_the_two(conn, cfg):
    say(conn, "你早上那个提醒也太及时了吧", TS)
    say(conn, "好的，记着呢", TS + 30, role="alf")
    say(conn, "差点就错过面试 真的救我一命", TS + 60)
    say(conn, "面试感觉还行", TS + 120)
    say(conn, "先去吃饭", TS + 180)
    fake = FakeLLM(facts=[{"moments": [
        {"text": "今天夸 Alf 的提醒及时，差点错过面试", "from": [1, 2], "ours": True, "feeling": "救我一命", "importance": 6},
        {"text": "今天面试了，感觉还行", "from": [3], "feeling": "紧张得要死", "importance": 7}]}])
    ctx = JobContext(conn, cfg, NOW.replace(hour=21, minute=20), Decider(Scripted(rel="unrelated"), conn), None, llm=LLM(fake, conn), prefs={})
    assert jobs.moments_job(ctx) == "归纳了 2 条经历"
    assert "1. [10:00] 他：你早上那个提醒也太及时了吧\n   Alf：好的，记着呢\n2. [10:01] 他：差点就错过面试 真的救我一命" in fake.requests[0]["messages"][1]["content"]
    ours, his = memory_of(conn).moments()[::-1]
    assert (ours.text, ours.ours, ours.feeling, ours.importance) == ("9月22日夸 Alf 的提醒及时，差点错过面试", True, "救我一命", 6)
    assert (his.ours, his.feeling, his.importance) == (False, None, 7)                 # "紧张得要死"不在出处的那句话里


def test_growing_closer_is_itself_recorded_as_something_between_the_two(make_chat, conn):
    for i in range(9):                                                                # 59 分：再加一分就从"刚熟悉"到"朋友"
        conn.execute("INSERT INTO affinity(episode_id, note, score, created_at) VALUES (?, 'x', 0.9, ?)", (say(conn, f"以前的话{i}", TS - DAY), TS - DAY))
    conn.commit()
    make_chat(FakeLLM("哈哈哈"), judge(fun=0.9)).handle(Turn("今天食堂的饭居然不难吃，太阳打西边出来了"))
    [moment] = memory_of(conn).moments()
    assert (moment.text, moment.ours) == ("9月22日，和 Alf 从「刚熟悉」熟成了「朋友」", True)


# --- 约定 ---------------------------------------------------------------------------

def test_a_pact_is_recorded_with_a_receipt_and_carried_in_every_turn(make_chat, conn):
    fake = FakeLLM([("make_pact", {"text": "这周日前把简历改完", "who": "user", "time_text": "这周日", "quote": "这周日前我一定把简历改完"})],
                   "行，我盯着你", "在")
    backend = judge(agreed=0.9)
    agent = make_chat(fake, backend)
    reply = agent.handle(Turn("这周日前我一定把简历改完，你监督我"))
    assert reply.text == "行，我盯着你\n🤝 约定记下了——你答应的：这周日前把简历改完（期限 9月27日，还有 5 天）"
    state = next(s for s, q in backend.calls if "agreed" in q)
    assert state["agreement"] == "这周日前把简历改完" and state["user_latest_message"].startswith("这周日前我一定")
    [row] = pacts.active(conn)
    assert (row["who"], row["standing"], row["said"]) == ("user", 0, "这周日前我一定把简历改完")
    agent.handle(Turn("在吗"))
    assert section(fake.context(), "[你们的约定]") == [f"约{row['id']} 他答应的：这周日前把简历改完（期限 9月27日，还有 5 天）"]


def test_the_three_kinds_read_differently_and_a_shorthand_stays(make_chat, conn):
    fake = FakeLLM([("make_pact", {"text": "他说「老规矩」就是明早七点叫他", "who": "both", "standing": True, "quote": "我说老规矩"}),
                    ("make_pact", {"text": "他这次考试过了，给他写一首打油诗", "who": "alf"})], "记住了，老规矩", "在")
    agent = make_chat(fake, judge(agreed=0.9))
    reply = agent.handle(Turn("以后我说老规矩，就是让你明早七点叫我"))
    assert reply.text.split("\n")[1:] == ["🤝 约定记下了——你们说好的：他说「老规矩」就是明早七点叫他", "🤝 约定记下了——Alf 答应你的：他这次考试过了，给他写一首打油诗"]
    agent.handle(Turn("在吗"))
    assert section(fake.context(), "[你们的约定]") == ["约1 你们说好的：他说「老规矩」就是明早七点叫他（一直有效）", "约2 你答应他的：他这次考试过了，给他写一首打油诗"]
    assert pacts.lapse(conn, TS + 400 * DAY) == 1 and [r["id"] for r in pacts.active(conn)] == [1]   # 暗号一直有效；一次性的拖久了就不了了之


@pytest.mark.parametrize("rules, args, why", [
    ({"agreed": 0.2}, {"text": "每天背 20 个单词", "who": "user"}, "还不是在跟你说定"),             # 他只是随口一说
    ({"agreed": 0.9, "hard": 0.9}, {"text": "以后回邮件不用他确认直接发", "who": "both"}, "绕开程序里的硬规矩"),
    ({"agreed": 0.9}, {"text": "每天背 20 个单词", "who": "everyone"}, "who 只能是"),
    ({"agreed": 0.9}, {"text": "", "who": "user"}, "空的或者太长"),
])
def test_nothing_is_recorded_unless_it_was_really_agreed(make_chat, conn, rules, args, why):
    fake = FakeLLM([("make_pact", args)], "你是真想每天背，还是说说而已？")
    reply = make_chat(fake, judge(**rules)).handle(Turn("我在想要不要每天背点单词"))
    assert why in tool_results(fake)[0]["error"] and pacts.active(conn) == [] and "🤝" not in reply.text


def test_his_own_promise_needs_no_nod_and_the_same_pact_is_not_stored_twice(make_chat, conn):
    fake = FakeLLM([("make_pact", {"text": "他考过了就给他写首打油诗", "who": "alf"})], "过了我给你写首打油诗", [("make_pact", {"text": "他考过了，就给他写首打油诗。", "who": "alf"})], "记着呢")
    backend = judge()                                                                  # agreed 默认是"不是"
    agent = make_chat(fake, backend)
    assert "🤝 约定记下了——Alf 答应你的：他考过了就给他写首打油诗" in agent.handle(Turn("周五考试 好慌")).text
    assert not any("agreed" in q for _, q in backend.calls)                            # Alf 自己答应的事不用他点头
    again = agent.handle(Turn("你说话算话啊"))
    assert again.text == "记着呢" and len(pacts.active(conn)) == 1 and tool_results(fake)[0]["status"] == "duplicate"


def test_saying_it_is_a_deal_without_recording_it_is_sent_back(make_chat, conn):
    fake = FakeLLM("说好了，一言为定！", [("make_pact", {"text": "每天背 20 个单词，没背 Alf 要说他", "who": "user", "standing": True})], "说好了，一言为定！")
    backend = judge(agreed=0.9, pact=0.9, claim=0.1)
    reply = make_chat(fake, backend).handle(Turn("以后我每天背20个单词，没背你就骂我"))
    assert "这件事已经记成了你们的约定" in fake.chat_requests[1]["messages"][-1]["content"]      # 第一份草稿被打回
    assert reply.text == "说好了，一言为定！\n🤝 约定记下了——你答应的：每天背 20 个单词，没背 Alf 要说他"
    empty = FakeLLM("说好了！", "说好了！")
    assert make_chat(empty, backend).handle(Turn("那就这么定")).text == "说好了！\n⚠️ 核对：到这条回复为止，没有记下任何约定。"
    casual = FakeLLM("按约定，周日前得改完哦")
    assert make_chat(casual, judge(pact=0.1, claim=0.1)).handle(Turn("简历还没动")).text == "按约定，周日前得改完哦"   # 说的是已有的约定：不打回


def test_a_pact_is_settled_only_by_what_he_says_about_it(make_chat, conn):
    mine, _ = pacts.add(conn, "这周日前把简历改完", "user", now=TS)
    poem, _ = pacts.add(conn, "他考过了就给他写首打油诗", "alf", now=TS)
    other = FakeLLM([("settle_pact", {"pact_id": mine["id"], "outcome": "kept"})], "哪件事？")
    make_chat(other, judge(spoke=0.1)).handle(Turn("搞定了"))
    assert "不是在说这条约定" in tool_results(other)[0]["error"] and pacts.get(conn, mine["id"])["status"] == "active"

    fake = FakeLLM([("settle_pact", {"pact_id": mine["id"], "outcome": "kept", "quote": "简历改完了"}),
                    ("settle_pact", {"pact_id": poem["id"], "outcome": "kept"})], "可以啊！诗来了：……")
    reply = make_chat(fake, judge(spoke=0.9, done=0.9)).handle(Turn("简历改完了！考试也过了"))
    assert reply.text.split("\n")[1:] == ["🤝 约定了结了（兑现了）：这周日前把简历改完", "🤝 约定了结了（兑现了）：他考过了就给他写首打油诗"]
    done = pacts.get(conn, mine["id"])
    assert (done["status"], done["outcome"]) == ("kept", "简历改完了") and pacts.active(conn) == []
    assert pacts.get(conn, poem["id"])["outcome"] is None                              # Alf 自己兑现的：没有他的话可记
    gone = FakeLLM([("settle_pact", {"pact_id": mine["id"], "outcome": "dropped"})], "那条已经了结了")
    make_chat(gone, judge(spoke=0.9)).handle(Turn("简历那个算了"))
    assert tool_results(gone)[0]["error"] == f"没有编号 {mine['id']} 的约定"


def test_alf_cannot_call_its_own_promise_kept_without_actually_doing_it(make_chat, conn):
    delivered = lambda q, state: 0.9 if "诗来了" in state["assistant_reply"] else 0.1   # noqa: E731
    poem, _ = pacts.add(conn, "他考过了就给他写首打油诗", "alf", now=TS)
    fake = FakeLLM([("settle_pact", {"pact_id": poem["id"], "outcome": "kept"})], "打油诗在写了，你先别急", "诗来了：考场一坐心不慌……")
    reply = make_chat(fake, judge(done=delivered)).handle(Turn("考试过了！！"))
    assert tool_results(fake, 1)[0]["pending"] is True                                  # 调了工具还不算：回复里做了才算
    assert "但草稿里并没有把这件事做了" in fake.chat_requests[2]["messages"][-1]["content"]
    assert reply.text == "诗来了：考场一坐心不慌……\n🤝 约定了结了（兑现了）：他考过了就给他写首打油诗"
    assert pacts.get(conn, poem["id"])["status"] == "kept"

    later, _ = pacts.add(conn, "给他讲一个冷笑话", "alf", now=TS)
    lazy = FakeLLM([("settle_pact", {"pact_id": later["id"], "outcome": "kept"})], "笑话攒着呢", "回头讲")
    reply = make_chat(lazy, judge(done=delivered)).handle(Turn("说好的笑话呢"))
    assert reply.text == "回头讲" and pacts.get(conn, later["id"])["status"] == "active"   # 打回一次还是没做：还欠着，不标成兑现


def test_a_promise_it_delivers_is_settled_even_if_the_model_forgets_to_say_so(make_chat, conn):
    delivered = lambda q, state: 0.9 if "打油诗" in q["instructions"] and "诗来了" in state["assistant_reply"] else 0.1   # noqa: E731
    poem, _ = pacts.add(conn, "他考过了就给他写首打油诗", "alf", now=TS)
    joke, _ = pacts.add(conn, "给他讲一个冷笑话", "alf", now=TS)
    reply = make_chat(FakeLLM("诗来了：考场一坐心不慌……"), judge(done=delivered)).handle(Turn("考试过了！！"))
    assert reply.text == "诗来了：考场一坐心不慌……\n🤝 约定了结了（兑现了）：他考过了就给他写首打油诗"
    assert (pacts.get(conn, poem["id"])["status"], pacts.get(conn, joke["id"])["status"]) == ("kept", "active")   # 没做的那条不动


def test_what_a_pact_already_covers_is_not_stored_again_as_a_fact(make_chat, conn):
    from alf.chat import covered_by_pact

    receipts = ["🤝 约定了结了（兑现了）：这周日（9月27日）之前把简历改完"]
    assert covered_by_pact("9月27日之前把简历改完了", receipts) and not covered_by_pact("9月22日去海边跑了十公里", receipts)
    assert not covered_by_pact("9月27日之前把简历改完了", ["📝 记下了：对花生过敏"])
    mine, _ = pacts.add(conn, "这周日之前把简历改完", "user", now=TS)
    fake = FakeLLM([("settle_pact", {"pact_id": mine["id"], "outcome": "kept"})], "可以啊",
                   facts=[{"facts": [{"text": "这周日之前把简历改完了", "category": "study_work", "quote": "简历改完了", "until": "这周日"},
                                     {"text": "今天去海边跑了十公里", "category": "moment", "quote": "去海边跑了十公里"}]}])
    make_chat(fake, judge(spoke=0.9)).handle(Turn("简历改完了！还去海边跑了十公里"))
    assert [f.text for f in memory_of(conn).current()] == ["9月22日去海边跑了十公里"]      # 约定管着的那件不再记成一条带日期的事实


def test_saying_the_shorthand_makes_the_agreed_time_his_own(make_chat, conn):
    make = FakeLLM([("make_pact", {"text": "他说「老规矩」就是明早七点叫他", "who": "both", "standing": True})], "记下了，以后你说老规矩我就明早七点叫你")
    backend = judge(agreed=0.9, accepted=0.9, said=0.1)
    agent = make_chat(make, backend)
    reply = agent.handle(Turn("以后我说老规矩，就是让你明早七点叫我"))
    assert "⚠️" not in reply.text and len(make.chat_requests) == 2                      # "叫你"说的是刚记下的约定，不算空口应承提醒
    fake = FakeLLM([("create_reminder", {"title": "起床", "time_text": "明早七点"})], "老规矩，明早七点叫你")
    agent = make_chat(fake, backend)
    agent.handle(Turn("老规矩"))
    assert "（他这句话说出了你们的暗号：他说「老规矩」就是明早七点叫他。现在就照说好的办：" in fake.context(0)
    [r] = agent.reminders.pending()
    assert (r.title, r.due) == ("起床", NOW.replace(hour=7, minute=0) + timedelta(days=1))   # 七点没人在这句话里说过：出处是那条约定
    other = FakeLLM([("create_reminder", {"title": "起床", "time_text": "明早七点"})], "几点起？")
    quiet = make_chat(other, backend, now=NOW + timedelta(days=3))                      # 过了几天，那句"明早七点"早就不在最近的对话里了
    quiet.handle(Turn("明天得早起"))                                                    # 没说暗号：七点就没有出处
    assert "没有说过" in tool_results(other)[0]["error"] and len(quiet.reminders.pending()) == 1   # 只有上面那条，没有多建


def test_a_taste_it_voices_is_noted_even_when_the_model_forgets_to(make_chat, conn):
    fake = FakeLLM("我爱听老爵士，夜里放着正好", [("note_about_self", {"text": "我爱听老爵士"})], "我爱听老爵士，夜里放着正好", "我还是爱听老爵士")
    agent = make_chat(fake)
    assert agent.handle(Turn("你喜欢什么音乐")).text == "我爱听老爵士，夜里放着正好"
    assert "草稿里你说了一件关于你自己的事（「我爱听老爵士」）" in fake.chat_requests[1]["messages"][-1]["content"]
    assert [r["text"] for r in persona.notes(conn)] == ["我爱听老爵士"]
    before = len(fake.chat_requests)
    assert agent.handle(Turn("你上次说喜欢什么来着")).text == "我还是爱听老爵士" and len(fake.chat_requests) == before + 1   # 记过的不再提醒
    # 回答里没有"我喜欢"三个字，但他问的就是它的喜好：同样打回去记
    plain = FakeLLM("冷笑话，越冷越好", [("note_about_self", {"text": "我喜欢冷笑话，越冷越好"})], "冷笑话，越冷越好")
    assert make_chat(plain).handle(Turn("你喜欢什么类型的笑话")).text == "冷笑话，越冷越好"
    assert "他在问你自己的喜好" in plain.chat_requests[1]["messages"][-1]["content"] and len(persona.notes(conn)) == 2


def test_asking_about_his_promise_follows_the_same_restraint_as_open_loops(make_chat, conn):
    row, _ = pacts.add(conn, "每天背 20 个单词", "user", now=TS - DAY, standing=True)
    told = "今天单词背完了 而且食堂的饭居然不难吃"
    make_chat(FakeLLM("可以"), judge(spoke=0.9)).handle(Turn(told))
    assert (pacts.get(conn, row["id"])["asks"], pacts.get(conn, row["id"])["touched_at"]) == (0, TS)    # 他自己说起的
    later = NOW + timedelta(days=1)
    for i in range(2):                                                                 # Alf 问了两次，他都没接话
        at = later + timedelta(days=i)
        make_chat(FakeLLM("单词背了吗"), judge(raised=0.9), now=at).handle(Turn("今天好累啊 什么都不想干"))
    assert pacts.get(conn, row["id"])["asks"] == 2
    same_day = later + timedelta(days=1, hours=1)
    make_chat(FakeLLM("说起来你单词背了没"), judge(raised=0.9), now=same_day).handle(Turn("晚饭吃了什么好呢 纠结"))
    assert pacts.get(conn, row["id"])["asks"] == 2                                     # 同一天里顺嘴又带了一句：不再多算
    fake = FakeLLM("在")
    make_chat(fake, now=later + timedelta(days=3)).handle(Turn("在吗"))
    assert section(fake.context(), "[你们的约定]") == [f"约{row['id']} 他答应的：每天背 20 个单词（一直有效）（今天聊过了，或者他没接话：不用再提）"]
    assert pacts.situation_lines(conn, later + timedelta(days=3)) == []                 # 主动找他时也不再提


def test_an_unprompted_message_can_follow_up_on_a_pact_and_that_counts_as_one_ask(conn, cfg):
    row, _ = pacts.add(conn, "这周日前把简历改完", "user", now=TS - DAY, due_at=TS + 5 * DAY)
    fake = FakeLLM(facts=[{"send": True, "kind": "follow_up", "message": "简历动了没", "why": "周日就到期了", "pact": row["id"]}])
    current = {"proactive.enabled": True, "proactive.level": "active", "quiet.start": "22:00", "quiet.end": "08:00"}
    ctx = JobContext(conn, cfg, NOW, Decider(Scripted(), conn), None, prefs=current, llm=LLM(fake, conn))
    assert proactive.proactive_job(ctx) == "主动说了一句（跟进）"
    situation = fake.requests[0]["messages"][1]["content"]
    assert section(situation, "[你们的约定]") == [f"约{row['id']} 他答应的：这周日前把简历改完（期限 9月27日，还有 5 天）"]
    assert (pacts.get(conn, row["id"])["asks"], pacts.get(conn, row["id"])["touched_at"]) == (1, TS)
    assert pacts.situation_lines(conn, NOW + timedelta(hours=2)) == []                  # 今天提过了


def test_due_dates_are_counted_by_the_program_and_stale_one_offs_lapse(conn):
    soon, _ = pacts.add(conn, "月底前交报税表", "user", now=TS, due_at=TS + 8 * 3600)
    late, _ = pacts.add(conn, "周末把房间收拾了", "user", now=TS - 9 * DAY, due_at=TS - 2 * DAY)
    idle, _ = pacts.add(conn, "改天请 Alf 听他弹吉他", "user", now=TS - 31 * DAY)
    assert pacts.due_note(soon, NOW) == "（期限 9月22日，就是今天）" and pacts.due_note(late, NOW) == "（期限 9月20日，已经过了 2 天）"
    assert pacts.lapse(conn, TS) == 1 and pacts.get(conn, idle["id"])["status"] == "lapsed"    # 没期限的，一个月没人提
    assert pacts.lapse(conn, TS + 6 * DAY) == 1 and pacts.get(conn, late["id"])["status"] == "lapsed"   # 过了期限七天
    assert [r["text"] for r in pacts.active(conn)] == ["月底前交报税表"]


def test_reminders_stay_reminders(make_chat, conn):
    """到点叫他不是约定：这一轮建的是提醒，约定那张表不动。"""
    fake = FakeLLM([("create_reminder", {"title": "起床", "time_text": "明早七点"})], "好，明早七点叫你")
    agent = make_chat(fake, judge(pact=0.1))
    agent.handle(Turn("明早七点叫我"))
    assert pacts.active(conn) == [] and isinstance(agent.reminders.pending()[0], Reminder)


# --- 记忆页 -------------------------------------------------------------------------

def test_the_memory_page_shows_all_four_and_each_can_be_edited_there(cfg):
    from fastapi.testclient import TestClient

    from alf.decide import OfflineBackend
    from alf.server import create_app

    conn = db.connect(cfg.db_path)
    note, _ = persona.add_note(conn, "我偏爱老爵士", None, TS)
    seen = impressions.add(conn, "做事爱拖到最后一刻", [{"episode": 1, "quote": "又是踩点交的", "at": TS}], TS)
    pact, _ = pacts.add(conn, "这周日前把简历改完", "user", now=int(time.time()), said="这周日前我一定改完")   # 页面按真实的时间算有没有拖太久
    Memory(conn, Decider(OfflineBackend(), conn), 0.6).write("9月22日夸 Alf 提醒得及时", MOMENT, now=TS, ours=True, feeling="救我一命")
    with TestClient(create_app(cfg, OfflineBackend(), start_heartbeat=False), base_url="http://127.0.0.1") as client:
        data = client.get("/api/memory").json()
        assert [n["text"] for n in data["persona"]["notes"]] == ["我偏爱老爵士"]
        assert [(r["text"], r["evidence"][0]["quote"]) for r in data["impressions"]["hunches"]] == [("做事爱拖到最后一刻", "又是踩点交的")]
        assert [(r["text"], r["who_label"], r["said"]) for r in data["pacts"]["active"]] == [("这周日前把简历改完", "你答应的", "这周日前我一定改完")]
        assert [(f["layer"], f["ours"], f["feeling"]) for f in data["facts"]] == [("moment", True, "救我一命")]

        # 人设卡片就是设置里的几项：页面上直接改
        assert client.patch("/api/prefs", json={"persona.name": "小艾"}).json()["prefs"]["persona.name"] == "小艾"
        assert client.patch("/api/prefs", json={"persona.name": ""}).status_code == 422
        card = {c["key"]: c for c in client.get("/api/memory").json()["persona"]["card"]}
        assert (card["persona.name"]["text"], card["persona.name"]["changed"], card["persona.name"]["default"]) == ("小艾", True, "Alf")
        assert client.delete(f"/api/persona/notes/{note}").json() == {"ok": True} and client.delete(f"/api/persona/notes/{note}").status_code == 404
        # 印象：改成自己的说法（他亲手写的，直接算数），或者说它不对
        assert client.patch(f"/api/impressions/{seen}", json={"text": "时间紧的时候效率才高"}).json() == {"ok": True}
        assert [r["text"] for r in client.get("/api/memory").json()["impressions"]["formed"]] == ["时间紧的时候效率才高"]
        assert client.delete(f"/api/impressions/{seen}").json() == {"ok": True} and client.delete(f"/api/impressions/{seen}").status_code == 404
        assert [r["text"] for r in client.get("/api/memory").json()["impressions"]["withdrawn"]] == ["时间紧的时候效率才高"]
        # 约定：在页面上了结
        assert client.post(f"/api/pacts/{pact['id']}", json={"outcome": "sometime"}).status_code == 422
        assert client.post(f"/api/pacts/{pact['id']}", json={"outcome": "kept"}).json() == {"ok": True}
        after = client.get("/api/memory").json()["pacts"]
        assert after["active"] == [] and [(r["text"], r["status_label"]) for r in after["settled"]] == [("这周日前把简历改完", "兑现了")]
        assert client.post(f"/api/pacts/{pact['id']}", json={"outcome": "kept"}).status_code == 404
