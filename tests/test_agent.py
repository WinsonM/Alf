from alf.agent import Turn
from alf.decide import DecisionError

from .conftest import Scripted, option_containing

NO_FACT = {"states_fact": 0.1, "fact_category": "other"}


def test_creates_reminder_and_echoes_resolved_time(make_agent):
    agent = make_agent(Scripted(intent="create_reminder", **NO_FACT))
    reply = agent.handle(Turn("明天下午三点提醒我交作业"))
    assert reply.text == "好的，明天 9月23日（周三）15:00 提醒你：交作业"
    [r] = agent.reminders.pending()
    assert (r.title, r.due.day, r.due.hour) == ("交作业", 23, 15)


def test_lead_time_and_guessed_ampm_are_spelled_out(make_agent):
    agent = make_agent(Scripted(intent="create_reminder", **NO_FACT))
    reply = agent.handle(Turn("3点开会，提前半小时提醒我"))
    assert reply.text.startswith("好的，「开会」在 今天 9月22日（周二）15:00，我会提前半小时（14:30）提醒你")
    assert "「3点」我理解成下午 15:00" in reply.text


def test_missing_time_is_asked_for_without_rerouting(make_agent):
    backend = Scripted(intent="create_reminder", **NO_FACT)
    agent = make_agent(backend)
    assert agent.handle(Turn("提醒我交作业")).text.startswith("要在什么时候提醒你交作业？")
    assert agent.handle(Turn("明天下午3点")).text == "好的，明天 9月23日（周三）15:00 提醒你：交作业"
    assert len(backend.calls) == 1


def test_past_time_is_refused(make_agent):
    agent = make_agent(Scripted(intent="create_reminder", **NO_FACT))
    assert "已经过去了" in agent.handle(Turn("今天上午9点提醒我开会")).text
    assert agent.reminders.pending() == []


def test_jev_picks_between_several_times(make_agent):
    agent = make_agent(Scripted(intent="create_reminder", when=option_containing("周四"), **NO_FACT))
    agent.handle(Turn("周五上午10点面试，周四晚上8点提醒我准备"))
    [r] = agent.reminders.pending()
    assert (r.title, r.due.day, r.due.hour) == ("准备", 24, 20)


def test_reminder_that_states_a_fact_is_also_memorised(make_agent):
    agent = make_agent(Scripted(intent="create_reminder", states_fact=0.9, fact_category="important_date",
                                clause=option_containing("生日")))
    reply = agent.handle(Turn("明天是我妈生日，提醒我买花"))
    assert reply.text == "好的，明天 9月23日（周三）09:00 提醒你：买花\n（你没说几点，先定在 09:00，不对的话告诉我）\n记住了：9月23日是我妈生日"
    [f] = agent.memory.current()
    assert (f.category, f.text) == ("important_date", "9月23日是我妈生日")


def test_low_confidence_asks_then_follows_the_choice(make_agent):
    probs = {"remember_fact": 0.4, "recall": 0.35, "other": 0.25}
    backend = Scripted(intent=("remember_fact", 0.4, probs), states_fact=0.5, fact_category="health")
    agent = make_agent(backend)
    reply = agent.handle(Turn("花生"))
    assert "1. 让我记住一件事" in reply.text and "2. 问我记得的事" in reply.text
    assert agent.handle(Turn("1")).text == "记住了：花生"
    assert len(backend.calls) == 1


def test_update_closes_the_old_fact(make_agent):
    agent = make_agent(Scripted(intent="remember_fact", states_fact=0.95, fact_category="location", rel="updates", same=0.9))
    agent.handle(Turn("我住在Newtown"))
    assert agent.handle(Turn("我搬到Glebe了")).text == "更新了记忆：「我住在Newtown」→「我搬到Glebe了」"
    [current] = agent.memory.current()
    old = next(f for f in agent.memory.history() if f.text == "我住在Newtown")
    assert current.text == "我搬到Glebe了"
    assert old.valid_to is not None and old.superseded_by == current.id


def test_duplicate_is_not_stored_twice(make_agent):
    agent = make_agent(Scripted(intent="remember_fact", states_fact=0.95, fact_category="location", rel="duplicate"))
    agent.handle(Turn("我住在Newtown"))
    assert agent.handle(Turn("记住：我住在Newtown")).text == "这个我已经记着了：我住在Newtown"
    assert len(agent.memory.history()) == 1


def test_recall_answers_with_the_stored_words(make_agent):
    backend = Scripted(intent="remember_fact", states_fact=0.9, fact_category="health", rel="unrelated")
    agent = make_agent(backend)
    agent.handle(Turn("我对花生过敏"))
    agent.handle(Turn("我喜欢吃辣"))
    backend.rules |= {"intent": "recall", "answer": option_containing("过敏")}
    assert agent.handle(Turn("我对什么过敏？")).text == "我记得：我对花生过敏（9月22日记下的）"


def test_cancel_picks_the_named_reminder(make_agent):
    backend = Scripted(intent="create_reminder", **NO_FACT)
    agent = make_agent(backend)
    agent.handle(Turn("明天下午三点提醒我交作业"))
    agent.handle(Turn("周五晚上8点提醒我给妈妈打电话"))
    backend.rules |= {"intent": "cancel_reminder", "target": option_containing("作业")}
    assert agent.handle(Turn("作业交了，不用提醒了")).text.startswith("已取消：交作业")
    assert [r.title for r in agent.reminders.pending()] == ["给妈妈打电话"]


def test_low_confidence_voice_fact_is_flagged(make_agent):
    agent = make_agent(Scripted(intent="remember_fact", states_fact=0.9, fact_category="health"))
    reply = agent.handle(Turn("我对花生过敏", source="voice", asr_confidence=0.4))
    assert "语音识别置信度较低" in reply.text
    assert agent.memory.current()[0].needs_confirmation


def test_jev_outage_is_reported_and_logged(make_agent, conn):
    class Down:
        name = "down"

        def system_one(self, state, questions):
            raise DecisionError("连不上 Jev（网络或超时）")

    agent = make_agent(Down())
    assert agent.handle(Turn("你好")).text == "抱歉，现在做不了判断：连不上 Jev（网络或超时）"
    row = conn.execute("SELECT ok, error FROM model_calls").fetchone()
    assert (row["ok"], row["error"]) == (0, "连不上 Jev（网络或超时）")
