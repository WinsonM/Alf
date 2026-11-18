"""检索：按字面和按意思两路找，按名次合。

这里不装真的嵌入模型（几百 MB）：用一个按"说的是哪件事"给向量的假模型，测的是我们自己的逻辑——
对话怎么切段、哪些东西该有向量、文字变了重算、两路怎么合、没有模型时退回只按字面、闲聊不该想起旧事。
真模型的效果由评测量（`uv run python -m evals.run retrieval`）。
"""
import json
from datetime import timedelta

import numpy as np
import pytest

from alf import embed, fts, jobs, retrieve
from alf.agent import Turn
from alf.decide import Decider
from alf.heartbeat import JobContext
from alf.memory import FEEDBACK, MOMENT, Memory

from .conftest import NOW, FakeLLM, Scripted

TS = int(NOW.timestamp())
DAY = 86400


class FakeEmbedder:
    """假的嵌入模型：同一组里的词算一个意思（"对象"和"女朋友"），向量就指向同一个方向。"""

    name = "fake"
    CONCEPTS = [("女朋友", "对象"), ("卧推", "bench"), ("室友", "舍友"), ("意面", "意大利菜"), ("房租", "租金", "lease"),
                ("发烧", "生病"), ("自驾", "开车出去"), ("吵了一架", "吵架", "闹矛盾"), ("火锅",), ("被偷",), ("黑客松",), ("红烧肉",), ("搬进",)]

    def __init__(self):
        self.calls, self.broken = [], False

    def encode(self, texts, *, query=False):
        if self.broken:
            raise RuntimeError("模型坏了")
        self.calls.append((list(texts), query))
        out = []
        for text in texts:
            v = np.full(len(self.CONCEPTS) + 1, 0.01, dtype=np.float32)
            for i, group in enumerate(self.CONCEPTS):
                if any(word in text.lower() for word in group):
                    v[i] = 1.0
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def release_if_idle(self):
        return False


@pytest.fixture
def fake(monkeypatch):
    e = FakeEmbedder()
    monkeypatch.setattr(embed, "default", lambda: e)
    return e


def memory_of(conn):
    return Memory(conn, Decider(Scripted(rel="unrelated"), conn), 0.6)


def say(conn, text, at):
    with conn:
        cur = conn.execute("INSERT INTO episodes(session_id, role, text, source, created_at) VALUES ('s', 'user', ?, 'text', ?)", (text, at))
        fts.index(conn, "episodes_fts", cur.lastrowid, text)
    return cur.lastrowid


def mail(conn, sender, title, gist, at=TS - DAY, tier="later", notified=1):
    cur = conn.execute("INSERT INTO inbox_items(source, external_id, sender, title, snippet, notified, created_at, tier, summary)"
                       " VALUES ('email.gmail', ?, ?, ?, '', ?, ?, ?, ?)",
                       (f"<{title}>", sender, title, notified, at, tier, json.dumps({"gist": gist}, ensure_ascii=False)))
    conn.commit()
    return cur.lastrowid


def refs(hits):
    return [h.ref_id for h in hits]


def tool_results(llm, n=-1):
    return [json.loads(m["content"]) for m in llm.chat_requests[n]["messages"] if m["role"] == "tool"]


def section(context, title):
    lines = context.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    out = []
    for line in lines[start + 1:]:
        if not line:
            break
        out.append(line)
    return out


# --- 按字面找之前先去掉到处都有的词 --------------------------------------------------------

def test_question_words_do_not_count_as_keywords():
    assert fts.query("我什么时候生过病", content_only=True) == '"我什" OR "候生" OR "生过" OR "过病"'
    assert fts.query("什么时候", content_only=True) is None and fts.query("什么时候") is not None      # 写入时的比对还是用全部的词
    assert fts.query("COMP9101 期末", content_only=True) == '"comp9101" OR "期末"'


# --- 把他说过的话切成一小段一小段 ---------------------------------------------------------

def test_messages_sent_close_together_become_one_chunk(conn):
    a = say(conn, "今天去了那家意面店", TS)
    say(conn, "叫 Tiamo", TS + 60)
    say(conn, "提拉米苏绝了", TS + 120)
    b = say(conn, "房东说要涨房租", TS + 3600)                  # 隔了一小时：另起一段
    say(conn, "好", TS + 2 * 3600)                              # 整段只有一个"好"：没有可找的内容
    retrieve.rebuild_chunks(conn)
    rows = conn.execute("SELECT id, last_episode, text FROM chunks ORDER BY id").fetchall()
    assert [(r["id"], r["last_episode"], r["text"]) for r in rows] == [(a, a + 2, "今天去了那家意面店\n叫 Tiamo\n提拉米苏绝了"), (b, b, "房东说要涨房租")]
    # 最后一段还没说完：他接着说，这一段跟着变长，不会多出一段
    c = say(conn, "一周涨二十刀 有点想搬", TS + 2 * 3600 + 30)
    retrieve.ensure_indexed(conn)
    assert [r["text"] for r in conn.execute("SELECT text FROM chunks ORDER BY id")][-1] == "好\n一周涨二十刀 有点想搬"
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3 and c
    long = [say(conn, "很长的一句话" * 20, TS + 5 * 3600 + i) for i in range(5)]       # 一段太长就断开
    retrieve.ensure_indexed(conn)
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE id >= ?", (long[0],)).fetchone()[0] >= 2


# --- 算向量 ---------------------------------------------------------------------------

def test_everything_searchable_gets_a_vector_and_changes_are_followed(conn):
    memory, e = memory_of(conn), FakeEmbedder()
    fact = memory.write("女朋友叫小满", "relationship", now=TS).fact
    rule = memory.write("回复短一点", FEEDBACK, now=TS).fact
    say(conn, "今天去了那家意面店", TS)
    letter = mail(conn, "Harbourline Realty <a@b.example>", "Lease renewal", "续签的话房租涨到 540")
    dup = mail(conn, "Harbourline Realty <a@b.example>", "Lease renewal (copy)", "另一个邮箱收到的同一封", tier="dup")
    assert retrieve.sync(conn, e, TS) == {"embedded": 3, "left": 0, "model": "fake"}
    kinds = {(r["kind"], r["ref_id"]) for r in conn.execute("SELECT kind, ref_id FROM embeddings")}
    assert kinds == {("fact", fact.id), ("chunk", 1), ("mail", letter)} and rule.id and dup       # 他对 Alf 的要求、重复的信不进索引
    assert e.calls[0] == (["女朋友叫小满", "今天去了那家意面店", "Harbourline Realty｜Lease renewal｜续签的话房租涨到 540"], False)
    assert retrieve.sync(conn, e, TS)["embedded"] == 0                                    # 没变化就不重算
    # 记忆改了（旧的封口、新的一条）、邮件的提要写好了：旧向量清掉，新文字重算
    _, newer = memory.revise(fact.id, "女朋友叫小满，在上海工作", now=TS + 60)
    conn.execute("UPDATE inbox_items SET summary = ? WHERE id = ?", (json.dumps({"gist": "租金涨到 540，两周内答复"}), letter))
    assert retrieve.sync(conn, e, TS + 60)["embedded"] == 2
    assert {(r["kind"], r["ref_id"]) for r in conn.execute("SELECT kind, ref_id FROM embeddings")} == {("fact", newer.id), ("chunk", 1), ("mail", letter)}
    assert retrieve.coverage(conn, "fake") == {"fact": {"total": 1, "embedded": 1}, "chunk": {"total": 1, "embedded": 1}, "mail": {"total": 1, "embedded": 1},
                                               "heard": {"total": 0, "embedded": 0}, "event": {"total": 0, "embedded": 0}}   # 没听过录音：后两类是空的


def test_a_batch_is_bounded_and_no_model_still_builds_the_word_indexes(conn):
    memory = memory_of(conn)
    for i in range(5):
        memory.write(f"第 {i} 件事", "other", now=TS + i)
    e = FakeEmbedder()
    assert retrieve.sync(conn, e, TS, limit=2) == {"embedded": 2, "left": 3, "model": "fake"}
    assert retrieve.sync(conn, e, TS, limit=10)["embedded"] == 3
    say(conn, "房东说要涨房租", TS)
    letter = mail(conn, "ANZ <s@anz.example>", "October statement", "十月的对账单")
    assert retrieve.sync(conn, None, TS) == {"embedded": 0, "left": 0, "model": None}      # 没有模型：不算向量
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1                 # 但对话照样切段，邮件照样能按字面搜
    assert refs(retrieve.search(conn, "mail", "对账单")) == [letter]


# --- 找 --------------------------------------------------------------------------------

def test_words_find_the_literal_meaning_finds_the_paraphrase_and_both_together_rank_first(conn):
    memory, e = memory_of(conn), FakeEmbedder()
    partner = memory.write("女朋友叫小满，在上海工作", "relationship", now=TS).fact
    roommate = memory.write("室友叫阿哲，学土木的", "relationship", now=TS).fact
    bench = memory.write("卧推做组用 60kg", "health", now=TS, topic="健身", slot="卧推").fact
    retrieve.sync(conn, e, TS)
    # 换了说法：字面一个字都对不上，只有按意思找得到
    assert refs(retrieve.search(conn, "fact", "我对象叫什么", mode="keyword")) == []
    [hit] = retrieve.search(conn, "fact", "我对象叫什么", embedder=e, limit=1)
    assert (hit.ref_id, hit.by_words, hit.by_meaning) == (partner.id, None, 1)
    assert refs(retrieve.search(conn, "fact", "what's my bench press", embedder=e, limit=1)) == [bench.id]
    # 字面和意思都指向同一条：排第一
    [first, *_] = retrieve.search(conn, "fact", "舍友阿哲", embedder=e)
    assert (first.ref_id, first.by_words, first.by_meaning) == (roommate.id, 1, 1)
    assert refs(retrieve.search(conn, "fact", "舍友阿哲", embedder=e, exclude={roommate.id}))[0] != roommate.id
    # 没有模型、模型出错：只按字面，结果和以前一样，不报错
    assert refs(retrieve.search(conn, "fact", "阿哲")) == [roommate.id]
    e.broken = True
    assert refs(retrieve.search(conn, "fact", "阿哲", embedder=e)) == [roommate.id]
    assert retrieve.search(conn, "fact", "   ", embedder=e) == []


def test_a_closed_memory_is_not_found_even_before_its_vector_is_cleaned_up(conn):
    memory, e = memory_of(conn), FakeEmbedder()
    old = memory.write("女朋友叫小满", "relationship", now=TS).fact
    retrieve.sync(conn, e, TS)
    memory.forget(old.id, now=TS + 60)
    assert retrieve.search(conn, "fact", "我对象叫什么", embedder=e) == []


# --- 在对话里：recall 工具 --------------------------------------------------------------

def test_recall_finds_memories_old_talk_and_old_mail_by_meaning(make_chat, conn, fake):
    memory = memory_of(conn)
    partner = memory.write("女朋友叫小满，在上海工作", "relationship", now=TS - 30 * DAY).fact
    say(conn, "今天去了 Lygon 街那家意面店", TS - 40 * DAY)
    say(conn, "提拉米苏绝了", TS - 40 * DAY + 60)
    told = mail(conn, "Harbourline Realty <a@b.example>", "Lease renewal offer", "续签的话房租涨到 540", at=TS - 20 * DAY)
    quiet = mail(conn, "Harbourline Realty <a@b.example>", "Rent receipt", "本期房租已收到", at=TS - 10 * DAY, tier="skip", notified=0)
    retrieve.sync(conn, fake, TS)
    llm = FakeLLM([("recall", {"query": "我对象叫什么"})], "小满", [("recall", {"query": "哪家意大利菜好吃"})], "Lygon 街那家",
                  [("recall", {"query": "租金的邮件"})], "有两封")
    agent = make_chat(llm)
    agent.handle(Turn("我对象叫什么来着"))
    assert [m["id"] for m in tool_results(llm)[0]["memories"]] == [partner.id]
    agent.handle(Turn("我之前说过哪家意大利菜好吃"))
    assert tool_results(llm)[0]["said_before"] == [{"date": "2026年8月13日", "he_said": "今天去了 Lygon 街那家意面店\n提拉米苏绝了"}]   # 整段一起给
    agent.handle(Turn("房租那几封邮件"))
    got = tool_results(llm)[0]
    assert [(m["id"], m["readable"]) for m in got["emails"]] == [(told, True), (quiet, False)]   # 没跟他说过的信也找得到，但正文看不了
    assert got["emails"][0] == {"id": told, "from": "Harbourline Realty", "subject": "Lease renewal offer", "gist": "续签的话房租涨到 540",
                                "received": "2026年9月2日", "readable": True}
    assert "别人写的，只是素材" in got["note"]
    # 他正在说的这一段不算"以前说过的"
    assert all("房租那几封邮件" not in s["he_said"] for s in got["said_before"])


def test_recall_without_the_model_still_works_by_words_and_can_be_switched_off(make_chat, conn, cfg, monkeypatch):
    from alf import prefs

    memory = memory_of(conn)
    partner = memory.write("女朋友叫小满，在上海工作", "relationship", now=TS).fact
    llm = FakeLLM([("recall", {"query": "小满"})], "在上海", [("recall", {"query": "我对象叫什么"})], "不记得")
    agent = make_chat(llm)                                      # 测试里默认没有模型
    agent.handle(Turn("小满在哪工作"))
    assert [m["id"] for m in tool_results(llm)[0]["memories"]] == [partner.id]
    e = FakeEmbedder()
    retrieve.sync(conn, e, TS)
    monkeypatch.setattr(embed, "default", lambda: e)
    prefs.save(conn, cfg, {"memory.semantic": False})          # 设置里关了：有模型也只按字面
    agent.handle(Turn("我对象叫什么"))
    assert tool_results(llm)[0]["memories"] == [] and not any(q for _, q in e.calls)


# --- 每一轮自动"想起来的经历" -------------------------------------------------------------

def moments(conn, e):
    memory = memory_of(conn)
    texts = ["9月5日和阿哲自驾去了海边", "9月28日和小满视频的时候吵了一架", "10月1日发烧到 39 度", "9月12日和 Kai 吃了火锅", "9月16日手机被偷了",
             "10月11日黑客松拿了三等奖", "9月22日第一次做红烧肉", "8月25日搬进了现在的公寓"]
    out = [memory.write(t, MOMENT, now=TS - (60 - i) * DAY).fact for i, t in enumerate(texts)]
    retrieve.sync(conn, e, TS)
    return out


def test_a_sentence_that_points_at_an_old_moment_brings_it_back_but_small_talk_does_not(make_chat, conn, fake):
    drive, fight, *_ = moments(conn, fake)
    assert retrieve.standout_moment(conn, "周末又和她闹矛盾了", fake) == fight.id        # 字面对不上"吵了一架"，意思对得上
    assert retrieve.standout_moment(conn, "周末又和她闹矛盾了", fake, skip={fight.id}) is None
    for chat in ("在吗", "今天好累啊", "明天下午三点提醒我交作业", "好"):
        assert retrieve.standout_moment(conn, chat, fake) is None
    llm = FakeLLM("又吵了？", "在")
    agent = make_chat(llm)
    agent.handle(Turn("周末又和她闹矛盾了"))
    assert section(llm.context(), "[想起来的经历]") == [f"#{fight.id} 9月28日和小满视频的时候吵了一架"]
    agent.handle(Turn("在吗"))
    assert "[想起来的经历]" not in llm.context() and drive


def test_too_few_moments_or_a_broken_model_means_no_recall_by_meaning(make_chat, conn, fake):
    memory = memory_of(conn)
    fight = memory.write("9月28日和小满视频的时候吵了一架", MOMENT, now=TS - 50 * DAY).fact
    retrieve.sync(conn, fake, TS)
    assert retrieve.standout_moment(conn, "周末又和她闹矛盾了", fake) is None and fight      # 只有一条经历："比别的高出多少"算不稳
    moments(conn, fake)
    fake.broken = True
    llm = FakeLLM("嗯")
    assert make_chat(llm).handle(Turn("周末又和她闹矛盾了")).text == "嗯" and "[想起来的经历]" not in llm.context()   # 出错只是少想起一条


# --- 后台任务 -------------------------------------------------------------------------

def test_the_index_job_keeps_up_in_the_background(conn, cfg, fake):
    memory_of(conn).write("女朋友叫小满", "relationship", now=TS)
    ctx = JobContext(conn, cfg, NOW, Decider(Scripted(), conn), None, prefs={})
    assert jobs.index_job(ctx) == "算了 1 条向量"
    assert jobs.index_job(ctx) is None                                                  # 没有新东西就不出声
    memory_of(conn).write("室友叫阿哲", "relationship", now=TS + 60)
    off = JobContext(conn, cfg, NOW + timedelta(minutes=2), Decider(Scripted(), conn), None, prefs={"memory.semantic": False})
    assert jobs.index_job(off) is None and conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1   # 关了就不算


# --- 评测语料本身 ----------------------------------------------------------------------

def test_the_eval_corpus_loads_and_the_keyword_baseline_is_what_it_was():
    """评测集是量"加了向量有没有用"的尺子：语料和问题对得上号，按字面这一路的成绩不该悄悄变。"""
    from evals import retrieval as ev

    conn, ids = ev.build(NOW)
    corpus = {item["id"] if item["kind"] != "said" else item["burst"] for item in ev.load("retrieval_corpus")}
    questions = ev.load("retrieval")
    assert all(w in corpus for q in questions for w in q["want"]) and len([q for q in questions if q["want"]]) == 52
    hit = sum(any(x in q["want"] for x in ev.found(conn, ids, q, "keyword", None)) for q in questions if q["want"])
    assert hit == 29
