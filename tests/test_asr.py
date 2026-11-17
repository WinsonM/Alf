from types import SimpleNamespace

from alf import asr


def test_prompt_carries_vocabulary_in_simplified_chinese():
    assert asr.build_prompt(["COMP5318", "Glebe"]) == "以下是普通话和英语混合的日常对话，使用简体中文。可能提到：COMP5318、Glebe。"


def test_confidence_is_duration_weighted():
    segs = [SimpleNamespace(start=0, end=3, avg_logprob=-0.1), SimpleNamespace(start=3, end=4, avg_logprob=-1.0)]
    assert asr.confidence(segs) == 0.723  # exp((3 * -0.1 + 1 * -1.0) / 4)
    assert asr.confidence([]) == 0.0


def test_vocabulary_merges_user_terms_and_known_names(conn):
    conn.execute("INSERT INTO vocabulary(term, created_at) VALUES ('Winson', 0)")
    conn.execute("INSERT INTO facts(category, text, valid_from, recorded_at) VALUES ('study_work', '我这学期在上COMP5318', 0, 0)")
    conn.execute("INSERT INTO reminders(title, due_at, notify_at, tz, created_at) VALUES ('去Glebe取快递', 0, 0, 'UTC', 0)")
    assert asr.vocabulary(conn) == ["Winson", "COMP5318", "Glebe"]


def test_vocabulary_includes_the_users_own_topic_and_item_names(conn):
    """实测（2026-11-17，small 模型）：不提示时"上斜哑铃卧推"被听成"上鞋雅灵握推"；把记忆里的动作名放进提示就对了。"""
    for i, (text, topic, slot) in enumerate([("上斜哑铃卧推单边 30kg", "健身", "上斜哑铃卧推"), ("绳索后拉 20kg", "健身", "绳索后拉"),
                                             ("11月21日约了leo去徒步", None, None), ("做了 3 sets and 10 reps", None, None)]):
        conn.execute("INSERT INTO facts(category, text, valid_from, recorded_at, topic, slot) VALUES ('other', ?, 0, ?, ?, ?)", (text, i, topic, slot))
    conn.execute("INSERT INTO facts(category, text, valid_from, valid_to, recorded_at, topic, slot) VALUES ('other', '旧的', 0, 1, 0, '读书', '第一章')")
    # 主题名、每一项的叫法、小写的人名（leo）都在；"kg""sets""reps""and"这些不是名字；已经过时的记忆不算
    assert asr.vocabulary(conn) == ["健身", "绳索后拉", "上斜哑铃卧推", "leo"]


def test_known_spellings_are_restored_in_the_transcript():
    assert asr.canonical("周五下午5点考comP5999,提前一个半小时提醒我", ["COMP5999", "leo", "健身"]) == "周五下午5点考COMP5999,提前一个半小时提醒我"
    assert asr.canonical("ask Leo about napoleon", ["leo"]) == "ask leo about napoleon"      # 整个词才换，词中间的不动
    assert asr.canonical("it is ok", ["ok"]) == "it is ok"                                  # 太短的不动
