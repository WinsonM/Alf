from datetime import datetime, timedelta

import pytest

from alf import timeparse as tp
from evals.run import check_timeparse, load

from .conftest import NOW, TZ

CASES = load("timeparse")


@pytest.mark.parametrize("case", CASES, ids=[f"{c['text']}@{c.get('now', '')}" for c in CASES])
def test_golden_case(case):
    assert check_timeparse(case) == []


def test_cn_int():
    assert [tp.cn_int(s) for s in ["十", "十五", "二十", "二十三", "两", "零五", "08", "三"]] == [10, 15, 20, 23, 2, 5, 8, 3]


def test_daily_recurrence_keeps_wall_clock_across_dst():
    before = datetime(2026, 10, 3, 9, 0, tzinfo=TZ)  # Sydney DST starts 2026-10-04 02:00
    nxt = tp.next_occurrence("DAILY", 8, 0, after=before)
    assert (nxt.month, nxt.day, nxt.hour, nxt.minute) == (10, 4, 8, 0)
    assert before.utcoffset() == timedelta(hours=10) and nxt.utcoffset() == timedelta(hours=11)


def test_relative_offset_counts_real_time_across_dst():
    now = datetime(2026, 10, 4, 1, 30, tzinfo=TZ)  # 30 min before clocks jump 02:00 -> 03:00
    [t] = tp.parse("一小时后提醒我", now).times
    assert t.when.timestamp() - now.timestamp() == 3600
    assert (t.when.hour, t.when.minute) == (3, 30)


def test_absolutize_only_rewrites_relative_dates():
    assert tp.absolutize("明天是我妈生日", NOW) == "9月23日是我妈生日"
    assert tp.absolutize("我老婆生日是3月14号", NOW) == "我老婆生日是3月14号"
    assert tp.absolutize("我每周三有瑜伽课", NOW) == "我每周三有瑜伽课"


def test_describe():
    assert tp.describe(datetime(2026, 9, 23, 15, 0, tzinfo=TZ), NOW) == "明天 9月23日（周三）15:00"
    assert tp.describe(datetime(2027, 3, 14, 9, 0, tzinfo=TZ), NOW) == "2027年3月14日（周日）09:00"
    assert tp.describe_recurrence("WEEKLY:0,2,4") == "每周一、三、五"


@pytest.mark.parametrize("original, text, expected", [
    ((23, 15, 0), "四点", (23, 16, 0)),                 # 只说钟点：留在原来那天，上午下午跟着原来的
    ((23, 15, 0), "改到下午四点半", (23, 16, 30)),
    ((23, 15, 0), "上午9点", (23, 9, 0)),
    ((22, 20, 0), "9点", (22, 21, 0)),                  # 原来是晚上 8 点，"9点"是晚上 9 点
    ((23, 8, 0), "9点", (23, 9, 0)),
    ((23, 15, 0), "后天", (24, 15, 0)),                 # 只说日期：保持原来的钟点
    ((23, 15, 0), "周五", (25, 15, 0)),
    ((23, 15, 0), "后天上午十点", (24, 10, 0)),          # 都说了：照说的来
    ((23, 15, 0), "半小时后", (22, 10, 30)),             # 相对现在的偏移
])
def test_reschedule_reads_the_new_time_relative_to_the_old_one(original, text, expected):
    day, hour, minute = original
    when, _ = tp.reschedule(datetime(2026, 9, day, hour, minute, tzinfo=TZ), text, NOW)
    assert (when.day, when.hour, when.minute) == expected


def test_reschedule_with_only_a_lead_keeps_the_time_and_reports_the_lead():
    """"提前一个半小时提醒"改的是提前量，不是事情的时间。"""
    original = datetime(2026, 9, 25, 17, 0, tzinfo=TZ)
    when, parsed = tp.reschedule(original, "提前一个半小时提醒", NOW)
    assert when == original and parsed.lead_minutes == 90 and not parsed.times


def test_reschedule_without_a_time_returns_nothing():
    assert tp.reschedule(datetime(2026, 9, 23, 15, 0, tzinfo=TZ), "随便吧", NOW) is None


@pytest.mark.parametrize("text, leftover", [
    ("明天下午三点", None), ("每周一三五晚上7点，提前半小时", None), ("改到下午四点", None), ("tomorrow at 9:30am", None),
    ("冬至那天下午三点", "冬 那天"), ("明天下午三点提醒我交作业", "交作业"), ("下周三考完试之后", "考完试之后"), ("发工资那天", "发工资那天"),
])
def test_anything_the_parser_did_not_understand_is_reported(text, leftover):
    assert tp.unparsed_time(text, tp.parse(text, NOW)) == leftover


@pytest.mark.parametrize("said, stored", [
    ("昨天和 leo 去徒步了", "9月21日和 leo 去徒步了"),
    ("前天考完试了", "9月20日考完试了"),
    ("大前天搬的家", "9月19日搬的家"),
    ("昨晚没睡好", "9月21日晚上没睡好"),
    ("以前天气好的时候常去", "以前天气好的时候常去"),          # "以前天气"里的"前天"不是日子
    ("昨天说好明天去", "9月21日说好9月23日去"),
])
def test_days_already_gone_are_stored_as_dates_too(said, stored):
    """记一件发生过的事（经历）时，"昨天"也要换成日期，否则过几天再看就不知道是哪天了。"""
    assert tp.absolutize(said, NOW) == stored
