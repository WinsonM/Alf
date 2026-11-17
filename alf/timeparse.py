"""Deterministic date/time understanding for reminders (Chinese first, basic English).

Jev "reads dates as text, not as ordered quantities" and "is not a calculator"
(TypeSafe's jev-1.13 limitations page), so every piece of date arithmetic lives
here. The parser is tuned for recall: it finds every time expression in a message
and resolves each one to an absolute datetime. When a message has several, Jev is
only asked which span is the reminder time; code owns the value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo

_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
       "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6, "七": 6,
           "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}
EN_WEEKDAY = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
              "friday": 4, "saturday": 5, "sunday": 6}
WEEKDAY_CN = "一二三四五六日"

# period -> (hour used when no clock time is given, how a 1-12 clock hour shifts)
PERIODS = {
    "凌晨": (5, "am"), "清晨": (7, "am"), "早上": (8, "am"), "早晨": (8, "am"),
    "上午": (9, "am"), "中午": (12, "noon"), "下午": (15, "pm"), "傍晚": (18, "pm"),
    "晚上": (20, "pm"), "夜里": (22, "pm"), "深夜": (23, "pm"),
}
_LATE = {"晚上", "夜里", "深夜"}  # "晚上12点" is midnight, not noon

_RELDAY = {"今天": 0, "今日": 0, "今儿": 0, "今早": 0, "今晚": 0, "today": 0, "tonight": 0,
           "明天": 1, "明日": 1, "明儿": 1, "明早": 1, "明晚": 1, "tomorrow": 1,
           "后天": 2, "大后天": 3}
_RELDAY_PERIOD = {"今早": "早上", "明早": "早上", "今晚": "晚上", "明晚": "晚上", "tonight": "晚上"}

_NUM = r"(?:\d{1,2}|[零〇一二两三四五六七八九十]{1,3})"
# An hour. A bare "一" is excluded after words like 快/早/多 so "快一点" (hurry up) is not 1 o'clock.
_HOUR = r"(?:\d{1,2}|(?<![快早晚多少好大小慢高低远近差轻重有这那])一|[零〇两二三四五六七八九十][零〇一二三四五六七八九十]{0,2})"
_WD = r"[一二三四五六日天七1-7]"
_EN_WD = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_PERIOD = "|".join(PERIODS)

_DATE = rf"""
    (?P<relday>大后天|后天|今天|今日|今儿|明天|明日|明儿|今早|今晚|明早|明晚
        |(?<![a-z])(?:today|tonight|tomorrow)(?![a-z]))
  | (?P<week>(?P<wkp>下下个?|下个?|这个?|本|上个?)?(?:周|星期|礼拜)(?P<wd>{_WD}))
  | (?P<enweek>(?<![a-z])(?:(?P<enwkp>next|this)\s+)?(?P<enwd>{_EN_WD})(?![a-z]))
  | (?P<iso>(?<!\d)(?P<iy>\d{{4}})[-/.](?P<imo>\d{{1,2}})[-/.](?P<id>\d{{1,2}})(?!\d))
  | (?P<md>(?:(?P<y>\d{{4}})年)?(?P<mo>{_NUM})月(?P<d>{_NUM})[日号]?)
  | (?P<mend>(?:(?P<mendmo>{_NUM})月|(?P<mendwhich>下个?月|本月|这个?月)|月)[底末])
  | (?P<mrel>(?P<mwhich>下个?月|本月|这个?月)(?P<d3>{_NUM})[日号]?)
  | (?P<dom>(?P<d2>{_NUM})号)
  | (?P<indays>(?P<nd>{_NUM})天(?:之)?后)
"""
_CLOCK = rf"""
    (?P<enclock>(?<![\d:])(?P<eh>\d{{1,2}})(?::(?P<em>\d{{2}}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)(?![a-z]))
  | (?P<h>{_HOUR})\s*(?P<sep>点钟?|时|[:：])\s*
    (?:(?P<half>半)|(?P<quarter>一刻|三刻)|(?P<mi>\d{{1,2}}|[零〇一二三四五六七八九十]{{1,3}})\s*分?)?
    (?:整)?
"""
_EXPR = re.compile(rf"""
    (?:{_DATE})?
    \s*的?\s*
    (?P<period>{_PERIOD})?
    \s*(?:at\s+)?
    (?:{_CLOCK})?
""", re.X | re.I)
_EXPR_KEYS = ("relday", "week", "enweek", "iso", "md", "mend", "mrel", "dom", "indays", "period", "h", "enclock")

_REL = re.compile(r"""
    (?P<rn>\d+|[零〇一二两三四五六七八九十]+|半)\s*个?\s*(?P<rhalf>半)?\s*(?P<ru>分钟|分|小时|钟头)\s*(?:之)?后
  | (?<![a-z])in\s+(?P<ern>\d+|an?|half\s+an?)\s*(?P<eru>minutes?|mins?|hours?|hrs?|h)(?![a-z])
""", re.X | re.I)

_LEAD = re.compile(r"提前\s*(?P<ln>\d+|[零〇一二两三四五六七八九十]+|半)?\s*个?\s*(?P<lhalf>半)?\s*(?P<lu>分钟|分|小时|钟头|天)")

_REC = re.compile(rf"""
    (?:
        (?P<daily>每天|每日|天天|每晚|每早
            |(?<![a-z])(?:every\s*day|daily|every\s+night|every\s+morning)(?![a-z]))
      | (?P<workday>每个?工作日|工作日|(?<![a-z])(?:every\s+)?weekdays(?![a-z]))
      | (?P<weekly>每个?(?:周|星期|礼拜)(?P<rwd>{_WD}(?:[、,，和]?{_WD})*))
      | (?P<enweekly>(?<![a-z])every\s+(?P<erwd>{_EN_WD})(?![a-z]))
    )
    \s*的?\s*
    (?P<period>{_PERIOD})?
    \s*(?:at\s+)?
    (?:{_CLOCK})?
""", re.X | re.I)


@dataclass
class TimeExpr:
    span: str
    start: int
    end: int
    when: datetime                 # tz-aware, in the caller's zone
    date_kind: str | None = None   # which date form was used; None = time-only / relative
    time_defaulted: bool = False   # no clock time given: used the period's or the default hour
    ampm_guessed: bool = False     # bare hour like "3点", read as 下午
    relative: bool = False         # an offset from now ("半小时后"), not a clock time

    @property
    def has_date(self) -> bool:
        return self.date_kind is not None


@dataclass
class Parsed:
    times: list[TimeExpr] = field(default_factory=list)
    lead_minutes: int = 0
    recurrence: str | None = None  # "DAILY" | "WEEKLY:0,2,4" (Mon=0)
    spans: list[tuple[int, int]] = field(default_factory=list)  # everything consumed, for title()


class _Invalid(Exception):
    """An expression naming an impossible date or time (2月30日, 25点)."""


def cn_int(s: str | None) -> int | None:
    """'15' / '十五' / '二十三' / '两' / '零五' -> int."""
    if not s:
        return None
    s = s.strip()
    if s.isdigit():
        return int(s)
    if len(s) == 2 and s[0] in "零〇":
        return _CN.get(s[1])
    if "十" in s:
        tens, _, ones = s.partition("十")
        t = _CN.get(tens) if tens else 1
        o = _CN.get(ones) if ones else 0
        return None if t is None or o is None else t * 10 + o
    return _CN.get(s) if len(s) == 1 else None


def _at(d: date, hour: int, minute: int, tz: tzinfo) -> datetime:
    if hour >= 24:
        d, hour = d + timedelta(days=hour // 24), hour % 24
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=tz)


def _shift(hour: int, period: str) -> int:
    kind = PERIODS[period][1]
    if kind == "pm":
        if hour == 12 and period in _LATE:
            return 24
        return hour + 12 if hour < 12 else hour
    if kind == "noon":
        return hour + 12 if hour < 11 else hour  # 中午1点 = 13:00
    if kind == "am" and hour == 12:
        return 0
    return hour


def _clock(m: re.Match) -> tuple[int, int, bool] | None:
    """(hour, minute, explicit) where explicit means am/pm or 24h form was unambiguous."""
    if m.group("enclock"):
        h, mi = int(m.group("eh")), int(m.group("em") or 0)
        if h > 12 or mi > 59:
            raise _Invalid
        pm = m.group("ap").lower().startswith("p")
        if pm and h < 12:
            h += 12
        elif not pm and h == 12:
            h = 0
        return h, mi, True
    if m.group("h"):
        h = cn_int(m.group("h"))
        if m.group("half"):
            mi = 30
        elif m.group("quarter"):
            mi = 15 if m.group("quarter") == "一刻" else 45
        else:
            mi = cn_int(m.group("mi")) if m.group("mi") else 0
        if h is None or mi is None or h > 24 or mi > 59:
            raise _Invalid
        zero_padded = m.group("sep") in (":", "：") and len(m.group("h")) == 2  # "08:00"
        return h, mi, zero_padded
    return None


def _time_of_day(m: re.Match, period: str | None, default_hour: int, day_known: bool):
    """(candidate hours, minute, time_defaulted, ampm_guessed), or None if the match names no time."""
    clock = _clock(m)
    if clock:
        h, mi, explicit = clock
        if period:
            return [_shift(h, period)], mi, False, False
        if explicit or h == 0 or h >= 13:
            return [h], mi, False, False
        if h == 12:
            return [12], mi, False, True
        if h <= 6:
            return [h + 12], mi, False, True   # nobody sets a reminder for 3am by saying "3点"
        # "8点" with no date: whichever of 08:00 / 20:00 comes next
        return ([h] if day_known else [h, h + 12]), mi, False, True
    if period:
        return [PERIODS[period][0]], 0, True, False
    if day_known:
        return [default_hour], 0, True, False
    return None


def _safe_date(y: int, mo: int | None, d: int | None) -> date:
    try:
        return date(y, mo, d)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise _Invalid from None


def _date_part(m: re.Match, today: date) -> tuple[date | None, str | None, bool, str | None]:
    """(date, kind, is_bare_weekday, implied_period)."""
    if rd := m.group("relday"):
        rd = rd.lower()
        return today + timedelta(days=_RELDAY[rd]), "relday", False, _RELDAY_PERIOD.get(rd)
    if m.group("week") or m.group("enweek"):
        if m.group("week"):
            wd, prefix = WEEKDAY[m.group("wd")], m.group("wkp") or ""
            weeks = 2 if prefix.startswith("下下") else 1 if prefix.startswith("下") else -1 if prefix.startswith("上") else 0
        else:
            wd, prefix = EN_WEEKDAY[m.group("enwd").lower()], (m.group("enwkp") or "").lower()
            weeks = 1 if prefix == "next" else 0
        if not prefix:  # bare "周三": next occurrence on or after today
            return today + timedelta(days=(wd - today.weekday()) % 7), "week", True, None
        monday = today - timedelta(days=today.weekday())
        return monday + timedelta(days=7 * weeks + wd), "week", False, None
    if m.group("iso"):
        return _safe_date(int(m.group("iy")), int(m.group("imo")), int(m.group("id"))), "iso", False, None
    if m.group("md"):
        mo, d = cn_int(m.group("mo")), cn_int(m.group("d"))
        if m.group("y"):
            return _safe_date(int(m.group("y")), mo, d), "md", False, None
        result = _safe_date(today.year, mo, d)
        if result < today:
            result = _safe_date(today.year + 1, mo, d)
        return result, "md", False, None
    if m.group("mend"):                     # 月底 = the month's last day
        def last_day(y: int, mo: int) -> date:
            return (date(y + 1, 1, 1) if mo == 12 else date(y, mo + 1, 1)) - timedelta(days=1)

        y, mo = today.year, today.month
        if m.group("mendmo"):               # "12月底": that month, next year if it has passed
            mo = cn_int(m.group("mendmo"))
            if mo is None or not 1 <= mo <= 12:
                raise _Invalid
            result = last_day(y, mo)
            return (result if result >= today else last_day(y + 1, mo)), "mend", False, None
        if (m.group("mendwhich") or "").startswith("下"):
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        return last_day(y, mo), "mend", False, None
    if m.group("mrel"):
        y, mo = today.year, today.month
        if m.group("mwhich").startswith("下"):
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        return _safe_date(y, mo, cn_int(m.group("d3"))), "mrel", False, None
    if m.group("dom"):
        d = cn_int(m.group("d2"))
        y, mo = today.year, today.month
        for _ in range(3):  # this month, else the next month that has that day
            try:
                result = date(y, mo, d)  # type: ignore[arg-type]
                if result >= today:
                    return result, "dom", False, None
            except (TypeError, ValueError):
                pass
            y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
        raise _Invalid
    if m.group("indays"):
        n = cn_int(m.group("nd"))
        if n is None:
            raise _Invalid
        return today + timedelta(days=n), "indays", False, None
    return None, None, False, None


def _absolute(m: re.Match, now: datetime, default_hour: int) -> TimeExpr | None:
    today = now.date()
    day, kind, bare_weekday, implied = _date_part(m, today)
    tod = _time_of_day(m, m.group("period") or implied, default_hour, day is not None)
    if tod is None:
        return None
    hours, mi, defaulted, guessed = tod
    if day is None:
        options = sorted(_at(today + timedelta(days=k), h, mi, now.tzinfo) for k in (0, 1) for h in hours)
        when = next(o for o in options if o > now)
    else:
        when = _at(day, hours[0], mi, now.tzinfo)
        if bare_weekday and when <= now:
            when = _at(day + timedelta(days=7), hours[0], mi, now.tzinfo)
    return TimeExpr("", 0, 0, when, kind, defaulted, guessed)


def _amount_minutes(n: str | None, half: str | None, unit: str) -> int:
    qty: float | None = 0.5 if n == "半" else cn_int(n) if n else 0
    if qty is None:
        return 0
    if half:
        qty += 0.5
    return int(qty * {"天": 1440, "小时": 60, "钟头": 60}.get(unit, 1))


def _recurrence(m: re.Match) -> tuple[str, str | None]:
    if word := m.group("daily"):
        word = re.sub(r"\s+", " ", word.lower())
        implied = "晚上" if word in ("每晚", "every night") else "早上" if word in ("每早", "every morning") else None
        return "DAILY", implied
    if m.group("workday"):
        return "WEEKLY:0,1,2,3,4", None
    if m.group("weekly"):
        days = sorted({WEEKDAY[c] for c in m.group("rwd") if c in WEEKDAY})
        return "WEEKLY:" + ",".join(map(str, days)), None
    return f"WEEKLY:{EN_WEEKDAY[m.group('erwd').lower()]}", None


def _recurrence_days(recurrence: str) -> list[int]:
    if recurrence == "DAILY":
        return list(range(7))
    return [int(x) for x in recurrence.split(":", 1)[1].split(",")]


def next_occurrence(recurrence: str, hour: int, minute: int, after: datetime) -> datetime:
    """First hour:minute (wall clock in `after`'s zone) on an allowed weekday, strictly after `after`."""
    days = _recurrence_days(recurrence)
    for k in range(8):
        d = after.date() + timedelta(days=k)
        if d.weekday() in days:
            candidate = _at(d, hour, minute, after.tzinfo)
            if candidate > after:
                return candidate
    raise ValueError(f"bad recurrence {recurrence!r}")


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and (text[a].isspace() or text[a] == "的"):
        a += 1
    while b > a and (text[b - 1].isspace() or text[b - 1] == "的"):
        b -= 1
    return a, b


def parse(text: str, now: datetime, default_hour: int = 9) -> Parsed:
    """Find every time expression in `text`, resolved against `now` (tz-aware)."""
    out = Parsed()

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in out.spans)

    for m in _LEAD.finditer(text):
        if minutes := _amount_minutes(m.group("ln"), m.group("lhalf"), m.group("lu")):
            out.lead_minutes = minutes
            out.spans.append(m.span())
            break

    for m in _REC.finditer(text):
        a, b = _trim(text, *m.span())
        if not free(a, b):
            continue
        try:
            recurrence, implied = _recurrence(m)
            hours, mi, defaulted, guessed = _time_of_day(m, m.group("period") or implied, default_hour, True)
        except _Invalid:
            continue
        first = next_occurrence(recurrence, hours[0], mi, after=now)
        out.recurrence = recurrence
        out.times.append(TimeExpr(text[a:b], a, b, first, None, defaulted, guessed))
        out.spans.append((a, b))
        break  # one recurrence per reminder

    for m in _REL.finditer(text):
        if not free(*m.span()):
            continue
        if m.group("ru"):
            minutes = _amount_minutes(m.group("rn"), m.group("rhalf"), m.group("ru"))
        else:
            n = m.group("ern").lower()
            qty = 0.5 if n.startswith("half") else 1 if n in ("a", "an") else int(n)
            minutes = int(qty * (60 if m.group("eru").lower().startswith("h") else 1))
        if not minutes:
            continue
        # Offsets are real elapsed time, so add them in UTC (a DST jump must not move them).
        when = (now.astimezone(timezone.utc) + timedelta(minutes=minutes)).astimezone(now.tzinfo)
        out.times.append(TimeExpr(m.group(), m.start(), m.end(), when, relative=True))
        out.spans.append(m.span())

    for m in _EXPR.finditer(text):
        a, b = _trim(text, *m.span())
        if a >= b or not free(a, b) or not any(m.group(k) for k in _EXPR_KEYS):
            continue
        try:
            expr = _absolute(m, now, default_hour)
        except _Invalid:
            continue
        if expr:
            expr.span, expr.start, expr.end = text[a:b], a, b
            out.times.append(expr)
            out.spans.append((a, b))

    out.times.sort(key=lambda t: t.start)
    return out


# Words that may sit next to a time phrase without changing which moment it names.
_TIME_FILLER = re.compile(r"[\s，,。.、:：~～!！]+|提醒我?|叫我|通知我|记得|时候|[的在到于至改成为把是]|\b(?:at|on|to|by|the|remind me)\b", re.I)


def unparsed_time(text: str, parsed: Parsed) -> str | None:
    """Whatever is left of a time phrase after removing what the parser understood and harmless filler.

    Anything left means the phrase was only partly understood: "冬至那天下午三点" would otherwise
    quietly become "today 15:00". Callers refuse to act on such a phrase rather than act on half of it.
    """
    chars = list(text)
    for a, b in parsed.spans:
        chars[a:b] = " " * (b - a)
    rest = _TIME_FILLER.sub(" ", "".join(chars))
    return " ".join(rest.split()) or None


_EN_MONTH = "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
_EN_DATE = re.compile(
    r"(?:\b(?:mon|tue(?:s)?|wed(?:nes)?|thu(?:rs?)?|fri|sat(?:ur)?|sun)(?:day)?,?\s+)?"       # a weekday in front adds nothing
    rf"(?:\b(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m1>{_EN_MONTH})\b\.?"
    rf"|\b(?P<m2>{_EN_MONTH})\b\.?\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\b)"
    r"(?:,?\s+(?P<y>\d{4})\b)?,?(?:\s+at\b)?", re.I)
_EN_MONTH_NUMBER = {name: i for i, name in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def english_dates(text: str, today: date) -> str:
    """Rewrite dates written with an English month name ("Friday 20 November 2026", "Nov 20th") as
    2026-11-20, which the parser reads. Without a year: this year, or next if that day has passed.

    Emails state deadlines this way. Numeric forms like 11/20 are left alone on purpose: day-first and
    month-first readers disagree about them, and the caller refuses phrases it does not fully understand.
    """
    def rewrite(m: re.Match) -> str:
        month = _EN_MONTH_NUMBER[(m.group("m1") or m.group("m2"))[:3].lower()]
        day = int(m.group("d1") or m.group("d2"))
        try:
            when = date(int(m.group("y")), month, day) if m.group("y") else date(today.year, month, day)
            if not m.group("y") and when < today:
                when = date(today.year + 1, month, day)
        except ValueError:
            return m.group(0)
        return f"{when.isoformat()} "

    return _EN_DATE.sub(rewrite, text)


def reschedule(original: datetime, text: str, now: datetime, default_hour: int = 9) -> tuple[datetime, Parsed] | None:
    """Move an existing reminder: "改到四点" keeps its day, "改到后天" keeps its time of day.

    A new time is read relative to what the reminder already was, not relative to now:
    tomorrow's 15:00 reminder moved "到四点" means tomorrow 16:00, not today's.
    Returns None when `text` holds no time expression.
    """
    parsed = parse(text, now, default_hour)
    if not parsed.times:
        # "提前一个半小时提醒" changes how early to be told, not when the thing happens.
        return (original, parsed) if parsed.lead_minutes else None
    t = parsed.times[0]
    if parsed.recurrence or t.relative or (t.has_date and not t.time_defaulted):
        return t.when, parsed                      # fully specified, or an offset from now
    if t.has_date:                                 # new day, old time of day
        return t.when.replace(hour=original.hour, minute=original.minute), parsed
    # Time of day only: same day as before. Re-read it from that day's midnight so that
    # "四点" lands on the reminder's own day rather than on today.
    midnight = original.replace(hour=0, minute=0, second=0, microsecond=0)
    t = parse(text, midnight - timedelta(seconds=1), default_hour).times[0]
    when = t.when
    if t.ampm_guessed:
        # "改到8点" for a 20:00 reminder means 20:00: of the two readings, take the one nearer the old time.
        other = when + timedelta(hours=12 if when.hour < 12 else -12)
        if other.date() == when.date() and abs(other - original) < abs(when - original):
            when = other
    return when, parsed


# --- turning a reminder request into its title ---------------------------------

_TRIGGER = re.compile(
    r"(?:记得|请|麻烦|帮我|给我|到时候?)*"
    r"(?:设(?:置)?(?:一?个)?提醒|提醒(?:一下)?我?(?:一下)?|叫我|喊我|通知我)"
    r"|(?<![a-z])(?:please\s+)?remind\s+me(?:\s+to)?(?![a-z])",
    re.I,
)
_CLAUSE = re.compile(r"[，。；;！!？?\n]|,\s*")
_LEADING = re.compile(r"^(?:[\s、:：]|我要|我得|我需要|我|要|的|to\s)+", re.I)
_TRAILING = re.compile(r"(?:[\s、:：.]|的|吧|哦|啊|呀|一下)+$")
_CJK_GAP = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")


def _clean(s: str) -> str:
    s = _TRIGGER.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    s = _CJK_GAP.sub("", s)
    s = _LEADING.sub("", s)
    return _TRAILING.sub("", s).strip()


def title(text: str, parsed: Parsed) -> str:
    """What to remind about: the request with its time expressions and trigger words removed.

    '明天下午三点开会，提前半小时提醒我' -> '开会'. The clause holding the trigger
    ("提醒我…") is preferred, so '明天是我妈生日，提醒我买花' -> '买花'.
    """
    chars = list(text)
    for a, b in parsed.spans:
        chars[a:b] = " " * (b - a)
    clauses = _CLAUSE.split("".join(chars))
    ordered = [c for c in clauses if _TRIGGER.search(c)] + [c for c in clauses if not _TRIGGER.search(c)]
    return next((t for c in ordered if (t := _clean(c))), "")


def clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE.split(text) if c.strip()]


# --- presentation ----------------------------------------------------------------

def describe(dt: datetime, now: datetime) -> str:
    """'明天 9月23日（周三）15:00': the exact wording Alf echoes back so mistakes are visible."""
    s = f"{dt.month}月{dt.day}日（周{WEEKDAY_CN[dt.weekday()]}）{dt:%H:%M}"
    if dt.year != now.year:
        s = f"{dt.year}年{s}"
    rel = {0: "今天", 1: "明天", 2: "后天"}.get((dt.date() - now.date()).days)
    return f"{rel} {s}" if rel else s


def describe_recurrence(recurrence: str) -> str:
    days = _recurrence_days(recurrence)
    if recurrence == "DAILY":
        return "每天"
    if days == [0, 1, 2, 3, 4]:
        return "每个工作日"
    return "每周" + "、".join(WEEKDAY_CN[d] for d in days)


# Days already gone. The parser above is for reminders and only looks forward, but a stored memory of something that
# happened ("昨天和 leo 去徒步了") must keep its date too. "以前天气" is not "前天".
_PAST_DAY = re.compile(r"大前天|(?<![以之目从提眼面先事])前天|昨天|昨日|昨儿|昨晚")
_PAST_OFFSET = {"大前天": 3, "前天": 2, "昨天": 1, "昨日": 1, "昨儿": 1, "昨晚": 1}


def absolutize(text: str, now: datetime) -> str:
    """Rewrite relative dates so a stored fact stays true: '明天是我妈生日' -> '9月23日是我妈生日'."""
    out = text
    for t in sorted(parse(text, now).times, key=lambda t: -t.start):
        if t.date_kind in ("relday", "week", "indays", "dom", "mrel", "mend"):
            day, clock = f"{t.when.month}月{t.when.day}日", "" if t.time_defaulted else f" {t.when:%H:%M}"
            before = out[: t.start].rstrip(" ，,、（(")
            if t.date_kind == "week" and re.search(r"\d{1,2}月\d{1,2}日$", before):
                # "9月21日（周一）""11月20日周五下午五点"：星期只是在给前面的日期做注，不是另一个日子。
                # 留着原样，只把钟点换成数字；否则会变成"9月21日（9月28日）"。
                new = clock or out[t.start : t.end]
            elif before.endswith(day):
                new = clock          # 同一天已经写在前面了，不再写一遍
            else:
                new = day + clock
            out = out[: t.start] + new + out[t.end :]

    def past(m: re.Match) -> str:
        day = now.date() - timedelta(days=_PAST_OFFSET[m.group()])
        return f"{day.month}月{day.day}日" + ("晚上" if m.group() == "昨晚" else "")

    out = _PAST_DAY.sub(past, out)
    # 模型有时两样都写："昨天（2026年11月20日）"。换完成了"11月20日（2026年11月20日）"，把重复的括号去掉。
    return re.sub(r"(\d{1,2}月\d{1,2}日)[（(](?:\d{4}年)?\1[）)]", r"\1", out)
