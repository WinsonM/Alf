"""Semantic memory: judged at write time, retrieved at read time.

Write: new statement -> similar current facts (same category + full-text match)
       -> Jev judges each pair (duplicate / updates / unrelated)
       -> code inserts, skips, or closes outdated facts (valid_to + superseded_by).
Read:  full-text candidates -> Jev picks the one that answers the question.

Three layers, all in the one `facts` table (2026-11-17):
- core    (topic IS NULL): who the user is, people, lasting preferences, how he wants Alf to behave.
                           Small, sent with every turn.
- topic   (topic set):     groups of detail that keep growing (training weights, a course, a trip).
                           Only a one-line index is always sent; a group is sent whole when it is relevant.
                           A keyword search cannot do this job: "健身记录" shares no word with "绳索下压 22.5kg".
- moment  (category):      dated things that happened. The latest few are sent; older ones come back when what he
                           says now touches them (a full-text match), or when they mattered a lot (`importance`).
                           `ours` marks things that happened between him and Alf rather than in his own life;
                           `feeling` keeps how he said it felt, in his words.
A fact with `expires_at` stops being current once that time passes ("11月20日有考试").

Open loops (2026-11-17, after Anima Brain's time-aware follow-up ranking): things that are not settled yet and are
worth asking about later - an upcoming dated event, a worry, a plan, a result he is waiting for. They are ordinary
facts with `follow_up` set (or with a date), ranked by how much they matter and how recently they came up.

The stored text is always the user's own words (with relative dates made
absolute): Jev chooses, code owns the string.
"""
from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from . import fts
from .decide import NONE, RELATE_TASK, RELATIONS, Decider, DecisionError, choice, pick, same_note_question

SAME_NOTE_FLOOR = 0.3   # below this P(same note), a "correction" is refused as an overwrite of something else
# Replacing an old fact on the model's say-so needs more than a relation verdict. Measured 2026-11-16:
# both Jev and DeepSeek called "绳索后拉用20kg" an update of "绳索下拉…22.5kg" (different exercises), with
# P(same note) = 0.63, while clear-cut updates scored 0.84-0.95. Below this bar both facts are kept and
# the user is asked.
SUPERSEDE_FLOOR = 0.8
MOMENT = "moment"         # a dated thing that happened; never part of the always-loaded layer
FEEDBACK = "feedback"     # how the user wants Alf to talk or work; always loaded, shown under its own heading
TOPIC_CHARS = 12
# Open-loop ranking, simplified from Anima Brain's  I x (1 + A) x exp(-t / (14 + 30A)) + U  (Alf has no arousal value):
#   priority = importance x exp(-days_since_it_came_up / LOOP_FADE_DAYS) + 2 / (1 + days_until_the_event)
# An exam three days away, mentioned today, importance 8: 8.5.  A worry from ten days ago, importance 6: 3.7.
# The same worry after two months: 0.3, below LOOP_MIN - by then it has faded without anyone closing it.
LOOP_FADE_DAYS = 21
LOOP_MIN = 1.0
LOOP_QUIET = 20 * 3600    # it came up within this long: no need to raise it again today
LOOP_MAX_ASKS = 2         # Alf raised it this many times and he did not pick it up: stop
DEFAULT_IMPORTANCE = 5
_SQUASH = re.compile(r"[\W_]+")
_COMMON = fts.COMMON     # bigrams too common to mean that two sentences are about the same thing


def _importance(value) -> int | None:
    try:
        return max(1, min(10, int(value))) if value is not None else None
    except (TypeError, ValueError):
        return None


def loop_priority(f: Fact, now: int) -> float:
    """How pressing an open loop is right now. See the constants above for the formula and worked numbers."""
    came_up = f.touched_at or f.recorded_at
    score = (f.importance or DEFAULT_IMPORTANCE) * math.exp(-max(0, now - came_up) / 86400 / LOOP_FADE_DAYS)
    if f.expires_at is not None:
        score += 2 / (1 + max(0, f.expires_at - now) / 86400)
    return score


def was_said(quote: str | None, *spoken: str) -> bool:
    """Is `quote` really something he said? Compared with spacing, punctuation and case removed, so a model that
    tidies a comma still passes, but one that quotes words he never used does not."""
    q = _SQUASH.sub("", quote or "").lower()
    return len(q) >= 2 and any(q in _SQUASH.sub("", text or "").lower() for text in spoken)


def same_wording(a: str, b: str) -> bool:
    """Identical once spacing, punctuation and case are ignored."""
    return _SQUASH.sub("", a).lower() == _SQUASH.sub("", b).lower()


@dataclass
class Fact:
    id: int
    category: str
    text: str
    recorded_at: int
    valid_to: int | None
    superseded_by: int | None
    source: str
    needs_confirmation: bool
    topic: str | None = None        # None = core layer
    slot: str | None = None         # which item of the topic this is the value of
    expires_at: int | None = None
    aftermath: str | None = None    # "open" once expired and not yet talked about; "done" after
    quote: str | None = None        # his own words this fact came from
    follow_up: bool = False         # an open loop: worth asking about later
    importance: int | None = None   # 1-10
    touched_at: int | None = None   # last time it came up in conversation
    asks: int = 0                   # times Alf raised it and he did not pick it up
    scope: str | None = None        # feedback only: where the rule applies. None = everywhere; chat | mail | proactive
    ours: bool = False              # moment only: it happened between him and Alf
    feeling: str | None = None      # moment only: how he said it felt, in his own words

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Fact:
        return cls(r["id"], r["category"], r["text"], r["recorded_at"], r["valid_to"],
                   r["superseded_by"], r["source"], bool(r["needs_confirmation"]),
                   r["topic"], r["slot"], r["expires_at"], r["aftermath"],
                   r["quote"], bool(r["follow_up"]), r["importance"], r["touched_at"], r["asks"], r["scope"],
                   bool(r["ours"]), r["feeling"])

    def recorded(self, tz) -> str:
        return f"{datetime.fromtimestamp(self.recorded_at, tz):%-m月%-d日}"


@dataclass
class WriteResult:
    fact: Fact
    status: str               # "added" | "updated" | "duplicate"
    superseded: list[Fact]
    maybe_same: list[Fact] = field(default_factory=list)   # kept alongside the new fact; the user should be asked


class Memory:
    def __init__(self, conn: sqlite3.Connection, decider: Decider, threshold: float):
        self.conn = conn
        self.decider = decider
        self.threshold = threshold

    def get(self, fact_id: int) -> Fact:
        return Fact.from_row(self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone())

    def current(self, category: str | None = None, limit: int = 100) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND (? IS NULL OR category = ?)"
            " ORDER BY recorded_at DESC, id DESC LIMIT ?", (category, category, limit))
        return [Fact.from_row(r) for r in rows]

    # --- layers -------------------------------------------------------------------

    def core(self, limit: int = 100) -> list[Fact]:
        """The always-loaded layer: current facts that belong to no topic and are not moments."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND topic IS NULL AND category != ?"
            " ORDER BY recorded_at DESC, id DESC LIMIT ?", (MOMENT, limit))
        return [Fact.from_row(r) for r in rows]

    def topics(self) -> list[tuple[str, int, int]]:
        """(topic, number of current facts, last recorded) - most recently touched first."""
        rows = self.conn.execute(
            "SELECT topic, COUNT(*) AS n, MAX(recorded_at) AS last FROM facts"
            " WHERE valid_to IS NULL AND topic IS NOT NULL AND category != ? GROUP BY topic ORDER BY last DESC, topic", (MOMENT,))
        return [(r["topic"], r["n"], r["last"]) for r in rows]

    def in_topic(self, topic: str, limit: int = 60) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND topic = ? AND category != ?"
            " ORDER BY recorded_at DESC, id DESC LIMIT ?", (self.topic_name(topic), MOMENT, limit))
        return [Fact.from_row(r) for r in rows]

    def topic_name(self, topic: str) -> str:
        """Reuse an existing topic's spelling ("健身" for " 健身 " or "健 身") so one topic does not split into two."""
        name = " ".join(str(topic).split())[:TOPIC_CHARS]
        return next((t for t, _, _ in self.topics() if same_wording(t, name)), name)

    def moments(self, limit: int = 5, since: int = 0) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND category = ? AND recorded_at >= ?"
            " ORDER BY recorded_at DESC, id DESC LIMIT ?", (MOMENT, since, limit))
        return [Fact.from_row(r) for r in rows]

    def moments_recalled(self, text: str, skip: set[int], limit: int = 3) -> list[Fact]:
        """Older moments that what he just said touches on: "蓝山" brings back the hike from two months ago.
        Plain full-text match, best first; `skip` holds the ones already shown."""
        asked = {t for t in fts.tokens(text) if len(t) > 1 and not t.isdigit() and t not in _COMMON}
        out = []
        for f in self.search(text, limit=limit + len(skip) + 8, category=MOMENT) if asked else []:
            shared = asked & set(fts.tokens(f.text))
            # Any shared bigram is a match for the index, and "考试" alone would drag in every exam he ever sat.
            # So: two shared words, or one that hardly occurs anywhere else in memory (a place, a name).
            if f.id not in skip and (len(shared) >= 2 or any(self._rare(t) for t in shared)):
                out.append(f)
        return out[:limit]

    def _rare(self, token: str) -> bool:
        return self.conn.execute("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?", (f'"{token}"',)).fetchone()[0] <= 2

    def moments_that_mattered(self, before: int, limit: int = 2, floor: int = 8) -> list[Fact]:
        """Moments from before `before` that were a big deal (importance >= floor), newest first. A friend does not
        forget the day you graduated just because it was more than a month ago."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND category = ? AND recorded_at < ? AND importance >= ?"
            " ORDER BY recorded_at DESC, id DESC LIMIT ?", (MOMENT, before, floor, limit))
        return [Fact.from_row(r) for r in rows]

    def expire(self, now: int) -> int:
        """Close facts whose date has passed. They are kept, and flagged so Alf can ask how it went."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE facts SET valid_to = expires_at, aftermath = 'open'"
                " WHERE valid_to IS NULL AND expires_at IS NOT NULL AND expires_at <= ?", (now,))
        return cur.rowcount

    def just_past(self, now: int, days: int = 3) -> list[Fact]:
        """Dated facts that ended in the last few days and have not been talked about since."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE aftermath = 'open' AND valid_to >= ? ORDER BY valid_to, id", (now - days * 86400,))
        return [Fact.from_row(r) for r in rows]

    def settle(self, fact_id: int) -> None:
        """He told how it went (or it no longer needs asking about): the loop is closed."""
        with self.conn:
            self.conn.execute("UPDATE facts SET aftermath = 'done' WHERE id = ? AND aftermath = 'open'", (fact_id,))
            self.conn.execute("UPDATE facts SET follow_up = 0 WHERE id = ?", (fact_id,))

    # --- open loops ---------------------------------------------------------------

    def open_loops(self, now: int, limit: int = 4, quiet: bool = True) -> list[tuple[Fact, float]] | list[Fact]:
        """Unsettled things worth bringing up now, most pressing first, with their priority.
        With quiet=False: the ones that already came up today as well (to notice if he returns to them), as bare facts."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND category != ? AND (follow_up = 1 OR expires_at IS NOT NULL)", (MOMENT,))
        ranked = []
        for f in map(Fact.from_row, rows):
            # Asked twice without an answer, or already came up today: not something to raise now. With quiet=False
            # these are kept, because he may still answer - and that has to be noticed.
            if quiet and (f.asks >= LOOP_MAX_ASKS or (f.touched_at is not None and now - f.touched_at < LOOP_QUIET)):
                continue
            score = loop_priority(f, now)
            if score >= LOOP_MIN:
                ranked.append((f, round(score, 2)))
        ranked = sorted(ranked, key=lambda pair: (-pair[1], pair[0].id))[:limit]
        return ranked if quiet else [f for f, _ in ranked]

    def touch(self, fact_id: int, now: int, *, by_user: bool) -> None:
        """This came up in conversation. If he talked about it, the count of unanswered asks starts over;
        if only Alf raised it, that is one more ask he did not pick up."""
        with self.conn:
            self.conn.execute("UPDATE facts SET touched_at = ?, asks = CASE WHEN ? THEN 0 ELSE asks + 1 END WHERE id = ?",
                              (now, by_user, fact_id))

    def retag(self, fact_id: int, *, topic: str | None, slot: str | None, expires_at: int | None,
              category: str | None = None) -> None:
        """Change how a fact is filed, not what it says (used when tidying facts stored before there were layers)."""
        topic = self.topic_name(topic) if topic else None
        with self.conn:
            self.conn.execute("UPDATE facts SET topic = ?, slot = ?, expires_at = ?, category = COALESCE(?, category) WHERE id = ?",
                              (topic, slot if topic else None, expires_at, category, fact_id))
            self._index(fact_id)

    def _index(self, fact_id: int) -> None:
        """Index the topic and slot with the text, so a search for "健身" finds the facts filed under it."""
        r = self.conn.execute("SELECT text, topic, slot FROM facts WHERE id = ?", (fact_id,)).fetchone()
        self.conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
        fts.index(self.conn, "facts_fts", fact_id, " ".join(p for p in (r["topic"], r["slot"], r["text"]) if p))

    def history(self, limit: int = 200) -> list[Fact]:
        rows = self.conn.execute("SELECT * FROM facts ORDER BY id DESC LIMIT ?", (limit,))
        return [Fact.from_row(r) for r in rows]

    def search(self, text: str, limit: int = 8, category: str | None = None) -> list[Fact]:
        q = fts.query(text)
        if not q:
            return []
        rows = self.conn.execute(
            "SELECT f.* FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid"
            " WHERE facts_fts MATCH ? AND f.valid_to IS NULL AND (? IS NULL OR f.category = ?) ORDER BY bm25(facts_fts) LIMIT ?",
            (q, category, category, limit))
        return [Fact.from_row(r) for r in rows]

    def write(self, text: str, category: str, *, now: int, source: str = "text",
              attachment_id: str | None = None, episode_id: int | None = None,
              confidence: float | None = None, needs_confirmation: bool = False,
              topic: str | None = None, slot: str | None = None, expires_at: int | None = None,
              quote: str | None = None, follow_up: bool = False, importance: int | None = None,
              scope: str | None = None, ours: bool = False, feeling: str | None = None) -> WriteResult:
        topic = self.topic_name(topic) if topic and str(topic).strip() else None
        slot = " ".join(str(slot).split()) or None if slot and topic else None
        group = self.in_topic(topic, limit=200) if topic else []
        # Same topic, same slot: this is the new value of that item. No model is asked - the 2026-11-16 failures
        # ("绳索后拉 20kg" judged an update of "绳索下拉 22.5kg") came from guessing which note a number belongs to.
        keyed = next((f for f in group if slot and f.slot and same_wording(f.slot, slot)), None)
        by_id = {f.id: f for f in (group[:5] if topic else self.current(category, limit=5)) + self.search(text, limit=5)}
        # Same topic, different slot: a different item by construction, never a candidate for replacement.
        candidates = [f for f in by_id.values()
                      if not (keyed and f.id == keyed.id)
                      and not (slot and f.topic == topic and f.slot and not same_wording(f.slot, slot))][:6]
        if same := next((f for f in [*([keyed] if keyed else []), *candidates] if same_wording(f.text, text)), None):
            return WriteResult(same, "duplicate", [])     # word-for-word the same: no need to ask a model
        duplicate, outdated, maybe_same = None, [keyed] if keyed else [], []
        if candidates:
            # One question per candidate, each carrying only its own note (keeps state small and relevant).
            questions = {f"rel_{f.id}": choice({"task": RELATE_TASK, "existing_note": f.text}, RELATIONS)
                         for f in candidates}
            answers = self.decider.ask("memory.relate", {"new_statement": text}, questions)
            for f in candidates:
                a = answers[f"rel_{f.id}"]
                if a.confidence < self.threshold:
                    continue
                if a.choice == "updates":
                    # Closing an old fact is the one destructive thing a write can do, so it needs a second,
                    # stricter check. Unsure means: keep both, and let the caller ask the user.
                    (outdated if self._same_note_score(f.text, text) >= SUPERSEDE_FLOOR else maybe_same).append(f)
                elif a.choice == "duplicate" and duplicate is None:
                    duplicate = f
            if duplicate and not outdated:
                return WriteResult(duplicate, "duplicate", [])
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO facts(category, text, valid_from, recorded_at, source, source_attachment,"
                " source_episode, confidence, needs_confirmation, topic, slot, expires_at, quote, follow_up, importance, scope,"
                " ours, feeling) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (category, text, now, now, source, attachment_id, episode_id, confidence, needs_confirmation, topic, slot,
                 expires_at, quote, bool(follow_up) and category != MOMENT, _importance(importance),
                 scope if category == FEEDBACK else None, bool(ours) and category == MOMENT,
                 (" ".join(str(feeling).split())[:40] or None) if feeling and category == MOMENT else None))
            new_id = cur.lastrowid
            for f in outdated:
                self.conn.execute("UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?", (now, new_id, f.id))
            self._index(new_id)
        return WriteResult(self.get(new_id), "updated" if outdated else "added", outdated, maybe_same)

    def _same_note_score(self, old_text: str, new_text: str) -> float:
        """P(new_text is a revision of old_text). Falls back to shared wording when no judge is available."""
        try:
            return self.decider.ask("memory.guard", {"existing_note": old_text, "proposed_text": new_text},
                                    {"same": same_note_question()})["same"].noul
        except DecisionError:
            return 1.0 if set(fts.tokens(old_text)) & set(fts.tokens(new_text)) else 0.0

    def recall(self, question: str) -> tuple[Fact | None, float]:
        candidates = self.search(question, limit=8)
        if not candidates:
            return None, 0.0
        a = self.decider.ask("memory.recall", {"question": question}, {
            "answer": pick("Which remembered note answers the user's question?",
                           {f"f{f.id}": f.text for f in candidates},
                           "None of these notes answers the question."),
        })["answer"]
        if a.choice == NONE or a.confidence < self.threshold:
            return None, a.confidence
        return next(f for f in candidates if f"f{f.id}" == a.choice), a.confidence

    def same_note(self, fact_id: int, text: str) -> bool | None:
        """Is `text` a revision of this fact (True), or a different fact that would overwrite it (False)?

        Returns None when there is no such current fact. If the decider is unavailable, falls back to
        whether the two share any wording at all: crude, but it still stops the worst overwrites.
        """
        old = self.conn.execute("SELECT text FROM facts WHERE id = ? AND valid_to IS NULL", (fact_id,)).fetchone()
        if not old:
            return None
        return self._same_note_score(old["text"], text) >= SAME_NOTE_FLOOR

    def revise(self, fact_id: int, text: str, *, now: int, source: str = "text",
               episode_id: int | None = None) -> tuple[Fact, Fact] | None:
        """The user corrected a fact: store the new wording and close the old one, linked to it."""
        old = self.conn.execute("SELECT * FROM facts WHERE id = ? AND valid_to IS NULL", (fact_id,)).fetchone()
        if not old:
            return None
        with self.conn:
            # The corrected wording stays filed where the old one was, and stays an open loop if it was one.
            cur = self.conn.execute(
                "INSERT INTO facts(category, text, valid_from, recorded_at, source, source_episode, topic, slot, expires_at,"
                " follow_up, importance, touched_at, asks, scope, ours, feeling) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (old["category"], text, now, now, source, episode_id, old["topic"], old["slot"], old["expires_at"],
                 old["follow_up"], old["importance"], old["touched_at"], old["asks"], old["scope"], old["ours"], old["feeling"]))
            self.conn.execute("UPDATE facts SET valid_to = ?, superseded_by = ? WHERE id = ?", (now, cur.lastrowid, fact_id))
            self._index(cur.lastrowid)
        return Fact.from_row(old), self.get(cur.lastrowid)

    def forget(self, fact_id: int, *, now: int) -> Fact | None:
        """Stop using a fact. It is closed, not deleted, so there is a record of what was once believed."""
        old = self.conn.execute("SELECT * FROM facts WHERE id = ? AND valid_to IS NULL", (fact_id,)).fetchone()
        if not old:
            return None
        with self.conn:
            self.conn.execute("UPDATE facts SET valid_to = ? WHERE id = ?", (now, fact_id))
        return Fact.from_row(old)

    def recorded_since(self, since: int) -> list[Fact]:
        """Current facts learned after `since`, oldest first (for the nightly digest)."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND recorded_at > ? ORDER BY id", (since,))
        return [Fact.from_row(r) for r in rows]

    def confirm(self, fact_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE facts SET needs_confirmation = 0 WHERE id = ?", (fact_id,))
