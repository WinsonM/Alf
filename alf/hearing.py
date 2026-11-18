"""听录音：手表一分钟一段传过来的录音 → 谁在什么时候说了什么。

一段录音走的路：
1. 收下（receive）：按内容去重，原声放到 data/heard/日期/ 下，登记一行。
2. 凑成一批（next_batch）：挨着的几段拼起来一起听——一分钟一段是为了好传，句子和对话并不在整分钟上断开。
3. 听（process）：先筛出有人声的几块（一整天的录音大半是安静的，这一步最省事）；每一块里切出谁在什么时候说，
   每个人算一条声纹去认人（people.py），每一句转成文字，记成一行 utterance。
4. 原声只留几天（prune），文字留着。

几条界线：
- 全在本机算。录音和转出来的文字都不出这台机器；只有每晚整理"今天的事"时，文字会交给 DeepSeek（见 days.py）。
- 听到的话是素材，不是指令：别人说什么都不会让 Alf 去做任何事，这里也没有任何能做事的入口。
- 没有人声的录音当场删掉原声，不等到期。
"""
from __future__ import annotations

import bisect
import hashlib
import json
import logging
import re
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from . import fts, people, voice
from .config import Settings
from .voice import SAMPLE_RATE, VoiceError

log = logging.getLogger("alf.hearing")

WINDOW = 300.0            # 一批最多听这么多秒：够装下一场对话的来回，切分模型也还跑得动
JOIN = 3.0                # 两段录音首尾隔不到这么久，算连着的
SETTLE = 150              # 一批没凑满时，最新的一段收到这么久之后就不等了（多半是停录了）
MIN_SPEECH = 1.5          # 一批里人声加起来不到这么多秒，当成没人说话
MAX_ATTEMPTS = 3
STALE_CLAIM = 900         # 一批录音被占住这么久还没听完，当成那个进程死了，放回去重听
MAX_SECONDS = 4 * 3600    # 手动导入的单个文件最长这么久
BUDGET = 40.0             # 后台一轮最多听这么久（墙上时间），到点就停，下一轮接着听
SITTING_GAP = 300.0       # 两句话隔了这么久，算另一场对话
PASSAGE_GAP = 120.0       # 检索用的"一小段"：隔了这么久就另起一段
PASSAGE_CHARS = 400
PASSAGE_LINES = 12
MAX_RATE = 14.0           # 一秒钟说不了这么多个字：超过的多半是模型把杂音硬听成了字
_CONTENT = re.compile(r"[\W_]+")


def audio_dir(cfg: Settings) -> Path:
    return cfg.db_path.parent / "heard"


def day_of(ts: float, tz) -> str:
    return datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- 收下 --------------------------------------------------------------------------------

def receive(conn: sqlite3.Connection, cfg: Settings, src: Path, started_at: float, device: str, now: int,
            *, move: bool = False) -> int | None:
    """收下一段录音，返回它的编号。这一段以前收过（内容一样）就返回 None——手机重传是常事，不该听两遍。"""
    digest = _sha256(src)
    if conn.execute("SELECT 1 FROM recordings WHERE sha256 = ?", (digest,)).fetchone():
        if move:
            src.unlink(missing_ok=True)
        return None
    seconds = voice.duration(src)
    if not seconds or seconds < 0.5:
        raise VoiceError("这个文件不像一段录音（读不出时长）")
    if seconds > MAX_SECONDS:
        raise VoiceError(f"这段录音有 {seconds / 3600:.1f} 小时，太长了；请切成 {MAX_SECONDS // 3600} 小时以内的再导入")
    suffix = src.suffix.lower() if re.fullmatch(r"\.[a-z0-9]{1,5}", src.suffix.lower()) else ".m4a"
    dest = audio_dir(cfg) / day_of(started_at, cfg.tz) / f"{digest[:16]}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    (shutil.move if move else shutil.copyfile)(str(src), str(dest))
    with conn:
        return conn.execute("INSERT INTO recordings(sha256, path, device, started_at, seconds, received_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (digest, str(dest), device[:40], started_at, seconds, now)).lastrowid


# --- 凑成一批 ----------------------------------------------------------------------------

def next_batch(conn: sqlite3.Connection, now: int) -> list[sqlite3.Row]:
    """下一批该听的录音：最早的那一串连着的。凑满了一批、或者最新的一段已经放了一阵，才返回；否则再等等。"""
    with conn:      # 占住了却一直没听完的（进程中途被杀）：放回去重听
        conn.execute("UPDATE recordings SET state = 'new' WHERE state = 'hearing' AND heard_at < ?", (now - STALE_CLAIM,))
    rows = conn.execute("SELECT * FROM recordings WHERE state = 'new' ORDER BY started_at LIMIT 500").fetchall()
    if not rows:
        return []
    run, total, full = [rows[0]], rows[0]["seconds"], False
    for r in rows[1:]:
        last = run[-1]
        if r["device"] != last["device"] or r["started_at"] - (last["started_at"] + last["seconds"]) > JOIN:
            full = True          # 这一串到头了，后面的是另一串：不用再等
            break
        if total + r["seconds"] > WINDOW:
            full = True
            break
        run.append(r)
        total += r["seconds"]
    if full or total >= WINDOW or now - max(r["received_at"] for r in run) >= SETTLE:
        return run
    return []


# --- 听 ----------------------------------------------------------------------------------

def usable(text: str, seconds: float) -> bool:
    """转出来的这句话值不值得留：得有两个字以上的内容，而且语速得是人说得出来的。"""
    content = _CONTENT.sub("", text)
    return len(content) >= 2 and len(content) / max(seconds, 0.2) <= MAX_RATE


def local_speakers(ears, piece) -> tuple[list, dict[int, tuple]]:
    """一块人声里，谁在什么时候说（turns），以及每个人的声纹（{这一块里的编号: (向量, 秒数)}）。

    切分模型宁可把一个人分成两个（见 voice.py），所以这里再并一次：两个"人"的声纹够像（people.LOCAL_SAME），
    就是同一个人。并了之后同一个人挨着的几句才连得起来，转写时句子才完整。说得太短、算不出声纹的不并，也认不了人。
    """
    import numpy as np

    turns = ears.turns(piece)
    by_speaker: dict[int, list] = {}
    for t in turns:
        by_speaker.setdefault(t.speaker, []).append(t)
    prints: dict[int, tuple] = {}
    for speaker, own in by_speaker.items():
        own = sorted(own, key=lambda t: -t.seconds)           # 最长的几句在前：声纹最多只用 PRINT_SECONDS 秒
        clip = np.concatenate([piece[int(t.start * SAMPLE_RATE): int(t.end * SAMPLE_RATE)] for t in own])
        vec = ears.voiceprint(clip)
        if vec is not None:
            prints[speaker] = (vec, min(len(clip) / SAMPLE_RATE, voice.PRINT_SECONDS))
    renamed: dict[int, int] = {}
    while len(prints) > 1:
        keys = sorted(prints)
        score, a, b = max((float(prints[a][0] @ prints[b][0]), a, b) for i, a in enumerate(keys) for b in keys[i + 1:])
        if score < people.LOCAL_SAME:
            break
        (va, sa), (vb, sb) = prints[a], prints.pop(b)
        merged = va * sa + vb * sb
        prints[a] = (merged / float(np.linalg.norm(merged)), min(sa + sb, voice.PRINT_SECONDS))
        renamed[b] = a
    for t in turns:
        while t.speaker in renamed:
            t.speaker = renamed[t.speaker]
    return turns, prints


def process(conn: sqlite3.Connection, cfg: Settings, ears, batch: list[sqlite3.Row], now: int) -> dict:
    """听一批连着的录音，把里面的每一句话记下来。返回这一批有多长、多少秒人声、记了几句。"""
    import numpy as np

    parts, starts, offset = [], [], 0.0          # starts：每段录音在拼起来的音频里从第几秒开始
    kept = []
    for r in batch:
        try:
            samples = voice.decode(r["path"]) if r["path"] else None
        except VoiceError as e:
            samples = None
            log.warning("录音 %d 读不出来：%s", r["id"], e)
        if samples is None or not len(samples):
            with conn:
                conn.execute("UPDATE recordings SET state = 'failed', error = ?, heard_at = ? WHERE id = ?", ("读不出来", now, r["id"]))
            continue
        kept.append(r)
        starts.append(offset)
        parts.append(samples)
        offset += len(samples) / SAMPLE_RATE
    if not kept:
        return {"seconds": 0.0, "speech": 0.0, "lines": 0}
    audio = np.concatenate(parts)

    def wall(t: float) -> tuple[float, sqlite3.Row]:
        """拼起来的音频里的第 t 秒，是哪段录音、墙上时间几点几分。"""
        i = max(0, bisect.bisect_right(starts, t) - 1)
        return kept[i]["started_at"] + (t - starts[i]), kept[i]

    blocks = voice.islands(audio)
    speech = sum(b - a for a, b in blocks)
    lines, days = 0, set()
    if speech >= MIN_SPEECH:
        for a, b in blocks:
            piece = audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)]
            turns, prints = local_speakers(ears, piece)
            # 每个人拿声纹去认。说得多的先认：新建"没听过的人"时，用的是听得最久的那个声音。
            who: dict[int, people.Match] = {}
            present: set[int] = set()
            for speaker, (vec, seconds) in sorted(prints.items(), key=lambda kv: -kv[1][1]):
                match = people.identify(conn, vec, seconds, ears.name, now, ref=f"rec:{kept[0]['id']}", present=present)
                who[speaker] = match
                if match.person_id is not None:
                    present.add(match.person_id)
            for t in voice.join_turns(turns):
                heard = ears.transcribe(piece[int(t.start * SAMPLE_RATE): int(t.end * SAMPLE_RATE)])
                if not usable(heard.text, t.seconds):
                    continue
                began, rec = wall(a + t.start)
                match = who.get(t.speaker) or people.Match(None, None)
                with conn:
                    conn.execute("INSERT INTO utterances(recording_id, person_id, started_at, ended_at, day, text, language, voice_score)"
                                 " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                 (rec["id"], match.person_id, began, began + t.seconds, day_of(began, cfg.tz), heard.text,
                                  heard.language, match.score))
                lines += 1
                days.add(day_of(began, cfg.tz))
    with conn:
        for i, r in enumerate(kept):
            end = starts[i] + len(parts[i]) / SAMPLE_RATE
            own = sum(max(0.0, min(b, end) - max(a, starts[i])) for a, b in blocks)
            conn.execute("UPDATE recordings SET state = ?, speech = ?, heard_at = ?, error = NULL WHERE id = ?",
                         ("done" if speech >= MIN_SPEECH and own > 0 else "empty", round(own, 2), now, r["id"]))
    for r in kept:                                 # 没有人声的当场删掉原声
        if conn.execute("SELECT state FROM recordings WHERE id = ?", (r["id"],)).fetchone()["state"] == "empty":
            _drop_audio(conn, r)
    if days:
        rebuild_passages(conn, days)
    return {"seconds": offset, "speech": speech, "lines": lines}


def hear_pending(conn: sqlite3.Connection, cfg: Settings, ears, budget: float = BUDGET, clock=time.time) -> dict:
    """把攒着的录音一批一批听掉，直到没有了或者这一轮的时间用完。"""
    started = time.monotonic()
    done = {"seconds": 0.0, "speech": 0.0, "lines": 0, "batches": 0}
    while time.monotonic() - started < budget:
        now = int(clock())
        batch = next_batch(conn, now)
        if not batch:
            break
        # 先占住再听：后台服务和手动导入的命令可能同时在跑，同一段录音不能听两遍
        with conn:
            mine = [r for r in batch if conn.execute("UPDATE recordings SET state = 'hearing', heard_at = ? WHERE id = ? AND state = 'new'",
                                                      (now, r["id"])).rowcount]
        if len(mine) != len(batch):
            with conn:
                conn.executemany("UPDATE recordings SET state = 'new' WHERE id = ? AND state = 'hearing'", [(r["id"],) for r in mine])
            break
        try:
            result = process(conn, cfg, ears, batch, now)
        except Exception as e:
            # 这一批出了岔子：记一次，下一轮重来；连着三次都不行就放弃，免得一段坏录音把后面的全堵住
            log.exception("听录音出错（录音 %s）", [r["id"] for r in batch])
            with conn:
                conn.executemany("UPDATE recordings SET state = 'new', attempts = attempts + 1, error = ? WHERE id = ? AND state = 'hearing'",
                                 [(f"{type(e).__name__}: {e}"[:200], r["id"]) for r in batch])
                conn.execute("UPDATE recordings SET state = 'failed', heard_at = ? WHERE state = 'new' AND attempts >= ?", (now, MAX_ATTEMPTS))
            break
        for key in ("seconds", "speech", "lines"):
            done[key] += result[key]
        done["batches"] += 1
    return done


def _drop_audio(conn: sqlite3.Connection, r: sqlite3.Row) -> None:
    if r["path"]:
        Path(r["path"]).unlink(missing_ok=True)
        with conn:
            conn.execute("UPDATE recordings SET path = NULL WHERE id = ?", (r["id"],))


def prune(conn: sqlite3.Connection, keep_days: int, now: int) -> int:
    """听完超过 keep_days 天的录音，删掉原声。文字留着。"""
    old = conn.execute("SELECT * FROM recordings WHERE path IS NOT NULL AND state NOT IN ('new', 'hearing') AND started_at < ?",
                       (now - keep_days * 86400,)).fetchall()
    for r in old:
        _drop_audio(conn, r)
    return len(old)


# --- 他本人的声纹：从发给 Alf 的语音消息里取 -----------------------------------------------

def enroll_from_messages(conn: sqlite3.Connection, ears, since: int, now: int) -> tuple[int, int]:
    """他发给 Alf 的语音消息一定是他本人说的：拿来当他的声纹，不用他专门录一段。返回（用了几条，看到哪个时间了）。"""
    import numpy as np

    used, seen = 0, since
    for r in conn.execute("SELECT id, path, created_at FROM attachments WHERE kind = 'audio' AND created_at > ? ORDER BY created_at LIMIT 20", (since,)):
        seen = r["created_at"]
        try:
            audio = voice.decode(r["path"])
        except VoiceError:
            continue                            # 文件没了或者读不出：跳过，不再回头看
        spans = voice.islands(audio)
        if not spans:
            continue
        clip = np.concatenate([audio[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)] for a, b in spans])
        vec = ears.voiceprint(clip)
        if vec is None or len(clip) / SAMPLE_RATE < 2.0:
            continue
        people.enroll_me(conn, vec, min(len(clip) / SAMPLE_RATE, voice.PRINT_SECONDS), ears.name, "message", f"att:{r['id']}", now)
        used += 1
    return used, seen


# --- 一场一场的对话、检索用的一小段 -------------------------------------------------------

def sittings(conn: sqlite3.Connection, start: float, end: float, *, undigested: bool = False) -> list[list[sqlite3.Row]]:
    """这段时间里听到的话，按"一场对话"分开：两句隔了 SITTING_GAP 以上就算另一场。"""
    rows = conn.execute("SELECT * FROM utterances WHERE started_at >= ? AND started_at < ?" + (" AND digested_at IS NULL" if undigested else "")
                        + " ORDER BY started_at, id", (start, end)).fetchall()
    out: list[list[sqlite3.Row]] = []
    for r in rows:
        if out and r["started_at"] - out[-1][-1]["ended_at"] <= SITTING_GAP:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def rebuild_passages(conn: sqlite3.Connection, days) -> None:
    """把这几天听到的话重新切成检索用的一小段一小段。每段写成"谁：说了什么"，所以改了名字、并了人之后也要重切。"""
    names = people.labels(conn)
    with conn:
        for day in sorted(set(days)):
            conn.execute("DELETE FROM heard_fts WHERE rowid IN (SELECT id FROM heard_passages WHERE day = ?)", (day,))
            conn.execute("DELETE FROM heard_passages WHERE day = ?", (day,))
            group: list[sqlite3.Row] = []
            rows = conn.execute("SELECT * FROM utterances WHERE day = ? ORDER BY started_at, id", (day,)).fetchall()
            for r in [*rows, None]:
                if r is not None and group and r["started_at"] - group[-1]["ended_at"] <= PASSAGE_GAP and len(group) < PASSAGE_LINES \
                        and sum(len(x["text"]) for x in group) + len(r["text"]) <= PASSAGE_CHARS:
                    group.append(r)
                    continue
                if group:
                    text = "\n".join(f"{names.get(x['person_id'], '有人')}：{x['text']}" for x in group)
                    conn.execute("INSERT INTO heard_passages(id, last_utterance, day, text, started_at, ended_at) VALUES (?, ?, ?, ?, ?, ?)",
                                 (group[0]["id"], group[-1]["id"], day, text, group[0]["started_at"], group[-1]["ended_at"]))
                    fts.index(conn, "heard_fts", group[0]["id"], text)
                group = [r] if r is not None else []


def event_index_text(conn: sqlite3.Connection, text: str, who: list[int]) -> str:
    """一件事拿什么去建字面索引：换上了名字的正文，加上在场的人——这样按名字也找得到。"""
    names = people.labels(conn)
    return " ".join([people.render(text, names), *(names.get(p, "") for p in who)])


def _reindex_events(conn: sqlite3.Connection, person_id: int) -> None:
    """这个人有了名字（或者并到了别人名下）：提到他的那些事，字面索引里要换上新名字。"""
    rows = conn.execute("SELECT id, text, people FROM day_events WHERE removed_at IS NULL AND (text LIKE ? OR people LIKE ?)",
                        (f"%声{person_id}%", f"%{person_id}%")).fetchall()
    with conn:
        for r in rows:
            conn.execute("DELETE FROM events_fts WHERE rowid = ?", (r["id"],))
            fts.index(conn, "events_fts", r["id"], event_index_text(conn, r["text"], json.loads(r["people"])))


def name_voice(conn: sqlite3.Connection, person_id: int, new_name: str | None, now: int, *, is_me: bool = False) -> sqlite3.Row | None:
    """告诉 Alf 这个声音是谁（people.name），并把这个人说过话的那些天、提到他的那些事的检索索引重做。"""
    days = people.days_of(conn, person_id)
    row = people.name(conn, person_id, new_name, now, is_me=is_me)
    if row is not None:
        rebuild_passages(conn, set(days) | set(people.days_of(conn, row["id"])))
        _reindex_events(conn, row["id"])
    return row


def forget_voice(conn: sqlite3.Connection, person_id: int) -> bool:
    days = people.days_of(conn, person_id)
    done = people.forget(conn, person_id)
    if done:
        rebuild_passages(conn, days)
    return done


def forget_day(conn: sqlite3.Connection, day: str) -> int:
    """把某一天听到的全删掉：话、整理出来的事、原声。录音的登记留着（只剩摘要和时间），免得手机重传时又收一遍。"""
    for r in conn.execute("SELECT r.* FROM recordings r WHERE r.path IS NOT NULL AND r.id IN (SELECT recording_id FROM utterances WHERE day = ?)", (day,)).fetchall():
        _drop_audio(conn, r)
    with conn:
        conn.execute("DELETE FROM events_fts WHERE rowid IN (SELECT id FROM day_events WHERE day = ?)", (day,))
        conn.execute("DELETE FROM day_events WHERE day = ?", (day,))
        gone = conn.execute("DELETE FROM utterances WHERE day = ?", (day,)).rowcount
    rebuild_passages(conn, [day])
    return gone


# --- 状态、后台任务 ----------------------------------------------------------------------

def status(conn: sqlite3.Connection, cfg: Settings, now: datetime) -> dict:
    """设置页上看的：模型在不在、还有多少没听、今天听了多少。"""
    today = now.strftime("%Y-%m-%d")
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    waiting = conn.execute("SELECT COUNT(*) AS n, COALESCE(SUM(seconds), 0) AS s FROM recordings WHERE state IN ('new', 'hearing')").fetchone()
    heard = conn.execute("SELECT COALESCE(SUM(seconds), 0) AS s, COALESCE(SUM(speech), 0) AS speech FROM recordings"
                         " WHERE state IN ('done', 'empty') AND started_at >= ?", (midnight,)).fetchone()
    kept = conn.execute("SELECT COUNT(*) AS n FROM recordings WHERE path IS NOT NULL").fetchone()
    return {"models": voice.status()["present"],
            "waiting": waiting["n"], "waiting_minutes": round(waiting["s"] / 60, 1),
            "today_minutes": round(heard["s"] / 60, 1), "today_speech_minutes": round(heard["speech"] / 60, 1),
            "today_lines": conn.execute("SELECT COUNT(*) FROM utterances WHERE day = ?", (today,)).fetchone()[0],
            "failed": conn.execute("SELECT COUNT(*) FROM recordings WHERE state = 'failed'").fetchone()[0],
            "audio_kept": kept["n"],
            "people": conn.execute("SELECT COUNT(*) FROM people WHERE name IS NOT NULL OR is_me = 1").fetchone()[0],
            "unnamed": conn.execute("SELECT COUNT(*) FROM people WHERE name IS NULL AND is_me = 0").fetchone()[0],
            "me": people.me(conn) is not None}


def hearing_job(ctx) -> str | None:
    """后台的那一轮：把攒着的录音听掉，顺手更新他本人的声纹、清掉过期的原声。没装模型或者关着就什么都不做。"""
    if not ctx.prefs.get("hearing.enabled", True):
        return None
    ears = voice.default()
    if ears is None:
        return None
    state = json.loads(ctx.cursor or "{}")
    now = int(ctx.now.timestamp())
    used, seen = enroll_from_messages(ctx.conn, ears, int(state.get("messages", 0)), now)
    state["messages"] = seen
    done = hear_pending(ctx.conn, ctx.cfg, ears)
    prune(ctx.conn, int(ctx.prefs.get("hearing.keep_days", 7)), now)
    ctx.cursor = json.dumps(state)
    if not done["batches"]:
        ears.release_if_idle()
        return f"用 {used} 条语音消息更新了他的声纹" if used else None
    return (f"听了 {done['seconds'] / 60:.1f} 分钟录音（人声 {done['speech'] / 60:.1f} 分钟），记下 {done['lines']} 句"
            + (f"；用 {used} 条语音消息更新了他的声纹" if used else ""))
