"""Speech-to-text with faster-whisper, running locally.

Jev accepts text only ("Transcribe or caption first"), so audio becomes text here,
before any decision is made. Two things matter more than raw accuracy for a
memory-first agent:

- Vocabulary biasing: names and course codes Alf already knows go into Whisper's
  `initial_prompt`, the main fix for mangled proper nouns.
- Confidence: kept on the attachment row, so facts heard with low confidence are
  stored as unconfirmed instead of silently entering memory.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000


@dataclass
class Transcript:
    text: str
    confidence: float        # exp(duration-weighted mean avg_logprob), 0..1
    language: str | None
    segments: list[dict]


class Recognizer:
    def __init__(self, model_size: str = "small", language: str | None = None):
        self.model_size = model_size
        self.language = language
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import, only when voice is used

            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def cached(self) -> bool:
        """识别模型在不在本机。不在的话第一次识别要先下载几百 MB，调用方可以提前跟用户说一声。"""
        import os

        if self._model is not None or Path(self.model_size).exists():
            return True
        hub = Path(os.getenv("HF_HUB_CACHE") or Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
        return any((hub / f"models--Systran--faster-whisper-{self.model_size}").glob("snapshots/*/model.bin"))

    def transcribe(self, path: Path, vocabulary: list[str]) -> Transcript:
        segments, info = self._load().transcribe(
            str(path),
            language=self.language,          # None = auto-detect: copes better with 中英混说
            initial_prompt=build_prompt(vocabulary),
            vad_filter=True,                 # drop silence before/after speech
            beam_size=5,
        )
        segs = list(segments)
        return Transcript(
            text=canonical("".join(s.text for s in segs).strip(), vocabulary),
            confidence=confidence(segs),
            language=info.language,
            segments=[{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text,
                       "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3)}
                      for s in segs],
        )


def build_prompt(vocabulary: list[str]) -> str:
    # Written in simplified Chinese: it also steers Whisper away from traditional characters.
    prompt = "以下是普通话和英语混合的日常对话，使用简体中文。"
    if vocabulary:
        prompt += "可能提到：" + "、".join(vocabulary[:40]) + "。"
    return prompt


def canonical(text: str, vocabulary: list[str]) -> str:
    """Restore the known spelling of Latin-script terms: Whisper heard "comP5999" for COMP5999 (measured 2026-11-17)."""
    for term in vocabulary:
        if term.isascii() and len(term) >= 3:
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", term, text, flags=re.I)
    return text


def confidence(segments) -> float:
    if not segments:
        return 0.0
    weights = [max(s.end - s.start, 0.01) for s in segments]
    mean_logprob = sum(s.avg_logprob * w for s, w in zip(segments, weights)) / sum(weights)
    return round(math.exp(mean_logprob), 3)


NOT_NAMES = {"the", "and", "for", "with", "kgs", "lbs", "reps", "sets", "max", "min"}


def vocabulary(conn: sqlite3.Connection, limit: int = 40) -> list[str]:
    """User-added terms first; then the topics and item names Alf has filed memories under; then Latin-script
    names and codes from current facts and reminders.

    The topic and item names are the user's own jargon, which is exactly what gets mangled. Measured 2026-11-17 with
    the `small` model: "上斜哑铃卧推" came out as "上鞋雅灵握推", and "周五下午五点考 COMP5999" as "中午下午5点comP5999";
    with the item names in the prompt both were heard correctly. (Whisper's `hotwords` option did worse than the prompt.)
    """
    terms = [r["term"] for r in conn.execute("SELECT term FROM vocabulary ORDER BY created_at DESC")]
    filed = conn.execute("SELECT topic, slot FROM facts WHERE valid_to IS NULL AND topic IS NOT NULL ORDER BY recorded_at DESC").fetchall()
    terms += [r["topic"] for r in filed] + [r["slot"] for r in filed if r["slot"]]
    rows = conn.execute("SELECT text FROM facts WHERE valid_to IS NULL"
                        " UNION ALL SELECT title FROM reminders WHERE status = 'pending'")
    for (text,) in rows:
        # Capitalised words and codes with digits, plus lower-case names such as "leo" (three letters or more, so not "kg").
        terms += [w for w in re.findall(r"[A-Za-z]{3,}\d*|[A-Za-z]+\d+", text) if w.lower() not in NOT_NAMES]
    return list(dict.fromkeys(terms))[:limit]


def store(conn: sqlite3.Connection, src: Path, attachment_dir: Path) -> tuple[str, Path]:
    """Copy an audio file into Alf's attachment store and register it."""
    attachment_id = uuid.uuid4().hex[:12]
    attachment_dir.mkdir(parents=True, exist_ok=True)
    dest = attachment_dir / f"{attachment_id}{src.suffix.lower() or '.wav'}"
    shutil.copy2(src, dest)
    with conn:
        conn.execute("INSERT INTO attachments(id, kind, path, sha256, created_at) VALUES (?, 'audio', ?, ?, ?)",
                     (attachment_id, str(dest), hashlib.sha256(dest.read_bytes()).hexdigest(), int(time.time())))
    return attachment_id, dest


def save_transcript(conn: sqlite3.Connection, attachment_id: str, t: Transcript) -> None:
    with conn:
        conn.execute("UPDATE attachments SET transcript = ?, confidence = ?, extracted = ? WHERE id = ?",
                     (t.text, t.confidence, json.dumps({"language": t.language, "segments": t.segments},
                                                       ensure_ascii=False), attachment_id))


def record(dest: Path) -> Path:
    """Push-to-talk from the default microphone: records until Enter, saves 16 kHz mono WAV."""
    import numpy as np
    import sounddevice as sd

    frames: list = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        callback=lambda data, *_: frames.append(data.copy())):
        input("🎙  录音中…说完按回车 ")
    audio = np.concatenate(frames) if frames else np.zeros((0, 1), dtype="int16")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())
    return dest
