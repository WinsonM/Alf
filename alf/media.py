"""收到的语音和图片：存进 Alf 自己的附件库，再转成识别和看图用得上的格式。

转换用的是 macOS 自带的两个命令行工具：afconvert（iMessage 的语音是 .caf 里装的 Opus，Whisper 那边的
解码库不一定认）和 sips（iPhone 拍的照片是 HEIC，模型只收 JPEG、PNG、GIF、WebP）。
"""
from __future__ import annotations

import base64
import hashlib
import shutil
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

AUDIO_EXT = {".caf", ".m4a", ".mp3", ".wav", ".amr", ".aac", ".ogg", ".opus", ".aiff"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".gif", ".webp", ".tiff", ".tif", ".bmp"}
MAX_IMAGE_EDGE = 1568           # 再大对看清文字没有帮助，只会让请求变慢变贵
MAX_BYTES = 30 * 1024 * 1024


class MediaError(RuntimeError):
    """这个附件用不了。消息是给用户看的。"""


def kind_of(path: Path | None, mime: str = "", audio_message: bool = False) -> str | None:
    """"audio"、"image"，或者 None（视频、文件等现在处理不了的）。"""
    suffix = path.suffix.lower() if path else ""
    if audio_message or mime.startswith("audio/") or suffix in AUDIO_EXT:
        return "audio"
    if mime.startswith("image/") or suffix in IMAGE_EXT:
        return "image"
    return None


def store(conn: sqlite3.Connection, src: Path, attachment_dir: Path, kind: str) -> tuple[str, Path]:
    """把文件复制进附件库并登记。之后由它产生的提醒和记忆都能追溯回这份原件。"""
    if src.stat().st_size > MAX_BYTES:
        raise MediaError("这个附件太大了")
    attachment_id = uuid.uuid4().hex[:12]
    attachment_dir.mkdir(parents=True, exist_ok=True)
    dest = attachment_dir / f"{attachment_id}{src.suffix.lower()}"
    shutil.copy2(src, dest)
    with conn:
        conn.execute("INSERT INTO attachments(id, kind, path, sha256, created_at) VALUES (?, ?, ?, ?, ?)",
                     (attachment_id, kind, str(dest), hashlib.sha256(dest.read_bytes()).hexdigest(), int(time.time())))
    return attachment_id, dest


def _run(args: list[str]) -> bool:
    try:
        return subprocess.run(args, capture_output=True, timeout=60, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def to_wav(src: Path) -> Path:
    """转成 16kHz 单声道 WAV。转不了就原样返回，让识别那边自己试——它认得的格式也不少。"""
    if src.suffix.lower() == ".wav":
        return src
    dest = src.with_suffix(".wav")
    return dest if _run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(src), str(dest)]) and dest.exists() else src


def to_jpeg(src: Path) -> Path:
    """转成长边不超过 MAX_IMAGE_EDGE 的 JPEG。"""
    dest = src.with_name(src.stem + ".view.jpg")
    ok = _run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "80", "-Z", str(MAX_IMAGE_EDGE), str(src), "--out", str(dest)])
    if not ok or not dest.exists() or dest.stat().st_size == 0:
        raise MediaError("这张图片我打不开")
    return dest


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
