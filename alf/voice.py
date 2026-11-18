"""本机的语音模型：哪一段有人在说话、是几个人轮流说的、这是谁的声音、说了什么。

听手表录音（hearing.py）要用四样东西，全在本机算，录音不出这台机器：

- 有没有人声：Silero VAD。faster-whisper 自带，不用另外下载。一整天的录音大半是安静的，先用它筛掉。
- 谁在什么时候说：pyannote 的切分模型（经 sherpa-onnx 跑）。它只管"这一小段里是几个人轮流说的"，
  编号只在这一小段里有意义；跨段、跨天认出同一个人靠的是下面的声纹。
- 声纹：3D-Speaker 的 CAM++（中英文都训过）。一段声音 → 一个 192 维的向量，同一个人的向量靠得近。
- 说了什么：SenseVoice。选它而不是 Whisper small，是因为在同一段中文上试过：Whisper small 把"初稿"听成
  "出报"、"之前"听成"追钱"；SenseVoice 这几处都对，而且快几十倍（一天的录音几分钟转完）。

模型文件要先下载（`uv run alf hear --setup`，一共约 200 MB）。没下载时 default() 返回 None，听录音这件事
整个不启动——和这个项目别处一样，少一样能力不该让别的事停下来。

两件用的时候要知道的事：
- SenseVoice 不给"把握"的分数，所以听到的话没有置信度；防听错靠的是后面的几道核对（见 days.py）。
- 它对没有人声的声音也会硬给出一两个字（常是韩文的"그."）。所以转写前先过人声检测，转写后只留认得的语言。
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("alf.voice")

SAMPLE_RATE = 16_000
HOME = Path(os.getenv("ALF_VOICE_MODELS") or Path.home() / ".cache" / "alf" / "voice")
_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
_TURNS = "sherpa-onnx-pyannote-segmentation-3-0"
_WORDS = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
PRINT_MODEL = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced"   # 声纹和算它的模型绑在一起：换了模型，旧声纹作废


@dataclass(frozen=True)
class ModelFile:
    path: str               # 放在 HOME 下的哪里
    url: str
    member: str | None      # 压缩包里的哪一个文件；None = 下载下来的就是它
    sha256: str             # 下载之后核对；对不上就不用
    mb: float


FILES = {
    "turns": ModelFile(f"{_TURNS}/model.onnx", f"{_RELEASES}/speaker-segmentation-models/{_TURNS}.tar.bz2", f"{_TURNS}/model.onnx",
                       "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079", 7.0),
    "print": ModelFile(f"{PRINT_MODEL}.onnx", f"{_RELEASES}/speaker-recongition-models/{PRINT_MODEL}.onnx", None,
                       "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2", 28.3),
    "words": ModelFile(f"{_WORDS}/model.int8.onnx", f"{_RELEASES}/asr-models/{_WORDS}.tar.bz2", f"{_WORDS}/model.int8.onnx",
                       "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51", 163.0),
    "tokens": ModelFile(f"{_WORDS}/tokens.txt", f"{_RELEASES}/asr-models/{_WORDS}.tar.bz2", f"{_WORDS}/tokens.txt",
                        "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc", 0.0),
}
LANGUAGES = ("zh", "en", "yue")     # 只留这几种语言的转写；别的多半是把杂音硬听成了字
IDLE = 600                          # 这么久没用就把模型从内存里放掉，下次用再装（一秒左右）
MIN_PRINT = 1.0                     # 不到这么多秒的声音算不出靠得住的声纹
PRINT_SECONDS = 30.0                # 算一个人的声纹最多用这么多秒：再多也不会更准，只会更慢
ISLAND_GAP = 3.0                    # 两段人声隔不到这么久，算同一场对话，一起切分
ISLAND_PAD = 0.3
ISLAND_MAX = 300.0
TURN_GAP = 0.8                      # 同一个人挨着的两句隔不到这么久，合成一句去转写
TURN_MAX = 25.0                     # SenseVoice 一次最多听三十秒；留点余量


class VoiceError(RuntimeError):
    """模型或者音频出了问题。消息是给人看的。"""


@dataclass
class Turn:
    start: float            # 秒，相对于这段音频的开头
    end: float
    speaker: int            # 这一小段对话里的第几个人；出了这一小段没有意义

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class Heard:
    text: str
    language: str           # zh | en | yue……；没听出字来是 ""


def files() -> dict[str, str] | None:
    """四个模型文件在本机的路径。缺任何一个就是 None。只看文件在不在，不联网。"""
    paths = {name: HOME / f.path for name, f in FILES.items()}
    return {name: str(p) for name, p in paths.items()} if all(p.is_file() for p in paths.values()) else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(say=print) -> None:
    """把缺的模型文件下载下来（`alf hear --setup` 调的）。

    压缩包里还带着别人的脚本，这里只取出点了名的那一个文件，别的不落盘、更不执行；每个文件都核对摘要，
    对不上就丢掉——模型是从网上来的，不该下到什么就用什么。
    """
    HOME.mkdir(parents=True, exist_ok=True)
    missing = {name: f for name, f in FILES.items() if not (HOME / f.path).is_file()}
    by_url: dict[str, list[ModelFile]] = {}
    for f in missing.values():
        by_url.setdefault(f.url, []).append(f)
    for url, wanted in by_url.items():
        say(f"下载 {url.rsplit('/', 1)[1]}（约 {max(sum(f.mb for f in wanted), 0.1):.0f} MB）……")
        with tempfile.TemporaryDirectory(dir=HOME) as tmp:
            fetched = Path(tmp) / "download"
            with urllib.request.urlopen(url, timeout=60) as r, fetched.open("wb") as out:
                shutil.copyfileobj(r, out, 1 << 20)
            for f in wanted:
                staged = Path(tmp) / "file"
                if f.member is None:
                    shutil.copyfile(fetched, staged)
                else:
                    with tarfile.open(fetched, "r:bz2") as tar:
                        src = tar.extractfile(tar.getmember(f.member))
                        if src is None:
                            raise VoiceError(f"压缩包里没有 {f.member}")
                        with staged.open("wb") as out:
                            shutil.copyfileobj(src, out, 1 << 20)
                if _sha256(staged) != f.sha256:
                    raise VoiceError(f"{f.path} 下载下来和预期的不一样，没有用它。可能是网络出了岔子，重跑一次试试")
                dest = HOME / f.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, dest)
    say("模型都在了。" if files() else "还有模型没下全。")


def decode(path: Path | str):
    """一个音频文件 → 16 kHz 单声道的 float32。用 faster-whisper 带的解码器（PyAV），m4a、wav、mp3 都认，
    不依赖系统里装没装 ffmpeg：服务是 launchd 起的，PATH 里通常没有 Homebrew。"""
    from faster_whisper.audio import decode_audio

    try:
        return decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    except Exception as e:
        raise VoiceError(f"这个录音文件读不出来：{type(e).__name__}") from e


def duration(path: Path | str) -> float | None:
    """录音有多长（秒）。只读文件头，不解码；读不出来是 None。"""
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration:
                return container.duration / 1_000_000
            stream = container.streams.audio[0]
            return float(stream.duration * stream.time_base) if stream.duration else None
    except Exception:
        return None


def islands(audio) -> list[tuple[float, float]]:
    """这段音频里有人声的几块（秒）。挨得近的并成一块，前后各留一点余量，太长的切开。没有人声就是空的。"""
    from faster_whisper import vad

    spans = vad.get_speech_timestamps(audio, vad.VadOptions(min_silence_duration_ms=500, speech_pad_ms=200))
    total = len(audio) / SAMPLE_RATE
    out: list[list[float]] = []
    for s in spans:
        start, end = max(0.0, s["start"] / SAMPLE_RATE - ISLAND_PAD), min(total, s["end"] / SAMPLE_RATE + ISLAND_PAD)
        if out and start - out[-1][1] <= ISLAND_GAP and end - out[-1][0] <= ISLAND_MAX:
            out[-1][1] = end
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def join_turns(turns: list[Turn]) -> list[Turn]:
    """同一个人挨着的几句合成一句（中间只隔了换气的工夫），太长的不合。合了之后转写有上下文，句子也完整。"""
    out: list[Turn] = []
    for t in sorted(turns, key=lambda t: t.start):
        last = out[-1] if out else None
        if last and last.speaker == t.speaker and t.start - last.end <= TURN_GAP and t.end - last.start <= TURN_MAX:
            last.end = max(last.end, t.end)
        else:
            out.append(Turn(t.start, t.end, t.speaker))
    return out


class Ears:
    """三个模型装在一起，懒加载：第一次用到才装进内存。听录音的线程和导入录音的命令可能同时用，用锁串起来。"""

    name = PRINT_MODEL

    def __init__(self, paths: dict[str, str]):
        self._paths = paths
        self._lock = threading.Lock()
        self._turns = self._print = self._words = None
        self._used = 0.0

    def _load(self) -> None:
        import sherpa_onnx as so

        started = time.monotonic()
        printer = so.SpeakerEmbeddingExtractorConfig(model=self._paths["print"], num_threads=2)   # 后台的活，不抢他正在用的 CPU
        config = so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=self._paths["turns"]), num_threads=2),
            embedding=printer,
            # 宁可把一个人分成两个，也不把两个人并成一个：分多了，后面按声纹还能认回同一个人；并错了就分不开了。
            clustering=so.FastClusteringConfig(num_clusters=-1, threshold=0.5),
            min_duration_on=0.3, min_duration_off=0.5)
        if not config.validate():
            raise VoiceError("语音模型的文件不对，重跑 `uv run alf hear --setup`")
        self._turns = so.OfflineSpeakerDiarization(config)
        self._print = so.SpeakerEmbeddingExtractor(printer)
        self._words = so.OfflineRecognizer.from_sense_voice(model=self._paths["words"], tokens=self._paths["tokens"],
                                                            num_threads=2, use_itn=True, language="auto")
        log.info("语音模型装好了（%.1f 秒）", time.monotonic() - started)

    def _ready(self) -> None:
        if self._turns is None:
            self._load()
        self._used = time.monotonic()

    def turns(self, audio) -> list[Turn]:
        """这一块人声里，谁在什么时候说。太短的一块切分模型分不了，当成一个人说的一句。"""
        seconds = len(audio) / SAMPLE_RATE
        if seconds < 1.5:
            return [Turn(0.0, seconds, 0)] if seconds >= 0.3 else []
        with self._lock:
            self._ready()
            result = self._turns.process(audio).sort_by_start_time()
        return [Turn(float(r.start), min(float(r.end), seconds), int(r.speaker)) for r in result if r.end - r.start >= 0.3]

    def voiceprint(self, samples):
        """一段声音 → 归一化的声纹向量；太短算不准，返回 None。"""
        import numpy as np

        if len(samples) < MIN_PRINT * SAMPLE_RATE:
            return None
        with self._lock:
            self._ready()
            stream = self._print.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples[: int(PRINT_SECONDS * SAMPLE_RATE)])
            stream.input_finished()
            vec = np.array(self._print.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else None

    def transcribe(self, samples) -> Heard:
        with self._lock:
            self._ready()
            stream = self._words.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            self._words.decode_stream(stream)
            result = stream.result
        language = (result.lang or "").strip("<|>")
        text = " ".join((result.text or "").split())
        return Heard(text, language) if language in LANGUAGES else Heard("", language)

    def release_if_idle(self, idle: float = IDLE) -> bool:
        with self._lock:
            if self._turns is not None and time.monotonic() - self._used > idle:
                self._turns = self._print = self._words = None
                log.info("语音模型有一阵没用了，从内存里放掉")
                return True
        return False


_default: Ears | None = None
_checked = 0.0


def _release_at_exit() -> None:
    """和 embed.py 一样：进程退出前先放掉模型，免得解释器收尾时 onnxruntime 的线程已经没了。"""
    if _default is not None:
        _default._turns = _default._print = _default._words = None


atexit.register(_release_at_exit)


def default() -> Ears | None:
    """全进程共用的那一个。模型不在本机（或者没装 sherpa-onnx）就是 None；隔一分钟再看一次。"""
    global _default, _checked
    if _default is None and time.monotonic() - _checked > 60:
        _checked = time.monotonic()
        import importlib.util

        paths = files() if importlib.util.find_spec("sherpa_onnx") else None
        if paths:
            _default = Ears(paths)
    return _default


def status() -> dict:
    """给自检和设置页看的：模型在不在本机。不装模型，也不联网。"""
    return {"present": files() is not None, "home": str(HOME)}
