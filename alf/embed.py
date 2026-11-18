"""本机的嵌入模型：把一段话变成一个向量，意思相近的话，向量也相近。

用的是 multilingual-e5-small（ONNX，CPU 上一句话几毫秒），中英混说都认。全程在本机算，内容不出这台机器。
模型文件要先下载（`uv run alf index` 会下，约 490 MB，和 Whisper 放在同一个缓存目录）；没下载时这里返回 None，
检索就退回只按字面找——和这个项目别处一样，少一样能力不该让整件事停下来。

两件用的时候要知道的事：
- E5 的约定：要找的那句话前面加 "query: "，被找的内容前面加 "passage: "。不加的话准确度明显下降。
- 相似度的绝对值不能当门槛：E5 的余弦全挤在 0.7–0.9 之间，一句"在吗"和哪条记忆都有 0.83。所以别处只用名次，不用分数。
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
import time

log = logging.getLogger("alf.embed")

MODEL = os.getenv("ALF_EMBED_MODEL", "intfloat/multilingual-e5-small")
FILES = ("onnx/model.onnx", "onnx/tokenizer.json")
MAX_TOKENS = 256          # 记忆、对话片段、邮件提要都很短；再长的截掉
BATCH = 16
IDLE = 600                # 这么久没用就把模型从内存里放掉（装着要占几百 MB），下次用再装，不到一秒


def files(download: bool = False) -> tuple[str, str] | None:
    """模型和分词器在本机的路径。download=False 时只看缓存，绝不联网；没有就是 None。"""
    try:
        from huggingface_hub import hf_hub_download

        return tuple(hf_hub_download(MODEL, name, local_files_only=not download) for name in FILES)   # type: ignore[return-value]
    except Exception as e:
        if download:
            raise
        log.debug("嵌入模型不在本机：%s", e)
        return None


class E5:
    """懒加载：第一次用到才把模型装进内存。几个线程（聊天、心跳）共用一份，用锁串起来。"""

    def __init__(self, model_path: str, tokenizer_path: str):
        self.name = MODEL
        self._paths = (model_path, tokenizer_path)
        self._lock = threading.Lock()
        self._session = self._tokenizer = None
        self._used = 0.0

    def _load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        started = time.monotonic()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2            # 后台的活，不抢他正在用的 CPU
        self._session = ort.InferenceSession(self._paths[0], sess_options=options, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(self._paths[1])
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding(pad_id=self._tokenizer.token_to_id("<pad>") or 1, pad_token="<pad>")
        self._needs_types = any(i.name == "token_type_ids" for i in self._session.get_inputs())
        log.info("嵌入模型装好了（%.1f 秒）", time.monotonic() - started)

    def encode(self, texts: list[str], *, query: bool = False):
        """一组话 → 一组归一化的向量（float32，N × 维数）。query=True 表示这是要找的那句话。"""
        import numpy as np

        prefix = "query: " if query else "passage: "
        out = []
        with self._lock:
            if self._session is None:
                self._load()
            self._used = time.monotonic()
            for i in range(0, len(texts), BATCH):
                enc = self._tokenizer.encode_batch([prefix + " ".join(t.split()) for t in texts[i:i + BATCH]])
                ids = np.array([e.ids for e in enc], dtype=np.int64)
                mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
                feed = {"input_ids": ids, "attention_mask": mask}
                if self._needs_types:
                    feed["token_type_ids"] = np.zeros_like(ids)
                hidden = self._session.run(None, feed)[0]
                pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)     # 按有效的词取平均
                out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), dtype=np.float32)

    def release_if_idle(self, idle: float = IDLE) -> bool:
        with self._lock:
            if self._session is not None and time.monotonic() - self._used > idle:
                self._session = self._tokenizer = None
                log.info("嵌入模型有一阵没用了，从内存里放掉")
                return True
        return False


_default: E5 | None = None
_checked = 0.0


def _release_at_exit() -> None:
    """进程退出前先把模型放掉。留给解释器收尾时再销毁的话，onnxruntime 的线程和锁可能已经没了，
    退出时会报一个 C++ 的异常（recursive_mutex lock failed）——不影响结果，但日志里难看，launchd 还会当成崩溃。"""
    if _default is not None:
        _default._session = _default._tokenizer = None


atexit.register(_release_at_exit)


def default() -> E5 | None:
    """全进程共用的那一个。模型不在本机（或者没装 onnxruntime）就是 None；隔一分钟再看一次，免得每句话都去翻缓存目录。"""
    global _default, _checked
    if _default is None and time.monotonic() - _checked > 60:
        _checked = time.monotonic()
        import importlib.util

        # 没装 onnxruntime 的环境里，模型在不在都用不了
        paths = files() if importlib.util.find_spec("onnxruntime") else None
        if paths:
            _default = E5(*paths)
    return _default


def status() -> dict:
    """给自检和设置页看的：模型在不在本机。不装模型，也不联网。"""
    return {"model": MODEL, "present": files() is not None}
