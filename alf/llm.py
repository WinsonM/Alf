"""生成式模型（DeepSeek）：负责听懂人话、抽取要点、像朋友一样回话。

接口是 OpenAI 兼容的 /chat/completions。几个按官方文档做的选择：
- 关掉思考模式：聊天要在几秒内回，而且思考模式下带工具的多轮请求必须回传 reasoning_content。
- 不用测试版的 strict 工具模式，所以模型给的工具参数要自己校验（文档明说可能不是合法 JSON）。
- 缓存是服务端自动做的前缀匹配，这里只负责把命中的 token 数记下来。
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .config import LLMConfig

RETRYABLE = {408, 429, 500, 503, 504}
RETRY_WAITS = (1.0, 3.0)   # 最多重试两次；再久用户就等不起了


class LLMError(RuntimeError):
    """模型这次没答上来。消息是给用户看的。"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None   # None = 模型给的参数不是合法的 JSON 对象


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    message: dict = field(default_factory=dict)   # 原样的 assistant 消息，工具往返时要原样放回去
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None


class LLMBackend(Protocol):
    name: str
    model: str

    def complete(self, messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False,
                 max_tokens: int = 1024, temperature: float | None = None) -> LLMReply: ...


class DeepSeek:
    name = "deepseek"

    def __init__(self, cfg: LLMConfig, client: httpx.Client | None = None, sleep=time.sleep):
        self.model = cfg.model
        self._url = f"{cfg.base_url}/chat/completions"
        self._headers = {"Authorization": f"Bearer {cfg.api_key}"}
        self._client = client or httpx.Client(timeout=httpx.Timeout(cfg.timeout, connect=10.0))
        self._sleep = sleep

    def complete(self, messages, *, tools=None, json_mode=False, max_tokens=1024, temperature=None) -> LLMReply:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "thinking": {"type": "disabled"}, "stream": False}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if temperature is not None:
            body["temperature"] = temperature
        data = self._post(body)
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("DeepSeek 返回了看不懂的内容") from e
        if choice.get("finish_reason") == "content_filter":
            raise LLMError("这条内容被 DeepSeek 的内容审核拦下了")
        calls = []
        for c in message.get("tool_calls") or []:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None
            calls.append(ToolCall(c.get("id", ""), fn.get("name", ""), args if isinstance(args, dict) else None))
        usage = data.get("usage") or {}
        return LLMReply(
            content=message.get("content") or "",
            tool_calls=calls,
            message=message,
            usage={"model": data.get("model", self.model), "input_tokens": usage.get("prompt_tokens"),
                   "output_tokens": usage.get("completion_tokens"), "cached_tokens": usage.get("prompt_cache_hit_tokens")},
            finish_reason=choice.get("finish_reason"),
        )

    def _post(self, body: dict) -> dict:
        last = "没有响应"
        for attempt in range(len(RETRY_WAITS) + 1):
            try:
                r = self._client.post(self._url, headers=self._headers, json=body)
            except httpx.TimeoutException:
                last = "DeepSeek 响应超时"
            except httpx.HTTPError as e:
                last = f"连不上 DeepSeek（{type(e).__name__}）"
            else:
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError as e:
                        raise LLMError("DeepSeek 返回的不是 JSON") from e
                detail = _error_message(r)
                if r.status_code == 401:
                    raise LLMError("DeepSeek 的 API key 无效，检查 .env 里的 DEEPSEEK_API_KEY")
                if r.status_code == 402:
                    raise LLMError("DeepSeek 账户余额不足，需要充值")
                if r.status_code not in RETRYABLE:
                    raise LLMError(f"DeepSeek 拒绝了请求（{r.status_code}）：{detail}")
                last = "DeepSeek 限流了" if r.status_code == 429 else f"DeepSeek 服务端出错（{r.status_code}）"
            if attempt < len(RETRY_WAITS):
                self._sleep(RETRY_WAITS[attempt])
        raise LLMError(last)


def _error_message(r: httpx.Response) -> str:
    try:
        return str(r.json()["error"]["message"])[:200]
    except Exception:
        return r.text[:200]


class LLM:
    """把后端和一个数据库连接绑在一起：每次调用都记一行 model_calls（延迟、token、缓存命中、失败原因）。"""

    def __init__(self, backend: LLMBackend, conn: sqlite3.Connection):
        self.backend = backend
        self.conn = conn

    def chat(self, purpose: str, messages: list[dict], **kwargs) -> LLMReply:
        started = time.monotonic()
        usage: dict = {}
        ok, error = False, None
        try:
            reply = self.backend.complete(messages, **kwargs)
            usage, ok = reply.usage, True
            return reply
        except Exception as e:
            error = str(e) if isinstance(e, LLMError) else repr(e)
            raise
        finally:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO model_calls(ts, purpose, model, input_tokens, output_tokens, cached_tokens,"
                    " latency_ms, ok, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (int(time.time()), purpose, usage.get("model", self.backend.model), usage.get("input_tokens"),
                     usage.get("output_tokens"), usage.get("cached_tokens"),
                     int((time.monotonic() - started) * 1000), ok, error))

    def json(self, purpose: str, messages: list[dict], **kwargs) -> dict:
        """要一个 JSON 对象。文档提醒：JSON 模式下内容偶尔会是空的，所以空了或坏了就再要一次。"""
        for _ in range(2):
            reply = self.chat(purpose, messages, json_mode=True, **kwargs)
            try:
                data = json.loads(reply.content)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        raise LLMError("模型没有按要求输出 JSON")
