"""DeepSeek 客户端：HTTP 被拦截，请求和响应的形状按官方文档（api-docs.deepseek.com，2026-11）。"""
import json

import httpx
import pytest

from alf.config import LLMConfig
from alf.decide import Decider, DecisionError, LLMDecisionBackend, choice, noul
from alf.llm import LLM, DeepSeek, LLMError

from .conftest import FakeLLM


def completion(content=None, tool_calls=None, finish="stop"):
    return {"id": "x", "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": finish, "message": {
                "role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940,
                      "prompt_cache_hit_tokens": 640, "prompt_cache_miss_tokens": 260}}


def client(*responses, seen=None):
    """按顺序返回 responses（(状态码, JSON) 或异常）；seen 收集请求。"""
    queue = list(responses)

    def handler(request):
        if seen is not None:
            seen.append({"url": str(request.url), "auth": request.headers["authorization"], "body": json.loads(request.content)})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body)

    waits = []
    ds = DeepSeek(LLMConfig(api_key="sk-test-not-real"), client=httpx.Client(transport=httpx.MockTransport(handler)),
                  sleep=waits.append)
    ds.waits = waits
    return ds


def test_request_follows_the_documented_shape():
    seen = []
    ds = client((200, completion("在的")), seen=seen)
    tools = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}]
    reply = ds.complete([{"role": "user", "content": "在吗"}], tools=tools, max_tokens=800)
    [req] = seen
    assert req["url"] == "https://api.deepseek.com/chat/completions" and req["auth"] == "Bearer sk-test-not-real"
    assert req["body"] == {"model": "deepseek-flash", "messages": [{"role": "user", "content": "在吗"}], "max_tokens": 800,
                           "thinking": {"type": "disabled"}, "stream": False, "tools": tools}
    assert reply.content == "在的" and reply.tool_calls == [] and reply.finish_reason == "stop"
    assert reply.usage == {"model": "deepseek-flash", "input_tokens": 900, "output_tokens": 40, "cached_tokens": 640}


def test_json_mode_and_temperature_are_optional_extras():
    seen = []
    client((200, completion("{}")), seen=seen).complete([{"role": "user", "content": "x"}], json_mode=True, temperature=0)
    assert seen[0]["body"]["response_format"] == {"type": "json_object"} and seen[0]["body"]["temperature"] == 0
    assert "tools" not in seen[0]["body"]


def test_tool_calls_are_parsed_and_broken_arguments_are_flagged():
    calls = [{"id": "c1", "type": "function", "function": {"name": "create_reminder", "arguments": '{"title": "交作业", "time_text": "明天"}'}},
             {"id": "c2", "type": "function", "function": {"name": "cancel_reminder", "arguments": '{"id": 3'}},
             {"id": "c3", "type": "function", "function": {"name": "cancel_reminder", "arguments": "[1]"}}]
    reply = client((200, completion(None, calls, "tool_calls"))).complete([{"role": "user", "content": "x"}])
    assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == [
        ("c1", "create_reminder", {"title": "交作业", "time_text": "明天"}), ("c2", "cancel_reminder", None), ("c3", "cancel_reminder", None)]
    assert reply.content == "" and reply.message["tool_calls"] == calls     # 原样留着，下一轮要原样发回去


def test_transient_failures_are_retried_then_succeed():
    ds = client((503, {"error": {"message": "overloaded"}}), httpx.ReadTimeout("slow"), (200, completion("好")))
    assert ds.complete([{"role": "user", "content": "x"}]).content == "好"
    assert ds.waits == [1.0, 3.0]


@pytest.mark.parametrize("responses, message, retries", [
    ([(401, {"error": {"message": "Authentication Fails"}})], "DeepSeek 的 API key 无效", 0),
    ([(402, {"error": {"message": "Insufficient Balance"}})], "DeepSeek 账户余额不足", 0),
    ([(400, {"error": {"message": "bad tools"}})], r"DeepSeek 拒绝了请求（400）：bad tools", 0),
    ([(429, {})] * 3, "DeepSeek 限流了", 2),
    ([httpx.ConnectError("no route")] * 3, r"连不上 DeepSeek（ConnectError）", 2),
    ([(200, {"choices": []})], "DeepSeek 返回了看不懂的内容", 0),
    ([(200, completion("…", finish="content_filter"))], "内容审核", 0),
])
def test_failures_become_readable_errors(responses, message, retries):
    ds = client(*responses)
    with pytest.raises(LLMError, match=message):
        ds.complete([{"role": "user", "content": "x"}])
    assert len(ds.waits) == retries


def test_every_call_is_logged_with_cache_hits_and_failures(conn):
    llm = LLM(client((200, completion("好")), (401, {})), conn)
    llm.chat("chat", [{"role": "user", "content": "x"}])
    with pytest.raises(LLMError):
        llm.chat("chat", [{"role": "user", "content": "x"}])
    rows = conn.execute("SELECT purpose, model, input_tokens, output_tokens, cached_tokens, ok, error FROM model_calls").fetchall()
    assert [tuple(r) for r in rows] == [("chat", "deepseek-flash", 900, 40, 640, 1, None),
                                        ("chat", "deepseek-flash", None, None, None, 0, "DeepSeek 的 API key 无效，检查 .env 里的 DEEPSEEK_API_KEY")]


def test_json_helper_asks_again_when_the_content_is_empty_or_broken(conn):
    fake = FakeLLM(facts=["   ", {"facts": []}])
    assert LLM(fake, conn).json("memory.extract", []) == {"facts": []} and len(fake.requests) == 2
    with pytest.raises(LLMError, match="没有按要求输出 JSON"):
        LLM(FakeLLM(facts=["nope", "[1, 2]"]), conn).json("memory.extract", [])


QUESTIONS = {"rel": choice({"task": "compare", "existing_note": "住在 Glebe"}, {"duplicate": "same", "updates": "changed", "unrelated": "other"}),
             "fact": noul("States a lasting fact.")}


def test_generative_model_can_stand_in_for_jev_on_closed_questions(conn):
    fake = FakeLLM(facts=[{"answers": {"rel": {"choice": "updates", "confidence": 0.9}, "fact": {"noul": 0.25}}}])
    answers = Decider(LLMDecisionBackend(fake), conn).ask("memory.relate", {"new_statement": "搬到 Newtown 了"}, QUESTIONS)
    assert (answers["rel"].choice, answers["rel"].confidence) == ("updates", 0.9)
    assert (answers["fact"].noul, answers["fact"].confidence) == (0.25, 0.75)
    sent = json.loads(fake.requests[0]["messages"][1]["content"])
    assert sent == {"state": {"new_statement": "搬到 Newtown 了"}, "questions": QUESTIONS} and fake.requests[0]["json_mode"]
    assert tuple(conn.execute("SELECT purpose, model, input_tokens, ok FROM model_calls").fetchone()) == ("memory.relate", "fake-model", 100, 1)


@pytest.mark.parametrize("raw", [
    {"answers": {"rel": {"choice": "maybe"}, "fact": {"noul": 0.5}}},     # 选项之外的答案
    {"answers": {"rel": {"choice": "updates"}}},                          # 漏答
    {"result": "?"}, "不是 JSON", LLMError("DeepSeek 限流了"),
])
def test_malformed_stand_in_answers_are_refused_not_guessed(conn, raw):
    with pytest.raises(DecisionError):
        Decider(LLMDecisionBackend(FakeLLM(facts=[raw])), conn).ask("memory.relate", {}, QUESTIONS)
