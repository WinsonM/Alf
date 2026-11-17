"""The real typesafe-sdk code path, with HTTP intercepted: no key, no network."""
import json

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient

from alf.decide import Decider, DecisionError, JevBackend, route_questions


def client_returning(status, body, seen=None):
    def handler(request):
        if seen is not None:
            seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx2.Response(status, json=body)
    return TypeSafeClient(api_key="test-not-a-real-key", transport=httpx2.MockTransport(handler))


def test_route_call_shape_and_answer_parsing(conn):
    seen = {}
    client = client_returning(200, {
        "model": "jev-1.13.0",
        "answers": {
            "intent": {"type": "choice", "choice": "recall", "confidence": 0.88,
                       "probabilities": {"recall": 0.88, "other": 0.12}},
            "states_fact": {"type": "noul", "noul": 0.2},
            "fact_category": {"type": "choice", "choice": "other", "confidence": 0.5, "probabilities": {"other": 0.5}},
        },
        "usage": {"input_tokens": 420, "output_tokens": 0},
    }, seen)
    answers = Decider(JevBackend(client), conn).ask("route", {"message": "我老婆生日是哪天"}, route_questions())

    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["body"]["state"] == {"message": "我老婆生日是哪天"}
    assert set(seen["body"]["questions"]) == {"intent", "states_fact", "fact_category"}
    assert (answers["intent"].choice, answers["intent"].confidence) == ("recall", 0.88)
    assert (answers["states_fact"].noul, answers["states_fact"].confidence) == (0.2, 0.8)
    row = conn.execute("SELECT model, input_tokens, ok FROM model_calls").fetchone()
    assert tuple(row) == ("jev-1.13.0", 420, 1)


def test_bad_key_becomes_a_readable_error(conn):
    client = client_returning(401, {"error": {"type": "authentication_error", "message": "invalid key"}})
    with pytest.raises(DecisionError, match="API key"):
        Decider(JevBackend(client), conn).ask("route", {"message": "你好"}, route_questions())
