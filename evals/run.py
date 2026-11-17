"""Eval runner.

    uv run python -m evals.run timeparse          # deterministic, no API calls, free
    uv run python -m evals.run routing --reps 2   # calls Jev (needs TYPESAFE_API_KEY)
    uv run python -m evals.run memory --reps 2
    uv run python -m evals.run guards             # the yes/no checks before deleting, replacing, or trusting "done"

Accuracy is printed next to its noise floor (about 1/sqrt(n * reps)): a change
smaller than that cannot be told apart from run-to-run variation. The seed sets
are small on purpose; grow them from real usage (the episodes table) toward 100+.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from alf import db, timeparse
from alf.config import load_dotenv
from alf.decide import RELATE_TASK, RELATIONS, Decider, JevBackend, OfflineBackend, choice, route_questions

CASES = Path(__file__).parent / "cases"
TZ = ZoneInfo("Australia/Sydney")
DEFAULT_NOW = "2026-09-22T10:00"  # a Tuesday; case expectations are written against it


def load(name: str) -> list[dict]:
    lines = (CASES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check_timeparse(case: dict) -> list[str]:
    """How the parser's output differs from the case's expectation; empty means pass."""
    now = datetime.fromisoformat(case.get("now", DEFAULT_NOW)).replace(tzinfo=TZ)
    p = timeparse.parse(case["text"], now)
    errors = []
    got = [t.when.strftime("%Y-%m-%dT%H:%M") for t in p.times]
    if got != case["times"]:
        errors.append(f"times {got} != {case['times']}")
    if p.lead_minutes != case.get("lead", 0):
        errors.append(f"lead {p.lead_minutes} != {case.get('lead', 0)}")
    if p.recurrence != case.get("recurrence"):
        errors.append(f"recurrence {p.recurrence} != {case.get('recurrence')}")
    if "title" in case and (title := timeparse.title(case["text"], p)) != case["title"]:
        errors.append(f"title {title!r} != {case['title']!r}")
    return errors


def run_timeparse(args) -> int:
    cases = load("timeparse")
    failed = 0
    for case in cases:
        if errors := check_timeparse(case):
            failed += 1
            print(f"✗ {case['text']}\n    " + "\n    ".join(errors))
    print(f"timeparse: {len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


def _decider(args) -> tuple[Decider, object]:
    conn = db.connect(":memory:")
    if args.offline:
        print("⚠ --offline: keyword rules, not Jev. Only checks the runner itself.\n")
        return Decider(OfflineBackend(), conn), conn
    load_dotenv()
    if args.backend == "deepseek":
        # The same closed questions, answered by the generative model instead of Jev.
        from alf.config import settings
        from alf.decide import LLMDecisionBackend
        from alf.llm import DeepSeek

        cfg = settings()
        if not cfg.llm:
            sys.exit("DEEPSEEK_API_KEY is not set (put it in .env)")
        return Decider(LLMDecisionBackend(DeepSeek(cfg.llm)), conn), conn
    if not os.getenv("TYPESAFE_API_KEY"):
        sys.exit("TYPESAFE_API_KEY is not set (put it in .env)")
    return Decider(JevBackend(), conn), conn


def run_routing(args) -> int:
    decider, conn = _decider(args)
    rows = []
    for _ in range(args.reps):
        for case in load("routing"):
            a = decider.ask("route", {"message": case["text"]}, route_questions())["intent"]
            rows.append((case["text"], case["intent"], a.choice, a.confidence))
    report(rows, args.threshold)
    report_cost(conn)
    return 0


def run_memory(args) -> int:
    decider, conn = _decider(args)
    rows = []
    for _ in range(args.reps):
        for case in load("memory"):
            q = choice({"task": RELATE_TASK, "existing_note": case["existing"]}, RELATIONS)
            a = decider.ask("memory.relate", {"new_statement": case["new"]}, {"rel": q})["rel"]
            rows.append((f"{case['existing']} → {case['new']}", case["relation"], a.choice, a.confidence))
    report(rows, args.threshold)
    report_cost(conn)
    return 0


def run_guards(args) -> int:
    """The yes/no checks that sit between the chat model and the store. For target/replace/send/said a wrong "yes" is
    the dangerous miss (the wrong thing gets deleted or overwritten, an email goes out without a clear go-ahead, or a
    reminder is created at a time the user never said); for claim/accepted it only costs a re-check, and for
    interesting (the affinity point) a miss costs nothing but a point."""
    from alf.affinity import INTEREST_FLOOR
    from alf.chat import (ACCEPT_FLOOR, ANNOYED_FLOOR, CLAIM_FLOOR, DISPUTE_FLOOR, FULFILL_FLOOR, HARD_FLOOR, LOOP_FLOOR, PACT_FLOOR, PACT_SAID_FLOOR,
                          RULE_FLOOR, SEND_FLOOR, SETTING_FLOOR, TARGET_FLOOR, TIME_SAID_FLOOR)
    from alf.decide import (accepted_question, against_safeguard_question, agreed_question, annoyed_question, asked_setting_question,
                            broke_rule_question, claim_question, confirm_send_question, disputes_question, fulfilled_question, impression_label_question,
                            impression_supported_question, inbox_questions, interesting_question, pact_said_question, raised_question,
                            same_note_question, spoke_of_question, target_question, time_said_question)
    from alf.impressions import LABEL_FLOOR, SUPPORT_FLOOR
    from alf.jobs import KIND_FLOOR, UNWANTED_FLOOR
    from alf.memory import SUPERSEDE_FLOOR

    decider, conn = _decider(args)
    setups = {
        "claim": (CLAIM_FLOOR, claim_question, lambda c: (
            {"user_latest_message": c["said"], "assistant_reply": c["reply"]}, f"{c['said']} / {c['reply']}")),
        "accepted": (ACCEPT_FLOOR, accepted_question, lambda c: (
            {"user_latest_message": c["said"], "assistant_reply": c["reply"]}, f"{c['said']} / {c['reply']}")),
        "target": (TARGET_FLOOR, target_question, lambda c: (
            {"recent_conversation": c["recent"], "user_latest_message": c["said"], "item": c["item"]}, f"「{c['said']}」→ {c['item']}")),
        "send": (SEND_FLOOR, confirm_send_question, lambda c: (
            {"recent_conversation": c["recent"], "user_latest_message": c["said"], "draft": c["draft"]}, f"「{c['said']}」→ 发出草稿")),
        "replace": (SUPERSEDE_FLOOR, same_note_question, lambda c: (
            {"existing_note": c["old"], "proposed_text": c["new"]}, f"{c['old']} → {c['new']}")),
        "said": (TIME_SAID_FLOOR, time_said_question, lambda c: (
            {"now": c["now"], "recent_conversation": c["recent"], "user_latest_message": c["recent"][-1][3:],
             "reminder": {"what": c["what"], "when": c["when"]}}, f"{c['recent'][-1]} → {c['when']}")),
        "interesting": (INTEREST_FLOOR, interesting_question, lambda c: (
            {"assistant_previous_message": c.get("before", ""), "user_latest_message": c["said"]}, c["said"])),
        # 邮件要不要立刻告诉他。门槛是设置里的 email.threshold，这里按默认值 0.6 考。
        "mailnow": (0.6, lambda: inbox_questions()["needs_now"], lambda c: (
            {"from": c["from"], "subject": c["subject"], "body_start": c["body"]}, c["subject"])),
        "annoyed": (ANNOYED_FLOOR, annoyed_question, lambda c: (
            {"assistant_previous_message": c.get("before", ""), "user_latest_message": c["said"]}, c["said"])),
        "spoke": (LOOP_FLOOR, lambda c: spoke_of_question(c["item"]), lambda c: (
            {"assistant_previous_message": "", "user_latest_message": c["said"], "assistant_reply": c["reply"]}, f"他：{c['said']} → {c['item']}")),
        "raised": (LOOP_FLOOR, lambda c: raised_question(c["item"]), lambda c: (
            {"assistant_previous_message": "", "user_latest_message": c["said"], "assistant_reply": c["reply"]}, f"Alf：{c['reply']} → {c['item']}")),
        # 听了会改（2026-11-17）：回复有没有违反他提过的要求；改这项设置（或定这条规矩）是不是他要的。
        "broke": (RULE_FLOOR, lambda c: broke_rule_question(c["rule"]), lambda c: (
            {"user_latest_message": c["said"], "assistant_reply": c["reply"]}, f"「{c['rule']}」← {c['reply']}")),
        "setting": (SETTING_FLOOR, asked_setting_question, lambda c: (
            {"recent_conversation": c["recent"], "user_latest_message": c["said"], "change": c["change"]}, f"「{c['said']}」→ {c['change']}")),
        # 一条要求是不是在让 Alf 绕开硬规矩（发信确认、如实汇报、外部内容不算指示）：是的话不收。
        "hardrule": (HARD_FLOOR, lambda c: against_safeguard_question(c["rule"]), lambda c: ({"proposed_instruction": c["rule"]}, c["rule"])),
        # 新来的信和他表过态的那封是不是同一类（是的话程序照他当时说的办）。
        "mailkind": (KIND_FLOOR, lambda c: inbox_questions(earlier={"1": (c["earlier_from"], c["earlier_subject"])})["kind_1"], lambda c: (
            {"from": c["from"], "subject": c["subject"], "body_start": c["body"]}, f"{c['subject']} ≈ {c['earlier_subject']}")),
        # 他专门给邮件定过规矩时：该不该立刻说（ask=now，按默认门槛 0.6）；是不是根本不用告诉他（ask=unwanted）。
        "mailrule": (None, lambda c: inbox_questions(rules=True)["needs_now" if c["ask"] == "now" else "unwanted"], lambda c: (
            {"from": c["from"], "subject": c["subject"], "body_start": c["body"], "user_instructions_about_email": c["rules"]},
            f"[{c['ask']}] {c['subject']} ← {c['rules'][0]}")),
        # 约定：这件事是不是真的说好了（是才记）；回复是不是在跟他约定一件事（没记下来就这么说要打回）。
        "pact": (PACT_FLOOR, agreed_question, lambda c: (
            {"recent_conversation": c["recent"], "user_latest_message": c["said"], "agreement": c["agreement"]}, f"「{c['said']}」→ {c['agreement']}")),
        "pactsaid": (PACT_SAID_FLOOR, pact_said_question, lambda c: (
            {"user_latest_message": c["said"], "assistant_reply": c["reply"]}, f"{c['said']} / {c['reply']}")),
        # Alf 把自己答应的事标成兑现之前：回复里是不是真的做了。
        "fulfilled": (FULFILL_FLOOR, lambda c: fulfilled_question(c["promise"]), lambda c: ({"assistant_reply": c["reply"]}, f"{c['promise']} ← {c['reply']}")),
        # 印象：他的话撑不撑得起它；它是不是在贴标签；他是不是在说它不对。
        "impression": (SUPPORT_FLOOR, lambda c: impression_supported_question(c["text"]), lambda c: (
            {"user_messages": c["messages"]}, f"{c['text']} ← {' / '.join(c['messages'])}")),
        "label": (LABEL_FLOOR, lambda c: impression_label_question(c["text"]), lambda c: ({}, c["text"])),
        "dispute": (DISPUTE_FLOOR, lambda c: disputes_question(c["text"]), lambda c: (
            {"assistant_previous_message": c.get("before", ""), "user_latest_message": c["said"]}, f"「{c['said']}」→ {c['text']}")),
    }
    failed, total, unsafe = 0, 0, 0
    scores: dict[str, dict[bool, list[float]]] = {}
    for _ in range(args.reps):
        for case in load("guards"):
            if args.kind and case["kind"] not in args.kind.split(","):
                continue
            floor, question, build = setups[case["kind"]]
            if case["kind"] == "mailrule":
                floor = 0.6 if case["ask"] == "now" else UNWANTED_FLOOR
            state, label = build(case)
            q = question(case) if case["kind"] in ("spoke", "raised", "broke", "hardrule", "mailkind", "mailrule",
                                                   "impression", "label", "dispute", "fulfilled") else question()   # 这几类的问题里带着具体内容
            p = decider.ask(f"eval.{case['kind']}", state, {"q": q})["q"].noul
            scores.setdefault(case["kind"], {True: [], False: []})[case["yes"]].append(p)
            if args.kind:
                print(f"  {case['kind']:<11}{p:.2f} {'是' if case['yes'] else '否'}  {label[:80]}")
            total += 1
            if (p >= floor) != case["yes"]:
                failed += 1
                unsafe += p >= floor and case["kind"] in ("target", "replace", "send", "said")
                print(f"✗ {case['kind']:<8}{p:.2f} (floor {floor}, want {'yes' if case['yes'] else 'no'})  {label[:70]}")
    if args.kind:      # 校门槛时用：每一类里"该是"和"不该是"的两堆分数各落在哪
        for kind, by in scores.items():
            yes, no = by[True], by[False]
            print(f"  {kind:<11}是：{min(yes):.2f}–{max(yes):.2f}（{len(yes)}）  不是：{min(no):.2f}–{max(no):.2f}（{len(no)}）")
    print(f"guards: {total - failed}/{total} passed · {unsafe} unsafe (would have deleted, overwritten, sent or scheduled something it should not)")
    report_cost(conn)
    return 1 if unsafe else 0


def report(rows: list[tuple[str, str, str, float]], threshold: float) -> None:
    n = len(rows)
    correct = sum(want == got for _, want, got, _ in rows)
    acted = [(want, got) for _, want, got, conf in rows if conf >= threshold]
    print(f"accuracy              {correct / n:6.1%}  ({correct}/{n})   noise floor ≈ ±{1 / math.sqrt(n):.0%}")
    print(f"acted on              {len(acted) / n:6.1%}  (confidence ≥ {threshold}; Alf asks about the rest)")
    if acted:
        right = sum(want == got for want, got in acted)
        print(f"accuracy when acting  {right / len(acted):6.1%}  ← the wrong actions a user would actually see")
    labels = sorted({want for _, want, _, _ in rows} | {got for _, _, got, _ in rows})
    print(f"\n{'label':<18}{'precision':>10}{'recall':>8}{'n':>5}")
    for label in labels:
        tp = sum(1 for _, want, got, _ in rows if want == got == label)
        predicted = sum(1 for _, _, got, _ in rows if got == label)
        actual = sum(1 for _, want, _, _ in rows if want == label)
        print(f"{label:<18}{tp / predicted if predicted else 0:>10.0%}{tp / actual if actual else 0:>8.0%}{actual:>5}")
    failures = Counter((text, want, got) for text, want, got, _ in rows if want != got)
    if failures:
        print("\nfailures:")
        for (text, want, got), k in failures.most_common():
            print(f"  ✗ {text}   want {want}, got {got}  ×{k}")


def report_cost(conn) -> None:
    rows = conn.execute("SELECT latency_ms, input_tokens FROM model_calls WHERE ok = 1").fetchall()
    if not rows:
        return
    latency = sorted(r["latency_ms"] for r in rows)
    tokens = sum(r["input_tokens"] or 0 for r in rows)
    p95 = latency[min(len(latency) - 1, int(len(latency) * 0.95))]
    print(f"\n{len(rows)} calls · latency p50 {latency[len(latency) // 2]} ms, p95 {p95} ms · {tokens} input tokens")


def main() -> int:
    parser = argparse.ArgumentParser(prog="evals.run")
    parser.add_argument("suite", choices=["timeparse", "routing", "memory", "guards"])
    parser.add_argument("--reps", type=int, default=1, help="repeat each case to measure run-to-run variance")
    parser.add_argument("--threshold", type=float, default=0.6, help="confidence Alf needs before acting")
    parser.add_argument("--backend", choices=["jev", "deepseek"], default="jev", help="who answers the closed questions")
    parser.add_argument("--offline", action="store_true", help="use keyword rules instead of Jev (runner check only)")
    parser.add_argument("--kind", default="", help="guards only: run just these kinds (comma-separated) and print each kind's score ranges")
    args = parser.parse_args()
    return {"timeparse": run_timeparse, "routing": run_routing, "memory": run_memory, "guards": run_guards}[args.suite](args)


if __name__ == "__main__":
    sys.exit(main())
