"""Apple「日历」和「提醒事项」的读写。

实际干活的是一个用系统 EventKit 接口写的小 App（app/alf-calendar.swift，编译后是 dist/AlfCalendar.app）。
- 用 EventKit 而不是 AppleScript，是因为后者查不到重复日程在后续日期上的那一次——每周的课就属于这种。
- 包成 App 并用系统的 open 来启动，是因为 macOS 只让正式的 App 申请这两项权限；直接运行会被不弹窗地拒绝。

这里所有函数都返回一个 dict：成功是结果，失败是 {"error": 给用户看的一句话}，不抛异常——调用方（心跳任务、
对话里的工具）都是把这句话原样转述出去。

写入有两条保护：Alf 建的条目在备注里带一个记号，小 App 删除前会核对它，所以删不到用户自己的东西；
上层另外有一张 apple_items 表记着 Alf 建过什么，撤销只能从那张表里选。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import ROOT

LABELS = {"event": "日历", "reminder": "提醒事项"}
NEEDS_PERMISSION = {"event": "还没授权读日历：在 Alf 的设置里点「授权日历和提醒事项」",
                    "reminder": "还没授权读提醒事项：在 Alf 的设置里点「授权日历和提醒事项」"}
SETTLE = 3.0   # open 偶尔会在小 App 写完结果之前就返回，所以最多再等这么久


def helper_path() -> Path:
    return Path(os.getenv("ALF_CALENDAR_HELPER") or ROOT / "dist" / "AlfCalendar.app")


def launch(helper: Path, output: Path, command: str, *args: str) -> None:
    """经由系统启动小 App 并等它跑完。-g 不抢前台，-n 每次新开一个，-W 等它退出。"""
    subprocess.run(["open", "-W", "-n", "-g", "-a", str(helper), "--args", str(output), command, *args],
                   capture_output=True, text=True, timeout=100, check=False)


def call(command: str, *args, payload: dict | None = None, helper: Path | None = None, run=None) -> dict:
    entity = "reminder" if "reminder" in command else "event"
    what, verb = LABELS[entity], ("读" if command in ("events", "reminders") else "写")
    helper, real = helper or helper_path(), run is None
    run = run or launch
    if not helper.exists():
        return {"error": f"{verb}{what}的小 App 还没编译：在项目目录运行 ./app/build.sh"}
    with tempfile.TemporaryDirectory() as tmp:
        output, extra = Path(tmp) / "out.json", [str(a) for a in args]
        if payload is not None:
            source = Path(tmp) / "in.json"
            source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            extra = [str(source), *extra]
        try:
            run(helper, output, command, *extra)
        except subprocess.TimeoutExpired:
            return {"error": f"{verb}{what}超时：可能在等你允许访问{what}（看看有没有弹窗）"}
        data = _result(output, SETTLE if real else 0)
    if data is None:
        return {"error": f"{verb}{what}失败：小 App 没有给出结果"}
    error = data.get("error")
    if not error:
        return data
    if error == "not_determined":
        return {"error": NEEDS_PERMISSION[entity]}
    if error == "timeout":
        return {"error": f"等了一分多钟没有等到授权。看看有没有「Alf 日历助手想要访问{what}」的弹窗，点允许后再试"}
    if error == "denied":
        return {"error": f"没有访问{what}的权限：系统设置 → 隐私与安全性 → {what}，给 Alf（或 python）打开"}
    if error == "not_found":
        return {"error": f"在{what}里找不到这一条，可能已经被删掉了"}
    if error == "not_by_alf":
        return {"error": "这一条不是 Alf 建的，不能由 Alf 删除"}
    if error == "no_calendar":
        return {"error": f"「{what}」里没有可以写入的默认列表"}
    return {"error": f"{verb}{what}失败：{error}"}


def _result(output: Path, wait: float) -> dict | None:
    deadline = time.monotonic() + wait
    while True:
        try:
            data = json.loads(output.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.1)


def read_events(days: int = 2, ask: bool = False, **kw) -> dict:
    """ask=False（后台读取）：没权限就直接说没权限，不弹窗。ask=True（用户点了授权按钮）：向用户申请。"""
    return call("events", max(1, min(int(days), 31)), *(["ask"] if ask else []), **kw)


def read_reminders(days: int = 7, ask: bool = False, **kw) -> dict:
    return call("reminders", max(1, min(int(days), 31)), *(["ask"] if ask else []), **kw)


def add_event(title: str, start: datetime, end: datetime, *, all_day: bool = False, location: str = "", **kw) -> dict:
    return call("add-event", payload={"title": title, "start": start.isoformat(timespec="seconds"),
                                      "end": end.isoformat(timespec="seconds"), "all_day": all_day, "location": location}, **kw)


def delete_event(event_id: str, **kw) -> dict:
    return call("delete-event", event_id, **kw)


def add_reminder(title: str, due: datetime | None = None, *, has_time: bool = True, **kw) -> dict:
    payload: dict = {"title": title}
    if due is not None:
        payload.update(due=due.isoformat(timespec="seconds"), has_time=has_time)
    return call("add-reminder", payload=payload, **kw)


def delete_reminder(item_id: str, **kw) -> dict:
    return call("delete-reminder", item_id, **kw)
