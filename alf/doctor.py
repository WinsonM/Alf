"""自检：Alf 要跑起来得凑齐好几样东西（两个 key、iMessage 地址、两项系统权限），这里逐项检查并说明怎么补。

命令行（`alf doctor`）和设置页用的是同一套检查。注意权限是按"哪个程序在运行"给的：iMessage 能不能用，
以后台服务自己的报告为准，而不是看运行这条命令的终端有没有权限。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass

from .config import Settings
from .imessage import ChannelError, ChatDB, imessage_account_count, send
from .llm import DeepSeek, LLMError

TEST_MESSAGE = "这是 Alf 的测试消息，收到说明发送正常。"


@dataclass
class Check:
    key: str
    ok: bool | None      # True 正常 / False 要处理 / None 只是说明
    title: str
    detail: str = ""


def fetch_status(port: int) -> dict | None:
    """问正在运行的后台服务要状态；没在跑就返回 None。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3) as r:
            return json.load(r)
    except Exception:
        return None


def collect(cfg: Settings, status: dict | None, send_test: bool = False) -> list[Check]:
    """status 是后台服务的状态（/api/status 的内容）；None 表示服务没在运行。"""
    out: list[Check] = []
    channel = next((c for c in (status or {}).get("channels", []) if c["name"] == "imessage" and c.get("configured")), None)

    # 1. 生成式模型
    if not cfg.llm:
        out.append(Check("deepseek", False, "DeepSeek", "在 .env 里填 DEEPSEEK_API_KEY（platform.deepseek.com 申请）"))
    else:
        try:
            started = time.monotonic()
            reply = DeepSeek(cfg.llm).complete([{"role": "user", "content": "只回复两个字：在的"}], max_tokens=10)
            out.append(Check("deepseek", True, f"DeepSeek（{cfg.llm.model}）", f"{reply.content.strip()!r}，{time.monotonic() - started:.1f} 秒"))
        except LLMError as e:
            out.append(Check("deepseek", False, "DeepSeek", str(e)))

    # 2. Jev（可选）
    if os.getenv("TYPESAFE_API_KEY"):
        out.append(_jev())
    else:
        out.append(Check("jev", None, "Jev", "没填 key：快速判断也交给 DeepSeek（可以以后再配）"))

    # 3. iMessage
    if not cfg.imessage:
        out.append(Check("owner", False, "iMessage 主人地址",
                         "在 .env 里填 ALF_IMESSAGE_OWNER=你自己的 iMessage 地址（手机号写成 +61412345678 这种，或邮箱）"))
    else:
        out.append(Check("owner", True, "iMessage 主人地址", f"只听 {'、'.join(cfg.imessage.owners)} 的话"))
        if channel is None:
            # 后台服务没在跑（或还没带上 iMessage 配置），只能用当前进程的权限试一下。
            try:
                ChatDB(cfg.imessage.db_path).max_rowid()
                out.append(Check("read", True, "读取「信息」数据库（当前终端的权限）"))
            except ChannelError as e:
                out.append(Check("read", False, "读取「信息」数据库（当前终端的权限）", str(e)))
        try:
            n = imessage_account_count()
            out.append(Check("account", n > 0, "「信息」App 已登录 iMessage 账号" if n else "「信息」App 没有登录 iMessage 账号",
                             "" if n else "打开「信息」App 登录 Alf 用的 Apple ID"))
        except ChannelError as e:
            out.append(Check("account", False, "控制「信息」App", str(e)))
        if send_test:
            try:
                send(cfg.imessage.owners[0], TEST_MESSAGE)
                out.append(Check("send", True, "发送测试消息", f"已发给 {cfg.imessage.owners[0]}，看看手机"))
            except ChannelError as e:
                out.append(Check("send", False, "发送测试消息", str(e)))

    # 4. 后台服务自己的看法
    if status is None:
        out.append(Check("service", False, "后台服务", "没有响应。用 `alf service install` 安装，或看 `alf service status`"))
    else:
        out.append(Check("service", status["heartbeat"]["alive"], "后台服务在运行", f"对话：{status.get('brain')}，判断：{status.get('backend')}"))
        if channel:
            if channel.get("ok") is None:
                out.append(Check("channel", None, "后台服务的 iMessage", "刚启动，还没跑完第一轮，过几秒再查"))
            else:
                out.append(Check("channel", channel["ok"], "后台服务的 iMessage", channel.get("error") or "能读到「信息」数据库"))
        elif cfg.imessage:
            out.append(Check("channel", False, "后台服务的 iMessage", "服务启动时还没有这项配置。改完 .env 后要重新运行 `alf service install`"))
        jobs = {j["name"]: j for j in status.get("jobs", [])}
        # 5. 第二期接入的来源：以后台任务最近一次的结果为准，不在这里另外登录一遍（多登录几次有的邮箱会临时封号）。
        boxes = [c for c in status.get("connectors", []) if c.get("configured")]
        if not boxes:
            out.append(Check("email", None, "邮箱", "还没接入。在设置页「邮箱」里填地址和授权码"))
        for box in boxes:
            job = jobs.get(box["name"], {})
            title = f"{box['label']}（{box.get('user', '')}）"
            if job.get("last_run") is None:
                out.append(Check(box["name"], None, title, "刚接入，还没查过第一次"))
            else:
                out.append(Check(box["name"], bool(job.get("ok")), title, job.get("error") or job.get("result") or "正常"))
        if cal := jobs.get("calendar"):
            waiting = (cal.get("result") or "").startswith("还没同步")
            if cal.get("last_run") is None:
                out.append(Check("calendar", None, "日历", "刚启动，还没同步过"))
            elif waiting:
                out.append(Check("calendar", None, "日历和提醒事项", "还没授权。在设置页点「授权日历和提醒事项」"))
            else:
                out.append(Check("calendar", bool(cal.get("ok")), "日历", cal.get("error") or cal.get("result") or "正常"))
    # 6. 按意思检索：只看模型在不在本机，不装它、也不联网。没有也能用，只是换了说法就找不到。
    from . import embed

    found = embed.status()
    out.append(Check("embed", True if found["present"] else None, "按意思检索",
                     f"嵌入模型在本机（{found['model']}）" if found["present"]
                     else "嵌入模型还没下载：现在只按字面找。运行 `uv run alf index` 下载（约 490 MB）并建好索引"))
    # 7. 听录音：同样只看模型在不在。没有的话别的照常用，只是手表传来的录音不会被听。
    from . import voice

    out.append(Check("hearing", True if voice.status()["present"] else None, "听录音",
                     "语音模型在本机" if voice.status()["present"]
                     else "语音模型还没下载：手表传来的录音不会被听。运行 `uv run alf hear --setup` 下载（约 200 MB）"))
    return out


def _jev() -> Check:
    from .decide import DecisionError, JevBackend, noul

    try:
        started = time.monotonic()
        _, usage = JevBackend().system_one({"message": "ping"}, {"test": noul("The message is a short test or greeting.")})
        return Check("jev", True, f"Jev（{usage.get('model', '?')}）", f"用来做快速判断，{time.monotonic() - started:.1f} 秒")
    except DecisionError as e:
        return Check("jev", False, "Jev", str(e))
    except Exception as e:   # key 格式不对时 SDK 在构造阶段就会抛
        return Check("jev", False, "Jev", f"{type(e).__name__}: {e}")


def as_dicts(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]


def run(cfg: Settings, send_test: bool = False) -> int:
    checks = collect(cfg, fetch_status(cfg.web_port), send_test)
    for c in checks:
        mark = {True: "✓", False: "✗", None: "–"}[c.ok]
        print(f"{mark} {c.title}" + (f"\n    {c.detail}" if c.detail else ""))
    problems = sum(c.ok is False for c in checks)
    print("\n全部就绪。" if not problems else f"\n还有 {problems} 项要处理。")
    return 1 if problems else 0
