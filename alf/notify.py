"""把消息送到 Alf 界面之外的地方。现在只有 macOS 通知中心；以后推手机也加在这里。"""
from __future__ import annotations

import subprocess
import sys


def notify_desktop(title: str, body: str) -> None:
    """macOS 通知中心；其他系统什么都不做。通知发不出去不应该影响提醒本身，所以失败不报错。"""
    if sys.platform != "darwin":
        return

    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    try:
        subprocess.run(["osascript", "-e", f'display notification {q(body)} with title {q(title)} sound name "default"'],
                       capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass
