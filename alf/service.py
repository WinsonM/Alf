"""让 Alf 常驻：注册成 macOS 的 LaunchAgent。

launchd 负责三件事：登录后自动启动、崩溃或被杀后自动拉起、把输出写进日志文件。
它管不了的是睡眠：Mac 合盖睡着时进程是冻结的，醒来后心跳会把错过的提醒补发。
要真正 7×24 不间断，得把 Alf 放到一台不睡觉的机器上（见 README）。
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path

from .config import ROOT

LABEL = "com.alf.agent"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
# macOS 的隐私保护（TCC）会拦后台进程读这几个目录，项目放在里面时第一次启动可能要点授权。
PROTECTED = ("Desktop", "Documents", "Downloads")


def build_plist(python: Path, root: Path, port: int, *, offline: bool = False, keep_awake: bool = False) -> dict:
    args = [str(python), "-m", "alf", "serve", "--port", str(port)]
    if offline:
        args.append("--offline")
    if keep_awake:
        # caffeinate -s：接着电源时不让系统睡眠（合盖且没外接显示器时仍然会睡）。
        args = ["/usr/bin/caffeinate", "-s", *args]
    log = str(root / "data" / "alf.log")
    return {
        "Label": LABEL,
        "ProgramArguments": args,
        "WorkingDirectory": str(root),
        "RunAtLoad": True,        # 登录后自动启动
        "KeepAlive": True,        # 退出后自动拉起
        "ThrottleInterval": 30,   # 一启动就失败时（比如端口被占），最多 30 秒重试一次，不空转
        "StandardOutPath": log,
        "StandardErrorPath": log,
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},   # 日志立刻落盘，出问题时看得到最后一行
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def _replace_service(sleep=time.sleep) -> subprocess.CompletedProcess:
    """卸掉旧的（如果有），再注册新的。

    launchd 卸载一个服务不是立刻完成的：它还在收尾时去注册同名服务会失败（Bootstrap failed: 5）。
    所以先等旧的真正消失，注册不上再重试几次。
    """
    target = f"{_domain()}/{LABEL}"
    _launchctl("bootout", target)                        # 没装过会报错，忽略
    for _ in range(20):
        if _launchctl("print", target).returncode != 0:  # 查不到了，说明已经卸干净
            break
        sleep(0.5)
    r = _launchctl("bootstrap", _domain(), str(PLIST))
    for _ in range(5):
        if r.returncode == 0:
            return r
        sleep(1.0)
        r = _launchctl("bootstrap", _domain(), str(PLIST))
    if r.returncode != 0 and _launchctl("print", target).returncode == 0:
        # 注册不上，但服务明明在：多半是另一个终端刚刚装过。那就让现有的重启一次，照样读到新配置。
        return _launchctl("kickstart", "-k", target)
    return r


def wait_ready(port: int, timeout: float = 40.0) -> bool:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def install(port: int, *, offline: bool, keep_awake: bool) -> int:
    if sys.platform != "darwin":
        print("常驻安装目前只支持 macOS（launchd）。其他系统请用 systemd 之类跑 `alf serve`。")
        return 1
    (ROOT / "data").mkdir(exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    PLIST.write_bytes(plistlib.dumps(build_plist(Path(sys.executable), ROOT, port, offline=offline, keep_awake=keep_awake)))
    r = _replace_service()
    if r.returncode != 0:
        print(f"注册失败：{r.stderr.strip() or r.stdout.strip()}")
        return 1
    print(f"已注册为登录自启服务：{PLIST}")
    print(f"界面：http://127.0.0.1:{port}    日志：{ROOT / 'data' / 'alf.log'}")
    # 注册成功不等于已经能用：进程启动要几秒。等它真的能响应了再说"好了"。
    if wait_ready(port):
        print("服务已就绪。")
        return 0
    print("服务还没有响应，用 `alf service status` 看日志。")
    try:
        inside = ROOT.relative_to(Path.home()).parts[:1]
    except ValueError:
        inside = ()
    if inside and inside[0] in PROTECTED:
        print(f"项目在 ~/{inside[0]} 里：macOS 可能弹窗问是否允许 python 访问该文件夹，请点允许。\n"
              "如果日志里有 Operation not permitted，把项目移出这个文件夹再装一次。")
    return 1


def uninstall() -> int:
    _launchctl("bootout", f"{_domain()}/{LABEL}")
    PLIST.unlink(missing_ok=True)
    print("已停止并取消登录自启。数据和日志都还在 data/ 里。")
    return 0


def parse_print(output: str) -> dict[str, str]:
    """`launchctl print` 的 "键 = 值" 行。同一个键会在后面的子段落里再出现（比如端点的 state = active），
    服务本身的那一行在最前面，所以只认第一次出现的。"""
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, sep, value = line.partition(" = ")
        if sep:
            fields.setdefault(key.strip(), value.strip())
    return fields


def status() -> int:
    r = _launchctl("print", f"{_domain()}/{LABEL}")
    if r.returncode != 0:
        print("没有注册常驻服务。用 `alf service install` 注册。")
        return 1
    fields = parse_print(r.stdout)
    print(f"状态：{fields.get('state', '未知')}    pid：{fields.get('pid', '—')}    上次退出码：{fields.get('last exit code', '—')}")
    log = ROOT / "data" / "alf.log"
    if log.exists():
        print(f"--- {log} 最后 10 行")
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]))
    return 0
