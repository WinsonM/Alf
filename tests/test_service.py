import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from alf import service


def test_plist_keeps_alf_alive_and_starts_it_at_login(tmp_path):
    p = service.build_plist(Path("/venv/bin/python"), Path("/proj"), 8765)
    assert p["ProgramArguments"] == ["/venv/bin/python", "-m", "alf", "serve", "--port", "8765"]
    assert (p["RunAtLoad"], p["KeepAlive"], p["WorkingDirectory"]) == (True, True, "/proj")
    assert p["StandardOutPath"] == p["StandardErrorPath"] == "/proj/data/alf.log"


def test_options_change_the_command_line():
    p = service.build_plist(Path("/venv/bin/python"), Path("/proj"), 9000, offline=True, keep_awake=True)
    assert p["ProgramArguments"] == ["/usr/bin/caffeinate", "-s", "/venv/bin/python", "-m", "alf", "serve",
                                     "--port", "9000", "--offline"]


@pytest.mark.skipif(sys.platform != "darwin", reason="plutil 只在 macOS 上有")
def test_plist_passes_macos_lint(tmp_path):
    path = tmp_path / "alf.plist"
    path.write_bytes(plistlib.dumps(service.build_plist(Path(sys.executable), tmp_path, 8765)))
    assert subprocess.run(["plutil", "-lint", str(path)], capture_output=True).returncode == 0


def test_status_reads_the_service_state_not_a_nested_one():
    output = "gui/501/com.alf.agent = {\n\tstate = running\n\truns = 1\n\tpid = 34916\n\tlast exit code = (never exited)\n\tendpoints = {\n\t\tstate = active\n\t}\n}"
    fields = service.parse_print(output)
    assert (fields["state"], fields["pid"], fields["last exit code"]) == ("running", "34916", "(never exited)")


def test_reinstall_waits_for_the_old_service_to_go_away_and_retries(monkeypatch):
    """重装时 launchd 还没卸完旧服务，第一次注册会失败（实际遇到过：Bootstrap failed: 5）。"""
    calls, waits = [], []
    still_there = iter([0, 0, 113])                      # print：前两次还查得到，第三次没了
    bootstrap = iter([5, 5, 0])                          # bootstrap：前两次失败，第三次成功

    def fake(*args):
        calls.append(args[0])
        code = {"bootout": 0, "print": lambda: next(still_there), "bootstrap": lambda: next(bootstrap)}[args[0]]
        return subprocess.CompletedProcess(args, code() if callable(code) else code, "", "Bootstrap failed: 5")

    monkeypatch.setattr(service, "_launchctl", fake)
    assert service._replace_service(sleep=waits.append).returncode == 0
    assert calls == ["bootout", "print", "print", "print", "bootstrap", "bootstrap", "bootstrap"]
    assert waits == [0.5, 0.5, 1.0, 1.0]


def test_reinstall_restarts_a_service_that_someone_else_just_loaded(monkeypatch):
    """实际遇到过：两个终端同时安装，后到的那个注册失败，但服务其实已经在跑。"""
    calls = []
    present = iter([113, 0])                             # 先查不到（卸掉了），注册失败后再查：又在了

    def fake(*args):
        calls.append(args[0])
        code = {"bootout": 0, "print": lambda: next(present), "bootstrap": 5, "kickstart": 0}[args[0]]
        return subprocess.CompletedProcess(args, code() if callable(code) else code, "", "")

    monkeypatch.setattr(service, "_launchctl", fake)
    assert service._replace_service(sleep=lambda s: None).returncode == 0
    assert calls == ["bootout", "print"] + ["bootstrap"] * 6 + ["print", "kickstart"]
