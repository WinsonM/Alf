from dataclasses import replace

from alf import doctor
from alf.config import IMessageConfig, LLMConfig
from alf.imessage import ChannelError
from alf.llm import LLMError, LLMReply


def test_nothing_configured_lists_what_to_do(cfg, capsys):
    assert doctor.run(replace(cfg, web_port=1)) == 1          # 端口 1 上没有服务
    out = capsys.readouterr().out
    assert "✗ DeepSeek\n    在 .env 里填 DEEPSEEK_API_KEY" in out
    assert "✗ iMessage 主人地址\n    在 .env 里填 ALF_IMESSAGE_OWNER=" in out
    assert "✗ 后台服务\n    没有响应" in out and out.rstrip().endswith("还有 3 项要处理。")


def test_configured_pieces_are_checked_for_real(cfg, capsys, monkeypatch, tmp_path):
    class Model:
        def __init__(self, cfg):
            pass

        def complete(self, messages, **kw):
            return LLMReply("在的")

    sent = []
    monkeypatch.setattr(doctor, "DeepSeek", Model)
    monkeypatch.setattr(doctor, "imessage_account_count", lambda: 1)
    monkeypatch.setattr(doctor, "send", lambda handle, text: sent.append((handle, text)))
    full = replace(cfg, web_port=1, llm=LLMConfig(api_key="sk-test"),
                   imessage=IMessageConfig(owners=("+61412345678",), db_path=tmp_path / "missing.db"))
    assert doctor.run(full, send_test=True) == 1
    out = capsys.readouterr().out
    assert "✓ DeepSeek（deepseek-flash）\n    '在的'" in out
    assert "✓ iMessage 主人地址\n    只听 +61412345678 的话" in out
    assert "✗ 读取「信息」数据库（当前终端的权限）" in out          # 后台服务没在跑时，才用终端的权限试
    assert "✓ 「信息」App 已登录 iMessage 账号" in out
    assert "✓ 发送测试消息" in out and sent == [("+61412345678", "这是 Alf 的测试消息，收到说明发送正常。")]
    assert "sk-test" not in out


def test_failures_show_the_reason(cfg, capsys, monkeypatch, tmp_path):
    class Broken:
        def __init__(self, cfg):
            pass

        def complete(self, messages, **kw):
            raise LLMError("DeepSeek 的 API key 无效，检查 .env 里的 DEEPSEEK_API_KEY")

    def no_permission():
        raise ChannelError("没有控制「信息」App 的权限：系统设置 → 隐私与安全性 → 自动化，给 python 勾上「信息」")

    monkeypatch.setattr(doctor, "DeepSeek", Broken)
    monkeypatch.setattr(doctor, "imessage_account_count", no_permission)
    full = replace(cfg, web_port=1, llm=LLMConfig(api_key="sk-test"),
                   imessage=IMessageConfig(owners=("a@b.c",), db_path=tmp_path / "missing.db"))
    assert doctor.run(full) == 1
    out = capsys.readouterr().out
    assert "✗ DeepSeek\n    DeepSeek 的 API key 无效" in out and "✗ 控制「信息」App\n    没有控制「信息」App 的权限" in out


def test_running_service_is_the_authority_on_imessage(cfg, capsys, monkeypatch, tmp_path):
    """实际遇到过：后台服务已经能读「信息」数据库，doctor 却因为终端自己读不了而报"还有问题"。"""
    import io
    import json

    status = {"heartbeat": {"alive": True}, "brain": "deepseek", "backend": "jev",
              "channels": [{"name": "imessage", "configured": True, "ok": True, "error": None}]}
    monkeypatch.setattr(doctor.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(json.dumps(status).encode()))
    monkeypatch.setattr(doctor, "imessage_account_count", lambda: 1)

    class Model:
        def __init__(self, cfg):
            pass

        def complete(self, messages, **kw):
            return LLMReply("在的")

    monkeypatch.setattr(doctor, "DeepSeek", Model)
    full = replace(cfg, llm=LLMConfig(api_key="sk-test"),
                   imessage=IMessageConfig(owners=("a@b.c",), db_path=tmp_path / "terminal-cannot-read.db"))
    assert doctor.run(full) == 0
    out = capsys.readouterr().out
    assert "当前终端的权限" not in out and "✓ 后台服务的 iMessage\n    能读到「信息」数据库" in out
    assert out.rstrip().endswith("全部就绪。")
