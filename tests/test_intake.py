"""收录音的门：只进不出。配对码换口令、口令才能交录音、交进来的只会进待听的队列。"""
import pytest
from fastapi.testclient import TestClient

from alf import db, intake, voice
from alf.decide import OfflineBackend
from alf.server import create_app

from .conftest import NOW

T = NOW.timestamp()


@pytest.fixture
def gate(cfg, monkeypatch):
    monkeypatch.setattr(voice, "duration", lambda path: 60.0)          # 不真的解码：这里测的是门，不是录音
    monkeypatch.setattr(intake, "hostname", lambda: "study-mac.local")
    monkeypatch.setattr(intake, "computer_name", lambda: "书房的 Mac")
    return intake.Gate(cfg, clock=lambda: T)


@pytest.fixture
def phone(gate):
    return TestClient(intake.create_app(gate))


def pair(gate, phone):
    code = gate.begin_pairing()["code"]
    return phone.post("/pair", json={"code": code}).json()["token"]


def put(phone, token, body=b"audio-bytes", at=T - 30, **headers):
    return phone.put("/recordings", content=body, headers={"authorization": f"Bearer {token}", "x-started-at": str(at), **headers})


def test_nothing_gets_in_without_pairing(gate, phone, cfg):
    assert phone.get("/hello").status_code == 401                       # 从没配对过：没有口令可用
    assert put(phone, "guess").status_code == 401
    assert phone.post("/pair", json={"code": "000000"}).status_code == 403    # 没人在 Mac 上点过配对
    assert not (intake.home(cfg) / "token").exists()
    assert sorted(r.path for r in intake.create_app(gate).routes) == ["/hello", "/pair", "/recordings"]   # 就这三个口，没有一个往外给数据


def test_the_code_is_short_lived_single_use_and_wrong_guesses_burn_it(gate, phone, cfg):
    code = gate.begin_pairing()["code"]
    assert len(code) == 6 and code.isdigit()
    wrong = "111111" if code != "111111" else "222222"
    assert phone.post("/pair", json={"code": wrong}).status_code == 403
    got = phone.post("/pair", json={"code": code}).json()
    assert got["host"] == "study-mac.local" and got["port"] == gate.port and len(got["token"]) >= 40
    assert (intake.home(cfg) / "token").stat().st_mode & 0o777 == 0o600           # 口令只有本人可读
    assert phone.post("/pair", json={"code": code}).status_code == 403            # 用过一次就作废
    assert phone.get("/hello", headers={"authorization": f"Bearer {got['token']}"}).json() == {"ok": True}
    # 第二台设备配对：拿到的是同一个口令，前一台不受影响
    assert pair(gate, phone) == got["token"]
    # 错五次：这个码作废，要回 Mac 上重新点
    code = gate.begin_pairing()["code"]
    for _ in range(intake.CODE_TRIES - 1):
        assert phone.post("/pair", json={"code": wrong if wrong != code else "333333"}).status_code == 403
    assert phone.post("/pair", json={"code": wrong if wrong != code else "333333"}).status_code == 429
    assert phone.post("/pair", json={"code": code}).status_code == 403
    # 过了五分钟
    code = gate.begin_pairing()["code"]
    gate.clock = lambda: T + intake.CODE_TTL + 1
    assert phone.post("/pair", json={"code": code}).status_code == 403
    assert phone.post("/pair", json={"code": "12345"}).status_code == 422


def test_a_paired_phone_can_hand_in_recordings_and_nothing_else(gate, phone, cfg):
    token = pair(gate, phone)
    first = put(phone, token, x_device="watch")
    assert first.status_code == 200 and first.json() == {"ok": True, "id": 1}
    assert put(phone, token).json() == {"ok": True, "duplicate": True}                # 重传：内容一样，不收第二遍
    conn = db.connect(cfg.db_path)
    row = conn.execute("SELECT * FROM recordings").fetchone()
    assert (row["state"], row["device"], row["started_at"], row["seconds"]) == ("new", "watch", T - 30, 60.0)
    assert not list((intake.home(cfg) / "tmp").glob("*"))                              # 临时文件不留
    assert gate.status() | {"check": None} == {"running": False, "port": gate.port, "paired": True, "host": "study-mac.local", "name": "书房的 Mac",
                                               "check": None, "received": 2, "last_at": T, "pairings": 1}
    assert "token" not in str(gate.status())


def test_what_is_handed_in_is_checked(gate, phone, monkeypatch):
    token = pair(gate, phone)
    assert put(phone, token, body=b"").status_code == 422
    assert phone.put("/recordings", content=b"x", headers={"authorization": f"Bearer {token}"}).status_code == 422       # 没说什么时候录的
    assert put(phone, token, at=T - 15 * 86400).status_code == 422                     # 两周以前的不收
    assert put(phone, token, at=T + 3600).status_code == 422                           # 来自未来的不收
    assert put(phone, token, body=b"x" * (intake.MAX_BYTES + 1)).status_code == 413
    monkeypatch.setattr(voice, "duration", lambda path: None)
    assert put(phone, token, body=b"not audio").status_code == 422                     # 不是录音
    odd = put(phone, token, body=b"named", **{"x-device": "../../etc/passwd"})
    monkeypatch.setattr(voice, "duration", lambda path: 60.0)
    assert odd.status_code == 422 or odd.json()["ok"]


def test_unpairing_locks_the_old_phone_out(gate, phone):
    token = pair(gate, phone)
    gate.unpair()
    assert put(phone, token).status_code == 401 and phone.get("/hello", headers={"authorization": f"Bearer {token}"}).status_code == 401
    assert pair(gate, phone) != token


def test_the_announcement_lives_in_this_process_and_leaves_no_helper_behind(cfg, monkeypatch):
    calls = []

    class Lib:      # 顶替系统的 Bonjour 接口：测试里不真的往局域网里报名字
        class _Fn:
            def __init__(self, name):
                self.name = name

            def __call__(self, *args):
                calls.append((self.name, args[3:5] + args[7:8] if self.name == "DNSServiceRegister" else ()))
                return 0

        def __getattr__(self, name):
            fn = self._Fn(name)
            setattr(self, name, fn)
            return fn

    lib = Lib()
    monkeypatch.setattr(intake.ctypes, "CDLL", lambda name: lib)
    announcer = intake.Announcer()
    assert announcer.start("书房的 Mac", 8766)
    name, (label, kind, port) = calls[0]
    assert (name, label.decode(), kind.decode(), port) == ("DNSServiceRegister", "书房的 Mac", intake.SERVICE, intake.socket.htons(8766))
    announcer.stop()
    announcer.stop()                                                    # 停两次没关系
    assert [c[0] for c in calls] == ["DNSServiceRegister", "DNSServiceRefDeallocate"]
    assert "Popen" not in open(intake.__file__).read()                  # 不再起子进程：父进程被杀，就不会留下一个还在报名字的孤儿


def test_the_certificate_is_made_once_and_its_fingerprint_is_stable(cfg):
    cert, key = intake.ensure_cert(cfg)
    assert key.stat().st_mode & 0o777 == 0o600
    first = intake.fingerprint(cert)
    assert intake.ensure_cert(cfg) == (cert, key) and intake.fingerprint(cert) == first and len(first) == 64
    assert intake.check_code(first) == f"{first[:4]}-{first[4:8]}".upper()


def test_the_settings_page_opens_and_closes_the_gate_and_chat_cannot(cfg, monkeypatch):
    from alf import prefs

    events = []
    gate = intake.Gate(cfg)
    monkeypatch.setattr(gate, "start", lambda: events.append("start"))
    monkeypatch.setattr(gate, "stop", lambda: events.append("stop"))
    monkeypatch.setattr(intake.Gate, "running", property(lambda self: events[-1:] == ["start"]))
    assert "hearing.intake" not in prefs.CHAT_SETTINGS and prefs.defaults(cfg)["hearing.intake"] is False    # 默认关；聊天里一句话开不了
    with TestClient(create_app(cfg, OfflineBackend(), gate=gate), base_url="http://127.0.0.1") as c:
        assert events == ["stop"]                                                    # 启动时照设置来：默认是关
        assert c.post("/api/intake/pair").status_code == 409                         # 没开着，不给配对码
        c.patch("/api/prefs", json={"hearing.intake": True})
        assert events == ["stop", "start"]
        got = c.post("/api/intake/pair").json()
        assert len(got["code"]) == 6 and got["expires_in"] == intake.CODE_TTL
        # 页面拿这份回答重画"配对手机"那一行：少了 available 或 running，那一行会在配对码出来的同时消失（真出过这个错）
        assert got["available"] and got["running"] and got["pairings"] == 0
        status = c.get("/api/intake").json()
        assert status["available"] and status["running"] and not status["paired"] and "token" not in status
        c.patch("/api/prefs", json={"hearing.intake": False})
        assert events[-1] == "stop"
        assert c.delete("/api/intake/pairing").json() == {"ok": True}
    with TestClient(create_app(cfg, OfflineBackend(), start_heartbeat=False), base_url="http://127.0.0.1") as c:
        assert c.get("/api/intake").json() == {"available": False}
