"""收录音的门：Alf 在局域网上唯一对外的口子，只进不出。

聊天网页和所有接口都只听 127.0.0.1（见 server.py）：那里有记忆和邮件，又没有登录。可手表录的音要经手机传进来，
总得有个口子。这个口子单独开一个端口，和主服务完全分开，能做的事只有三样：

- PUT /recordings  交一段录音。交进来的只会被放进待听的队列（hearing.receive），别的什么都碰不到。
- POST /pair       拿配对码换口令。配对码是他在这台 Mac 的设置页上点出来的六位数，五分钟内有效，错五次作废。
- GET /hello       手机用来确认"口令还管用"。

没有任何一个接口往外给数据：拿到口令的人最多能往队列里塞录音，读不到一个字。

几道防线：
- 默认不开。设置页上打开"接收手表传来的录音"才启动，关掉就停。
- 全程 TLS。证书是这台机器自己签的；手机在配对时记住它的指纹，以后只认这一张。配对那一下是"第一次见面就信"，
  所以两边各显示一个核对码（指纹的头几位），他扫一眼对得上，就不是别人冒充的。
- 口令是随机的 32 字节，放在 data/intake/ 下，只有本人可读。设置页上点"解除配对"会换一个新的，旧手机就进不来了。
- 一次最多 8 MB（一分钟的录音不到 300 KB）；录音的时间得在最近两周之内。
"""
from __future__ import annotations

import ctypes
import hashlib
import hmac
import logging
import os
import secrets
import socket
import ssl
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import db, hearing
from .config import Settings
from .voice import VoiceError

log = logging.getLogger("alf.intake")

PORT = int(os.getenv("ALF_INTAKE_PORT", "8766"))
SERVICE = "_alf-intake._tcp"            # 手机靠这个名字在局域网里找到这台 Mac（Bonjour）
MAX_BYTES = 8 * 1024 * 1024
CODE_TTL = 300
CODE_TRIES = 5
EARLIEST = 14 * 86400                   # 比这更早的录音不收
LATEST = 600                            # 手机和 Mac 的钟差一点没关系，差太多就不对了


def home(cfg: Settings) -> Path:
    return cfg.db_path.parent / "intake"


def _private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def ensure_cert(cfg: Settings) -> tuple[Path, Path]:
    """这台机器自己签的证书，没有就现签一张（十年）。用系统自带的 openssl，不多带一个依赖。"""
    cert, key = home(cfg) / "cert.pem", home(cfg) / "key.pem"
    if not (cert.is_file() and key.is_file()):
        cert.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        done = subprocess.run(["/usr/bin/openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "3650",
                               "-subj", "/CN=Alf intake", "-keyout", str(key), "-out", str(cert)], capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            raise RuntimeError(f"证书没签成：{done.stderr.strip()[:200]}")
        key.chmod(0o600)
    return cert, key


def fingerprint(cert: Path) -> str:
    """证书的 SHA-256 指纹（十六进制）。手机配对时记下它，以后只认这一张证书。"""
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.read_text())).hexdigest()


def check_code(print_hex: str) -> str:
    """给人核对用的短码：指纹的头八位，分成两组。配对时 Mac 和手机上各显示一个，一样才对。"""
    return f"{print_hex[:4]}-{print_hex[4:8]}".upper()


def token(cfg: Settings) -> str | None:
    path = home(cfg) / "token"
    return path.read_text().strip() or None if path.is_file() else None


def new_token(cfg: Settings) -> str:
    """换一个新口令。以前配对过的手机都得重新配对。"""
    value = secrets.token_urlsafe(32)
    _private(home(cfg) / "token", value)
    return value


def hostname() -> str:
    """这台 Mac 在局域网里的名字（xxx.local）。手机以后靠它找过来，IP 变了也不怕。"""
    try:
        name = subprocess.run(["/usr/sbin/scutil", "--get", "LocalHostName"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return f"{name}.local" if name else "localhost"


def computer_name() -> str:
    """这台 Mac 给人看的名字（"某某的 MacBook Air"）。手机的配对列表里显示的就是它：家里有两台 Mac 也分得清。"""
    try:
        name = subprocess.run(["/usr/sbin/scutil", "--get", "ComputerName"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name[:60] or "Alf"


class Announcer:
    """在局域网里报个名字（Bonjour），手机好找到这台 Mac。

    直接向系统登记，不另起进程：登记挂在本进程的一个连接上，进程一没（正常退出、被杀、原地重启），连接断了，
    系统自己就把名字撤掉。第一版是起一个 `dns-sd -R` 子进程替我们登记——父进程被杀之后它还活着，
    手机上就多出一个点了没反应的"Alf"，真的那台被挤成了"Alf (2)"。
    """

    def __init__(self):
        self._ref: ctypes.c_void_p | None = None

    def start(self, name: str, port: int) -> bool:
        self.stop()
        try:
            lib = ctypes.CDLL(None)
            lib.DNSServiceRegister.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p,
                                               ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint16, ctypes.c_uint16,
                                               ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            lib.DNSServiceRegister.restype = ctypes.c_int32
            ref = ctypes.c_void_p()
            # 端口要按网络字节序给；名字重了系统会自己加上"(2)"
            err = lib.DNSServiceRegister(ctypes.byref(ref), 0, 0, name.encode(), SERVICE.encode(), None, None, socket.htons(port),
                                         0, None, None, None)
        except (OSError, AttributeError) as e:      # 不是 macOS，或者系统里没有这个接口
            log.warning("没能在局域网里登记服务，手机得手动填这台 Mac 的地址：%s", e)
            return False
        if err != 0:
            log.warning("没能在局域网里登记服务（错误 %d），手机得手动填这台 Mac 的地址", err)
            return False
        self._ref = ref
        return True

    def stop(self) -> None:
        if self._ref is not None:
            lib = ctypes.CDLL(None)
            lib.DNSServiceRefDeallocate.argtypes = [ctypes.c_void_p]
            lib.DNSServiceRefDeallocate.restype = None
            lib.DNSServiceRefDeallocate(self._ref)
            self._ref = None


class PairIn(BaseModel):
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class Gate:
    """这个口子的开关和状态。主服务（设置页）和收录音的小服务共用同一个对象：配对码在这边生成，在那边核对。"""

    def __init__(self, cfg: Settings, port: int = PORT, clock=time.time):
        self.cfg, self.port, self.clock = cfg, port, clock
        self._lock = threading.Lock()
        self._code: str | None = None
        self._expires = 0.0
        self._tries = 0
        self.received = 0                    # 这次启动以来收了几段
        self.pairings = 0                    # 这次启动以来配对成功了几次（设置页靠它知道"手机刚配上了"）
        self.last_at: float | None = None
        self._server = None
        self._thread: threading.Thread | None = None
        self._announcer = Announcer()

    # --- 配对 ---

    def begin_pairing(self) -> dict:
        """在设置页上点"配对手机"：出一个六位数，五分钟内有效。"""
        with self._lock:
            self._code, self._expires, self._tries = f"{secrets.randbelow(1_000_000):06d}", self.clock() + CODE_TTL, 0
            return {"code": self._code, "expires_in": CODE_TTL}

    def redeem(self, code: str) -> str:
        """手机拿配对码来换口令。码不对、过期、错太多次都换不到；换到一次这个码就作废。"""
        with self._lock:
            if self._code is None or self.clock() > self._expires:
                self._code = None
                raise HTTPException(403, "配对码过期了，在 Mac 上重新点一次")
            if not hmac.compare_digest(code, self._code):
                self._tries += 1
                if self._tries >= CODE_TRIES:
                    self._code = None
                    raise HTTPException(429, "错太多次了，在 Mac 上重新点一次")
                raise HTTPException(403, "配对码不对")
            self._code = None
            self.pairings += 1
        return token(self.cfg) or new_token(self.cfg)

    def unpair(self) -> None:
        new_token(self.cfg)
        with self._lock:
            self._code = None

    def allowed(self, header: str | None) -> bool:
        known = token(self.cfg)
        given = header[7:].strip() if header and header.lower().startswith("bearer ") else ""
        return bool(known and given) and hmac.compare_digest(given, known)

    # --- 开、关 ---

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        cert = home(self.cfg) / "cert.pem"
        return {"running": self.running, "port": self.port, "paired": token(self.cfg) is not None, "host": hostname(),
                "name": computer_name(),
                "check": check_code(fingerprint(cert)) if cert.is_file() else None,
                "received": self.received, "last_at": self.last_at, "pairings": self.pairings}

    def start(self) -> None:
        """开门：起一个只收录音的 HTTPS 服务，并在局域网里报上这台 Mac 的名字，手机好找到。已经开着就什么都不做。"""
        if self.running:
            return
        import uvicorn

        cert, key = ensure_cert(self.cfg)
        config = uvicorn.Config(create_app(self), host="0.0.0.0", port=self.port, ssl_certfile=str(cert), ssl_keyfile=str(key),
                                access_log=False, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None      # 信号归主服务管
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="alf-intake")
        self._thread.start()
        self._announcer.start(computer_name(), self.port)
        log.info("收录音的端口开了：%d（局域网，只收不发）", self.port)

    def stop(self) -> None:
        self._announcer.stop()
        if self._server is not None:
            self._server.should_exit = True
            self._server = None
            log.info("收录音的端口关了")
        self._thread = None

    def sync(self, enabled: bool) -> None:
        """照设置开或关。开不起来（端口被占、证书签不成）只记一笔，不让主服务跟着出错。"""
        try:
            self.start() if enabled else self.stop()
        except Exception:
            log.exception("收录音的端口没开起来")


def create_app(gate: Gate) -> FastAPI:
    cfg = gate.cfg
    conn = db.connect(cfg.db_path, check_same_thread=False)
    lock = threading.Lock()
    app = FastAPI(title="Alf intake", docs_url=None, redoc_url=None, openapi_url=None)

    def authed(request: Request) -> None:
        if not gate.allowed(request.headers.get("authorization")):
            raise HTTPException(401, "没有配对，或者配对已经解除了")

    @app.get("/hello")
    def hello(request: Request) -> dict:
        authed(request)
        return {"ok": True}

    @app.post("/pair")
    def pair(body: PairIn) -> dict:
        value = gate.redeem(body.code)
        cert = home(cfg) / "cert.pem"
        log.info("有一台设备配对成功")
        return {"token": value, "host": hostname(), "port": gate.port,
                "check": check_code(fingerprint(cert)) if cert.is_file() else None}

    def take(data: bytes, started_at: float, device: str) -> int | None:
        tmp = home(cfg) / "tmp" / f"{uuid.uuid4().hex}.m4a"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        try:
            with lock:
                return hearing.receive(conn, cfg, tmp, started_at, device, int(gate.clock()), move=True)
        finally:
            tmp.unlink(missing_ok=True)

    @app.put("/recordings")
    async def recording(request: Request) -> dict:
        authed(request)
        try:
            started_at = float(request.headers.get("x-started-at", ""))
            size = int(request.headers.get("content-length", "0"))
        except ValueError:
            raise HTTPException(422, "缺少录音的开始时间") from None
        now = gate.clock()
        if not now - EARLIEST <= started_at <= now + LATEST:
            raise HTTPException(422, "录音的时间不对：只收最近两周的")
        if size > MAX_BYTES:
            raise HTTPException(413, "这一段太大了")
        data = await request.body()
        if not data or len(data) > MAX_BYTES:
            raise HTTPException(413 if data else 422, "这一段太大了" if data else "录音是空的")
        device = "".join(ch for ch in request.headers.get("x-device", "watch") if ch.isalnum() or ch in "-_")[:20] or "watch"
        try:
            got = await run_in_threadpool(take, data, started_at, device)
        except VoiceError as e:
            raise HTTPException(422, str(e)) from e
        gate.received += 1
        gate.last_at = now
        if got is None:
            return {"ok": True, "duplicate": True}
        log.info("收到一段录音（%s，%s 开始）", device, datetime.fromtimestamp(started_at, cfg.tz).strftime("%m-%d %H:%M:%S"))
        return {"ok": True, "id": got}

    return app
