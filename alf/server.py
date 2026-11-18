"""常驻服务：一个进程里同时有 HTTP 接口、聊天网页和心跳线程。

只监听 127.0.0.1。Alf 里有你的记忆和邮件摘要，而这个服务没有登录，所以不对局域网
开放；要从手机访问，得先加鉴权再放开，不能只改监听地址。
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import (affinity, apple, asr, connectors, db, doctor, embed, feedback, impressions, lessons, pacts, persona, prefs, proactive, retrieve,
               timeline, websearch)
from .agent import Reply, Turn
from .chat import CATEGORY_LABELS, make_agent
from .config import MAIL_PROVIDERS, Settings, env_file, normalize_handle, update_dotenv
from .connectors import ConnectorError, email_smtp
from .connectors.email_imap import ImapEmail
from .decide import Backend, Decider
from .heartbeat import Heartbeat, job_status, load_state
from .imessage import CHANNEL as IMESSAGE, ChannelError, IMessageChannel
from .jobs import FAST_JOBS, build_jobs, source_errors
from .llm import LLM, LLMBackend
from .mcp_servers.calendar import read_events
from .memory import FEEDBACK, MOMENT, Memory

log = logging.getLogger("alf.server")

INDEX = Path(__file__).parent / "web" / "index.html"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
AUDIO_EXT = {"audio/webm": ".webm", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".m4a",
             "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/ogg": ".ogg", "audio/mpeg": ".mp3"}


MAIL_KEYS = tuple(f"ALF_{name.upper()}_{part}" for name in MAIL_PROVIDERS for part in ("USER", "PASSWORD"))
SECRET_KEYS = ("DEEPSEEK_API_KEY", "TYPESAFE_API_KEY", *(k for k in MAIL_KEYS if k.endswith("PASSWORD")))
ENV_KEYS = ("DEEPSEEK_API_KEY", "TYPESAFE_API_KEY", "ALF_IMESSAGE_OWNER", *MAIL_KEYS)
PRIVACY_URL = "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_AllFiles"


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class SetupIn(BaseModel):
    """只写传了的项；传空字符串表示清空。"""
    DEEPSEEK_API_KEY: str | None = Field(default=None, max_length=300)
    TYPESAFE_API_KEY: str | None = Field(default=None, max_length=300)
    ALF_IMESSAGE_OWNER: str | None = Field(default=None, max_length=300)
    ALF_GMAIL_USER: str | None = Field(default=None, max_length=200)
    ALF_GMAIL_PASSWORD: str | None = Field(default=None, max_length=200)
    ALF_163_USER: str | None = Field(default=None, max_length=200)
    ALF_163_PASSWORD: str | None = Field(default=None, max_length=200)
    ALF_QQ_USER: str | None = Field(default=None, max_length=200)
    ALF_QQ_PASSWORD: str | None = Field(default=None, max_length=200)


class FactIn(BaseModel):
    text: str | None = Field(default=None, min_length=1, max_length=200)
    scope: str | None = None      # 只对"对 Alf 的要求"有用：这条要求管哪里（all | chat | mail | proactive）


class ProposalIn(BaseModel):
    accept: bool


class ImpressionIn(BaseModel):
    text: str = Field(min_length=1, max_length=impressions.TEXT_CHARS)


class PactIn(BaseModel):
    outcome: str = Field(pattern="^(kept|dropped)$")


def read_env(path: Path) -> dict[str, str]:
    """只把 .env 解析出来看，不改当前进程的环境变量。"""
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def reexec() -> None:
    """用启动时的同一条命令原地重启自己，好让 .env 的改动生效。进程号不变，launchd 不会察觉。
    稍等片刻再执行，让触发重启的那个请求先把响应发回去。"""
    args = [sys.executable, "-m", "alf", *sys.argv[1:]]
    threading.Timer(0.5, lambda: os.execv(sys.executable, args)).start()


def open_privacy_settings() -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", PRIVACY_URL], capture_output=True, timeout=10, check=False)


def _check_inbox(box) -> None:
    ImapEmail(box).preview(1)


def create_app(cfg: Settings, backend: Backend, *, llm_backend: LLMBackend | None = None,
               recognizer: asr.Recognizer | None = None, start_heartbeat: bool = True,
               imessage: IMessageChannel | None = None, env_path: Path | None = None,
               restart: Callable[[], None] = reexec, open_settings: Callable[[], None] = open_privacy_settings,
               read_calendar: Callable[..., dict] = read_events,
               read_reminders: Callable[..., dict] = apple.read_reminders,
               check_inbox: Callable = _check_inbox, check_outbox: Callable = email_smtp.check) -> FastAPI:
    env_path = env_path or env_file()
    # 请求跑在线程池里，所以连接允许跨线程，各配一把锁。分成两个连接是因为一轮对话要等模型好几秒：
    # 对话用的连接被占着的时候，网页读时间线和状态不应该跟着卡住。
    conn = db.connect(cfg.db_path, check_same_thread=False)        # 读时间线、状态、存录音
    lock = threading.Lock()
    agent = make_agent(db.connect(cfg.db_path, check_same_thread=False), cfg, backend, llm_backend)
    turn_lock = threading.Lock()                                   # 同一时刻只处理一轮对话，不管它从哪个入口来
    active = connectors.build(cfg)
    # 任务表按当前设置生成，心跳每次都重新生成一遍：界面上改了时间或开关，几秒内生效。
    # 分两条车道：到点提醒单独一条，每 5 秒必到；要联网、要等模型的（汇总信息、主动判断、邮箱、早报）
    # 在另一条里慢慢跑，再慢也耽误不了提醒。
    def lane(current: dict, fast: bool) -> list:
        return [j for j in build_jobs(cfg, active, current) if (j.name in FAST_JOBS) == fast]

    heartbeat = Heartbeat(cfg, lambda current: lane(current, True), make_decider=lambda c: Decider(backend, c))
    background = Heartbeat(cfg, lambda current: lane(current, False), make_decider=lambda c: Decider(backend, c),
                           make_llm=(lambda c: LLM(llm_backend, c)) if llm_backend else None, name="alf-background")
    recognizer = recognizer or asr.Recognizer(cfg.asr_model, cfg.asr_language)
    loaded_env = read_env(env_path)   # 启动时 .env 的样子，用来判断"改了但还没重启"

    def handle_turn(turn: Turn, on_reply=None) -> Reply:
        with turn_lock:
            return agent.handle(turn, on_reply)

    if imessage is None and cfg.imessage is not None:
        imessage = IMessageChannel(cfg, handle_turn, recognizer=recognizer)
    elif imessage is not None:
        imessage.handle = handle_turn
        imessage.recognizer = imessage.recognizer or recognizer

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_heartbeat:
            heartbeat.start()
            background.start()
            if imessage:
                imessage.start()
        yield
        heartbeat.stop()
        background.stop()
        if imessage:
            imessage.stop()

    app = FastAPI(title="Alf", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    # 只认本机的 Host 头：挡住 DNS 重绑定（恶意网页把自己的域名解析到 127.0.0.1 来读你的数据）。
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.state.heartbeat = heartbeat
    app.state.background = background

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next):
        # 会改动数据的请求只接受本页面自己发的。浏览器在跨站请求上一定会带 Origin，
        # 所以别的网站（哪怕是本机另一个端口上的页面）没法替用户点"重启"或"发测试消息"。
        # 没有 Origin 的是 curl、脚本这类本机程序，放行。
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "只接受 Alf 自己页面的请求"}, status_code=403)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/api/timeline")
    def get_timeline(after: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)) -> dict:
        with lock:
            return {"items": timeline.fetch(conn, after, limit)}

    @app.post("/api/chat")
    def chat(body: ChatIn) -> dict:
        text = body.text.strip()
        if not text:
            raise HTTPException(422, "消息是空的")
        reply = handle_turn(Turn(text))
        return {"reply": reply.text, "intent": reply.intent, "confidence": reply.confidence}

    @app.post("/api/voice")
    async def voice(request: Request) -> dict:
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if not content_type.startswith("audio/"):
            raise HTTPException(415, "需要 audio/* 类型的录音")
        data = await request.body()
        if not data:
            raise HTTPException(422, "录音是空的")
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "录音太大了")
        return await run_in_threadpool(handle_voice, data, AUDIO_EXT.get(content_type, ".webm"))

    def handle_voice(data: bytes, ext: str) -> dict:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            tmp = Path(f.name)
        try:
            with lock:
                attachment_id, stored = asr.store(conn, tmp, cfg.attachment_dir)
                vocabulary = asr.vocabulary(conn)
        finally:
            tmp.unlink(missing_ok=True)
        try:
            # 识别不持锁：要好几秒，第一次还要下载模型，期间不能卡住页面的轮询。
            t = recognizer.transcribe(stored, vocabulary)
        except Exception as e:
            log.exception("语音识别失败")
            raise HTTPException(503, f"语音识别失败：{e}") from e
        with lock:
            asr.save_transcript(conn, attachment_id, t)
        if not t.text:
            return {"transcript": "", "confidence": t.confidence, "reply": None}
        reply = handle_turn(Turn(t.text, source="voice", attachment_id=attachment_id, asr_confidence=t.confidence))
        return {"transcript": t.text, "confidence": t.confidence, "reply": reply.text}

    @app.get("/api/status")
    def status() -> dict:
        now = datetime.now(cfg.tz)
        with lock:
            job_rows = job_status(conn, build_jobs(cfg, active, prefs.load(conn, cfg)), now)
            counts = {
                "reminders": conn.execute("SELECT COUNT(*) FROM reminders WHERE status = 'pending'").fetchone()[0],
                "facts": conn.execute("SELECT COUNT(*) FROM facts WHERE valid_to IS NULL").fetchone()[0],
                "inbox": conn.execute("SELECT COUNT(*) FROM inbox_items").fetchone()[0],
            }
        last = heartbeat.last_tick
        # 连续三个周期没心跳就算停了（留点余量给跑得慢的任务，比如一次查邮件）。
        alive = last is not None and (now - last).total_seconds() < max(cfg.heartbeat_tick * 3, 90)
        return {
            # brain：谁在跟用户对话；backend：谁在做"选哪个/是不是"的判断
            "brain": llm_backend.name if llm_backend else backend.name,
            "model": llm_backend.model if llm_backend else None,
            "backend": backend.name,
            "tz": cfg.tz.key,
            "now": int(now.timestamp()),
            "heartbeat": {"alive": alive, "tick": cfg.heartbeat_tick,
                          "last_tick": int(last.timestamp()) if last else None},
            "jobs": job_rows,
            "counts": counts,
            "connectors": [{"name": m.name, "label": m.label, "configured": True, "user": m.user, "can_send": bool(m.smtp_host)}
                           for m in cfg.mailboxes] or [{"name": "email", "label": "邮箱", "configured": False}],
            "channels": [{"name": "imessage", "label": "iMessage", "configured": imessage is not None,
                          **(imessage.status if imessage else {})}],
        }

    @app.post("/api/jobs/{name}/run", status_code=202)
    def run_job(name: str) -> dict:
        if not (heartbeat.request(name) or background.request(name)):
            raise HTTPException(404, "没有这个任务")
        return {"queued": name}

    # --- 主动消息：它最近是怎么判断的、信息来源是否正常 -----------------------------------

    @app.get("/api/proactive")
    def get_proactive() -> dict:
        now = datetime.now(cfg.tz)
        with lock:
            current = prefs.load(conn, cfg)
            entries = proactive.recent(conn)
            stats = feedback.stats(conn, now)
            try:
                snapshot = json.loads(load_state(conn, "sources").cursor or "{}")
            except ValueError:
                snapshot = {}
        errors = source_errors(snapshot)
        level = proactive.LEVELS.get(current["proactive.level"], proactive.LEVELS["active"])
        return {
            "level": {"key": current["proactive.level"], "label": level.label, "per_day": level.per_day,
                      "gap_minutes": level.gap, "wake_minutes": level.wake, "threshold": level.threshold},
            "levels": {k: {"label": v.label, "per_day": v.per_day, "gap_minutes": v.gap} for k, v in proactive.LEVELS.items()},
            "quiet_now": proactive.in_quiet_hours(now, current),
            "gated_by_jev": backend.name == "jev",
            "feedback": stats,
            "log": entries,
            "sources": {"at": snapshot.get("at"),
                        "state": {name: errors.get(name, "ok") for name in snapshot.get("sources", {})}},
        }

    # --- 设置：改了立刻生效的那部分 ---------------------------------------------------

    @app.get("/api/prefs")
    def get_prefs() -> dict:
        with lock:
            return {"prefs": prefs.load(conn, cfg), "defaults": prefs.defaults(cfg)}

    @app.patch("/api/prefs")
    def patch_prefs(changes: dict) -> dict:
        try:
            with lock:
                return {"prefs": prefs.save(conn, cfg, changes)}
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    # --- 环境：key 和 iMessage 地址，在 .env 里，改了要重启 -----------------------------

    def setup_view() -> dict:
        on_disk = read_env(env_path)
        env = {}
        for key in ENV_KEYS:
            value = on_disk.get(key, "")
            # key 永远不发回页面，只说填没填、多长；地址是用户自己要核对的，原样给。
            env[key] = {"set": bool(value), "length": len(value)} if key in SECRET_KEYS else {"set": bool(value), "value": value}
        return {"env": env, "python": os.path.realpath(sys.executable), "env_path": str(env_path),
                "pending_restart": any(on_disk.get(k, "") != loaded_env.get(k, "") for k in ENV_KEYS)}

    @app.get("/api/setup")
    def get_setup() -> dict:
        return setup_view()

    @app.post("/api/setup")
    def post_setup(body: SetupIn) -> dict:
        changes = {k: v.strip() for k, v in body.model_dump().items() if v is not None}
        for key, value in changes.items():
            if "\n" in value or "\r" in value:
                raise HTTPException(422, f"{key} 里不能有换行")
        if owner := changes.get("ALF_IMESSAGE_OWNER"):
            for part in owner.split(","):
                handle = normalize_handle(part)
                if "@" not in handle and not (handle.startswith("+") and len(handle) >= 9):
                    raise HTTPException(422, "手机号要带国家码，写成 +61412345678 这种；或者填 Apple ID 邮箱")
            changes["ALF_IMESSAGE_OWNER"] = ",".join(normalize_handle(p) for p in owner.split(",") if p.strip())
        for key in MAIL_KEYS:
            if key.endswith("USER") and changes.get(key) and not email_smtp.ADDRESS.fullmatch(changes[key]):
                raise HTTPException(422, f"{changes[key]} 不像一个邮箱地址")
        if changes:
            update_dotenv(env_path, changes)
        return setup_view()

    @app.post("/api/email/test")
    def test_email() -> dict:
        """逐个邮箱试一次登录（收信和发信各一次）。只登录，不读信的内容，也不发任何东西。"""
        results = []
        for box in cfg.mailboxes:
            row = {"name": box.name, "label": box.label, "user": box.user, "receive": None, "send": None}
            for part, attempt in (("receive", lambda: check_inbox(box)), ("send", lambda: check_outbox(box))):
                try:
                    attempt()
                    row[part] = "ok"
                except ConnectorError as e:
                    row[part] = str(e)
            results.append(row)
        return {"mailboxes": results}

    @app.post("/api/restart", status_code=202)
    def post_restart() -> dict:
        restart()
        return {"restarting": True}

    @app.get("/api/doctor")
    def get_doctor() -> dict:
        return {"checks": doctor.as_dicts(doctor.collect(cfg, status()))}

    @app.get("/api/imessage/recent")
    def imessage_recent(limit: int = Query(8, ge=1, le=30)) -> dict:
        """查"这条消息为什么没回"：最近几条消息的形态和处理结论。没有正文，也没有对方的地址。"""
        if imessage is None:
            raise HTTPException(409, "还没接入 iMessage：先填 iMessage 地址并重启")
        try:
            rows = imessage.chatdb.recent(imessage.im.owners, limit)
        except ChannelError as e:
            raise HTTPException(503, str(e)) from e
        with lock:
            cursor = load_state(conn, IMESSAGE).cursor
        return {"cursor": int(cursor) if cursor else None, "messages": rows}

    @app.post("/api/test-message", status_code=202)
    def test_message() -> dict:
        if imessage is None:
            raise HTTPException(409, "还没接入 iMessage：先填 iMessage 地址并重启")
        with lock:
            timeline.post(conn, "chat", doctor.TEST_MESSAGE, int(datetime.now(cfg.tz).timestamp()), IMESSAGE)
        return {"queued": True}

    @app.post("/api/calendar/connect")
    def connect_calendar() -> dict:
        """用户点了"授权日历和提醒事项"：这一次允许弹出系统的授权框（两样各问一次），然后马上重新同步。"""
        result = read_calendar(2, ask=True)
        todos = read_reminders(7, ask=True)
        background.request("sources")
        background.request("calendar")
        return {"ok": "events" in result, "error": result.get("error"), "events": len(result.get("events", [])),
                "reminders_ok": "reminders" in todos, "reminders_error": todos.get("error")}

    @app.post("/api/open-privacy-settings", status_code=202)
    def open_privacy() -> dict:
        open_settings()
        return {"opened": True}

    # --- 记忆：看、改、忘 -------------------------------------------------------------

    def memory() -> Memory:
        return Memory(conn, Decider(backend, conn), cfg.decision_threshold)

    def fact_view(f) -> dict:
        # layer：这条记忆平时怎么进对话——core 每轮都带 | topic 提到这个主题时整组带 | moment 经历，带最近几条 |
        # rule 他对 Alf 的要求（现在生效的在「学到的」那一栏里单独列，带范围）
        layer = "rule" if f.category == FEEDBACK else "moment" if f.category == MOMENT else "topic" if f.topic else "core"
        return {"id": f.id, "category": f.category, "label": CATEGORY_LABELS.get(f.category, "其他"), "text": f.text,
                "recorded_at": f.recorded_at, "source": f.source, "unconfirmed": f.needs_confirmation,
                "current": f.valid_to is None, "superseded_by": f.superseded_by,
                "layer": layer, "topic": f.topic, "expires_at": f.expires_at, "past": f.aftermath is not None,
                # quote：他当时的原话；open_loop：还没了结、Alf 惦记着要问一句的事
                "quote": f.quote, "open_loop": f.valid_to is None and f.category != MOMENT and (f.follow_up or f.expires_at is not None),
                # 只对经历有意义：是不是发生在他和 Alf 之间、他当时说的感受、这件事有多大
                "ours": f.ours, "feeling": f.feeling, "importance": f.importance if f.category == MOMENT else None}

    @app.get("/api/memory")
    def get_memory(all: bool = False) -> dict:
        with lock:
            memory().expire(int(datetime.now(cfg.tz).timestamp()))
            facts = memory().history(limit=500) if all else memory().current(limit=500)
            now = int(datetime.now(cfg.tz).timestamp())
            warmth = affinity.view(conn, now)
            learned = lessons.view(conn, now)
            who = persona.view(conn, prefs.load(conn, cfg))
            seen = impressions.view(conn, now)
            agreed = pacts.view(conn, now)
        return {"facts": [fact_view(f) for f in facts], "affinity": warmth, "lessons": learned,
                "persona": who, "impressions": seen, "pacts": agreed}

    @app.patch("/api/memory/{fact_id}")
    def edit_fact(fact_id: int, body: FactIn) -> dict:
        if body.text is None and body.scope is None:
            raise HTTPException(422, "没说要改什么")
        with lock:
            if body.scope is not None and not lessons.set_scope(conn, fact_id, body.scope):
                raise HTTPException(404, "没有这条要求")
            if body.text is None:
                return {"fact": fact_view(memory().get(fact_id))}
            revised = memory().revise(fact_id, " ".join(body.text.split()), now=int(datetime.now(cfg.tz).timestamp()))
        if not revised:
            raise HTTPException(404, "没有这条记忆")
        return {"fact": fact_view(revised[1])}

    # --- 学到的：例子、聊天里改的设置、等他点头的规矩 ------------------------------------

    @app.delete("/api/lessons/examples/{example_id}")
    def drop_example(example_id: int) -> dict:
        with lock:
            if not lessons.remove_example(conn, example_id, int(datetime.now(cfg.tz).timestamp())):
                raise HTTPException(404, "没有这个例子")
        return {"ok": True}

    @app.post("/api/lessons/settings/{change_id}/undo")
    def undo_setting(change_id: int) -> dict:
        with lock:
            problem = lessons.undo_setting(conn, cfg, change_id, int(datetime.now(cfg.tz).timestamp()))
        if problem:
            raise HTTPException(409, problem)
        return {"ok": True}

    @app.post("/api/lessons/proposals/{proposal_id}")
    def answer_proposal(proposal_id: int, body: ProposalIn) -> dict:
        """在页面上对一条规矩提议点头或摇头。点头 = 记成一条要求，出处的例子从此由它取代。"""
        now = int(datetime.now(cfg.tz).timestamp())
        with lock:
            proposal = next((p for p in lessons.open_proposals(conn, now) if p["id"] == proposal_id), None)
            if not proposal:
                raise HTTPException(404, "没有这条提议")
            rule = None
            if body.accept:
                rule = memory().write(proposal["text"], FEEDBACK, now=now, scope=lessons.scope_of(proposal["scope"])).fact.id
            lessons.settle_proposal(conn, proposal_id, rule_id=rule)
        return {"ok": True, "rule_id": rule}

    @app.get("/api/retrieval")
    def retrieval_status() -> dict:
        """按意思检索的状态：嵌入模型在不在本机，每一类有多少条已经建好索引。只有数目，没有内容。"""
        found = embed.status()
        with lock:
            return {**found, "coverage": retrieve.coverage(conn, found["model"] if found["present"] else None)}

    @app.get("/api/search/recent")
    def recent_searches() -> dict:
        """最近上网查过什么：问的是什么、DeepSeek 实际搜了哪几条、看了哪几个网页。给设置页用。"""
        with lock:
            return {"searches": websearch.view(conn)}

    # --- 人设里它说过的话、它对他的印象、约定：在页面上删、改、了结 ---------------------------

    @app.delete("/api/persona/notes/{note_id}")
    def drop_persona_note(note_id: int) -> dict:
        with lock:
            if not persona.remove_note(conn, note_id, int(datetime.now(cfg.tz).timestamp())):
                raise HTTPException(404, "没有这一条")
        return {"ok": True}

    @app.patch("/api/impressions/{impression_id}")
    def edit_impression(impression_id: int, body: ImpressionIn) -> dict:
        with lock:
            if not impressions.rewrite(conn, impression_id, body.text):
                raise HTTPException(404, "没有这条印象")
        return {"ok": True}

    @app.delete("/api/impressions/{impression_id}")
    def drop_impression(impression_id: int) -> dict:
        """在页面上说一条印象不对：收回，以后不再形成同样的。"""
        with lock:
            if not impressions.withdraw(conn, impression_id, int(datetime.now(cfg.tz).timestamp()), "在记忆页上删掉的"):
                raise HTTPException(404, "没有这条印象")
        return {"ok": True}

    @app.post("/api/pacts/{pact_id}")
    def settle_pact(pact_id: int, body: PactIn) -> dict:
        with lock:
            if not pacts.settle(conn, pact_id, body.outcome, int(datetime.now(cfg.tz).timestamp()), "在记忆页上了结的"):
                raise HTTPException(404, "没有这条约定")
        return {"ok": True}

    @app.delete("/api/memory/{fact_id}")
    def forget_fact(fact_id: int) -> dict:
        with lock:
            old = memory().forget(fact_id, now=int(datetime.now(cfg.tz).timestamp()))
        if not old:
            raise HTTPException(404, "没有这条记忆")
        return {"forgot": old.text}

    return app


def serve(cfg: Settings, backend: Backend, port: int, llm_backend: LLMBackend | None = None) -> None:
    import uvicorn

    # force=True：有的依赖（MCP 库）在导入时就给根日志器装了自己的处理器，不强制的话这里的格式不生效，
    # 日志里就没有时间了。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%m-%d %H:%M:%S", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)   # 不然每次调模型都打一行请求日志
    app = create_app(cfg, backend, llm_backend=llm_backend)
    log.info("Alf 启动中 http://127.0.0.1:%d  对话=%s  判断=%s  iMessage=%s  时区=%s  数据=%s", port,
             llm_backend.name if llm_backend else "模板", backend.name, "开" if cfg.imessage else "未配置", cfg.tz.key, cfg.db_path)
    # access_log 关掉：页面每 3 秒轮询一次，开着的话一天几万行日志。
    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False, log_level="warning")
