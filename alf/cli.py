"""命令行入口。

    alf                  终端里对话（打字，或 /rec 录音、/voice <文件>）
    alf serve            启动常驻服务：聊天网页 + 心跳（提醒、邮箱、每日播报）
    alf doctor           逐项检查 key、iMessage 权限、后台服务，告诉你还差什么
    alf app              打开 Alf.app（原生窗口；第一次会先编译）
    alf organize         整理旧记忆：补上主题和到期日
    alf check-email      自检邮箱配置：能不能登录、最新几封能不能正常解析
    alf service install  注册成登录自启的后台服务（macOS）
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import asr, db
from .agent import Turn, format_reminders
from .chat import make_agent
from .config import Settings, load_dotenv, settings
from .decide import Decider, DecisionError, make_backend

JEV_PRICE_PER_MTOK = 0.042  # 只算输入；Jev 的输出 token 免费（docs.typesafe.ai/models）
ICONS = {"reminder": "🔔", "email": "📧", "briefing": "☀️", "digest": "📝", "nudge": "💬"}

COMMANDS = """命令：
  /rec              录音（按回车结束），识别后当作一句话发给 Alf
  /voice <文件>     识别音频文件（wav/m4a/mp3…）
  /list             待办提醒
  /cancel <编号>    取消提醒
  /facts [all]      当前记忆；all 连同被更新掉的旧版本一起看
  /confirm <编号>   确认一条语音来源的记忆
  /vocab [add <词>] 语音识别热词（人名、课程代码…）
  /usage            Jev 调用次数、延迟、token 和费用
  /help  /quit"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alf", description="Alf 个人助手")
    offline_help = "用关键词规则代替 Jev，只用来试流程（没有真正的判断能力）"
    parser.add_argument("--offline", action="store_true", help=offline_help)
    sub = parser.add_subparsers(dest="command")
    # 子命令里的同名参数用 SUPPRESS 做默认值，否则会把写在前面的 `alf --offline serve` 覆盖回 False。
    p = sub.add_parser("serve", help="启动常驻服务：聊天网页 + 心跳")
    p.add_argument("--port", type=int, help="监听端口（默认取 ALF_WEB_PORT，再默认 8765）")
    p.add_argument("--offline", action="store_true", default=argparse.SUPPRESS, help=offline_help)
    p = sub.add_parser("doctor", help="检查 key、iMessage 权限和后台服务")
    p.add_argument("--send-test", action="store_true", help="顺便给你自己发一条测试 iMessage")
    sub.add_parser("app", help="打开 Alf.app（原生窗口；第一次会先编译）")
    sub.add_parser("check-email", help="自检邮箱配置（只读）")
    sub.add_parser("organize", help="整理旧记忆：补上主题和到期日（只改归类，不改内容）")
    sub.add_parser("index", help="下载按意思检索用的嵌入模型（约 490 MB），并给已有的记忆、对话、邮件建好索引")
    p = sub.add_parser("hear", help="听录音：下载语音模型（--setup），或者导入几段录音让它听")
    p.add_argument("files", nargs="*", help="要导入的录音文件（m4a、wav、mp3）；几个文件按顺序算是连着录的")
    p.add_argument("--setup", action="store_true", help="下载语音模型（约 200 MB，存到 ~/.cache/alf/voice）")
    p.add_argument("--at", help="第一个文件是什么时候开始录的，写成「2026-11-18 14:05」；不写就按文件的修改时间往前推")
    p.add_argument("--show", action="store_true", help="听完把转出来的话打印出来（默认只报数目）")
    p = sub.add_parser("service", help="把 Alf 注册成登录自启的后台服务（macOS）")
    p.add_argument("action", choices=["install", "uninstall", "status"])
    p.add_argument("--port", type=int)
    p.add_argument("--offline", action="store_true", default=argparse.SUPPRESS, help=offline_help)
    p.add_argument("--keep-awake", action="store_true", help="接着电源时不让 Mac 睡眠")
    args = parser.parse_args(argv)

    load_dotenv()
    cfg = settings()
    if args.command == "serve":
        from .server import serve

        llm_backend = llm_for(cfg, args.offline)
        serve(cfg, make_backend(args.offline, llm_backend), args.port or cfg.web_port, llm_backend)
        return 0
    if args.command == "doctor":
        from . import doctor

        return doctor.run(cfg, send_test=args.send_test)
    if args.command == "app":
        return open_app()
    if args.command == "check-email":
        return check_email(cfg)
    if args.command == "organize":
        return organize_memory(cfg, offline=args.offline)
    if args.command == "index":
        return build_index(cfg)
    if args.command == "hear":
        return hear(cfg, args.files, setup=args.setup, at=args.at, show=args.show)
    if args.command == "service":
        from . import service

        if args.action == "install":
            return service.install(args.port or cfg.web_port, offline=args.offline, keep_awake=args.keep_awake)
        return service.uninstall() if args.action == "uninstall" else service.status()
    return chat(cfg, offline=args.offline)


def build_index(cfg: Settings) -> int:
    """下载嵌入模型（没有的话），把库里现有的东西全部建好检索索引。后台服务平时会自己跟上新增的；这条命令用在第一次。"""
    from datetime import datetime

    from . import embed, retrieve

    if embed.files() is None:
        print(f"下载嵌入模型 {embed.MODEL}（约 490 MB，存到 ~/.cache/huggingface）……")
        try:
            embed.files(download=True)
        except Exception as e:
            print(f"没下载成：{e}")
            return 1
    embedder = embed.default()
    if embedder is None:
        print("嵌入模型用不了（缺 onnxruntime？运行 uv sync 试试）。")
        return 1
    conn = db.connect(cfg.db_path)
    try:
        while True:
            done = retrieve.sync(conn, embedder, int(datetime.now(cfg.tz).timestamp()))
            if done["embedded"]:
                print(f"  算了 {done['embedded']} 条，还剩 {done['left']} 条")
            if not done["left"]:
                break
        label = {"fact": "记忆", "chunk": "对话片段", "mail": "邮件", "heard": "听到的话", "event": "每天的事"}
        print("索引好了：" + "，".join(f"{label[k]} {v['embedded']}/{v['total']}" for k, v in retrieve.coverage(conn, embedder.name).items()))
    finally:
        conn.close()
    return 0


def hear(cfg: Settings, files: list[str], *, setup: bool, at: str | None, show: bool) -> int:
    """下载语音模型，或者手动导入几段录音。平时录音是手表经手机传过来、后台自己听的；这条命令用在第一次装模型，
    和手头有一段现成的录音想让它听的时候。默认只报数目——转出来的话是他身边人说的，不该随手打在终端里。"""
    from datetime import datetime

    from . import hearing, people, voice

    if setup:
        try:
            voice.download()
        except Exception as e:
            print(f"没下载成：{e}")
            return 1
    ears = voice.default()
    conn = db.connect(cfg.db_path)
    try:
        now = datetime.now(cfg.tz)
        if not files:
            s = hearing.status(conn, cfg, now)
            print(("语音模型在本机。" if s["models"] else "语音模型还没下载，运行 uv run alf hear --setup。")
                  + f"今天听了 {s['today_minutes']} 分钟（人声 {s['today_speech_minutes']} 分钟），记下 {s['today_lines']} 句；"
                  + f"还有 {s['waiting']} 段没听；认识 {s['people']} 个人，另有 {s['unnamed']} 个没名字的声音。")
            return 0
        if ears is None:
            print("语音模型还没下载，先运行 uv run alf hear --setup。")
            return 1
        try:
            start = datetime.strptime(at, "%Y-%m-%d %H:%M").replace(tzinfo=cfg.tz).timestamp() if at else None
        except ValueError:
            print("--at 写成「2026-11-18 14:05」这种格式。")
            return 1
        before = conn.execute("SELECT COALESCE(MAX(id), 0) FROM utterances").fetchone()[0]
        for name in files:
            path = Path(name).expanduser()
            seconds = voice.duration(path) if path.is_file() else None
            if not seconds:
                print(f"{path.name}：读不出来，跳过")
                continue
            began = start if start is not None else path.stat().st_mtime - seconds
            try:
                got = hearing.receive(conn, cfg, path, began, "import", int(now.timestamp()))
            except voice.VoiceError as e:
                print(f"{path.name}：{e}")
                continue
            print(f"{path.name}：{'收下了' if got else '以前收过，跳过'}（{seconds / 60:.1f} 分钟）")
            start = began + seconds if start is not None else None
        # 手动导入的不用等"是不是还有下一段"：把钟拨快一点，让凑批的那道等待直接过去
        done = hearing.hear_pending(conn, cfg, ears, budget=3600, clock=lambda: time.time() + hearing.SETTLE)
        print(f"听了 {done['seconds'] / 60:.1f} 分钟（人声 {done['speech'] / 60:.1f} 分钟），记下 {done['lines']} 句。"
              "后台会在对话说完二十分钟后把它们整理成「每天的事」。")
        if show:
            names = people.labels(conn)
            for u in conn.execute("SELECT * FROM utterances WHERE id > ? ORDER BY started_at", (before,)):
                print(f"  {datetime.fromtimestamp(u['started_at'], cfg.tz):%m-%d %H:%M:%S} {names.get(u['person_id'], '有人')}：{u['text']}")
    finally:
        conn.close()
    return 0


def organize_memory(cfg: Settings, offline: bool) -> int:
    from datetime import datetime

    from . import db
    from .decide import Decider
    from .llm import LLM, LLMError
    from .organize import organize

    backend = llm_for(cfg, offline)
    if backend is None:
        print("整理记忆要用到 DeepSeek：先在设置里填好 key。")
        return 1
    conn = db.connect(cfg.db_path)
    try:
        changes = organize(conn, LLM(backend, conn), Decider(make_backend(offline, backend), conn), cfg, datetime.now(cfg.tz))
    except LLMError as e:
        print(f"没整理成：{e}")
        return 1
    print("\n".join(changes) if changes else "没有需要整理的：记忆都已经归好类了。")
    return 0


def llm_for(cfg: Settings, offline: bool):
    """生成式模型的客户端；离线模式或没填 key 时没有。"""
    if offline or not cfg.llm:
        return None
    from .llm import DeepSeek

    return DeepSeek(cfg.llm)


def open_app() -> int:
    """打开原生窗口外壳。它只是界面，真正干活的是后台服务（alf service install）。"""
    import subprocess

    from .config import ROOT

    bundle = ROOT / "Alf.app"   # 放在项目第一层，方便在访达里直接双击
    if not bundle.exists():
        print("第一次使用，先编译 Alf.app（需要 Xcode 或命令行工具）…")
        if subprocess.run([str(ROOT / "app" / "build.sh")], check=False).returncode != 0:
            print("编译失败。可以先用浏览器打开 http://127.0.0.1:8765")
            return 1
    return subprocess.run(["open", str(bundle)], check=False).returncode


def check_email(cfg: Settings) -> int:
    from .connectors import ConnectorError
    from .connectors.email_imap import ImapEmail

    from .connectors import email_smtp

    if not cfg.mailboxes:
        print("还没接入邮箱：在 Alf 的设置页「邮箱」里填地址和授权码，或在 .env 里填 ALF_GMAIL_USER / ALF_GMAIL_PASSWORD 等（见 .env.example）。")
        return 1
    problems = 0
    for box in cfg.mailboxes:
        try:
            items = ImapEmail(box).preview(3)
        except ConnectorError as e:
            print(f"✗ {box.label}（{box.user}）：{e}")
            problems += 1
            continue
        print(f"✓ {box.label}（{box.user}）收信正常。最新 {len(items)} 封（只读，没有标成已读，也没有发给模型）：")
        for item in items:
            print(f"    · {item.title}\n      {item.sender}")
        try:
            email_smtp.check(box)
            print("  ✓ 发信登录正常（没有发出任何邮件）")
        except ConnectorError as e:
            print(f"  ✗ 发信：{e}")
            problems += 1
    return 1 if problems else 0


def chat(cfg: Settings, offline: bool) -> int:
    from .heartbeat import Heartbeat, Job
    from .jobs import reminders_job

    conn = db.connect(cfg.db_path)
    llm_backend = llm_for(cfg, offline)
    if not offline and llm_backend is None and not os.getenv("TYPESAFE_API_KEY"):
        print("还没有可用的模型：在 .env 里填 DEEPSEEK_API_KEY（或只填 TYPESAFE_API_KEY 用模板模式）；"
              "也可以加 --offline 先试流程。")
        return 1
    backend = make_backend(offline, llm_backend)
    agent = make_agent(conn, cfg, backend, llm_backend)
    recognizer = asr.Recognizer(cfg.asr_model, cfg.asr_language)

    def on_post(kind: str, text: str) -> None:
        print(f"\n{ICONS.get(kind, '•')} {text}\n你 › ", end="", flush=True)

    # 终端版只带"到期提醒"这一个任务；邮箱和每日播报由常驻服务（alf serve）负责。
    # 两边同时开着也没关系：提醒是原子认领的，只会响一次。
    Heartbeat(cfg, [Job("reminders", "到期提醒", reminders_job, every=cfg.reminder_every)],
              make_decider=lambda c: Decider(backend, c), on_post=on_post).start()

    mode = ("离线规则模式（不是真的模型）" if offline else f"DeepSeek · {llm_backend.model}" if llm_backend
            else f"Jev · {os.getenv('TYPESAFE_DEFAULT_MODEL', 'jev-latest')}（模板回复）")
    print(f"Alf 已启动 [{mode}]  时区 {cfg.tz.key}  数据 {cfg.db_path}\n直接打字对话，/help 看命令。")

    def say(turn: Turn) -> None:
        print("Alf ›", agent.handle(turn).text)

    def voice(path: Path | None) -> None:
        try:
            if path is None:
                path = asr.record(Path(tempfile.gettempdir()) / f"alf-{uuid.uuid4().hex[:8]}.wav")
            elif not path.exists():
                print(f"找不到文件：{path}")
                return
            attachment_id, stored = asr.store(conn, path, cfg.attachment_dir)
            print(f"识别中…（Whisper {cfg.asr_model}，首次使用会下载模型）")
            t = recognizer.transcribe(stored, asr.vocabulary(conn))
        except Exception as e:  # 麦克风没授权、模型没下载、文件读不了……
            print(f"语音识别失败：{e}")
            return
        asr.save_transcript(conn, attachment_id, t)
        if not t.text:
            print("没听清，再说一次？")
            return
        print(f"🎙  「{t.text}」（置信度 {t.confidence:.2f}，语言 {t.language}）")
        say(Turn(t.text, source="voice", attachment_id=attachment_id, asr_confidence=t.confidence))

    while True:
        try:
            line = input("你 › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        try:
            if not line.startswith("/"):
                say(Turn(line))
            elif cmd in ("/quit", "/exit"):
                return 0
            elif cmd == "/help":
                print(COMMANDS)
            elif cmd == "/rec":
                voice(None)
            elif cmd == "/voice":
                if arg:
                    voice(Path(arg).expanduser())
                else:
                    print("用法：/voice <音频文件>")
            elif cmd == "/list":
                items = agent.reminders.pending()
                print(format_reminders(items, agent.clock()) if items else "现在没有待办提醒。")
            elif cmd == "/cancel":
                r = agent.reminders.cancel(int(arg)) if arg.isdigit() else None
                print(f"已取消：{r.title}" if r else "没有这个待办提醒编号。")
            elif cmd == "/facts":
                show_facts(agent, everything=arg == "all")
            elif cmd == "/confirm":
                if arg.isdigit():
                    agent.memory.confirm(int(arg))
                    print("已确认。")
                else:
                    print("用法：/confirm <编号>")
            elif cmd == "/vocab":
                vocab(conn, arg)
            elif cmd == "/usage":
                usage(conn)
            else:
                print("不认识这个命令，/help 看全部。")
        except DecisionError as e:
            print(f"Alf › 抱歉，现在做不了判断：{e}")


def show_facts(agent, everything: bool) -> None:
    facts = agent.memory.history() if everything else agent.memory.current()
    if not facts:
        print("还没有记忆。")
    for f in facts:
        flags = "".join([" [旧版本→#%d]" % f.superseded_by if f.superseded_by else "",
                         " [语音·未确认]" if f.needs_confirmation else ""])
        print(f"  #{f.id}  [{f.category}] {f.text}  ({f.recorded(agent.cfg.tz)}){flags}")


def vocab(conn, arg: str) -> None:
    if arg.startswith("add "):
        term = arg[4:].strip()
        with conn:
            conn.execute("INSERT OR IGNORE INTO vocabulary(term, created_at) VALUES (?, ?)", (term, int(time.time())))
        print(f"已加入热词：{term}")
    else:
        terms = asr.vocabulary(conn)
        print("识别热词：" + ("、".join(terms) if terms else "（无）用 /vocab add <词> 添加"))


def usage(conn) -> None:
    rows = conn.execute(
        "SELECT purpose, COUNT(*) n, SUM(1 - ok) failed, AVG(latency_ms) avg_ms, MAX(latency_ms) max_ms,"
        " SUM(COALESCE(input_tokens, 0)) tokens FROM model_calls GROUP BY purpose ORDER BY n DESC").fetchall()
    if not rows:
        print("还没有 Jev 调用。")
        return
    total = sum(r["tokens"] for r in rows)
    print(f"{'用途':<18}{'次数':>6}{'失败':>6}{'平均ms':>9}{'最大ms':>9}{'输入tokens':>12}")
    for r in rows:
        print(f"{r['purpose']:<18}{r['n']:>6}{r['failed']:>6}{r['avg_ms']:>9.0f}{r['max_ms']:>9}{r['tokens']:>12}")
    print(f"合计输入 {total} tokens ≈ ${total / 1e6 * JEV_PRICE_PER_MTOK:.6f}（输出免费）")


if __name__ == "__main__":
    sys.exit(main())
