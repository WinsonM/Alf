"""iMessage 里的语音和图片（场景 16、17），以及 👍👎 反馈。假的「信息」数据库加上附件表；
格式转换用的是系统自带的 afconvert 和 sips，真的跑一遍。"""
import json
import sqlite3
import struct
import wave
import zlib
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from alf import asr, db, feedback, lessons, media, mood, timeline
from alf.chat import ChatAgent
from alf.config import IMessageConfig
from alf.decide import Decider
from alf.imessage import UNSUPPORTED, ChatDB, IMessageChannel
from alf.llm import LLM

from .conftest import NOW, FakeLLM, Scripted
from .test_imessage import OWNER, Outbox, receive


def png(path: Path, width=40, height=30) -> Path:
    """一张纯色的小 PNG，够让 sips 当成真图片来转。"""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = b"".join(b"\x00" + b"\xd0\x40\x40" * width for _ in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def wav(path: Path, seconds=0.3) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * int(16000 * seconds))
    return path


class Listener:
    """语音识别的替身。"""

    def __init__(self, text="明天下午三点 remind me 交 assignment", confidence=0.88, ready=True):
        self.text, self.confidence, self.ready, self.heard = text, confidence, ready, []

    def cached(self):
        return self.ready

    def transcribe(self, path, vocabulary):
        self.heard.append(path)
        if isinstance(self.text, Exception):
            raise self.text
        return asr.Transcript(self.text, self.confidence, "zh", [])


@pytest.fixture
def rig(cfg, chatdb, tmp_path):
    c = sqlite3.connect(chatdb)
    c.executescript("""
        ALTER TABLE message ADD COLUMN is_audio_message INTEGER DEFAULT 0;
        ALTER TABLE message ADD COLUMN guid TEXT;
        ALTER TABLE message ADD COLUMN associated_message_guid TEXT;
        CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, mime_type TEXT);
        CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
    """)
    c.commit()
    c.close()
    root = tmp_path / "Attachments"
    root.mkdir()
    cfg = replace(cfg, imessage=IMessageConfig(owners=(OWNER,), db_path=chatdb, poll=0.01, attachments_root=root))
    fake, out, clock, ears = FakeLLM(), Outbox(), [NOW.timestamp()], Listener()
    conn = db.connect(cfg.db_path)
    agent = ChatAgent(conn, LLM(fake, conn), Decider(Scripted(rel="unrelated", claim=0.05), conn), cfg, clock=lambda: NOW)
    channel = IMessageChannel(cfg, agent.handle, sender=out, clock=lambda: clock[0], recognizer=ears)
    channel.step()                                        # 第一轮只定起点
    return type("Rig", (), {"cfg": cfg, "fake": fake, "out": out, "clock": clock, "conn": conn, "agent": agent, "channel": channel,
                            "chatdb": chatdb, "root": root, "ears": ears})


def attach(path, rowid, filename, mime, audio=False):
    c = sqlite3.connect(path)
    cur = c.execute("INSERT INTO attachment(filename, mime_type) VALUES (?, ?)", (str(filename) if filename else None, mime))
    c.execute("INSERT INTO message_attachment_join VALUES (?, ?)", (rowid, cur.lastrowid))
    c.execute("UPDATE message SET is_audio_message = ? WHERE ROWID = ?", (int(audio), rowid))
    c.commit()
    c.close()


# --- 语音 ----------------------------------------------------------------------------

def test_a_voice_message_is_transcribed_echoed_and_handled_like_typing(rig):
    src = wav(rig.root / "Audio Message.wav")
    src.rename(src.with_suffix(".caf"))                    # 「信息」里的语音是 .caf
    attach(rig.chatdb, receive(rig.chatdb, "￼", attachments=1), rig.root / "Audio Message.caf", "audio/x-caf", audio=True)
    rig.fake.script += [[("create_reminder", {"title": "交 assignment", "time_text": "明天下午三点"})], "好嘞"]
    rig.channel.step()
    # 回复开头不再附"🎙 听到：……"（用户觉得不像聊天）。这里回复没说时间，所以提醒那行照附，他能看出听成了什么。
    assert rig.out.sent == [(OWNER, "好嘞\n⏰ 交 assignment｜明天 9月23日（周三）15:00")]
    assert "[用户刚发来的消息]（语音转写，可能有错字）\n明天下午三点 remind me 交 assignment" in rig.fake.context(0)
    row = rig.conn.execute("SELECT kind, transcript, confidence, path FROM attachments").fetchone()
    assert (row["kind"], row["transcript"], row["confidence"]) == ("audio", "明天下午三点 remind me 交 assignment", 0.88)
    assert Path(row["path"]).parent == rig.cfg.attachment_dir             # 原件存进了 Alf 自己的附件库
    episode = rig.conn.execute("SELECT source, attachment_id FROM episodes WHERE role = 'user'").fetchone()
    assert episode["source"] == "voice" and episode["attachment_id"]
    assert rig.agent.reminders.pending()[0].source == "voice"


def test_voice_that_cannot_be_used_gets_a_plain_answer_instead_of_a_guess(rig):
    def voice(name="a.caf", exists=True):
        if exists:
            wav(rig.root / "x.wav").rename(rig.root / name)
        attach(rig.chatdb, receive(rig.chatdb, None, attachments=1), rig.root / name, "audio/x-caf", audio=True)
        rig.clock[0] += 1
        rig.channel.step()
        return rig.out.sent[-1][1]

    rig.ears.text = ""
    assert voice() == "这条语音我没听清，再说一遍或者打字？"
    rig.ears.text = RuntimeError("model download failed")
    assert voice("b.caf") == "这条语音没识别成（RuntimeError），先打字跟我说吧。"
    rig.channel.recognizer = None
    assert voice("c.caf") == "语音我现在听不了，先打字跟我说吧。"
    assert rig.fake.requests == []                                         # 这几种情况都没有去打扰模型


def test_first_voice_message_warns_that_the_model_has_to_download(rig, monkeypatch):
    monkeypatch.setattr("alf.imessage.ATTACHMENT_WAIT", 0)
    rig.ears.ready = False
    wav(rig.root / "x.wav").rename(rig.root / "a.caf")
    attach(rig.chatdb, receive(rig.chatdb, None, attachments=1), rig.root / "a.caf", "audio/x-caf", audio=True)
    rig.fake.script.append("在呢")
    rig.channel.step()
    assert [t for _, t in rig.out.sent][0] == "收到语音。第一次听要先下载识别模型（几百 MB），可能要等几分钟。"
    attach(rig.chatdb, receive(rig.chatdb, None, attachments=1), rig.root / "not-downloaded-yet.caf", "audio/x-caf", audio=True)
    rig.clock[0] += 1
    rig.channel.step()
    assert rig.out.sent[-1][1] == "这条语音还没下载到这台 Mac 上，过几秒再发一次？"


def test_audio_is_really_converted_for_the_recogniser(tmp_path):
    caf = tmp_path / "msg.caf"
    assert media._run(["afconvert", "-f", "caff", "-d", "aac", str(wav(tmp_path / "src.wav", 0.5)), str(caf)]) and caf.exists()
    out = media.to_wav(caf)
    with wave.open(str(out)) as w:
        assert (out.suffix, w.getframerate(), w.getnchannels()) == (".wav", 16000, 1) and w.getnframes() > 4000
    broken = tmp_path / "broken.caf"
    broken.write_bytes(b"not audio")
    assert media.to_wav(broken) == broken                                   # 转不了就把原件交给识别那边自己试


# --- 图片 ----------------------------------------------------------------------------

def test_a_picture_reaches_the_model_as_an_image_and_nothing_is_created_unasked(rig):
    attach(rig.chatdb, receive(rig.chatdb, "￼看看这个", attachments=1), png(rig.root / "IMG_0001.png", 3000, 40), "image/png")
    rig.fake.script.append("这是迎新晚会的海报：周五晚上七点，在 Manning Bar。要不要帮你建个提醒？")
    rig.channel.step()
    assert rig.out.sent == [(OWNER, "这是迎新晚会的海报：周五晚上七点，在 Manning Bar。要不要帮你建个提醒？")]
    text, image = rig.fake.chat_requests[0]["messages"][1]["content"]
    assert text["type"] == "text" and text["text"].endswith("[用户刚发来的消息]（附了 1 张图片，在这条消息后面）\n看看这个")
    assert image["type"] == "image_url" and image["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/")   # 真的转成了 JPEG
    row = rig.conn.execute("SELECT kind, path FROM attachments").fetchone()
    view = Path(row["path"]).with_name(Path(row["path"]).stem + ".view.jpg")
    assert row["kind"] == "image" and view.exists()
    width = int(media.subprocess.run(["sips", "-g", "pixelWidth", str(view)], capture_output=True, text=True).stdout.split()[-1])
    assert width == media.MAX_IMAGE_EDGE                                    # 3000 像素宽的图缩到了上限
    assert [r["text"] for r in rig.conn.execute("SELECT text FROM episodes WHERE role = 'user'")] == ["[图片×1] 看看这个"]
    assert rig.agent.reminders.pending() == []
    # 下一句"好"：海报上的时间地点在 Alf 刚才那句回复里，模型接得上
    rig.fake.script += [[("create_reminder", {"title": "迎新晚会", "time_text": "周五晚上七点"})], "建好了"]
    receive(rig.chatdb, "好，建一个")
    rig.clock[0] += 1
    rig.channel.step()
    assert "Alf：这是迎新晚会的海报：周五晚上七点，在 Manning Bar。" in rig.fake.context(1)
    assert rig.out.sent[-1][1] == "建好了\n⏰ 迎新晚会｜9月25日（周五）19:00"


def test_only_pictures_from_the_messages_folder_are_read_and_other_files_are_declined(rig, tmp_path, monkeypatch):
    monkeypatch.setattr("alf.imessage.ATTACHMENT_WAIT", 0)
    outside = png(tmp_path / "secret.png")                                  # 数据库里写了一个附件目录之外的路径
    attach(rig.chatdb, receive(rig.chatdb, None, attachments=1), outside, "image/png")
    rig.channel.step()
    assert rig.out.sent[-1][1] == "图片还没下载到这台 Mac 上，过几秒再发一次？" and rig.fake.requests == []
    (rig.root / "bad.png").write_bytes(b"this is not a picture")
    attach(rig.chatdb, receive(rig.chatdb, None, attachments=1), rig.root / "bad.png", "image/png")
    rig.clock[0] += 1
    rig.channel.step()
    assert rig.out.sent[-1][1] == "这张图片我打不开，换一种方式发给我试试？"
    (rig.root / "notes.pdf").write_bytes(b"%PDF-1.4")
    attach(rig.chatdb, receive(rig.chatdb, None, attachments=1), rig.root / "notes.pdf", "application/pdf")
    rig.clock[0] += 1
    rig.channel.step()
    assert rig.out.sent[-1][1] == UNSUPPORTED and rig.conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == 1   # 只登记了那张打不开的图


def test_attachment_kinds_are_told_apart():
    assert media.kind_of(Path("a.caf"), "", True) == "audio" and media.kind_of(Path("IMG.HEIC"), "") == "image"
    assert media.kind_of(Path("x"), "audio/amr") == "audio" and media.kind_of(Path("x"), "image/webp") == "image"
    assert media.kind_of(Path("clip.mov"), "video/quicktime") is None and media.kind_of(None, "") is None


# --- 👍👎 和"没用" --------------------------------------------------------------------

def tapback(path, kind, target_guid):
    c = sqlite3.connect(path)
    cur = c.execute("INSERT INTO message(text, handle_id, associated_message_type, associated_message_guid) VALUES (?, 1, ?, ?)",
                    ("踩了一条消息", kind, f"p:0/{target_guid}"))
    c.execute("INSERT INTO chat_message_join VALUES (1, ?)", (cur.lastrowid,))
    c.commit()
    c.close()


def sent_by_alf(rig, kind, text, guid):
    """Alf 发出一条主动消息：进时间线、被渠道送出去，「信息」数据库里也出现一条自己发的。"""
    episode = timeline.post(rig.conn, kind, text, int(rig.clock[0]), "imessage")
    rig.channel.step()
    c = sqlite3.connect(rig.chatdb)
    c.execute("INSERT INTO message(text, is_from_me, guid) VALUES (?, 1, ?)", (rig.out.sent[-1][1], guid))
    c.commit()
    c.close()
    return episode


def test_thumbs_on_a_proactive_message_are_counted_and_can_be_taken_back(rig):
    news = sent_by_alf(rig, "nudge", "OpenAI 今天发了新模型，跟你做的东西有点关系", "G-1")
    brief = sent_by_alf(rig, "briefing", "今天还有 1 个安排", "G-2")
    tapback(rig.chatdb, 2002, "G-1")                                        # 👎
    tapback(rig.chatdb, 2001, "G-2")                                        # 👍
    tapback(rig.chatdb, 2003, "G-2")                                        # 哈哈：不算评价
    tapback(rig.chatdb, 2002, "G-unknown")
    rig.clock[0] += 1
    rig.channel.step()
    assert [tuple(r) for r in rig.conn.execute("SELECT episode_id, rating, source FROM feedback ORDER BY episode_id")] == [
        (news, -1, "tapback"), (brief, 1, "tapback")]
    assert feedback.stats(rig.conn, NOW) == {"days": 7, "sent": 2, "down": 1, "up": 1, "useless_rate": 0.5}
    # 👎 同时是一个"这种别再发"的例子，下次要不要主动开口时模型看得到；👍 的那条不是
    assert lessons.example_lines(rig.conn, "proactive") == ["你发的「OpenAI 今天发了新模型，跟你做的东西有点关系」——他说：👎"]
    assert rig.fake.requests == [] and len(rig.out.sent) == 2               # 点一下不会触发一轮对话
    # 短期情绪也跟着动：👎 留个疙瘩（1.0），随后的 👍 消掉一半
    assert round(mood.current(rig.conn, int(rig.clock[0]))[1], 2) == 0.5 and mood.view(rig.conn, int(rig.clock[0]))["reason"] == "对一条消息点了 👎"
    tapback(rig.chatdb, 3002, "G-1")                                        # 取消 👎
    rig.clock[0] += 1
    rig.channel.step()
    assert feedback.stats(rig.conn, NOW)["down"] == 0 and lessons.example_lines(rig.conn, "proactive") == []


def test_saying_useless_right_after_a_proactive_message_counts_but_not_after_a_chat_reply(rig):
    sent_by_alf(rig, "nudge", "要不要出去走走？", "G-1")
    rig.fake.script += ["好，这类以后少发", "哪条？", "嗯"]
    receive(rig.chatdb, "没用")
    rig.clock[0] += 1
    rig.channel.step()
    assert feedback.stats(rig.conn, NOW)["down"] == 1
    assert "（程序已经记下：他觉得你上一条主动发的消息没用。简单回应一句就好，不用辩解。）" in rig.fake.context()
    receive(rig.chatdb, "没用")                                              # 这次上一条是聊天回复，不是主动消息
    rig.clock[0] += 1
    rig.channel.step()
    assert feedback.stats(rig.conn, NOW)["down"] == 1 and "程序已经记下" not in rig.fake.context()
    # 用户自己设的提醒不算主动消息：对它说"没用"不计入，统计的分母里也没有它
    timeline.post(rig.conn, "reminder", "喝水（10:00）", int(rig.clock[0]), "imessage")
    receive(rig.chatdb, "这条没用")
    rig.clock[0] += 1
    rig.channel.step()
    assert feedback.stats(rig.conn, NOW + timedelta(minutes=1)) == {"days": 7, "sent": 1, "down": 1, "up": 0, "useless_rate": 1.0}
    assert [bool(feedback.USELESS.match(t)) for t in ("没用", "这条没用", "这种消息没啥用", "别发这种了", "没用的东西你也发", "有用")] == [True, True, True, True, False, False]


def test_old_message_database_without_the_new_columns_still_works(cfg, tmp_path):
    """早先版本的 macOS 没有 is_audio_message 等列，也没有附件表：照常读文字，附件当作不认识的。"""
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, style INTEGER);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB, handle_id INTEGER, is_from_me INTEGER DEFAULT 0,
            cache_has_attachments INTEGER DEFAULT 0, associated_message_type INTEGER DEFAULT 0, item_type INTEGER DEFAULT 0);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        INSERT INTO handle VALUES (1, '+61412345678');
        INSERT INTO message VALUES (1, '你好', NULL, 1, 0, 0, 0, 0), (2, NULL, NULL, 1, 0, 1, 0, 0), (3, 'liked', NULL, 1, 0, 0, 2001, 0);
    """)
    c.commit()
    c.close()
    reactions = []
    messages, _ = ChatDB(path).new_messages(0, (OWNER,), reactions)
    assert [(m.text, m.has_attachments, m.attachments) for m in messages] == [("你好", False, []), ("", True, [])] and reactions == []


def test_recent_messages_are_described_by_shape_only_for_diagnosing_a_missing_reply(rig):
    """2026-11-17：用户发了一张图，四分钟没有回复。查下来是消息还没送到这台 Mac——为了能分清"没送到"和"被跳过"，
    服务可以报出最近几条消息的形态。这里面不能有正文，也不能有对方的地址。"""
    png(rig.root / "IMG_1.png")
    attach(rig.chatdb, receive(rig.chatdb, "看得见是啥不", attachments=1), rig.root / "IMG_1.png", "image/png")
    receive(rig.chatdb, "这台 Mac 自己发的", from_me=1)
    receive(rig.chatdb, "陌生人说的话", handle=2, chat=3)
    receive(rig.chatdb, "赞了一条消息", tapback=2001)
    attach(rig.chatdb, receive(rig.chatdb, None, attachments=1), rig.root / "还没下载.heic", "image/heic")
    rows = ChatDB(rig.chatdb).recent((OWNER,), limit=5)[::-1]
    assert [r["verdict"] for r in rows] == ["会处理", "跳过：是这台 Mac 自己发出的", "跳过：发信地址不在「主人」名单里",
                                           "跳过：是对某条消息的回应（类型 2001）", "会处理"]
    assert (rows[0]["text_chars"], rows[0]["from_owner"], rows[0]["address_kind"]) == (6, True, "电话")
    assert [(a["kind"], a["mime"], a["ext"], a["on_disk"]) for a in rows[0]["attachments"]] == [("image", "image/png", ".png", True)]
    assert rows[4]["attachments"][0]["on_disk"] is False                    # 记录到了，文件还没下载下来
    dumped = json.dumps(rows, ensure_ascii=False)
    assert "看得见" not in dumped and "陌生人" not in dumped and OWNER not in dumped and "stranger@" not in dumped
