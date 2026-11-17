"""SQLite 表结构。Alf 的全部记忆都在这一个文件里：好备份，能直接用 `sqlite3` 翻，搬家也只搬一个文件。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 原始输入（现在是音频，以后是图片）。事实和提醒都指回这里，
-- 记错了的时候能一路追到当时那段录音。
CREATE TABLE IF NOT EXISTS attachments (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('audio', 'image')),
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    transcript  TEXT,
    extracted   TEXT,              -- JSON：识别分段、检测到的语言
    confidence  REAL,
    created_at  INTEGER NOT NULL
);

-- 时间线：对话记录 + Alf 主动发出的消息（提醒、新邮件、每日播报）。
-- 网页和终端看到的都是这一张表，所以谁发的消息两边都能看到。
CREATE TABLE IF NOT EXISTS episodes (
    id            INTEGER PRIMARY KEY,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('user', 'alf')),
    kind          TEXT NOT NULL DEFAULT 'chat',   -- chat | reminder | email | briefing | digest
    text          TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'text' CHECK (source IN ('text', 'voice')),
    attachment_id TEXT REFERENCES attachments(id),
    created_at    INTEGER NOT NULL,
    -- 用户消息：从哪个渠道来的；Alf 的消息：要送到哪个渠道。NULL = 只留在本机（网页、终端能看到）。
    channel       TEXT,
    delivered_at  INTEGER,            -- 送达渠道的时间；0 = 过期放弃，没有发
    send_attempts INTEGER NOT NULL DEFAULT 0
);

-- 事实记忆，双时间轴：valid_from/valid_to 是事实成立的时间段，
-- recorded_at 是 Alf 得知的时间。事实从不删除，只会被"封口"。
CREATE TABLE IF NOT EXISTS facts (
    id                 INTEGER PRIMARY KEY,
    category           TEXT NOT NULL,
    text               TEXT NOT NULL,
    valid_from         INTEGER NOT NULL,
    valid_to           INTEGER,
    superseded_by      INTEGER REFERENCES facts(id),
    recorded_at        INTEGER NOT NULL,
    source             TEXT NOT NULL DEFAULT 'text',
    source_attachment  TEXT REFERENCES attachments(id),
    source_episode     INTEGER REFERENCES episodes(id),
    confidence         REAL,
    needs_confirmation INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS facts_current ON facts(category) WHERE valid_to IS NULL;

CREATE TABLE IF NOT EXISTS reminders (
    id                INTEGER PRIMARY KEY,
    title             TEXT NOT NULL,
    due_at            INTEGER NOT NULL,   -- 事件时间，UTC 秒
    lead_minutes      INTEGER NOT NULL DEFAULT 0,
    notify_at         INTEGER NOT NULL,   -- due_at 减去提前量
    tz                TEXT NOT NULL,      -- 重复提醒按这个时区的墙上时间推进（夏令时安全）
    recurrence        TEXT,               -- DAILY | WEEKLY:0,2,4（周一=0）
    status            TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'cancelled')),
    notified_at       INTEGER,
    source            TEXT NOT NULL DEFAULT 'text',
    source_attachment TEXT REFERENCES attachments(id),
    source_episode    INTEGER REFERENCES episodes(id),
    created_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS reminders_due ON reminders(status, notify_at);

-- 语音识别该"预期"听到的词（人名、课程代码……）。
CREATE TABLE IF NOT EXISTS vocabulary (
    term       TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL
);

-- 每次 Jev 调用一行：成本和延迟统计的原始数据。
CREATE TABLE IF NOT EXISTS model_calls (
    id            INTEGER PRIMARY KEY,
    ts            INTEGER NOT NULL,
    purpose       TEXT NOT NULL,
    model         TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    latency_ms    INTEGER NOT NULL,
    ok            INTEGER NOT NULL,
    error         TEXT,
    cached_tokens INTEGER             -- 输入里命中缓存的部分（DeepSeek 会报）
);

-- 心跳任务的运行状态。放在库里而不是内存里：重启后不会把"每天一次"的任务再跑一遍，
-- 连接器的游标（比如邮箱读到第几封）也不会丢。
CREATE TABLE IF NOT EXISTS job_state (
    name        TEXT PRIMARY KEY,
    last_run_at INTEGER,
    ok          INTEGER,
    error       TEXT,
    result      TEXT,               -- 最近一次有内容的结果，给状态面板看
    runs        INTEGER NOT NULL DEFAULT 0,
    cursor      TEXT
);

-- 连接器拉进来的外部消息（邮件……）。UNIQUE 保证同一封不会处理两次。
CREATE TABLE IF NOT EXISTS inbox_items (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,      -- 连接器名，如 email
    external_id TEXT NOT NULL,
    sender      TEXT,
    title       TEXT,
    snippet     TEXT,
    received_at INTEGER,
    attention   REAL,               -- Jev 给的"需要你处理"的概率；NULL = 没做分诊
    category    TEXT,
    notified    INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL,
    UNIQUE (source, external_id)
);

-- 在界面上改的设置（说话风格、主动消息的开关和时间……）。没有的项用默认值，见 prefs.py。
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,       -- JSON
    updated_at INTEGER NOT NULL
);

-- 每一次"要不要主动说话"的判断：发了、忍住了、还是没话说，以及理由。用来回头看它判断得对不对。
CREATE TABLE IF NOT EXISTS proactive_log (
    id      INTEGER PRIMARY KEY,
    ts      INTEGER NOT NULL,
    outcome TEXT NOT NULL,          -- sent 发了 | held 想说但没过门槛 | silent 模型认为没话可说
    kind    TEXT,
    message TEXT,
    why     TEXT,
    score   REAL                    -- Jev 给的"现在发合不合适"的概率；NULL = 没打分
);

-- 「日历」里未来几天的日程，每几分钟同步一次。用来发现新增、改期、取消，以及在开始前提醒。
CREATE TABLE IF NOT EXISTS calendar_events (
    key        TEXT PRIMARY KEY,    -- 日程标识 + 开始时间：重复日程的每一次各占一行
    event_id   TEXT NOT NULL,
    title      TEXT NOT NULL,
    start_at   INTEGER NOT NULL,
    end_at     INTEGER NOT NULL,
    all_day    INTEGER NOT NULL DEFAULT 0,
    location   TEXT,
    calendar   TEXT,
    by_alf     INTEGER NOT NULL DEFAULT 0,
    alerted_at INTEGER              -- 开始前的那次提醒是什么时候发的
);

-- Alf 写进「日历」和「提醒事项」的条目。撤销只能从这里选，所以碰不到用户自己建的东西。
CREATE TABLE IF NOT EXISTS apple_items (
    id             INTEGER PRIMARY KEY,
    kind           TEXT NOT NULL CHECK (kind IN ('event', 'reminder')),
    external_id    TEXT NOT NULL,
    title          TEXT NOT NULL,
    start_at       INTEGER,         -- 日程的开始，或提醒事项的到期时间；没有时间的提醒事项是 NULL
    end_at         INTEGER,
    all_day        INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'removed')),
    source_episode INTEGER,
    created_at     INTEGER NOT NULL
);

-- 回信草稿。只有 status 从 pending 变成 sent 的那一步会真的发信，而那一步要过"用户确认了这一份"的检查。
CREATE TABLE IF NOT EXISTS email_drafts (
    id            INTEGER PRIMARY KEY,
    inbox_id      INTEGER NOT NULL REFERENCES inbox_items(id),
    account       TEXT NOT NULL,    -- 从哪个邮箱发：收到那封信的邮箱
    to_addr       TEXT NOT NULL,
    subject       TEXT NOT NULL,
    body          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'discarded')),
    shown_episode INTEGER,          -- 草稿全文是在哪一条回复里给用户看的
    created_at    INTEGER NOT NULL,
    sent_at       INTEGER
);

-- 提醒发出之后的跟进：过一阵没回应就追问一次，第二天早报再提一次，之后不再追。
CREATE TABLE IF NOT EXISTS followups (
    episode_id  INTEGER PRIMARY KEY,   -- 那条提醒消息
    reminder_id INTEGER,
    title       TEXT NOT NULL,
    fired_at    INTEGER NOT NULL,
    asked_at    INTEGER,               -- 追问发出的时间；0 = 该追问时是安静时段，没有问
    briefed_at  INTEGER,               -- 在早报里提过的时间
    done        INTEGER NOT NULL DEFAULT 0   -- 1 = 这件事不再跟进（他回应了，或者该提的都提过了）
);

-- 用户对 Alf 主动发的消息的评价（iMessage 里点 👍👎，或回一句"没用"）。7 天试用按它算"没用的比例"。
CREATE TABLE IF NOT EXISTS feedback (
    episode_id INTEGER PRIMARY KEY,
    rating     INTEGER NOT NULL,       -- 1 有用 | -1 没用
    source     TEXT NOT NULL,          -- tapback | text
    created_at INTEGER NOT NULL
);

-- 好感度：每次聊到有意思的内容加一分。分数 = 50 + 行数（封顶 100），所以每一分都查得到是哪句话挣来的。
CREATE TABLE IF NOT EXISTS affinity (
    id         INTEGER PRIMARY KEY,
    episode_id INTEGER NOT NULL UNIQUE REFERENCES episodes(id),
    note       TEXT NOT NULL,          -- 他当时说的那句话（截短）
    score      REAL,                   -- 判断"有意思"的把握
    created_at INTEGER NOT NULL
);

-- 短期情绪：只有一行。warm 是刚聊得开心留下的余温，tension 是他不满意留下的疙瘩；两者都随时间衰减（见 mood.py）。
CREATE TABLE IF NOT EXISTS mood (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    warm       REAL NOT NULL DEFAULT 0,
    tension    REAL NOT NULL DEFAULT 0,
    reason     TEXT,                   -- 最近一次让他不满意的那句话（截短），给记忆页显示
    updated_at INTEGER NOT NULL
);

-- 针对某一次的纠正，存成例子（2026-11-17）："这封不用马上告诉我""这条没用""这句太长了"。
-- 下次判断同类情况时拿给模型和 Jev 看；同类的攒够了，晚上汇总时提议成一条规矩，由用户确认。
CREATE TABLE IF NOT EXISTS examples (
    id          INTEGER PRIMARY KEY,
    area        TEXT NOT NULL,          -- mail 邮件该不该立刻说 | proactive 主动发的消息 | chat 聊天里的某一句
    verdict     TEXT NOT NULL,          -- mail：now | later | skip；其余：bad
    about       TEXT NOT NULL,          -- 说的是哪一条：邮件写"发件人｜主题"，消息写原文（截短）
    said        TEXT NOT NULL,          -- 他当时怎么说的（原话，截短）；点 👎 的写"👎"
    inbox_id    INTEGER,                -- 那封邮件
    target      INTEGER,                -- 那条 Alf 发的消息（episodes.id）
    episode_id  INTEGER,                -- 他说这句话的那条消息
    created_at  INTEGER NOT NULL,
    removed_at  INTEGER,                -- 他在记忆页上撤销了
    proposed_at INTEGER,                -- 已经拿去提议过规矩（不重复提）
    rule_id     INTEGER                 -- 他同意之后变成了哪条规矩（facts.id）
);

-- 在聊天里改的设置：改了什么、原来是什么、他当时怎么说的。记忆页上看得到，可以改回去。
CREATE TABLE IF NOT EXISTS setting_changes (
    id         INTEGER PRIMARY KEY,
    key        TEXT NOT NULL,
    old        TEXT NOT NULL,           -- JSON
    new        TEXT NOT NULL,           -- JSON
    quote      TEXT,
    episode_id INTEGER,
    created_at INTEGER NOT NULL,
    undone_at  INTEGER
);

-- 等他点头的规矩提议：从几个同类的例子归纳出来的。他说好才变成规矩。
CREATE TABLE IF NOT EXISTS rule_proposals (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    scope      TEXT NOT NULL,
    examples   TEXT NOT NULL,           -- JSON：出自哪几个例子的编号
    status     TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'accepted', 'declined', 'expired')),
    rule_id    INTEGER,
    created_at INTEGER NOT NULL
);

-- 人设：Alf 说过的关于自己的话（"我偏爱老爵士"）。人设卡片本身在 settings 里（persona.*）；这里存的是卡片上没写、
-- 聊天时现想出来的那些，记下来是为了以后说法一致。
CREATE TABLE IF NOT EXISTS persona_notes (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    episode_id INTEGER,                 -- 是在回哪句话的时候说的
    created_at INTEGER NOT NULL,
    removed_at INTEGER                  -- 他在页面上删掉了
);

-- 印象：Alf 对他这个人的看法（"做事爱拖到最后一刻，但总能赶上"）。是推断，不是他说的事实，所以单独放：
-- 每条都挂着它依据的那几句原话；只凭一天的话不算数，另一天又印证了才算"形成"。他说不对就收回。
CREATE TABLE IF NOT EXISTS impressions (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    evidence   TEXT NOT NULL DEFAULT '[]',   -- JSON：[{"episode": 编号, "quote": "他的原话", "at": 时间}]
    days       INTEGER NOT NULL DEFAULT 1,   -- 有几个不同的日子印证过它
    status     TEXT NOT NULL DEFAULT 'current' CHECK (status IN ('current', 'withdrawn')),
    note       TEXT,                         -- 收回时他是怎么说的
    created_at INTEGER NOT NULL,
    seen_at    INTEGER NOT NULL,             -- 最近一次被印证
    formed_at  INTEGER,                      -- 印证够了、算形成的时间；NULL = 还只是苗头
    told_at    INTEGER                       -- 在每晚汇总里告诉过他
);

-- 约定：他和 Alf 说好的事。who 是谁答应的：user 他答应要做的（Alf 惦记着问）| alf Alf 答应他的（得兑现）|
-- both 两边说好的习惯和暗号。只在对话里当场记，而且得是真的说好了（见 chat.py 的 PACT_FLOOR）。
CREATE TABLE IF NOT EXISTS pacts (
    id         INTEGER PRIMARY KEY,
    text       TEXT NOT NULL,
    who        TEXT NOT NULL CHECK (who IN ('user', 'alf', 'both')),
    standing   INTEGER NOT NULL DEFAULT 0,   -- 1 = 一直有效（习惯、暗号）；0 = 办完就了结的一次性承诺
    due_at     INTEGER,                      -- 说好的期限，程序从他的原话算的
    said       TEXT,                         -- 他当时的原话
    episode_id INTEGER,
    status     TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'kept', 'dropped', 'lapsed')),
    outcome    TEXT,                         -- 了结时他怎么说的
    created_at INTEGER NOT NULL,
    settled_at INTEGER,
    touched_at INTEGER,                      -- 上一次聊到这件事
    asks       INTEGER NOT NULL DEFAULT 0    -- Alf 提起而他没接话的次数；到两次就不再提
);

-- 全文索引里存的是预先切好的词（中文按二元组，见 fts.py）；rowid 等于源表的 id。
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(tokens);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(tokens);
"""


# 老数据库缺的列。CREATE TABLE IF NOT EXISTS 不会给已有的表加列，所以要单独补。
MIGRATIONS = [
    ("episodes", "channel", "TEXT"),
    ("episodes", "delivered_at", "INTEGER"),
    ("episodes", "send_attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("model_calls", "cached_tokens", "INTEGER"),
    ("inbox_items", "account", "TEXT"),        # 哪个邮箱收到的（回信从它发出）
    ("inbox_items", "body", "TEXT"),           # 正文开头较长的一段：写摘要、起草回信时用
    ("inbox_items", "reply_to", "TEXT"),       # 回信该寄到的地址
    ("inbox_items", "message_id", "TEXT"),
    ("inbox_items", "refs", "TEXT"),
    ("inbox_items", "summary", "TEXT"),
    ("inbox_items", "deferred", "TEXT"),       # 安静时段里攒着、还没发给用户的通知全文
    ("inbox_items", "reminder_id", "INTEGER"),  # 因为这封信里的截止日期自动建的提醒
    # 邮件分三档（2026-11-17 用户要求）：now 立刻告诉他 | later 留到每天的汇总里 | skip 广告通知，只计数 | dup 另一个邮箱已经收到过
    ("inbox_items", "tier", "TEXT"),
    ("inbox_items", "urgency", "REAL"),        # Jev 给的"得立刻告诉他"的把握
    ("inbox_items", "reported_at", "INTEGER"),  # 在哪一次每日汇总里说过
    # 记忆分层（2026-11-17）。topic 为空 = 常驻层（每轮都带）；有 topic = 主题层（提到时整组带上）。
    ("facts", "topic", "TEXT"),
    ("facts", "slot", "TEXT"),                 # 这条是主题里哪一项的值（比如动作名）；同一项报新值时直接替换，不用模型猜
    ("facts", "expires_at", "INTEGER"),        # 带日期的事：过了这个时刻就不再是"现状"
    ("facts", "aftermath", "TEXT"),            # 过期之后：open = 还没聊过后来怎么样 | done = 聊过了
    # 2026-11-17，参考 Anima Brain 加的四样：
    ("facts", "quote", "TEXT"),                # 这条记忆出自他的哪句原话。后台记的事必须有，程序核对过它确实在他的话里
    ("facts", "follow_up", "INTEGER NOT NULL DEFAULT 0"),   # 1 = 还没了结、以后值得问一句的事（"惦记的事"）
    ("facts", "importance", "INTEGER"),        # 1–10，对他有多要紧；排"惦记的事"时用
    ("facts", "touched_at", "INTEGER"),        # 上一次聊到这件事的时间（谁提的都算）
    ("facts", "asks", "INTEGER NOT NULL DEFAULT 0"),        # Alf 提起而他没接话的次数；到两次就不再提
    # 2026-11-17「听了会改」：
    ("facts", "scope", "TEXT"),                # 只对"他对 Alf 的要求"有意义：这条要求管哪里。空 = 哪里都管；chat | mail | proactive
    ("episodes", "reply_to", "INTEGER"),       # 他在 iMessage 里引用回复的是哪一条（episodes.id）
    ("episodes", "quoted", "TEXT"),            # 被引用的那条的原文（截短）；对不上编号时至少留着这个
    ("inbox_items", "episode_id", "INTEGER"),  # 这封信是在哪条消息里告诉他的：他引用那条消息说话时，知道说的是哪封信
    # 经历记忆：
    ("facts", "ours", "INTEGER NOT NULL DEFAULT 0"),   # 只对经历有意义：1 = 这件事发生在他和 Alf 之间（"我们之间的事"）
    ("facts", "feeling", "TEXT"),              # 只对经历有意义：他当时自己说的感受，照抄的一小段
]


def migrate(conn: sqlite3.Connection) -> None:
    for table, column, ddl in MIGRATIONS:
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    # 这个索引用到新列，所以放在补列之后建。
    conn.execute("CREATE INDEX IF NOT EXISTS episodes_outbox ON episodes(channel, id)"
                 " WHERE delivered_at IS NULL AND channel IS NOT NULL AND role = 'alf'")
    conn.commit()


def connect(path: Path | str, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """check_same_thread=False 只给 HTTP 服务用：它的请求跑在线程池里，调用方必须自己加锁。"""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    migrate(conn)
    return conn
