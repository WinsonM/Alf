import sqlite3

from alf import db, timeline


def test_old_database_gains_the_new_columns_and_keeps_its_rows(tmp_path):
    """9 月和 11 月上半月建的库没有 channel 等列；打开时要自动补上，原有的行不能丢。"""
    path = tmp_path / "alf.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE episodes (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'chat', text TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'text',
            attachment_id TEXT, created_at INTEGER NOT NULL);
        CREATE TABLE model_calls (id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, purpose TEXT NOT NULL, model TEXT,
            input_tokens INTEGER, output_tokens INTEGER, latency_ms INTEGER NOT NULL, ok INTEGER NOT NULL, error TEXT);
        INSERT INTO episodes(session_id, role, text, created_at) VALUES ('s', 'user', '老消息', 1);
    """)
    old.commit()
    old.close()

    conn = db.connect(path)
    db.connect(path)                                     # 重复打开不报错
    [row] = timeline.fetch(conn)
    assert row["text"] == "老消息"
    assert tuple(conn.execute("SELECT channel, delivered_at, send_attempts FROM episodes").fetchone()) == (None, None, 0)
    conn.execute("INSERT INTO model_calls(ts, purpose, latency_ms, ok, cached_tokens) VALUES (1, 'chat', 5, 1, 64)")
    timeline.post(conn, "reminder", "交作业", 2, "imessage")
    assert conn.execute("SELECT channel FROM episodes WHERE id = 2").fetchone()[0] == "imessage"
