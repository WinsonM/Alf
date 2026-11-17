from alf import fts


def test_chinese_is_indexed_as_bigrams():
    assert fts.tokens("我老婆的生日") == ["我老", "老婆", "婆的", "的生", "生日"]


def test_latin_words_and_codes_stay_whole():
    assert fts.tokens("交COMP5318作业") == ["交", "comp5318", "作业"]


def test_bigram_query_finds_sentence(conn):
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(tokens)")
    fts.index(conn, "t", 1, "我老婆生日是3月14号")
    fts.index(conn, "t", 2, "我对花生过敏")
    hits = [r[0] for r in conn.execute("SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t)", (fts.query("老婆生日是哪天"),))]
    assert hits == [1]
