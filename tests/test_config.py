"""Tests for config.py constants and configuration."""


def test_canonical_topics_is_nonempty_list():
    from config import CANONICAL_TOPICS
    assert isinstance(CANONICAL_TOPICS, list)
    assert len(CANONICAL_TOPICS) > 100


def test_canonical_topics_contains_key_entries():
    from config import CANONICAL_TOPICS
    for expected in ["Agents", "Language Models", "OpenAI", "Anthropic", "Safety And Alignment"]:
        assert expected in CANONICAL_TOPICS, f"Missing topic: {expected}"


def test_download_paths_are_strings():
    from config import DOWNLOAD_ROOT, DOWNLOAD_DIR, PAGES_DIR, TEXT_DIR
    for path in [DOWNLOAD_ROOT, DOWNLOAD_DIR, PAGES_DIR, TEXT_DIR]:
        assert isinstance(path, str)
        assert len(path) > 0


def test_timeouts_are_positive():
    from config import REQUEST_TIMEOUT, SHORT_REQUEST_TIMEOUT, DOMAIN_RATE_LIMIT
    assert REQUEST_TIMEOUT > 0
    assert SHORT_REQUEST_TIMEOUT > 0
    assert DOMAIN_RATE_LIMIT > 0


def test_domain_skiplist_and_ignore_list():
    from config import DOMAIN_SKIPLIST, IGNORE_LIST
    assert isinstance(DOMAIN_SKIPLIST, list)
    assert isinstance(IGNORE_LIST, list)


def test_model_constants():
    from config import CLAUDE_SONNET, CLAUDE_HAIKU, DEFAULT_MODEL
    assert "sonnet" in CLAUDE_SONNET.lower() or "claude" in CLAUDE_SONNET.lower()
    assert "haiku" in CLAUDE_HAIKU.lower() or "claude" in CLAUDE_HAIKU.lower()
    assert DEFAULT_MODEL in (CLAUDE_SONNET, CLAUDE_HAIKU)


def test_default_concurrency():
    from config import DEFAULT_CONCURRENCY
    assert isinstance(DEFAULT_CONCURRENCY, int)
    assert DEFAULT_CONCURRENCY > 0


def test_newsagentdb_constant():
    from config import NEWSAGENTDB
    assert NEWSAGENTDB.endswith(".db")
