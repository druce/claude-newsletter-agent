"""Tests for db.py SQLite models."""
import os
import sqlite3
from datetime import datetime
import pytest

TEST_DB = "test_newsletter.db"


@pytest.fixture(autouse=True)
def clean_db():
    """Remove test DB before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestSQLiteModel:
    def test_create_table_and_insert(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://example.com", final_url="http://example.com/final", title="Test", source="rss")
        u.insert(TEST_DB)
        assert u.id is not None

    def test_get_by_id(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="html")
        u.insert(TEST_DB)
        fetched = Url.get(TEST_DB, u.id)
        assert fetched is not None
        assert fetched.initial_url == "http://a.com"

    def test_get_all(self):
        from db import Url
        Url.create_table(TEST_DB)
        Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="s").insert(TEST_DB)
        Url(initial_url="http://b.com", final_url="http://b.com", title="B", source="s").insert(TEST_DB)
        all_urls = Url.get_all(TEST_DB)
        assert len(all_urls) == 2

    def test_update(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="s")
        u.insert(TEST_DB)
        u.title = "Updated"
        u.update(TEST_DB)
        fetched = Url.get(TEST_DB, u.id)
        assert fetched.title == "Updated"

    def test_delete(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="s")
        u.insert(TEST_DB)
        rid = u.id
        u.delete(TEST_DB)
        assert Url.get(TEST_DB, rid) is None

    def test_upsert_inserts_then_updates(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="s")
        u.upsert(TEST_DB)
        assert u.id is not None
        first_id = u.id
        u.title = "Updated"
        u.upsert(TEST_DB)
        assert u.id == first_id
        fetched = Url.get(TEST_DB, first_id)
        assert fetched.title == "Updated"

    def test_unique_constraint_on_initial_url(self):
        from db import Url
        Url.create_table(TEST_DB)
        Url(initial_url="http://same.com", final_url="http://same.com", title="A", source="s").insert(TEST_DB)
        with pytest.raises(sqlite3.IntegrityError):
            Url(initial_url="http://same.com", final_url="http://same.com", title="B", source="s").insert(TEST_DB)


class TestUrlMetadata:
    def test_url_with_published_and_summary_roundtrips(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(
            initial_url="http://example.com/article",
            final_url="http://example.com/article",
            title="Test",
            source="rss",
            published="Thu, 13 Feb 2026 12:00:00 GMT",
            summary="A clean summary",
        )
        u.insert(TEST_DB)
        fetched = Url.get(TEST_DB, u.id)
        assert fetched.published == "Thu, 13 Feb 2026 12:00:00 GMT"
        assert fetched.summary == "A clean summary"

    def test_url_without_metadata_defaults_to_none(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://example.com/no-meta", final_url="http://example.com/no-meta", title="T", source="s")
        u.insert(TEST_DB)
        fetched = Url.get(TEST_DB, u.id)
        assert fetched.published is None
        assert fetched.summary is None


class TestMigrateTable:
    def test_adds_missing_columns(self):
        """Simulate an old schema missing published/summary, then migrate."""
        from db import Url, _connect
        # Create table with old schema (no published/summary)
        with _connect(TEST_DB) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    initial_url TEXT UNIQUE NOT NULL,
                    final_url TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    isAI INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT
                )
            """)
        # Insert a row with old schema
        with _connect(TEST_DB) as conn:
            conn.execute("INSERT INTO urls (initial_url, final_url, title, source) VALUES (?, ?, ?, ?)",
                         ("http://old.com", "http://old.com", "Old", "s"))

        # Migrate should add published and summary columns
        Url.migrate_table(TEST_DB)

        # Now we can insert with new fields
        u = Url(initial_url="http://new.com", final_url="http://new.com", title="New", source="s",
                published="2026-02-13", summary="A summary")
        u.insert(TEST_DB)
        fetched = Url.get(TEST_DB, u.id)
        assert fetched.published == "2026-02-13"
        assert fetched.summary == "A summary"

        # Old row still readable, with None defaults
        old = Url.get(TEST_DB, 1)
        assert old.published is None
        assert old.summary is None

    def test_idempotent(self):
        from db import Url
        Url.create_table(TEST_DB)
        # Calling migrate on a table that already has all columns should be a no-op
        Url.migrate_table(TEST_DB)
        Url.migrate_table(TEST_DB)
        # Should still work fine
        u = Url(initial_url="http://x.com", final_url="http://x.com", title="X", source="s")
        u.insert(TEST_DB)
        assert Url.get(TEST_DB, u.id) is not None


class TestArticle:
    def test_create_and_retrieve(self):
        from db import Article
        Article.create_table(TEST_DB)
        a = Article(
            final_url="http://example.com/article",
            url="http://example.com/article",
            source="TechCrunch",
            title="AI News",
            domain="example.com",
        )
        a.insert(TEST_DB)
        fetched = Article.get(TEST_DB, a.id)
        assert fetched.title == "AI News"
        assert fetched.domain == "example.com"

    def test_unique_on_final_url(self):
        from db import Article
        Article.create_table(TEST_DB)
        Article(final_url="http://same.com", url="http://same.com", source="s", title="A", domain="same.com").insert(TEST_DB)
        with pytest.raises(sqlite3.IntegrityError):
            Article(final_url="http://same.com", url="http://same.com", source="s", title="B", domain="same.com").insert(TEST_DB)

    def test_topics_conversion(self):
        from db import Article
        assert Article.topics_list_to_string(["AI", "ML", "NLP"]) == "AI,ML,NLP"
        assert Article.topics_string_to_list("AI,ML,NLP") == ["AI", "ML", "NLP"]
        assert Article.topics_string_to_list("") == []
        assert Article.topics_string_to_list(None) == []

    def test_article_defaults(self):
        from db import Article
        a = Article(final_url="http://x.com", url="http://x.com", source="s", title="T", domain="x.com")
        assert a.isAI is False
        assert a.rating == 0.0
        assert a.content_length == 0


class TestSite:
    def test_create_and_retrieve(self):
        from db import Site
        Site.create_table(TEST_DB)
        s = Site(domain_name="example.com", site_name="Example", reputation=0.8)
        s.insert(TEST_DB)
        fetched = Site.get(TEST_DB, s.id)
        assert fetched.reputation == 0.8

    def test_unique_on_domain(self):
        from db import Site
        Site.create_table(TEST_DB)
        Site(domain_name="x.com", site_name="X", reputation=0.5).insert(TEST_DB)
        with pytest.raises(sqlite3.IntegrityError):
            Site(domain_name="x.com", site_name="X2", reputation=0.6).insert(TEST_DB)


class TestNewsletter:
    def test_create_and_retrieve(self):
        from db import Newsletter
        Newsletter.create_table(TEST_DB)
        n = Newsletter(session_id="s1", date=datetime.now(), final_newsletter="# Hello")
        n.insert(TEST_DB)
        fetched = Newsletter.get(TEST_DB, n.id)
        assert fetched.final_newsletter == "# Hello"


class TestAgentState:
    def test_create_and_retrieve(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        a = AgentState(session_id="s1", step_name="gather_urls", state_data='{"key": "val"}')
        a.insert(TEST_DB)
        fetched = AgentState.get(TEST_DB, a.id)
        assert fetched.state_data == '{"key": "val"}'

    def test_get_by_session_and_step(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="gather_urls", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s1", step_name="filter_urls", state_data='{"n":1}').insert(TEST_DB)
        fetched = AgentState.get_by_session_and_step(TEST_DB, "s1", "filter_urls")
        assert fetched is not None
        assert fetched.state_data == '{"n":1}'

    def test_get_latest_by_session(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="step_a", state_data="a").insert(TEST_DB)
        AgentState(session_id="s1", step_name="step_b", state_data="b").insert(TEST_DB)
        latest = AgentState.get_latest_by_session(TEST_DB, "s1")
        assert latest is not None
        assert latest.step_name == "step_b"

    def test_get_all_by_session(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="a", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s1", step_name="b", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s2", step_name="a", state_data="{}").insert(TEST_DB)
        results = AgentState.get_all_by_session(TEST_DB, "s1")
        assert len(results) == 2

    def test_list_sessions(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="a", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s2", step_name="a", state_data="{}").insert(TEST_DB)
        sessions = AgentState.list_sessions(TEST_DB)
        assert set(sessions) == {"s1", "s2"}

    def test_delete_session(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="a", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s1", step_name="b", state_data="{}").insert(TEST_DB)
        count = AgentState.delete_session(TEST_DB, "s1")
        assert count == 2
        assert AgentState.get_all_by_session(TEST_DB, "s1") == []

    def test_unique_session_step_constraint(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="same", state_data="{}").insert(TEST_DB)
        with pytest.raises(sqlite3.IntegrityError):
            AgentState(session_id="s1", step_name="same", state_data="{}").insert(TEST_DB)
