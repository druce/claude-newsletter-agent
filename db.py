"""SQLite persistence layer for the newsletter agent.

Uses dataclasses + raw sqlite3. A SQLiteModel base class eliminates
repetitive CRUD boilerplate across all models.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T", bound="SQLiteModel")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@dataclass
class SQLiteModel:
    """Base class providing generic CRUD for sqlite3-backed dataclasses.

    Subclasses must define:
        _table_name: ClassVar[str]
        _create_sql: ClassVar[str]  -- full CREATE TABLE statement
        _unique_columns: ClassVar[Tuple[str, ...]]  -- for upsert conflict detection
    """
    _table_name: ClassVar[str]
    _create_sql: ClassVar[str]
    _unique_columns: ClassVar[Tuple[str, ...]] = ()
    _indexes_sql: ClassVar[List[str]] = []

    id: Optional[int] = field(default=None, repr=False)

    @classmethod
    def create_table(cls, db_path: str) -> None:
        with _connect(db_path) as conn:
            conn.execute(cls._create_sql)
            for idx_sql in cls._indexes_sql:
                conn.execute(idx_sql)

    @classmethod
    def migrate_table(cls, db_path: str) -> None:
        """Add any missing columns to an existing table. Idempotent."""
        expected = {f.name for f in fields(cls)} - {"id"}
        with _connect(db_path) as conn:
            rows = conn.execute(f"PRAGMA table_info({cls._table_name})").fetchall()
            existing = {r["name"] for r in rows}
            for col in expected - existing:
                conn.execute(f"ALTER TABLE {cls._table_name} ADD COLUMN {col} TEXT")

    def _data_columns(self) -> List[str]:
        """Return field names excluding 'id'."""
        return [f.name for f in fields(self) if f.name != "id"]

    def _data_values(self) -> List[Any]:
        """Return field values excluding 'id', converting datetimes to ISO strings."""
        vals = []
        for col in self._data_columns():
            v = getattr(self, col)
            if isinstance(v, datetime):
                v = v.isoformat()
            vals.append(v)
        return vals

    def insert(self, db_path: str) -> None:
        cols = self._data_columns()
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {self._table_name} ({', '.join(cols)}) VALUES ({placeholders})"
        with _connect(db_path) as conn:
            cur = conn.execute(sql, self._data_values())
            self.id = cur.lastrowid

    def update(self, db_path: str) -> None:
        if self.id is None:
            raise ValueError("Cannot update a record without an id")
        cols = self._data_columns()
        set_clause = ", ".join(f"{c} = ?" for c in cols)
        sql = f"UPDATE {self._table_name} SET {set_clause} WHERE id = ?"
        with _connect(db_path) as conn:
            conn.execute(sql, self._data_values() + [self.id])

    def delete(self, db_path: str) -> None:
        if self.id is None:
            raise ValueError("Cannot delete a record without an id")
        with _connect(db_path) as conn:
            conn.execute(f"DELETE FROM {self._table_name} WHERE id = ?", (self.id,))

    def upsert(self, db_path: str) -> None:
        """Insert or update. If id is set, updates. Otherwise inserts."""
        if self.id is not None:
            self.update(db_path)
        else:
            self.insert(db_path)

    @classmethod
    def _row_to_instance(cls: Type[T], row: sqlite3.Row) -> T:
        col_names = [f.name for f in fields(cls)]
        kwargs = {}
        for col in col_names:
            if col in row.keys():
                kwargs[col] = row[col]
        return cls(**kwargs)

    @classmethod
    def get(cls: Type[T], db_path: str, record_id: int) -> Optional[T]:
        with _connect(db_path) as conn:
            row = conn.execute(
                f"SELECT * FROM {cls._table_name} WHERE id = ?", (record_id,)
            ).fetchone()
        if row is None:
            return None
        return cls._row_to_instance(row)

    @classmethod
    def get_all(cls: Type[T], db_path: str) -> List[T]:
        with _connect(db_path) as conn:
            rows = conn.execute(f"SELECT * FROM {cls._table_name}").fetchall()
        return [cls._row_to_instance(r) for r in rows]


@dataclass
class Url(SQLiteModel):
    """URL and headline tracking."""
    _table_name: ClassVar[str] = "urls"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            initial_url TEXT UNIQUE NOT NULL,
            final_url TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            isAI INTEGER NOT NULL DEFAULT 0,
            published TEXT,
            summary TEXT,
            created_at TEXT
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("initial_url",)

    initial_url: str = ""
    final_url: str = ""
    title: str = ""
    source: str = ""
    isAI: bool = False
    published: Optional[str] = None
    summary: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class Article(SQLiteModel):
    """Full article content and metadata."""
    _table_name: ClassVar[str] = "articles"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            final_url TEXT UNIQUE NOT NULL,
            url TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            published TEXT,
            date TEXT,
            rss_summary TEXT,
            description TEXT,
            summary TEXT,
            short_summary TEXT,
            isAI INTEGER NOT NULL DEFAULT 0,
            status TEXT,
            html_path TEXT,
            text_path TEXT,
            content_length INTEGER NOT NULL DEFAULT 0,
            rating REAL NOT NULL DEFAULT 0.0,
            cluster_label TEXT,
            topics TEXT,
            domain TEXT NOT NULL DEFAULT '',
            site_name TEXT NOT NULL DEFAULT '',
            reputation REAL,
            last_updated TEXT
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("final_url",)

    final_url: str = ""
    url: str = ""
    source: str = ""
    title: str = ""
    published: Optional[datetime] = None
    date: Optional[datetime] = None
    rss_summary: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    short_summary: Optional[str] = None
    isAI: bool = False
    status: Optional[str] = None
    html_path: Optional[str] = None
    text_path: Optional[str] = None
    content_length: int = 0
    rating: float = 0.0
    cluster_label: Optional[str] = None
    topics: Optional[str] = None
    domain: str = ""
    site_name: str = ""
    reputation: Optional[float] = None
    last_updated: Optional[datetime] = None

    @staticmethod
    def topics_list_to_string(topics: List[str]) -> str:
        return ",".join(topics)

    @staticmethod
    def topics_string_to_list(topics_str: Optional[str]) -> List[str]:
        if not topics_str:
            return []
        return [t.strip() for t in topics_str.split(",") if t.strip()]


@dataclass
class Site(SQLiteModel):
    """Source site reputation tracking."""
    _table_name: ClassVar[str] = "sites"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS sites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_name TEXT UNIQUE NOT NULL,
            site_name TEXT NOT NULL DEFAULT '',
            reputation REAL NOT NULL DEFAULT 0.0
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("domain_name",)

    domain_name: str = ""
    site_name: str = ""
    reputation: float = 0.0


@dataclass
class Newsletter(SQLiteModel):
    """Finalized newsletter storage."""
    _table_name: ClassVar[str] = "newsletters"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS newsletters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            date TEXT NOT NULL,
            final_newsletter TEXT NOT NULL DEFAULT ''
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("session_id",)

    session_id: str = ""
    date: Optional[datetime] = None
    final_newsletter: str = ""


@dataclass
class AgentState(SQLiteModel):
    """Workflow state persistence with session checkpointing."""
    _table_name: ClassVar[str] = "agent_state"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS agent_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            step_name TEXT NOT NULL,
            state_data TEXT NOT NULL DEFAULT '',
            updated_at TEXT,
            UNIQUE(session_id, step_name)
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("session_id", "step_name")
    _indexes_sql: ClassVar[List[str]] = [
        "CREATE INDEX IF NOT EXISTS idx_agent_state_session_id ON agent_state(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_agent_state_updated_at ON agent_state(updated_at)",
    ]

    session_id: str = ""
    step_name: str = ""
    state_data: str = ""
    updated_at: Optional[datetime] = None

    @classmethod
    def get_by_session_and_step(cls, db_path: str, session_id: str, step_name: str) -> Optional["AgentState"]:
        with _connect(db_path) as conn:
            row = conn.execute(
                f"SELECT * FROM {cls._table_name} WHERE session_id = ? AND step_name = ?",
                (session_id, step_name),
            ).fetchone()
        if row is None:
            return None
        return cls._row_to_instance(row)

    @classmethod
    def get_latest_by_session(cls, db_path: str, session_id: str) -> Optional["AgentState"]:
        with _connect(db_path) as conn:
            row = conn.execute(
                f"SELECT * FROM {cls._table_name} WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return cls._row_to_instance(row)

    @classmethod
    def get_all_by_session(cls, db_path: str, session_id: str) -> List["AgentState"]:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM {cls._table_name} WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [cls._row_to_instance(r) for r in rows]

    @classmethod
    def list_sessions(cls, db_path: str) -> List[str]:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT session_id FROM {cls._table_name} ORDER BY session_id"
            ).fetchall()
        return [r["session_id"] for r in rows]

    @classmethod
    def delete_session(cls, db_path: str, session_id: str) -> int:
        with _connect(db_path) as conn:
            cur = conn.execute(
                f"DELETE FROM {cls._table_name} WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount
