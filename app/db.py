"""Yerel SQLite veritabanı: video kütüphanesi, işlem durumu ve öneriler.

Bulut veritabanı yerine tek dosyalık SQLite. `init_db()` şemayı kurar
(idempotent — tekrar çağırmak güvenli).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    title        TEXT,
    channel      TEXT,
    description  TEXT,
    duration_sec REAL,
    ingested_at  TEXT DEFAULT (datetime('now'))
);

-- Boru hattı adımlarının durumu (her video için adım adım izleme).
-- stage: ingest | transcribe | audio | visual | fuse | analyze | export
-- status: pending | running | done | error
CREATE TABLE IF NOT EXISTS stages (
    video_id   TEXT NOT NULL,
    stage      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    detail     TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (video_id, stage),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

-- Claude'un ürettiği öneriler + üretim paketleri.
-- fmt: short | episode | podcast
CREATE TABLE IF NOT EXISTS recommendations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id   TEXT NOT NULL,
    fmt        TEXT NOT NULL,
    start_sec  REAL,
    end_sec    REAL,
    score      REAL,
    title      TEXT,
    payload    TEXT,        -- tam üretim paketi (JSON: hook, açıklama, gerekçe, kesimler...)
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rec_video ON recommendations(video_id, fmt);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Foreign key'leri açık, satırları sözlük gibi erişilebilir bağlantı."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Şemayı oluşturur (idempotent)."""
    with connect() as conn:
        conn.executescript(SCHEMA)


def upsert_video(meta: dict[str, Any]) -> None:
    """Video metadatasını ekler/günceller."""
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO videos (video_id, url, title, channel, description, duration_sec)
            VALUES (:video_id, :url, :title, :channel, :description, :duration_sec)
            ON CONFLICT(video_id) DO UPDATE SET
                url=excluded.url, title=excluded.title, channel=excluded.channel,
                description=excluded.description, duration_sec=excluded.duration_sec
            """,
            {
                "video_id": meta["video_id"],
                "url": meta.get("url"),
                "title": meta.get("title"),
                "channel": meta.get("channel"),
                "description": meta.get("description"),
                "duration_sec": meta.get("duration_sec"),
            },
        )


def set_stage(video_id: str, stage: str, status: str, detail: str | None = None) -> None:
    """Bir adımın durumunu işaretler."""
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO stages (video_id, stage, status, detail, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(video_id, stage) DO UPDATE SET
                status=excluded.status, detail=excluded.detail, updated_at=datetime('now')
            """,
            (video_id, stage, status, detail),
        )


def get_video(video_id: str) -> sqlite3.Row | None:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
        return cur.fetchone()


def get_stages(video_id: str) -> list[sqlite3.Row]:
    with connect() as conn:
        cur = conn.execute(
            "SELECT * FROM stages WHERE video_id = ? ORDER BY updated_at", (video_id,)
        )
        return cur.fetchall()


def list_videos() -> list[sqlite3.Row]:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM videos ORDER BY ingested_at DESC")
        return cur.fetchall()


def get_recommendations(video_id: str, fmt: str | None = None) -> list[sqlite3.Row]:
    """Önerileri puana göre azalan sırada döndürür (opsiyonel format filtresi)."""
    with connect() as conn:
        if fmt:
            cur = conn.execute(
                "SELECT * FROM recommendations WHERE video_id=? AND fmt=? "
                "ORDER BY score DESC",
                (video_id, fmt),
            )
        else:
            cur = conn.execute(
                "SELECT * FROM recommendations WHERE video_id=? "
                "ORDER BY fmt, score DESC",
                (video_id,),
            )
        return cur.fetchall()


def replace_recommendations(video_id: str, recs: list[dict[str, Any]]) -> None:
    """Bir video için önerileri tazeler (eskileri silip yenilerini yazar)."""
    with connect() as conn:
        conn.execute("DELETE FROM recommendations WHERE video_id = ?", (video_id,))
        conn.executemany(
            """
            INSERT INTO recommendations (video_id, fmt, start_sec, end_sec, score, title, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    video_id,
                    r.get("fmt"),
                    r.get("start_sec"),
                    r.get("end_sec"),
                    r.get("score"),
                    r.get("title"),
                    json.dumps(r.get("payload", {}), ensure_ascii=False),
                )
                for r in recs
            ],
        )
