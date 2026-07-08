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
-- fmt: short | episode | podcast | supercut
-- (supercut: farklı zamanlardan bağlanan çok-parçalı montaj; parçalar payload.spans'ta)
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

-- Kullanıcı dostu proje adı ↔ video_id eşlemesi (TUI'de takip için).
-- video_id başta boş olabilir (ingest bitince doldurulur).
CREATE TABLE IF NOT EXISTS projects (
    name         TEXT PRIMARY KEY,
    video_id     TEXT,
    url          TEXT,
    dir          TEXT,                     -- proje çıktı klasörü (mutlak yol)
    status       TEXT DEFAULT 'active',    -- active | done (bitti/arşivlendi)
    archive_path TEXT,                     -- <video_id>.zip yolu
    closed_at    TEXT,
    created_at   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE SET NULL
);
"""

# Eski DB'lerde projects tablosu eksik kolonlarla oluşmuş olabilir → idempotent ekle.
_PROJECT_COLS = {
    "dir": "TEXT", "status": "TEXT DEFAULT 'active'",
    "archive_path": "TEXT", "closed_at": "TEXT",
}


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
    """Şemayı oluşturur (idempotent) + eksik proje kolonlarını ekler."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        existing = {r[1] for r in conn.execute("PRAGMA table_info(projects)")}
        for col, decl in _PROJECT_COLS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE projects ADD COLUMN {col} {decl}")


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


# --- projeler (TUI) --------------------------------------------------------

def create_project(name: str, url: str | None = None, video_id: str | None = None) -> None:
    """Proje adını kaydeder (varsa url/video_id günceller). video_id sonra doldurulabilir."""
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO projects (name, url, video_id) VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                url=COALESCE(excluded.url, projects.url),
                video_id=COALESCE(excluded.video_id, projects.video_id)
            """,
            (name, url, video_id),
        )


def set_project_video(name: str, video_id: str) -> None:
    """Bir projeye üretilen video_id'yi bağlar (ingest bitince)."""
    with connect() as conn:
        conn.execute("UPDATE projects SET video_id=? WHERE name=?", (video_id, name))


def get_project(name: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()


def project_for_video(video_id: str) -> str | None:
    """Bir video_id'ye bağlı proje adı (varsa)."""
    with connect() as conn:
        row = conn.execute("SELECT name FROM projects WHERE video_id=?", (video_id,)).fetchone()
        return row["name"] if row else None


def list_projects() -> list[sqlite3.Row]:
    """Projeleri, bağlı video başlığı ve süresiyle birlikte (yeni → eski)."""
    with connect() as conn:
        return conn.execute(
            """
            SELECT p.name, p.video_id, p.url, p.created_at,
                   p.status, p.dir, p.archive_path,
                   v.title, v.duration_sec
            FROM projects p
            LEFT JOIN videos v ON v.video_id = p.video_id
            ORDER BY p.created_at DESC
            """
        ).fetchall()


def delete_project_full(name: str) -> str | None:
    """Projeyi ve bağlı videoyu (öneri/adımlar dâhil, FK cascade) siler. video_id döndürür."""
    with connect() as conn:
        row = conn.execute("SELECT video_id FROM projects WHERE name=?", (name,)).fetchone()
        vid = row["video_id"] if row else None
        if vid:
            conn.execute("DELETE FROM videos WHERE video_id=?", (vid,))   # recs/stages cascade
        conn.execute("DELETE FROM projects WHERE name=?", (name,))
    return vid


def mark_project_done(video_id: str, dir_path: str, archive_path: str | None) -> None:
    """Projeyi 'done' işaretler; çıktı klasörü ve arşiv yolunu yazar."""
    with connect() as conn:
        conn.execute(
            """
            UPDATE projects SET status='done', dir=?, archive_path=?,
                   closed_at=datetime('now')
            WHERE video_id=?
            """,
            (dir_path, archive_path, video_id),
        )


def replace_recommendations(video_id: str, recs: list[dict[str, Any]],
                            only_fmt: str | None = None) -> None:
    """Bir video için önerileri tazeler (eskileri silip yenilerini yazar).

    only_fmt verilirse yalnızca o formatın önerileri silinir (ör. supercut'ı,
    analyze'ın ürettiği short/episode/podcast'i silmeden yeniler).
    """
    with connect() as conn:
        if only_fmt:
            conn.execute(
                "DELETE FROM recommendations WHERE video_id = ? AND fmt = ?",
                (video_id, only_fmt),
            )
        else:
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
