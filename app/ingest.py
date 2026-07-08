"""Faz 1 — İçerik alma (yt-dlp ile indirme + metadata).

Bir YouTube URL'sinden:
  - video.mp4   (görüntü analizi için, <=1080p)
  - audio.wav   (16kHz mono — Whisper / VAD / librosa için ideal, ffmpeg ile)
  - meta.json   (başlık, kanal, açıklama, süre, bölüm işaretleri)
üretir ve metadatayı SQLite'a yazar. Çıktılar data/<video_id>/ altında.
"""
from __future__ import annotations

import json
import subprocess

import yt_dlp
from rich.console import Console

from . import db
from .config import video_dir

console = Console()

# 1080p ile sınırla: kalite/disk dengesi. Birleştirilmiş mp4 hedefliyoruz.
_FORMAT = "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"


def _extract_audio(video_path, audio_path) -> None:
    """video.mp4'ten 16kHz mono WAV çıkarır (Whisper/VAD/librosa için standart)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(audio_path),
        ],
        check=True,
    )


def ingest(url: str, progress_hook=None) -> str:
    """URL'yi indirir, metadatayı DB'ye yazar, video_id döndürür.

    İndirilmiş video varsa yeniden indirmez (önbellek).
    progress_hook verilirse yt-dlp indirme ilerlemesi ona iletilir (TUI ilerleme çubuğu).
    """
    # 1) Metadatayı çek (indirmeden)
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = info["id"]
    vdir = video_dir(video_id)
    video_path = vdir / "video.mp4"
    audio_path = vdir / "audio.wav"
    meta_path = vdir / "meta.json"

    title = info.get("title")
    console.print(f"[bold]{title}[/bold]  [dim]({video_id})[/dim]")

    # Metadatayı baştan derle ve DB'ye yaz (stages tablosu videos'a FK ile bağlı,
    # bu yüzden video satırı set_stage'den önce var olmalı).
    meta = {
        "video_id": video_id,
        "url": info.get("webpage_url") or url,
        "title": title,
        "channel": info.get("channel") or info.get("uploader"),
        "description": info.get("description"),
        "duration_sec": info.get("duration"),
        "chapters": info.get("chapters"),  # varsa resmi bölüm işaretleri
        "upload_date": info.get("upload_date"),
        "view_count": info.get("view_count"),
        "tags": info.get("tags"),
    }
    db.upsert_video(meta)
    db.set_stage(video_id, "ingest", "running")

    try:
        # 2) Videoyu indir (varsa atla)
        if video_path.exists():
            console.print("  [dim]video.mp4 zaten var, indirme atlandı[/dim]")
        else:
            console.print("  video indiriliyor…")
            opts = {
                "format": _FORMAT,
                "merge_output_format": "mp4",
                "outtmpl": str(vdir / "video.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
            }
            if progress_hook is not None:
                opts["progress_hooks"] = [progress_hook]
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            if not video_path.exists():
                # bazı durumlarda farklı uzantı oluşabilir; ilk video* dosyasını al
                cand = next((p for p in vdir.glob("video.*") if p.suffix != ".json"), None)
                if cand and cand != video_path:
                    cand.rename(video_path)

        # 3) Ses çıkar (varsa atla)
        if audio_path.exists():
            console.print("  [dim]audio.wav zaten var, çıkarma atlandı[/dim]")
        else:
            console.print("  ses çıkarılıyor (16kHz mono)…")
            _extract_audio(video_path, audio_path)

        # 4) Metadatayı diske de yaz (boru hattının sonraki adımları için)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        db.set_stage(video_id, "ingest", "done")
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "ingest", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    dur = meta["duration_sec"]
    console.print(
        f"  [green]✓[/green] alındı  •  süre: {dur:.0f}s" if dur else "  [green]✓[/green] alındı"
    )
    console.print(f"  [dim]{vdir}[/dim]")
    return video_id
