"""Faz 2 — Transkript omurgası (mlx-whisper, kelime düzeyinde zaman damgası).

audio.wav -> transcript.json
Çıktı yapısı (boru hattının omurgası — tüm sinyaller buna hizalanır):
{
  "language": "tr",
  "duration_sec": 1234.5,
  "text": "...",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.2, "text": "...",
     "words": [{"word": "...", "start": 0.0, "end": 0.4}]}
  ]
}
"""
from __future__ import annotations

import json

from rich.console import Console

from . import db
from .config import WHISPER_MODEL, video_dir

console = Console()


def transcribe(video_id: str) -> None:
    """audio.wav'dan kelime düzeyinde zaman damgalı transkript üretir."""
    import mlx_whisper  # ağır import — komut çalışınca yüklensin

    vdir = video_dir(video_id)
    audio_path = vdir / "audio.wav"
    out_path = vdir / "transcript.json"

    if not audio_path.exists():
        raise FileNotFoundError(
            f"audio.wav yok: {audio_path}. Önce 'l2s ingest' çalıştır."
        )

    console.print(f"transkript üretiliyor  [dim]({WHISPER_MODEL})[/dim]")
    console.print("  [dim]ilk çalıştırmada model indirilir; sonraki çalıştırmalar hızlıdır[/dim]")
    db.set_stage(video_id, "transcribe", "running")

    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=WHISPER_MODEL,
            word_timestamps=True,
        )

        segments = []
        for seg in result.get("segments", []):
            words = [
                {"word": w["word"], "start": w["start"], "end": w["end"]}
                for w in seg.get("words", [])
            ]
            segments.append(
                {
                    "id": seg.get("id"),
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "words": words,
                }
            )

        transcript = {
            "language": result.get("language"),
            "duration_sec": segments[-1]["end"] if segments else 0.0,
            "text": result.get("text", "").strip(),
            "segments": segments,
        }
        out_path.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        db.set_stage(
            video_id, "transcribe", "done",
            f"{len(segments)} segment, dil={transcript['language']}",
        )
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "transcribe", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    console.print(
        f"  [green]✓[/green] {len(segments)} segment  •  dil: {transcript['language']}"
    )
    console.print(f"  [dim]{out_path}[/dim]")
