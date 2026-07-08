"""Faz 5 — Birleştirme: tüm sinyalleri transkript omurgası üzerinde tek temsile toplar.

transcript.json (+ audio_signals.json + visual_signals.json + meta.json) -> fused.json

Bu, Claude'un "gördüğü" zenginleştirilmiş temsildir. Her transkript segmenti;
ses enerjisi, içinde bulunduğu sahne ve hemen ardından bir duraklama olup olmadığı
(iyi kesim noktası ipucu) ile etiketlenir. Ses/görüntü dosyaları yoksa bu alanlar
boş kalır — boru hattı yine de çalışır (sadece-transkript modu).
"""
from __future__ import annotations

import json

from rich.console import Console

from . import db
from .config import video_dir
from .sentences import build_sentences

console = Console()

_PAUSE_GAP_SEC = 0.6  # segment sonundan bu kadar süre içinde başlayan duraklama "ardından duraklama" sayılır


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fuse(video_id: str) -> None:
    """Tüm modaliteleri zaman-hizalı tek JSON'da birleştirir -> fused.json."""
    vdir = video_dir(video_id)

    transcript = _load(vdir / "transcript.json")
    if transcript is None:
        raise FileNotFoundError("transcript.json yok. Önce 'l2s transcribe' çalıştır.")

    audio = _load(vdir / "audio_signals.json")
    visual = _load(vdir / "visual_signals.json")
    meta = _load(vdir / "meta.json") or {}

    console.print("sinyaller birleştiriliyor")
    db.set_stage(video_id, "fuse", "running")

    try:
        energy = audio.get("energy", []) if audio else []
        pauses = audio.get("pauses", []) if audio else []
        scenes = visual.get("scenes", []) if visual else []

        def energy_in(start: float, end: float):
            vals = [e["rms"] for e in energy if start <= e["t"] <= end]
            if not vals:
                return None, None
            return round(sum(vals) / len(vals), 4), round(max(vals), 4)

        def scene_of(t: float):
            for s in scenes:
                if s["end"] is None or (s["start"] <= t <= s["end"]):
                    return s["id"]
            return None

        def pause_after(end: float) -> bool:
            return any(end <= p["start"] <= end + _PAUSE_GAP_SEC for p in pauses)

        def pause_before(start: float) -> bool:
            # segment başlangıcından hemen önce biten bir duraklama (temiz giriş noktası)
            return any(start - _PAUSE_GAP_SEC <= p["end"] <= start + 0.15 for p in pauses)

        fused_segments = []
        for seg in transcript["segments"]:
            avg_e, peak_e = energy_in(seg["start"], seg["end"])
            mid = (seg["start"] + seg["end"]) / 2.0
            fused_segments.append(
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "avg_energy": avg_e,
                    "peak_energy": peak_e,
                    "scene_id": scene_of(mid),
                    "pause_before": pause_before(seg["start"]),
                    "pause_after": pause_after(seg["end"]),
                }
            )

        fused = {
            "video_id": video_id,
            "title": meta.get("title"),
            "channel": meta.get("channel"),
            "duration_sec": transcript.get("duration_sec") or meta.get("duration_sec"),
            "language": transcript.get("language"),
            "chapters": meta.get("chapters"),
            "has_audio_signals": audio is not None,
            "has_visual_signals": visual is not None,
            "scene_count": len(scenes),
            "segments": fused_segments,
            # cümle indeksi (kelime zamanlaması + noktalama) — sınır yaslamanın
            # segment değil CÜMLE sınırına yapılması için tek kaynak
            "sentences": build_sentences(transcript["segments"]),
            "pauses": pauses,
        }
        (vdir / "fused.json").write_text(
            json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        db.set_stage(
            video_id, "fuse", "done",
            f"{len(fused_segments)} segment, ses={audio is not None}, görüntü={visual is not None}",
        )
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "fuse", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    console.print(
        f"  [green]✓[/green] {len(fused_segments)} segment birleştirildi"
        f"  •  ses: {'✓' if audio else '—'}  görüntü: {'✓' if visual else '—'}"
    )
    console.print(f"  [dim]{vdir / 'fused.json'}[/dim]")
