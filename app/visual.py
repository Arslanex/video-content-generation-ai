"""Faz 5 — Görüntü sinyalleri (PySceneDetect sahne kesimleri + ffmpeg kare örnekleme).

video.mp4 -> visual_signals.json (+ frames/*.jpg)
v1 kapsamı:
  - scenes : sahne kesim aralıkları (görsel kesim noktalarını doğrular)
  - frames : sahne başına temsilî kare (ileride OCR / Claude vision için hazır)

Sonraki iyileştirme (opsiyonel): kareler üzerinde OCR (ekran metni) ve
Claude vision ile kare açıklaması.
"""
from __future__ import annotations

import json
import subprocess

from rich.console import Console

from . import db
from .config import video_dir

console = Console()


def _extract_frame(video_path, t_sec: float, out_path) -> None:
    """Belirtilen saniyeden tek kare çıkarır (ffmpeg)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{t_sec:.2f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "3",
            str(out_path),
        ],
        check=True,
    )


def analyze_visual(video_id: str) -> None:
    """Sahne kesimlerini bulur ve sahne başına temsilî kare örnekler."""
    from scenedetect import ContentDetector, detect  # ağır import

    vdir = video_dir(video_id)
    video_path = vdir / "video.mp4"
    frames_dir = vdir / "frames"
    out_path = vdir / "visual_signals.json"

    if not video_path.exists():
        raise FileNotFoundError(
            f"video.mp4 yok: {video_path}. Önce 'l2s ingest' çalıştır."
        )

    console.print("görüntü sinyalleri çıkarılıyor  [dim](scenedetect)[/dim]")
    db.set_stage(video_id, "visual", "running")

    try:
        scene_list = detect(str(video_path), ContentDetector())

        frames_dir.mkdir(exist_ok=True)
        scenes = []
        for idx, (start_tc, end_tc) in enumerate(scene_list):
            start = start_tc.get_seconds()
            end = end_tc.get_seconds()
            mid = (start + end) / 2.0
            frame_path = frames_dir / f"scene_{idx:04d}.jpg"
            _extract_frame(video_path, mid, frame_path)
            scenes.append(
                {
                    "id": idx,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "frame": str(frame_path.relative_to(vdir)),
                }
            )

        # Hiç sahne kesimi bulunamazsa (tek sahne) ortadan tek kare al
        if not scenes:
            frame_path = frames_dir / "scene_0000.jpg"
            _extract_frame(video_path, 0.0, frame_path)
            scenes.append({"id": 0, "start": 0.0, "end": None,
                           "frame": str(frame_path.relative_to(vdir))})

        signals = {"scene_count": len(scenes), "scenes": scenes}
        out_path.write_text(
            json.dumps(signals, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        db.set_stage(video_id, "visual", "done", f"{len(scenes)} sahne")
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "visual", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    console.print(f"  [green]✓[/green] {len(scenes)} sahne kesimi + temsilî kare")
    console.print(f"  [dim]{out_path}[/dim]")
