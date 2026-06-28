"""Faz 4 — Ses sinyalleri (librosa: RMS enerji eğrisi + duraklama tespiti).

audio.wav -> audio_signals.json
PyTorch gerektiren VAD yerine librosa'nın kendi araçlarıyla (hafif, yerel):
  - energy : ~0.5s aralıklı RMS enerji eğrisi (heyecan/yoğunluk tepe noktaları)
  - pauses : sessizlik aralıkları (doğal kesim noktaları)
  - peaks  : en yüksek enerjili anlar (duygusal/vurgulu noktalar)
"""
from __future__ import annotations

import json

from rich.console import Console

from . import db
from .config import video_dir

console = Console()

_HOP_SEC = 0.5          # enerji eğrisi çözünürlüğü
_TOP_DB = 30            # bunun altındaki bölümler "sessiz" sayılır
_MIN_PAUSE_SEC = 0.4    # bu süreden kısa duraklamalar yok sayılır


def analyze_audio(video_id: str) -> None:
    """Ses enerjisi + duraklama sinyallerini çıkarır."""
    import librosa  # ağır import — komut çalışınca yüklensin
    import numpy as np

    vdir = video_dir(video_id)
    audio_path = vdir / "audio.wav"
    out_path = vdir / "audio_signals.json"

    if not audio_path.exists():
        raise FileNotFoundError(
            f"audio.wav yok: {audio_path}. Önce 'l2s ingest' çalıştır."
        )

    console.print("ses sinyalleri çıkarılıyor  [dim](librosa)[/dim]")
    db.set_stage(video_id, "audio", "running")

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        total_sec = float(len(y) / sr)

        # --- RMS enerji eğrisi (0.5s aralıklara indirgenmiş) ---
        hop = int(_HOP_SEC * sr)
        rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
        rms_max = float(rms.max()) or 1.0
        energy = [
            {"t": round(i * _HOP_SEC, 2), "rms": round(float(v) / rms_max, 4)}
            for i, v in enumerate(rms)
        ]

        # --- Duraklamalar: konuşma/aktif aralıkların dışındaki boşluklar ---
        intervals = librosa.effects.split(y, top_db=_TOP_DB, hop_length=hop)
        pauses = []
        prev_end = 0.0
        for start, end in intervals:
            s, e = start / sr, end / sr
            if s - prev_end >= _MIN_PAUSE_SEC:
                pauses.append({"start": round(prev_end, 2), "end": round(s, 2)})
            prev_end = e
        if total_sec - prev_end >= _MIN_PAUSE_SEC:
            pauses.append({"start": round(prev_end, 2), "end": round(total_sec, 2)})

        # --- Enerji tepe noktaları (en yüksek %10) ---
        if len(rms):
            thr = float(np.quantile(rms, 0.9))
            peaks = [
                {"t": round(i * _HOP_SEC, 2), "rms": round(float(v) / rms_max, 4)}
                for i, v in enumerate(rms)
                if v >= thr
            ]
        else:
            peaks = []

        signals = {
            "duration_sec": round(total_sec, 2),
            "hop_sec": _HOP_SEC,
            "energy": energy,
            "pauses": pauses,
            "peaks": peaks,
        }
        out_path.write_text(
            json.dumps(signals, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        db.set_stage(
            video_id, "audio", "done",
            f"{len(pauses)} duraklama, {len(peaks)} tepe",
        )
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "audio", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    console.print(
        f"  [green]✓[/green] {len(pauses)} duraklama  •  {len(peaks)} enerji tepesi"
    )
    console.print(f"  [dim]{out_path}[/dim]")
