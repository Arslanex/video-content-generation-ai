"""Konuşmacı ayrımı (diarization) — pyannote ile "kim ne zaman konuştu".

İki kullanım:
  - diarize(video_id)            : tüm audio.wav (uzun videoda yavaş) -> speakers.json
  - diarize_range(vid, s, e)     : sadece bir klip aralığı (hızlı; dublaj bunu kullanır)

pyannote.audio 4.x modeli kapalıdır: HF_TOKEN + model koşulları kabulü gerekir.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

from . import db
from .config import HF_TOKEN, video_dir

console = Console()

_MODEL = "pyannote/speaker-diarization-community-1"  # pyannote.audio 4.x modeli
_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN ayarlı değil. Konuşmacı ayrımı için:\n"
            "  1) huggingface.co/settings/tokens'tan ücretsiz token al\n"
            "  2) huggingface.co/pyannote/speaker-diarization-community-1 koşullarını kabul et\n"
            "  3) .env dosyasına HF_TOKEN=... ekle"
        )
    from pyannote.audio import Pipeline

    try:  # yeni sürüm: token=, eski: use_auth_token=
        p = Pipeline.from_pretrained(_MODEL, token=HF_TOKEN)
    except TypeError:
        p = Pipeline.from_pretrained(_MODEL, use_auth_token=HF_TOKEN)
    if p is None:
        raise RuntimeError(
            "Model yüklenemedi. HF_TOKEN geçerli mi ve "
            "pyannote/speaker-diarization-community-1 koşullarını kabul ettin mi?"
        )
    try:  # Apple GPU (MPS) varsa hızlandır — hata olursa CPU'da kalır
        import torch
        if torch.backends.mps.is_available():
            p.to(torch.device("mps"))
    except Exception:  # noqa: BLE001
        pass
    _pipeline = p
    return p


def _run(audio_path, offset: float = 0.0, num_speakers: int | None = None) -> list[dict]:
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    pipeline = _load_pipeline()
    try:
        out = pipeline(str(audio_path), **kwargs)
    except Exception:  # noqa: BLE001 — MPS'te desteklenmeyen op olabilir → CPU'ya düş
        try:
            import torch
            pipeline.to(torch.device("cpu"))
        except Exception:  # noqa: BLE001
            pass
        out = pipeline(str(audio_path), **kwargs)
    # pyannote 4.x: DiarizeOutput(.speaker_diarization) | 3.x: Annotation
    annotation = getattr(out, "speaker_diarization", out)
    return [
        {"start": round(seg.start + offset, 2), "end": round(seg.end + offset, 2),
         "speaker": spk}
        for seg, _, spk in annotation.itertracks(yield_label=True)
    ]


def diarize(video_id: str, force: bool = False):
    """Tüm audio.wav üzerinde konuşmacı ayrımı (uzun videoda yavaş) -> speakers.json."""
    vdir = video_dir(video_id)
    audio_path = vdir / "audio.wav"
    out_path = vdir / "speakers.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))
    if not audio_path.exists():
        raise FileNotFoundError("audio.wav yok. Önce 'l2s ingest' çalıştır.")

    console.print("konuşmacı ayrımı (pyannote)  [dim]uzun videoda dakikalar sürebilir[/dim]")
    db.set_stage(video_id, "diarize", "running")
    try:
        turns = _run(audio_path)
        speakers = sorted({t["speaker"] for t in turns})
        data = {"num_speakers": len(speakers), "speakers": speakers, "turns": turns}
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        db.set_stage(video_id, "diarize", "done", f"{len(speakers)} konuşmacı")
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "diarize", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise
    console.print(f"  [green]✓[/green] {len(speakers)} konuşmacı, {len(turns)} konuşma")
    return data


def diarize_range(video_id: str, start: float, end: float,
                  num_speakers: int | None = None) -> dict:
    """Sadece [start, end] aralığını diarize eder (hızlı). Turns mutlak zamanlı döner.

    num_speakers verilirse pyannote'a ipucu olarak geçer (daha kararlı sonuç).
    """
    vdir = video_dir(video_id)
    audio_path = vdir / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError("audio.wav yok. Önce 'l2s ingest' çalıştır.")

    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(audio_path),
             "-ac", "1", "-ar", "16000", str(clip)],
            check=True,
        )
        turns = _run(clip, offset=start, num_speakers=num_speakers)
    speakers = sorted({t["speaker"] for t in turns})
    return {"num_speakers": len(speakers), "speakers": speakers, "turns": turns}


def speaker_for(turns: list[dict], start: float, end: float) -> str | None:
    """Verilen [start,end] aralığıyla en çok örtüşen konuşmacıyı döndürür."""
    best, best_ov = None, 0.0
    for t in turns:
        ov = max(0.0, min(end, t["end"]) - max(start, t["start"]))
        if ov > best_ov:
            best, best_ov = t["speaker"], ov
    return best
