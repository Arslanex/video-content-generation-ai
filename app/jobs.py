"""Ağır işleri AYRI bir süreçte çalıştırır (TUI'nin kullandığı).

Neden ayrı süreç: Textual uygulaması terminalin standart akışlarını (fd 0/1/2) yönetir.
Aynı süreçte thread içinde mlx-whisper gibi bazı kütüphaneler alt-süreç açarken
'bad value(s) in fds_to_keep' hatası verir; ayrıca üçüncü-parti çıktı doğrudan
Textual ekranına sızıp bozar. Ayrı süreçte fd 0/1/2 devnull'a yönlendirilir →
her şey susturulur, UI ile yalnızca bir kuyruk (Queue) üzerinden konuşulur.
"""
from __future__ import annotations

import os


def _mute() -> None:
    """Çocuk süreçte tüm çıktıyı sustur (OS fd düzeyinde + modül konsolları)."""
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        try:
            os.dup2(devnull, fd)
        except OSError:
            pass
    import sys
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    from rich.console import Console
    null = Console(file=open(os.devnull, "w"))
    from . import analyze, audio, dub, fuse, ingest, render, supercut, transcribe
    for m in (ingest, transcribe, audio, fuse, analyze, render, supercut, dub):
        m.console = null


def _job_pipeline(report, name: str, url: str) -> None:
    """İndir → transkript → ses → fuse → analiz. Devam-farkında: çıktısı olan adım atlanır
    (yeni iş = hepsi çalışır; tekrar koş = kaldığı yerden sürer)."""
    from . import db
    from .analyze import analyze as AN
    from .audio import analyze_audio
    from .config import ANTHROPIC_API_KEY, video_dir
    from .fuse import fuse as FU
    from .ingest import ingest as ING
    from .lifecycle import write_project_json
    from .transcribe import transcribe as TR

    report("step", "indiriliyor")

    def hook(d):
        if d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate")
            if tot:
                report("download", min(100.0, d.get("downloaded_bytes", 0) / tot * 100))

    vid = ING(url, progress_hook=hook)          # video.mp4 varsa indirme atlanır
    db.set_project_video(name, vid)
    report("ok", f"indirildi · video_id {vid}")
    report("refresh", None)
    vdir = video_dir(vid)

    if (vdir / "transcript.json").exists():
        report("info", "transkript zaten var, atlandı")
    else:
        report("step", "transkript (mlx-whisper)")
        TR(vid)
        report("ok", "transkript")

    if (vdir / "audio_signals.json").exists():
        report("info", "ses zaten var, atlandı")
    else:
        report("step", "ses sinyalleri (librosa)")
        analyze_audio(vid)
        report("ok", "ses sinyalleri")

    report("step", "birleştirme")               # ucuz, her zaman
    FU(vid)
    report("ok", "birleştirme")

    if (vdir / "recommendations.json").exists():
        report("info", "analiz zaten var, atlandı")
    elif ANTHROPIC_API_KEY:
        report("step", "analiz (Claude)")
        AN(vid)
        report("ok", "analiz — öneriler hazır")
    else:
        report("warn", "ANTHROPIC_API_KEY yok → analiz atlandı")

    write_project_json(vid)
    report("ok", f"{name} hazır")


def _job_transcribe(report, video_id: str) -> None:
    from .transcribe import transcribe as TR
    report("step", "transkript (mlx-whisper)")
    TR(video_id)
    report("ok", "transkript")


def _job_audio(report, video_id: str) -> None:
    from .audio import analyze_audio
    report("step", "ses sinyalleri (librosa)")
    analyze_audio(video_id)
    report("ok", "ses sinyalleri")


def _job_fuse(report, video_id: str) -> None:
    from .fuse import fuse as FU
    report("step", "birleştirme")
    FU(video_id)
    report("ok", "birleştirme")


def _job_analyze(report, video_id: str) -> None:
    from .analyze import analyze as AN
    report("step", "analiz (Claude)")
    AN(video_id)
    report("ok", "analiz yenilendi")


def _job_supercut(report, video_id: str) -> None:
    from .supercut import supercut as SC
    report("step", "supercut (keşif → kurgu → tutarlılık kapısı)")
    SC(video_id)
    report("ok", "supercut önerileri hazır")


def _job_render(report, video_id: str, rec_id: int, opts: dict) -> None:
    from . import db
    from .render import render as R
    rec = next((r for r in db.get_recommendations(video_id) if r["id"] == rec_id), None)
    if rec is None:
        report("error", f"#{rec_id} bulunamadı")
        return
    tag = f" +dub {opts['lang']}" if opts["lang"] else ""
    if not opts["lang"] and opts.get("caplang"):
        tag = f" +altyazı {opts['caplang']}"
    report("step", f"render {rec['fmt']} #{rec_id}{tag}")
    R(video_id, str(rec_id), layout=opts["layout"], captions=opts["captions"],
      intro=opts["cover"], lang=opts["lang"], xfade=opts["xfade"],
      cap_lang=opts.get("caplang"))
    report("ok", f"render bitti #{rec_id} → ciktilar/")


def _job_finish(report, video_id: str, name: str) -> None:
    from .lifecycle import finish_project
    report("step", f"bitiriliyor: {name} (video sil + zip)")
    res = finish_project(video_id)
    msg = "silindi" if res["removed_video"] else "yoktu"
    report("ok", f"{name} bitirildi · video.mp4 {msg} · arşiv: {res['zip'].name}")


_JOBS = {
    "pipeline": _job_pipeline,
    "transcribe": _job_transcribe,
    "audio": _job_audio,
    "fuse": _job_fuse,
    "analyze": _job_analyze,
    "supercut": _job_supercut,
    "render": _job_render,
    "finish": _job_finish,
}


def child_entry(q, jobname: str, args: list) -> None:
    """Çocuk sürecin giriş noktası: sustur, işi çalıştır, mesajları kuyruğa yaz.

    setsid ile yeni bir süreç grubu lideri olur → TUI iptal edince ebeveyn,
    grubun tamamını (ffmpeg vb. torunlar dahil) tek seferde sonlandırabilir."""
    try:
        os.setsid()
    except OSError:
        pass
    _mute()

    def report(kind: str, payload=None) -> None:
        q.put((kind, payload))

    try:
        _JOBS[jobname](report, *args)
    except Exception as exc:  # noqa: BLE001
        report("error", str(exc))
    finally:
        q.put(("__end__", None))
