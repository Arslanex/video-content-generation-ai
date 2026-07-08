"""Proje yaşam döngüsü: project.json yazımı + 'bitir' (kapat + arşivle).

Bitir: orijinal video.mp4 silinir (en büyük dosya), ciktilar/ + project.json bir
zip'e sıkıştırılır, DB'de proje 'done' işaretlenir (dir + archive_path yazılır).
transcript/fused/recommendations korunur → gerekirse yeniden render mümkün.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from . import db
from .config import BASE_DIR, project_dir, video_dir


def _output_counts(pdir: Path) -> dict[str, int]:
    """ciktilar/ altındaki her klasördeki dosya sayısı."""
    counts: dict[str, int] = {}
    cikti = pdir / "ciktilar"
    if cikti.exists():
        for sub in cikti.iterdir():
            if sub.is_dir():
                counts[sub.name] = sum(
                    1 for f in sub.iterdir() if f.is_file() and not f.name.startswith("."))
    return counts


def write_project_json(video_id: str, status: str = "active") -> dict:
    """Proje meta bilgisini <proje>/project.json'a yazar (DB'den derlenir)."""
    pdir = project_dir(video_id)
    name = db.project_for_video(video_id) or video_id
    proj = db.get_project(name)
    v = db.get_video(video_id)

    rec_counts: dict[str, int] = {}
    for r in db.get_recommendations(video_id):
        rec_counts[r["fmt"]] = rec_counts.get(r["fmt"], 0) + 1

    payload = {
        "name": name,
        "video_id": video_id,
        "title": v["title"] if v else None,
        "url": proj["url"] if proj else None,
        "duration_sec": v["duration_sec"] if v else None,
        "status": status,
        "recommendations": rec_counts,
        "outputs": _output_counts(pdir),
        "dir": str(pdir),
    }
    (pdir / "project.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def finish_project(video_id: str) -> dict:
    """Projeyi bitirir: orijinal videoyu sil → ciktilar+meta zip → DB 'done'.

    Döndürür: {removed_video, zip, dir}.
    """
    pdir = project_dir(video_id)

    # 1) orijinal videoyu sil (en büyük dosya; ara dosyalar korunur)
    vfile = video_dir(video_id) / "video.mp4"
    removed = vfile.exists()
    if removed:
        vfile.unlink()

    # 2) güncel project.json (status=done)
    write_project_json(video_id, status="done")

    # 3) arşiv: ciktilar/ + project.json → <proje>/<video_id>.zip
    zip_path = pdir / f"{video_id}.zip"
    pjson = pdir / "project.json"
    cikti = pdir / "ciktilar"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        if pjson.exists():
            z.write(pjson, "project.json")
        if cikti.exists():
            for f in sorted(cikti.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    z.write(f, f.relative_to(pdir))

    # 4) DB
    db.mark_project_done(video_id, str(pdir), str(zip_path))
    return {"removed_video": removed, "zip": zip_path, "dir": pdir}


def delete_project(name: str) -> str | None:
    """Projeyi tümüyle siler: DB kayıtları + diskteki proje klasörü."""
    vid = db.delete_project_full(name)
    if vid:
        d = BASE_DIR / vid                 # project_dir() mkdir yapar; burada saf yol
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    return vid
