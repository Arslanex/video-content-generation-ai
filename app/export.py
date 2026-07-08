"""Faz 6 — Çıktı: onaylanan önerileri ffmpeg ile klip dosyalarına keser.

Seçim sözdizimi (--pick):
  "short:1"            -> en yüksek puanlı short
  "short:1,short:2"    -> ilk iki short
  "episode:1,short:3"  -> 1. bölüm + 3. short
  "short"              -> tüm short'lar
  "all"                -> tüm öneriler

Short formatında varsayılan olarak 9:16 dikey kırpma uygulanır (--no-vertical ile kapatılır).
Klipler data/<video_id>/clips/ altına yazılır.
"""
from __future__ import annotations

import subprocess

from rich.console import Console

from . import db
from .config import clip_dir, video_dir

console = Console()


def _parse_picks(pick: str, recs_by_fmt: dict[str, list]) -> list:
    """Seçim ifadesini öneri satırlarına çözer."""
    chosen = []
    for token in (t.strip() for t in pick.split(",") if t.strip()):
        if token == "all":
            for rows in recs_by_fmt.values():
                chosen.extend(rows)
            continue
        if ":" in token:
            fmt, rank = token.split(":", 1)
            rows = recs_by_fmt.get(fmt, [])
            idx = int(rank) - 1
            if 0 <= idx < len(rows):
                chosen.append(rows[idx])
            else:
                console.print(f"  [yellow]uyarı:[/yellow] {token} için aday yok")
        else:
            chosen.extend(recs_by_fmt.get(token, []))
    # tekilleştir (id bazlı)
    seen, unique = set(), []
    for r in chosen:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    return unique


def _cut(video_path, start: float, end: float, out_path, vertical: bool) -> None:
    """video.mp4'ten [start, end] aralığını keser (re-encode, kare hassas)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(video_path)]
    if vertical:
        # merkezden 9:16 kırp, 1080x1920'e ölçekle
        cmd += ["-vf", "crop=ih*9/16:ih,scale=1080:1920"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path)]
    subprocess.run(cmd, check=True)


def export(video_id: str, picks: str, vertical: bool = True) -> None:
    """Seçilen önerileri klip dosyalarına dönüştürür."""
    vdir = video_dir(video_id)
    video_path = vdir / "video.mp4"

    if not video_path.exists():
        raise FileNotFoundError("video.mp4 yok. Önce 'l2s ingest' çalıştır.")

    all_recs = db.get_recommendations(video_id)
    if not all_recs:
        raise RuntimeError("Öneri yok. Önce 'l2s analyze' çalıştır.")

    recs_by_fmt: dict[str, list] = {}
    for r in all_recs:
        recs_by_fmt.setdefault(r["fmt"], []).append(r)

    selected = _parse_picks(picks, recs_by_fmt)
    if not selected:
        raise RuntimeError(f"'{picks}' hiçbir öneriyle eşleşmedi.")

    # supercut çok-parçalı; tek-aralık kesimi yanlış olur → montaj için render'a yönlendir.
    sc = [r for r in selected if r["fmt"] == "supercut"]
    if sc:
        from .render import render as _render
        sc_picks = ",".join(str(r["id"]) for r in sc)
        console.print(f"  [dim]{len(sc)} supercut çok-parçalı → montaj için render'a "
                      f"yönlendiriliyor (renders/): {sc_picks}[/dim]")
        _render(video_id, sc_picks)
        selected = [r for r in selected if r["fmt"] != "supercut"]
        if not selected:
            return

    db.set_stage(video_id, "export", "running")

    try:
        for r in selected:
            is_short = r["fmt"] == "short"
            v = vertical and is_short
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_"
                                 for c in (r["title"] or "clip"))[:50].strip()
            name = f"{r['fmt']}_{r['id']}_{safe_title}.mp4".replace(" ", "_")
            out = clip_dir(video_id, r["fmt"]) / name   # ciktilar/<format>/…
            console.print(
                f"  kesiliyor: [cyan]{r['fmt']}[/cyan] "
                f"{r['start_sec']:.1f}-{r['end_sec']:.1f}s"
                f"{' (9:16)' if v else ''}  → {name}"
            )
            _cut(video_path, r["start_sec"], r["end_sec"], out, v)
        db.set_stage(video_id, "export", "done", f"{len(selected)} klip")
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "export", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    from .config import output_dir
    console.print(f"  [green]✓[/green] {len(selected)} klip  •  "
                  f"[dim]{output_dir(video_id)}[/dim]")
