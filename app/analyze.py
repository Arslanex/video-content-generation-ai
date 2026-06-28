"""Faz 3 — Claude ile analiz: segmentasyon + 3 format puanlama + üretim paketi.

fused.json -> recommendations (DB + recommendations.json)
Model: claude-opus-4-8, adaptive thinking, yapılandırılmış çıktı (json_schema).
"""
from __future__ import annotations

import json

from rich.console import Console

from . import db
from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, PROMPTS_DIR, video_dir

console = Console()

# Claude'un döndüreceği yapılandırılmış çıktı şeması.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fmt": {"type": "string", "enum": ["short", "episode", "podcast"]},
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "score": {"type": "number"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "description": {"type": "string"},
                    "reason": {"type": "string"},
                    "lens": {"type": "string"},
                },
                "required": ["fmt", "start_sec", "end_sec", "score", "title",
                             "hook", "description", "reason", "lens"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["recommendations"],
    "additionalProperties": False,
}


# Format başına süre sınırları (saniye): (min, max)
_DURATION = {"short": (12, 60), "episode": (90, 900), "podcast": (900, 1800)}

_PRIORITY = {
    "viral": "Önceliğin güçlü hook ve alıntılanabilirlik: çarpıcı, paylaşılası, "
             "tek başına anlamlı anları öne çıkar.",
    "educational": "Önceliğin öğreticilik ve bilgi yoğunluğu: net bir şey öğreten, "
                   "çerçeve/adım/liste içeren bölümleri öne çıkar.",
    "emotional": "Önceliğin duygusal ve yüksek enerjili anlar: gülme, heyecan, "
                 "samimi/etkileyici kısımlar.",
    "balanced": "",
}


def _preferences_text(formats: list[str], count, priority: str,
                      focus: str | None, exclude: str | None) -> str:
    """Kullanıcı tercihlerini Claude'a verilecek yönergeye çevirir."""
    lines = ["## Kullanıcı tercihleri (bunlara KESİNLİKLE uy)"]
    lines.append(f"Sadece şu format(lar) için aday üret: {', '.join(formats)}. "
                 "Başka format üretme.")
    if count:
        lines.append(f"Her format için en iyi ~{count} adayı üret "
                     "(zorlama; gerçekten iyi aday yoksa daha az ver).")
    if _PRIORITY.get(priority):
        lines.append(_PRIORITY[priority])
    if focus:
        lines.append(f"Konu odağı: özellikle şu konulara bak — {focus}. "
                     "Bu konularla ilgisiz bölümleri önerme.")
    if exclude:
        lines.append(f"Şunları içeren bölümleri kesinlikle atla: {exclude} "
                     "(ör. reklam, sponsor, jenerik, tanıtım).")
    lines.append("Her öneride 'lens' alanına o adayın baskın gücünü tek-iki kelimeyle "
                 "yaz (ör. hook, alıntı, duygu, öğreticilik, anlatı).")
    return "\n".join(lines)


def _refine(recs: list[dict], segments: list[dict],
            formats: list[str] | None = None, count: int | None = None
            ) -> tuple[list[dict], list[str]]:
    """Claude önerilerini doğrular/temizler:
    - start/end'i gerçek segment sınırlarına yaslar,
    - süreyi format sınırlarına çeker (uzunsa kuyruğu pause_after'a kırpar),
    - çok kısa/geçersizleri atar, aynı formatta ağır çakışmaları eler (yüksek puan kalır).
    Döndürür: (temiz_öneriler, değişiklik_notları).
    """
    notes0: list[str] = []
    # istenmeyen formatları ele (güvenlik: prompt'a rağmen gelirse)
    if formats:
        before = len(recs)
        recs = [r for r in recs if r["fmt"] in formats]
        if len(recs) < before:
            notes0.append(f"{before - len(recs)} istenmeyen-format önerisi elendi")

    if not segments:
        return _cap_per_format(recs, count), notes0

    seg_starts = sorted(s["start"] for s in segments)
    seg_ends = sorted(s["end"] for s in segments)
    pa_ends = sorted(s["end"] for s in segments if s.get("pause_after"))
    notes: list[str] = []

    def nearest(vals, x):
        return min(vals, key=lambda v: abs(v - x))

    refined = []
    for r in recs:
        fmt = r["fmt"]
        s = nearest(seg_starts, r["start_sec"])
        e = nearest(seg_ends, r["end_sec"])
        if e <= s:
            notes.append(f"{fmt} {r['start_sec']:.0f}s geçersiz (atıldı)")
            continue
        lo, hi = _DURATION.get(fmt, (0, 1e9))
        dur = e - s
        if dur > hi:
            limit = s + hi
            cands_pa = [v for v in pa_ends if s < v <= limit]
            cands = [v for v in seg_ends if s < v <= limit]
            new_e = max(cands_pa) if cands_pa else (max(cands) if cands else e)
            notes.append(f"{fmt} {dur:.0f}s → {new_e - s:.0f}s kırpıldı")
            e = new_e
        elif dur < lo:
            target = s + lo
            cands_pa = [v for v in pa_ends if v >= target]
            cands = [v for v in seg_ends if v >= target]
            if cands_pa or cands:
                e = min(cands_pa) if cands_pa else min(cands)
        if e - s < lo * 0.6 or e <= s:
            notes.append(f"{fmt} çok kısa (atıldı)")
            continue
        rr = dict(r)
        rr["start_sec"], rr["end_sec"] = round(s, 2), round(e, 2)
        refined.append(rr)

    # aynı formatta ağır çakışmaları ele (yüksek puanı tut)
    refined.sort(key=lambda r: (r["fmt"], -r["score"]))
    kept: list[dict] = []
    for r in refined:
        clash = False
        for k in kept:
            if k["fmt"] != r["fmt"]:
                continue
            inter = max(0.0, min(r["end_sec"], k["end_sec"]) - max(r["start_sec"], k["start_sec"]))
            if inter > 0.5 * min(r["end_sec"] - r["start_sec"], k["end_sec"] - k["start_sec"]):
                clash = True
                notes.append(f"{r['fmt']} çakışma elendi ({r['start_sec']:.0f}s)")
                break
        if not clash:
            kept.append(r)
    return _cap_per_format(kept, count), notes0 + notes


def _cap_per_format(recs: list[dict], count: int | None) -> list[dict]:
    """Her format için en yüksek puanlı `count` adayı tutar (count None ise hepsi)."""
    if not count:
        return recs
    out = []
    for fmt in {r["fmt"] for r in recs}:
        rows = sorted((r for r in recs if r["fmt"] == fmt), key=lambda r: -r["score"])
        out.extend(rows[:count])
    return out


def _build_user_content(fused: dict) -> str:
    """fused.json'dan token-verimli, sadece gerekli alanları içeren girdi üretir."""
    head = {
        "title": fused.get("title"),
        "channel": fused.get("channel"),
        "duration_sec": fused.get("duration_sec"),
        "language": fused.get("language"),
        "has_audio_signals": fused.get("has_audio_signals"),
        "has_visual_signals": fused.get("has_visual_signals"),
        "scene_count": fused.get("scene_count"),
        "chapters": fused.get("chapters"),
    }
    segs = [
        {
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "energy": s.get("avg_energy"),
            "scene": s.get("scene_id"),
            "pause_before": s.get("pause_before"),
            "pause_after": s.get("pause_after"),
        }
        for s in fused["segments"]
    ]
    return (
        "Video bilgisi:\n"
        + json.dumps(head, ensure_ascii=False, indent=2)
        + "\n\nZaman-hizalı segmentler:\n"
        + json.dumps(segs, ensure_ascii=False)
    )


_ALL_FORMATS = ["short", "episode", "podcast"]


def analyze(video_id: str, formats: list[str] | None = None, count: int | None = None,
            priority: str = "balanced", focus: str | None = None,
            exclude: str | None = None) -> None:
    """Claude'a zaman-hizalı temsili verir, kullanıcı tercihlerine göre öneri üretir.

    formats: üretilecek formatlar (None = hepsi). count: format başına hedef adet.
    priority: viral | educational | emotional | balanced. focus/exclude: konu odağı.
    """
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY ayarlı değil. .env dosyasına ekle "
            "(cp .env.example .env, sonra anahtarı doldur)."
        )

    formats = formats or _ALL_FORMATS
    vdir = video_dir(video_id)
    fused_path = vdir / "fused.json"
    if not fused_path.exists():
        raise FileNotFoundError("fused.json yok. Önce 'l2s fuse' çalıştır.")

    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    system_prompt = (PROMPTS_DIR / "analyze_system.md").read_text(encoding="utf-8")
    prefs = _preferences_text(formats, count, priority, focus, exclude)
    user_content = prefs + "\n\n" + _build_user_content(fused)

    console.print(f"Claude analiz ediyor  [dim]({CLAUDE_MODEL})[/dim]  "
                  f"[dim]formatlar: {','.join(formats)} · öncelik: {priority}[/dim]")
    db.set_stage(video_id, "analyze", "running")

    try:
        client = anthropic.Anthropic()
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=[
                {"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}
            ],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            msg = stream.get_final_message()

        if msg.stop_reason == "refusal":
            raise RuntimeError(f"Model isteği reddetti: {msg.stop_details}")

        text = next(b.text for b in msg.content if b.type == "text")
        data = json.loads(text)
        recs = data.get("recommendations", [])

        # doğrulama/temizleme: format filtresi, süre, sınır yaslama, çakışma, adet
        recs, notes = _refine(recs, fused.get("segments", []), formats, count)
        for n in notes:
            console.print(f"  [dim]· {n}[/dim]")

        # DB'ye yaz (payload = üretim paketinin zengin alanları)
        db.replace_recommendations(
            video_id,
            [
                {
                    "fmt": r["fmt"],
                    "start_sec": r["start_sec"],
                    "end_sec": r["end_sec"],
                    "score": r["score"],
                    "title": r["title"],
                    "payload": {
                        "hook": r["hook"],
                        "description": r["description"],
                        "reason": r["reason"],
                        "lens": r.get("lens", ""),
                    },
                }
                for r in recs
            ],
        )
        (vdir / "recommendations.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        by_fmt: dict[str, int] = {}
        for r in recs:
            by_fmt[r["fmt"]] = by_fmt.get(r["fmt"], 0) + 1
        detail = ", ".join(f"{k}:{v}" for k, v in by_fmt.items()) or "0"
        db.set_stage(video_id, "analyze", "done", detail)
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "analyze", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    console.print(f"  [green]✓[/green] {len(recs)} öneri  •  {detail}")
    console.print(f"  [dim]{vdir / 'recommendations.json'}[/dim]")
