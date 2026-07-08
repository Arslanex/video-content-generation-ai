"""Faz 3+ — Supercut: videonun farklı yerlerinden bağlanan anları tek montaja diz.

Değer, kesme mekaniğinde değil, dağınık anların ortak bir tez etrafında bir anlatı
yayı (kanca → gelişme → kapanış) kurmasında. Bu yüzden burası bir "yönetmen": Claude'u
iki geçişte ve bir tutarlılık kapısında çalıştırır.

  1) keşif  (discover) — videodan aday tezler/temalar çıkar
  2) kurgu  (assemble) — her tema için arc kuran span'leri seç + sırala
  3) kapı   (review)   — birleşik metnin akışını doğrula, sırayı düzelt, tutarsızı ele

fused.json -> öneriler (DB fmt=supercut, only_fmt ile analyze'ı silmeden) + supercut.json

Bu faz yalnızca MONTAJ PLANINI üretir (span'ler + roller + gerekçe). Video render'ı
ayrı bir faz (render tarafında fmt=supercut yolu) — burada dosya kesilmez.
"""
from __future__ import annotations

import json

from rich.console import Console

from . import db
from .analyze import _build_user_content       # segment temsili (DRY — analyze ile aynı)
from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, PROMPTS_DIR, video_dir

console = Console()

# Dikey supercut süre/parça bandı
_MIN_TOTAL, _MAX_TOTAL = 20.0, 90.0
_MIN_SPAN = 2.0            # bir parça bundan kısaysa atılır
_MIN_SPANS, _MAX_SPANS = 2, 7
_MIN_COHERENCE = 55.0     # tutarlılık kapısı eşiği (0–100)
_TERMINAL = (".", "!", "?", "…", ":")   # cümle-bütünlüğü işaretleri


# --- Yapılandırılmış çıktı şemaları ---------------------------------------

DISCOVER_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "thesis": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["id", "thesis", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["themes"],
    "additionalProperties": False,
}

_SPAN = {
    "type": "object",
    "properties": {
        "start_sec": {"type": "number"},
        "end_sec": {"type": "number"},
        "role": {"type": "string"},
        "text_preview": {"type": "string"},
    },
    "required": ["start_sec", "end_sec", "role", "text_preview"],
    "additionalProperties": False,
}

ASSEMBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "supercuts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "description": {"type": "string"},
                    "reason": {"type": "string"},
                    "score": {"type": "number"},
                    "spans": {"type": "array", "items": _SPAN},
                },
                "required": ["theme_id", "title", "hook", "description",
                             "reason", "score", "spans"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["supercuts"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "coherent": {"type": "boolean"},
                    "coherence_score": {"type": "number"},
                    "order": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": "string"},
                },
                "required": ["index", "coherent", "coherence_score", "order", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


# --- Claude çağrısı (ortak, önbellekli segment bloğu) ----------------------

def _call(system_prompt: str, shared: str, task: str, schema: dict,
          max_tokens: int = 8000) -> dict:
    """Tek bir yapılandırılmış Claude çağrısı.

    `shared` (segment temsili) her geçişte aynı kalır → prompt cache ile ucuzlar.
    `task` geçişe özel yönergedir.
    """
    import anthropic

    client = anthropic.Anthropic()
    content = [
        {"type": "text", "text": shared, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": task},
    ]
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"Model isteği reddetti: {msg.stop_details}")
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text)


# --- Tercihler -------------------------------------------------------------

def _prefs(count, priority, focus, exclude) -> str:
    lines = []
    if count:
        lines.append(f"En fazla ~{count} tema öner (zorlama).")
    if priority and priority != "balanced":
        lines.append(f"Öncelik: {priority} tonundaki temaları öne çıkar.")
    if focus:
        lines.append(f"Konu odağı: özellikle şu konulara bak — {focus}.")
    if exclude:
        lines.append(f"Şunları içeren anları kullanma: {exclude}.")
    return ("\n".join(lines) + "\n\n") if lines else ""


# --- Span doğrulama / yapılandırma ----------------------------------------

def _nearest(vals, x):
    return min(vals, key=lambda v: abs(v - x))


def _snap(x, exact_vals, preferred_vals, window=1.5):
    """x'i en yakın gerçek sınıra yaslar; o sınırın yakınında tercih edilen bir
    sınır (duraklama / cümle sonu) varsa onu seçer."""
    base = _nearest(exact_vals, x)
    near = [v for v in preferred_vals if abs(v - base) <= window]
    return min(near, key=lambda v: abs(v - base)) if near else base


def _span_text(start: float, end: float, segments: list[dict]) -> str:
    """Bir zaman aralığına düşen segment metinlerini birleştirir (tutarlılık için)."""
    parts = [s["text"] for s in segments
             if start <= (s["start"] + s["end"]) / 2.0 <= end]
    return " ".join(p.strip() for p in parts if p).strip()


def _refine_spans(spans: list[dict], segments: list[dict],
                  sentences: list[dict] | None = None) -> tuple[list[dict], list[str]]:
    """Span'leri CÜMLE sınırlarına yaslar (cümle yoksa segment sınırına — yedek),
    kısa/çakışanları eler. Sıra KORUNUR (anlatısal sıra zamansal değildir)."""
    from . import sentences as S

    notes: list[str] = []
    sents = sentences or []
    seg_starts = sorted(s["start"] for s in segments)
    seg_ends = sorted(s["end"] for s in segments)
    pb = [s["start"] for s in segments if s.get("pause_before")]
    good_ends = sorted({
        s["end"] for s in segments
        if s.get("pause_after") or (s["text"] or "").rstrip().endswith(_TERMINAL)
    })

    refined: list[dict] = []
    for sp in spans:
        if sents:                                    # CÜMLE-hassas yaslama
            s = S.snap_start(sp["start_sec"], sents)
            e = S.snap_end(sp["end_sec"], sents)
        else:                                         # yedek: segment sınırı
            s = _snap(sp["start_sec"], seg_starts, pb)
            e = _snap(sp["end_sec"], seg_ends, good_ends)
        if e - s < _MIN_SPAN:
            notes.append(f"parça {sp['start_sec']:.0f}s çok kısa (atıldı)")
            continue
        # aynı anı iki kez almayı önle (zaman-çakışması) — sıra bozulmadan
        clash = any(
            min(e, k["end"]) - max(s, k["start"]) > 0.5 * min(e - s, k["end"] - k["start"])
            for k in refined
        )
        if clash:
            notes.append(f"parça {s:.0f}s çakışma (atıldı)")
            continue
        refined.append({
            "start": round(s, 2), "end": round(e, 2),
            "role": sp.get("role", ""),
            "text_preview": sp.get("text_preview", ""),
        })
    return refined, notes


def _fit(spans: list[dict], notes: list[str]) -> list[dict]:
    """Parça sayısını ve toplam süreyi banda çeker: kanca ve kapanışı koruyarak
    ortadan en uzun parçaları atar."""
    def total():
        return sum(sp["end"] - sp["start"] for sp in spans)

    while (len(spans) > _MAX_SPANS or total() > _MAX_TOTAL) and len(spans) > _MIN_SPANS:
        i = max(range(1, len(spans) - 1), key=lambda j: spans[j]["end"] - spans[j]["start"])
        d = spans.pop(i)
        notes.append(f"kırpma: {d['start']:.0f}s parça çıkarıldı (süre/sayı sınırı)")
    return spans


def _apply_order(spans: list[dict], order: list[int]) -> list[dict]:
    """Denetçinin nihai sırası. `order`, TUTULACAK parça index'lerinin sırasıdır;
    listede olmayan index DÜŞÜRÜLÜR (akışı bozan parçayı ele). Geçersiz ya da çok fazla
    düşüren (min parça altına inen) sıra yok sayılır → parçalar olduğu gibi kalır."""
    if not order:
        return spans
    if not all(isinstance(i, int) and 0 <= i < len(spans) for i in order):
        return spans
    if len(set(order)) != len(order):        # tekrar eden index
        return spans
    if len(order) < _MIN_SPANS:              # aşırı düşürme → güvenli tarafta kal
        return spans
    return [spans[i] for i in order]


# --- Ana akış --------------------------------------------------------------

def supercut(video_id: str, count: int | None = None, priority: str = "balanced",
             focus: str | None = None, exclude: str | None = None) -> None:
    """Videodan supercut (montaj) önerileri üretir: keşif → kurgu → tutarlılık kapısı."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY ayarlı değil. `l2s set-anthropic` ile ekle.")

    vdir = video_dir(video_id)
    fused_path = vdir / "fused.json"
    if not fused_path.exists():
        raise FileNotFoundError("fused.json yok. Önce 'l2s fuse' çalıştır.")

    fused = json.loads(fused_path.read_text(encoding="utf-8"))
    segments = fused.get("segments", [])
    shared = _build_user_content(fused)
    prefs = _prefs(count, priority, focus, exclude)

    p_discover = (PROMPTS_DIR / "supercut_discover.md").read_text(encoding="utf-8")
    p_assemble = (PROMPTS_DIR / "supercut_assemble.md").read_text(encoding="utf-8")
    p_review = (PROMPTS_DIR / "supercut_review.md").read_text(encoding="utf-8")

    db.set_stage(video_id, "supercut", "running")
    try:
        # 1) keşif — aday tezler
        console.print(f"[bold]1/3[/bold] tema keşfi  [dim]({CLAUDE_MODEL})[/dim]")
        themes = _call(p_discover, shared,
                       prefs + "Bu videodan güçlü supercut'lar çıkarabilecek temaları öner.",
                       DISCOVER_SCHEMA, max_tokens=4000).get("themes", [])
        if not themes:
            console.print("  [yellow]uygun tema bulunamadı.[/yellow]")
            db.set_stage(video_id, "supercut", "done", "0 tema")
            return
        thesis_of = {t["id"]: t["thesis"] for t in themes}
        for t in themes:
            console.print(f"  [dim]· tema {t['id']}: {t['thesis']}[/dim]")

        # 2) kurgu — her tema için arc kuran span'ler
        console.print("[bold]2/3[/bold] montaj kurgusu")
        assemble_task = (
            "Aşağıdaki temalar için birer supercut kur:\n"
            + json.dumps(themes, ensure_ascii=False, indent=2)
        )
        raw = _call(p_assemble, shared, assemble_task, ASSEMBLE_SCHEMA,
                    max_tokens=12000).get("supercuts", [])

        from . import sentences as _S            # eski fused.json → transcript.json yedeği
        sentences = _S.load_sentences(vdir, fused)
        cuts: list[dict] = []
        for sc in raw:
            spans, notes = _refine_spans(sc.get("spans", []), segments, sentences)
            for n in notes:
                console.print(f"  [dim]· {n}[/dim]")
            spans = _fit(spans, notes)
            if len(spans) < _MIN_SPANS:
                console.print(f"  [dim]· '{sc.get('title', '')}' yeterli parça yok (atıldı)[/dim]")
                continue
            sc["spans"] = spans
            cuts.append(sc)
        if not cuts:
            console.print("  [yellow]geçerli montaj kurulamadı.[/yellow]")
            db.set_stage(video_id, "supercut", "done", "0 montaj")
            return

        # 3) tutarlılık kapısı — birleşik metnin akışı
        console.print("[bold]3/3[/bold] tutarlılık kapısı")
        blocks = []
        for i, sc in enumerate(cuts):
            lines = [f"### Supercut {i}: {sc.get('title', '')}"]
            for j, sp in enumerate(sc["spans"]):
                # GERÇEK kesilecek metin (cümle-hizalı) — kapı artık gerçek kesime kör değil
                txt = (_S.text_between(sp["start"], sp["end"], sentences) if sentences
                       else _span_text(sp["start"], sp["end"], segments))
                lines.append(f"[{j}] ({sp['role']}) {txt}")
            blocks.append("\n".join(lines))
        reviews = _call(p_review, shared,
                        "Aşağıdaki supercut'ları değerlendir:\n\n" + "\n\n".join(blocks),
                        REVIEW_SCHEMA, max_tokens=4000).get("reviews", [])
        review_by_idx = {rv["index"]: rv for rv in reviews}

        recs: list[dict] = []
        for i, sc in enumerate(cuts):
            rv = review_by_idx.get(i, {})
            coh = float(rv.get("coherence_score", 0.0))
            if not rv.get("coherent", True) or coh < _MIN_COHERENCE:
                console.print(f"  [dim]· '{sc.get('title', '')}' tutarlılık düşük "
                              f"(puan {coh:.0f}) — elendi[/dim]")
                continue
            before_n = len(sc["spans"])
            sc["spans"] = _apply_order(sc["spans"], rv.get("order", []))
            if len(sc["spans"]) < before_n:
                console.print(f"  [dim]· '{sc.get('title', '')}' {before_n - len(sc['spans'])} "
                              f"parça düşürüldü (akış)[/dim]")
            start = min(sp["start"] for sp in sc["spans"])
            end = max(sp["end"] for sp in sc["spans"])
            total = sum(sp["end"] - sp["start"] for sp in sc["spans"])
            score = round(0.5 * float(sc.get("score", 0)) + 0.5 * coh, 1)
            recs.append({
                "fmt": "supercut",
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "score": score,
                "title": sc.get("title", ""),
                "payload": {
                    "hook": sc.get("hook", ""),
                    "description": sc.get("description", ""),
                    "reason": sc.get("reason", ""),
                    "lens": "supercut",
                    "thesis": thesis_of.get(sc.get("theme_id"), ""),
                    "total_sec": round(total, 2),
                    "coherence": coh,
                    "coherence_note": rv.get("note", ""),
                    "spans": sc["spans"],
                },
            })

        if not recs:
            console.print("  [yellow]tutarlılık kapısından geçen montaj yok.[/yellow]")
            db.set_stage(video_id, "supercut", "done", "0 (kapı)")
            return

        recs.sort(key=lambda r: -r["score"])
        db.replace_recommendations(video_id, recs, only_fmt="supercut")
        (vdir / "supercut.json").write_text(
            json.dumps({"supercuts": recs}, ensure_ascii=False, indent=2), encoding="utf-8")
        db.set_stage(video_id, "supercut", "done", f"{len(recs)} montaj")
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "supercut", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    console.print(f"  [green]✓[/green] {len(recs)} supercut  •  "
                  f"[cyan]l2s recs {video_id}[/cyan]")
    console.print(f"  [dim]{vdir / 'supercut.json'}[/dim]")
