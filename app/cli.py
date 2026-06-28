"""long-to-shorts komut satırı arayüzü.

Kullanım:
    l2s init
    l2s ingest "https://youtube.com/watch?v=..."
    l2s transcribe <video_id>
    l2s audio <video_id>
    l2s visual <video_id>
    l2s fuse <video_id>
    l2s analyze <video_id>
    l2s export <video_id> --pick short:3,episode:1
    l2s status <video_id>
    l2s list
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from . import db
from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, DB_PATH, ensure_dirs

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="Uzun videolardan shorts / YouTube bölümü / podcast bölümü önerileri (yerel).",
)
console = Console()


_HF_MODEL = "pyannote/speaker-diarization-community-1"
_HF_LINK = f"https://huggingface.co/{_HF_MODEL}"


def _fail(msg: str) -> None:
    """Temiz hata: traceback yerine net mesaj + çıkış."""
    console.print(f"\n[bold red]Eksik:[/bold red] {msg}\n")
    raise typer.Exit(1)


def _need_anthropic() -> None:
    from .config import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        _fail("ANTHROPIC_API_KEY yok (analiz/çeviri için gerekli).\n"
              "  → `uv run l2s set-anthropic` ile ekle\n"
              "  → anahtarı buradan al: https://console.anthropic.com")


def _hf_status() -> str:
    """HF durumunu döndürür: ok | no_token | gated | bad_token | unknown."""
    from .config import HF_TOKEN
    if not HF_TOKEN:
        return "no_token"
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(_HF_MODEL, token=HF_TOKEN)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        from huggingface_hub.utils import GatedRepoError
        if isinstance(exc, GatedRepoError):
            return "gated"
        s = str(exc).lower()
        if "gated" in s or "restricted" in s or "awaiting" in s:
            return "gated"
        if "401" in s or "unauthor" in s or "invalid" in s or "token" in s:
            return "bad_token"
        return "unknown"


def _need_hf_hard() -> None:
    """Konuşmacı ayrımı zorunlu olan komutlar için (diarize)."""
    st = _hf_status()
    if st == "no_token":
        _fail("HF_TOKEN yok (konuşmacı ayrımı için gerekli).\n"
              "  → `uv run l2s set-hf` ile ekle\n"
              f"  → ve modeli onayla ('Agree and access repository'):\n    {_HF_LINK}")
    if st == "gated":
        _fail("pyannote modeli erişimin onaylı değil.\n"
              f"  → Şu sayfada 'Agree and access repository' de:\n    {_HF_LINK}")
    if st == "bad_token":
        _fail("HF_TOKEN geçersiz görünüyor.\n  → `uv run l2s set-hf` ile yenisini gir.")


def _warn_hf_soft() -> None:
    """Dublaj gibi HF olmadan da (tek ses) çalışan komutlar için yumuşak uyarı."""
    st = _hf_status()
    if st == "no_token":
        console.print("[yellow]Not:[/yellow] HF_TOKEN yok → konuşmacı ayrımı kapalı, "
                      "tek ses kullanılacak. Açmak için: [cyan]l2s set-hf[/cyan]")
    elif st == "gated":
        console.print("[yellow]Not:[/yellow] pyannote modeli onaylı değil → tek ses. "
                      f"Onaylamak için: [cyan]{_HF_LINK}[/cyan]")
    elif st == "bad_token":
        console.print("[yellow]Not:[/yellow] HF_TOKEN geçersiz → tek ses. "
                      "Düzeltmek için: [cyan]l2s set-hf[/cyan]")


def _set_env_key(key: str, value: str) -> None:
    """`.env`'de KEY=value satırını ekler/günceller (.env yoksa oluşturur)."""
    from .config import ROOT

    env = ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    out, found = [], False
    for line in lines:
        if line.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    env.write_text("\n".join(out) + "\n", encoding="utf-8")


@app.command("set-anthropic")
def set_anthropic(
    key: str = typer.Option(..., prompt="Anthropic API key", hide_input=True,
                            help="Anthropic (Claude) API anahtarı (girilince gizlenir)"),
) -> None:
    """Anthropic API anahtarını .env'e yazar (ANTHROPIC_API_KEY; analiz/çeviri için)."""
    _set_env_key("ANTHROPIC_API_KEY", key.strip())
    console.print("[green]✓[/green] ANTHROPIC_API_KEY .env'e yazıldı")


@app.command("set-hf")
def set_hf(
    key: str = typer.Option(..., prompt="HuggingFace token", hide_input=True,
                            help="HuggingFace token (girilince gizlenir)"),
) -> None:
    """HuggingFace token'ını .env'e yazar (HF_TOKEN; konuşmacı ayrımı/dublaj için)."""
    _set_env_key("HF_TOKEN", key.strip())
    console.print("[green]✓[/green] HF_TOKEN .env'e yazıldı")


@app.command()
def init() -> None:
    """Klasörleri ve veritabanı şemasını oluşturur (idempotent)."""
    ensure_dirs()
    db.init_db()
    console.print(f"[green]✓[/green] Veritabanı hazır: [dim]{DB_PATH}[/dim]")
    key_state = ("[green]ayarlı[/green]" if ANTHROPIC_API_KEY
                 else "[yellow]eksik — `l2s set-anthropic`[/yellow]")
    console.print(f"  Claude modeli: [cyan]{CLAUDE_MODEL}[/cyan]  •  API anahtarı: {key_state}")


@app.command()
def run(
    url: str,
    visual: bool = typer.Option(False, help="Sahne tespitini de çalıştır (yavaş, çoğu durumda gerekmez)"),
    render_pick: str = typer.Option(None, "--render", help="Bitince render et, örn: short:1,short:2"),
    force: bool = typer.Option(False, help="Çıktısı olan adımları da yeniden çalıştır"),
    formats: str = typer.Option("short,episode,podcast", help="Üretilecek formatlar, örn: short"),
    count: int = typer.Option(None, help="Format başına hedef aday sayısı"),
    priority: str = typer.Option("balanced", help="viral | educational | emotional | balanced"),
    focus: str = typer.Option(None, help="Konu odağı"),
    exclude: str = typer.Option(None, help="Atlanacak konular"),
    captions: bool = typer.Option(True, help="Render'da gömülü altyazı (--no-captions ile kapat)"),
    layout: bool = typer.Option(True, help="Render'da short'larda yüz-farkında yerleşim"),
    intro: bool = typer.Option(True, help="Render'da açılış kapağı"),
) -> None:
    """Uçtan uca: ingest → transcribe → audio → (visual) → fuse → analyze → (render).

    Her adımın çıktısı varsa atlanır; --force ile baştan yapılır.
    """
    _need_anthropic()   # analiz Claude gerektirir — indirmeden önce kontrol et
    from .config import video_dir
    from .ingest import ingest as _ingest

    def done(vid: str, fname: str) -> bool:
        return (video_dir(vid) / fname).exists() and not force

    console.rule("[bold]1/6 ingest")
    vid = _ingest(url)

    console.rule("[bold]2/6 transcribe")
    if done(vid, "transcript.json"):
        console.print("  [dim]transcript.json var, atlandı[/dim]")
    else:
        from .transcribe import transcribe as _t
        _t(vid)

    console.rule("[bold]3/6 audio")
    if done(vid, "audio_signals.json"):
        console.print("  [dim]audio_signals.json var, atlandı[/dim]")
    else:
        from .audio import analyze_audio
        analyze_audio(vid)

    console.rule("[bold]4/6 visual")
    if not visual:
        console.print("  [dim]atlandı (--visual ile açılır)[/dim]")
    elif done(vid, "visual_signals.json"):
        console.print("  [dim]visual_signals.json var, atlandı[/dim]")
    else:
        from .visual import analyze_visual
        analyze_visual(vid)

    console.rule("[bold]5/6 fuse")
    from .fuse import fuse as _fuse
    _fuse(vid)

    console.rule("[bold]6/6 analyze")
    if done(vid, "recommendations.json"):
        console.print("  [dim]recommendations.json var, atlandı[/dim]")
    else:
        from .analyze import analyze as _analyze
        fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
        _analyze(vid, formats=fmt_list, count=count, priority=priority,
                 focus=focus, exclude=exclude)

    if render_pick:
        console.rule("[bold]render")
        from .render import render as _render
        _render(vid, render_pick, layout=layout, captions=captions, intro=intro)

    console.rule("[bold green]bitti")
    console.print(f"  video_id: [cyan]{vid}[/cyan]  •  öneriler: [cyan]l2s recs {vid}[/cyan]")


@app.command()
def ingest(url: str) -> None:
    """Faz 1 — Bir YouTube URL'sini indirir ve metadatayı kaydeder."""
    from .ingest import ingest as _ingest

    _ingest(url)


@app.command()
def transcribe(video_id: str) -> None:
    """Faz 2 — Transkript üretir (mlx-whisper)."""
    from .transcribe import transcribe as _t

    _t(video_id)


@app.command()
def audio(video_id: str) -> None:
    """Faz 4 — Ses sinyallerini çıkarır."""
    from .audio import analyze_audio

    analyze_audio(video_id)


@app.command()
def visual(video_id: str) -> None:
    """Faz 5 — Görüntü sinyallerini çıkarır."""
    from .visual import analyze_visual

    analyze_visual(video_id)


@app.command()
def fuse(video_id: str) -> None:
    """Faz 5 — Tüm sinyalleri zaman-hizalı tek temsile birleştirir."""
    from .fuse import fuse as _fuse

    _fuse(video_id)


@app.command()
def analyze(
    video_id: str,
    formats: str = typer.Option("short,episode,podcast", help="Üretilecek formatlar, örn: short veya short,podcast"),
    count: int = typer.Option(None, help="Format başına hedef aday sayısı"),
    priority: str = typer.Option("balanced", help="viral | educational | emotional | balanced"),
    focus: str = typer.Option(None, help="Konu odağı, örn: girişimcilik,yatırım"),
    exclude: str = typer.Option(None, help="Atlanacak konular, örn: reklam,jenerik"),
) -> None:
    """Faz 3 — Claude ile, tercihlere göre öneri ve üretim paketleri üretir."""
    _need_anthropic()
    from .analyze import analyze as _analyze

    fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
    _analyze(video_id, formats=fmt_list, count=count, priority=priority,
             focus=focus, exclude=exclude)


@app.command()
def export(
    video_id: str,
    pick: str = typer.Option(..., help="Seçimler, örn: short:1,episode:1 | short | all"),
    vertical: bool = typer.Option(True, help="Short'larda 9:16 dikey kırpma"),
) -> None:
    """Faz 6 — Seçilen önerileri klip dosyalarına dönüştürür (ffmpeg)."""
    from .export import export as _export

    _export(video_id, pick, vertical=vertical)


@app.command()
def render(
    video_id: str,
    pick: str = typer.Option(..., help="Seçimler, örn: short:1,episode:1 | short | all"),
    layout: bool = typer.Option(True, help="Short'larda yüz-farkında dikey yerleşim"),
    captions: bool = typer.Option(True, help="Gömülü altyazı (cümle/öbek bloğu)"),
    intro: bool = typer.Option(
        True, "--cover/--no-cover", "--intro/--no-intro",
        help="Açılış kapağı (başlık + hook). Kapaksız için --no-cover"),
    pad: bool = typer.Option(True, help="Sınırları sessizliğe yasla + ses fade (ani başlangıcı önler)"),
) -> None:
    """Faz 7 — Seçilen önerileri yayınlanabilir kliplere render eder (yerleşim + altyazı + kapak)."""
    from .render import render as _render

    _render(video_id, pick, layout=layout, captions=captions, intro=intro, pad=pad)


@app.command()
def produce(
    video_id: str,
    ids: str = typer.Argument(..., help="recs'teki öneri ID'leri, örn: 14 veya 14,16,18"),
    layout: bool = typer.Option(True, help="Short'larda yüz-farkında dikey yerleşim"),
    captions: bool = typer.Option(True, help="Gömülü altyazı"),
    intro: bool = typer.Option(
        True, "--cover/--no-cover", "--intro/--no-intro",
        help="Açılış kapağı. Kapaksız üretmek için --no-cover"),
    lang: str = typer.Option(None, help="Dublajlı reel: hedef dil (en, es, fr, it, pt, hi)"),
) -> None:
    """Seçilen öneri ID'lerini yayınlanabilir kliplere üretir (recs → ID seç → üret).

    --lang verilirse dikey + dublaj sesi + hedef-dil altyazısı + marka (dublajlı reel).
    """
    if lang:                               # dublajlı reel: çeviri + (opsiyonel) konuşmacı ayrımı
        _need_anthropic()
        _warn_hf_soft()
    from .render import render as _render

    _render(video_id, ids, layout=layout, captions=captions, intro=intro, lang=lang)


@app.command()
def dub(
    video_id: str,
    rec_id: int = typer.Argument(..., help="recs'teki öneri ID'si"),
    lang: str = typer.Option("en", help="Hedef dil: en, en-gb, es, fr, it, pt, hi"),
    per_speaker: bool = typer.Option(True, help="Konuşmacı başına farklı ses (diarization; HF_TOKEN gerekir)"),
    captions: bool = typer.Option(True, help="Hedef-dil altyazı (kelime-kelime); --no-captions ile kapat"),
    speakers: int = typer.Option(None, help="Konuşmacı sayısı ipucu, örn: 2 (daha kararlı ayrım)"),
) -> None:
    """Bir öneriyi hedef dile dublajlar (çeviri + Kokoro seslendirme + altyazı + gömme)."""
    _need_anthropic()                      # çeviri için zorunlu
    if per_speaker:
        _warn_hf_soft()                    # HF/model yoksa tek-sese düşeceğini söyle
    from .dub import dub as _dub

    _dub(video_id, rec_id, lang=lang, per_speaker=per_speaker, captions=captions,
         speakers=speakers)


@app.command()
def diarize(video_id: str, force: bool = typer.Option(False, help="Yeniden çalıştır")) -> None:
    """Konuşmacı ayrımı (kim ne zaman konuştu) → speakers.json."""
    _need_hf_hard()
    from .diarize import diarize as _diarize

    _diarize(video_id, force=force)


@app.command()
def recs(video_id: str) -> None:
    """Bir video için üretilen önerileri (üretim paketleriyle) gösterir."""
    import json as _json

    rows = db.get_recommendations(video_id)
    if not rows:
        console.print("[dim]Öneri yok. Önce 'l2s analyze' çalıştır.[/dim]")
        return
    for r in rows:
        payload = _json.loads(r["payload"] or "{}")
        lens = payload.get("lens", "")
        console.print(
            f"[bold cyan]{r['fmt']}[/bold cyan] "
            f"[dim]#{r['id']}[/dim]  "
            f"{r['start_sec']:.1f}–{r['end_sec']:.1f}s  "
            f"[yellow]puan {r['score']:.0f}[/yellow]"
            + (f"  [magenta]◆ {lens}[/magenta]" if lens else "")
        )
        console.print(f"  [bold]{r['title']}[/bold]")
        if payload.get("hook"):
            console.print(f"  hook: {payload['hook']}")
        if payload.get("description"):
            console.print(f"  açıklama: {payload['description']}")
        if payload.get("reason"):
            console.print(f"  [dim]gerekçe: {payload['reason']}[/dim]")
        console.print()


@app.command()
def status(video_id: str) -> None:
    """Bir videonun boru hattı durumunu gösterir."""
    v = db.get_video(video_id)
    if v is None:
        console.print(f"[red]Video bulunamadı:[/red] {video_id}")
        raise typer.Exit(1)
    console.print(f"[bold]{v['title'] or video_id}[/bold]  [dim]({video_id})[/dim]")
    table = Table("Adım", "Durum", "Güncellendi", "Detay")
    for s in db.get_stages(video_id):
        table.add_row(s["stage"], s["status"], s["updated_at"], s["detail"] or "")
    console.print(table)


@app.command(name="list")
def list_cmd() -> None:
    """Kütüphanedeki videoları listeler."""
    rows = db.list_videos()
    if not rows:
        console.print("[dim]Kütüphane boş. Önce 'l2s ingest <url>' çalıştır.[/dim]")
        return
    table = Table("video_id", "Başlık", "Kanal", "Süre (sn)", "Alındı")
    for r in rows:
        table.add_row(
            r["video_id"], r["title"] or "", r["channel"] or "",
            f"{r['duration_sec']:.0f}" if r["duration_sec"] else "",
            r["ingested_at"],
        )
    console.print(table)


if __name__ == "__main__":
    app()
