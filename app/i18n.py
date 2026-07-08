"""Arayüz metinleri için basit iki-dilli (TR/EN) çeviri katmanı.

Kullanım:
    from . import i18n
    i18n.t("home_open")                 # aktif dilde metin
    i18n.t("log_started", name="x")     # yer tutuculu
    i18n.set_lang("en")                 # dili değiştir + kalıcı kaydet
    i18n.toggle()                       # TR <-> EN

Dil tercihi ROOT/.l2s_prefs.json içinde saklanır. Yalnızca ARAYÜZ (menü/ipucu/
yardım/bildirim/başlık) çevrilir; ağır iş süreçlerinin (indirme/transkript…) anlık
günlük satırları Türkçe kalır.
"""
from __future__ import annotations

import json

from .config import ROOT

_PREFS = ROOT / ".l2s_prefs.json"
LANGS = ("tr", "en")
_DEFAULT = "tr"


def _load_lang() -> str:
    try:
        v = json.loads(_PREFS.read_text(encoding="utf-8")).get("lang")
        return v if v in LANGS else _DEFAULT
    except Exception:  # noqa: BLE001
        return _DEFAULT


_lang = _load_lang()


def get_lang() -> str:
    return _lang


def set_lang(code: str) -> None:
    global _lang
    if code not in LANGS:
        return
    _lang = code
    try:
        _PREFS.write_text(json.dumps({"lang": code}), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def toggle() -> str:
    set_lang("en" if _lang == "tr" else "tr")
    return _lang


def t(key: str, **kw) -> str:
    """Aktif dilde metni döndürür (yer tutucular kw ile doldurulur)."""
    entry = _STR.get(key)
    if entry is None:
        return key
    s = entry.get(_lang) or entry.get(_DEFAULT) or key
    return s.format(**kw) if kw else s


# --------------------------------------------------------------------------
# Metin sözlüğü: key -> {"tr": ..., "en": ...}
# --------------------------------------------------------------------------
_STR: dict[str, dict[str, str]] = {
    # --- ortak / eylem etiketleri (Footer + yardım) ---
    "act_open": {"tr": "Aç", "en": "Open"},
    "act_new": {"tr": "Yeni proje", "en": "New project"},
    "act_delete": {"tr": "Sil", "en": "Delete"},
    "act_help": {"tr": "Yardım", "en": "Help"},
    "act_quit": {"tr": "Çıkış", "en": "Quit"},
    "act_back": {"tr": "◄ Geri", "en": "◄ Back"},
    "act_open_folder": {"tr": "Klasörü aç", "en": "Open folder"},
    "act_reveal": {"tr": "Finder'da göster", "en": "Reveal in Finder"},
    "act_copy_desktop": {"tr": "Masaüstüne kopyala", "en": "Copy to Desktop"},
    "act_finish": {"tr": "Bitir", "en": "Finish"},
    "act_cancel": {"tr": "İptal", "en": "Cancel"},
    "act_lang": {"tr": "Dil: EN", "en": "Lang: TR"},
    "lang_switched": {"tr": "Dil: Türkçe", "en": "Language: English"},
    "act_cancel_job": {"tr": "İşi iptal et", "en": "Cancel job"},
    "log_cancelled": {"tr": "işlem iptal edildi (arka plan durduruldu)",
                      "en": "operation cancelled (background stopped)"},
    "no_active_job": {"tr": "Çalışan iş yok.", "en": "No running job."},
    "confirm_cancel_job": {"tr": "Çalışan iş iptal edilsin mi?\nİlerleme kaybolabilir.",
                           "en": "Cancel the running job?\nProgress may be lost."},
    "help_cancel": {"tr": "Çalışan işi (transkript/analiz/render…) yarıda durdurur.",
                    "en": "Stops the running job (transcript/analysis/render…) midway."},

    # --- HomeScreen ---
    "home_title": {"tr": "Projeler", "en": "Projects"},
    "home_log": {"tr": "Günlük", "en": "Log"},
    "home_col_project": {"tr": "Proje", "en": "Project"},
    "home_col_status": {"tr": "Durum", "en": "Status"},
    "home_col_video": {"tr": "Video", "en": "Video"},
    "home_col_dur": {"tr": "Süre", "en": "Length"},
    "home_status_done": {"tr": "bitti ✓ arşiv", "en": "done ✓ archived"},
    "home_hint": {
        "tr": "[b]enter[/b] aç  ·  [b]n[/b] yeni  ·  [b]d[/b] sil  ·  [b]l[/b] dil  ·  [b]?[/b] yardım  ·  [b]q[/b] çıkış",
        "en": "[b]enter[/b] open  ·  [b]n[/b] new  ·  [b]d[/b] delete  ·  [b]l[/b] lang  ·  [b]?[/b] help  ·  [b]q[/b] quit",
    },
    "home_hint_empty": {
        "tr": "Henüz proje yok.  [b]n[/b] ile yeni bir video ekle  ·  [b]l[/b] dil  ·  [b]?[/b] yardım  ·  [b]q[/b] çıkış",
        "en": "No projects yet.  Add a video with [b]n[/b]  ·  [b]l[/b] lang  ·  [b]?[/b] help  ·  [b]q[/b] quit",
    },
    "home_no_key": {
        "tr": "ANTHROPIC_API_KEY yok — analiz/supercut/dublaj çalışmaz",
        "en": "no ANTHROPIC_API_KEY — analysis/supercut/dubbing won't run",
    },
    "home_help_title": {"tr": "Projeler — komutlar", "en": "Projects — commands"},
    "home_started": {"tr": "{name}: iş başladı", "en": "{name}: job started"},
    "log_downloading": {"tr": "indiriliyor  %{pct}", "en": "downloading  {pct}%"},
    "home_deleted": {"tr": "silindi: {name}", "en": "deleted: {name}"},
    "home_pick_first": {"tr": "Önce bir proje seç.", "en": "Select a project first."},
    "home_confirm_delete": {
        "tr": "'{name}' ve TÜM çıktıları silinsin mi?",
        "en": "Delete '{name}' and ALL its outputs?",
    },
    # yardım satırları (enter/n/d/l/nav/q/?)
    "help_home_open": {"tr": "Seçili projeyi açar (içindeki işlemler, öneriler, çıktılar).",
                       "en": "Opens the selected project (its operations, ideas, outputs)."},
    "help_home_new": {"tr": "Ad + YouTube URL sorar; indirip işlemeye başlar.",
                      "en": "Asks name + YouTube URL; downloads and starts processing."},
    "help_home_delete": {"tr": "Seçili projeyi ve tüm çıktılarını siler (onay ister).",
                         "en": "Deletes the selected project and all its outputs (asks to confirm)."},
    "nav_label": {"tr": "Gezin", "en": "Navigate"},
    "help_nav_updown": {"tr": "Listede gezinir.", "en": "Navigate the list."},
    "help_lang": {"tr": "Arayüz dilini TR ↔ EN değiştirir (kalıcı).",
                  "en": "Switches the interface language TR ↔ EN (persisted)."},
    "help_quit": {"tr": "Uygulamadan çıkar.", "en": "Quits the application."},
    "help_help": {"tr": "Bu paneli açar.", "en": "Opens this panel."},

    # --- ProjectScreen ---
    "proj_tab_ops": {"tr": "1 · İşlemler", "en": "1 · Operations"},
    "proj_tab_recs": {"tr": "2 · Öneriler", "en": "2 · Ideas"},
    "proj_tab_out": {"tr": "3 · Çıktılar", "en": "3 · Outputs"},
    "proj_hint_ops": {"tr": "[dim]↑/↓ ile seç, [b]enter[/b] ile o adımı (yeniden) çalıştır[/dim]",
                      "en": "[dim]↑/↓ to select, [b]enter[/b] to (re)run that step[/dim]"},
    "proj_hint_recs": {"tr": "[dim]↑/↓ ile seç, [b]enter[/b] ile seçili öneriyi render et[/dim]",
                       "en": "[dim]↑/↓ to select, [b]enter[/b] to render the selected idea[/dim]"},
    "proj_hint_out": {"tr": "[dim][b]enter[/b] oynat · [b]r[/b] Finder'da göster · [b]d[/b] masaüstüne kopyala · [b]o[/b] klasör[/dim]",
                      "en": "[dim][b]enter[/b] play · [b]r[/b] reveal in Finder · [b]d[/b] copy to Desktop · [b]o[/b] folder[/dim]"},
    "proj_log": {"tr": "İşlem günlüğü", "en": "Operation log"},
    "proj_help_title": {"tr": "{name} — komutlar", "en": "{name} — commands"},
    "proj_status_done": {"tr": "[magenta]bitti ✓ arşivlendi[/magenta]", "en": "[magenta]done ✓ archived[/magenta]"},
    "proj_status_active": {"tr": "[green]aktif[/green]", "en": "[green]active[/green]"},
    "proj_breadcrumb": {"tr": "Projeler ›", "en": "Projects ›"},
    "proj_vid_none": {"tr": "henüz yok", "en": "none yet"},
    "proj_video_none": {"tr": "Video henüz yok.", "en": "No video yet."},
    "proj_need_pipeline": {"tr": "Önce 'Tam boru hattı' ile indir/işle.",
                           "en": "First download/process via 'Full pipeline'."},
    "proj_no_url": {"tr": "URL yok.", "en": "No URL."},
    "proj_need_key_for": {"tr": "{label} için ANTHROPIC_API_KEY gerekli.",
                          "en": "{label} requires ANTHROPIC_API_KEY."},
    "proj_already_done": {"tr": "Bu proje zaten bitirildi.", "en": "This project is already finished."},
    "proj_confirm_finish": {
        "tr": "'{name}' bitirilsin mi?\nOrijinal video silinir, çıktılar zip'lenir.",
        "en": "Finish '{name}'?\nThe source video is deleted, outputs are zipped.",
    },
    # işlem adımları (ops listesi)
    "step_pipeline_label": {"tr": "Tam boru hattı", "en": "Full pipeline"},
    "step_pipeline_desc": {"tr": "İndir → transkript → ses → birleştir → analiz. Biten adımı atlar (kaldığı yerden).",
                           "en": "Download → transcript → audio → fuse → analyze. Skips finished steps (resumes)."},
    "step_transcribe_label": {"tr": "Transkript", "en": "Transcript"},
    "step_transcribe_desc": {"tr": "Konuşmayı kelime düzeyinde metne çevirir (mlx-whisper).",
                             "en": "Transcribes speech to word-level text (mlx-whisper)."},
    "step_audio_label": {"tr": "Ses sinyalleri", "en": "Audio signals"},
    "step_audio_desc": {"tr": "Enerji eğrisi + duraklama tespiti (librosa).",
                        "en": "Energy curve + pause detection (librosa)."},
    "step_fuse_label": {"tr": "Birleştirme", "en": "Fuse"},
    "step_fuse_desc": {"tr": "Tüm sinyalleri zaman-hizalı tek temsile toplar.",
                       "en": "Merges all signals into one time-aligned representation."},
    "step_analyze_label": {"tr": "Analiz (öneriler)", "en": "Analysis (ideas)"},
    "step_analyze_desc": {"tr": "Claude ile short/episode/podcast önerileri üretir.",
                          "en": "Generates short/episode/podcast ideas with Claude."},
    "step_supercut_label": {"tr": "Supercut öner", "en": "Suggest supercut"},
    "step_supercut_desc": {"tr": "Farklı anları bir anlatıya dizen montaj önerileri (Claude ×3).",
                           "en": "Montage ideas weaving moments into a narrative (Claude ×3)."},
    # rozetler / durumlar
    "badge_done": {"tr": "[green]✓ yapıldı[/green]", "en": "[green]✓ done[/green]"},
    "badge_pending": {"tr": "[yellow]○ bekliyor[/yellow]", "en": "[yellow]○ pending[/yellow]"},
    "act_run_again": {"tr": "[dim]enter: yeniden çalıştır[/dim]", "en": "[dim]enter: run again[/dim]"},
    "act_run": {"tr": "[dim]enter: çalıştır[/dim]", "en": "[dim]enter: run[/dim]"},
    "act_run_full": {"tr": "[cyan]enter: baştan sona çalıştır[/cyan]", "en": "[cyan]enter: run end-to-end[/cyan]"},
    "recs_rendered": {"tr": "üretildi", "en": "rendered"},
    "recs_empty": {"tr": "[dim]Öneri yok. İşlemler'de 'Analiz' / 'Supercut' çalıştır.[/dim]",
                   "en": "[dim]No ideas. Run 'Analysis' / 'Supercut' under Operations.[/dim]"},
    "outs_empty": {"tr": "[dim]Henüz çıktı yok. Öneriler'den render et.[/dim]",
                   "en": "[dim]No outputs yet. Render from Ideas.[/dim]"},
    "out_archive": {"tr": "arşiv", "en": "archive"},
    "out_moved": {"tr": "Dosya taşınmış/silinmiş; liste tazelendi.",
                  "en": "File moved/deleted; list refreshed."},
    "out_pick_first": {"tr": "Önce Çıktılar sekmesinde bir dosya seç.",
                       "en": "First select a file in the Outputs tab."},
    "out_opened": {"tr": "açıldı: {name}", "en": "opened: {name}"},
    "out_in_finder": {"tr": "Finder'da: {name}", "en": "In Finder: {name}"},
    "out_copy_fail": {"tr": "Kopyalanamadı: {err}", "en": "Copy failed: {err}"},
    "out_copied": {"tr": "masaüstüne kopyalandı: {name}  (orijinal yerinde)",
                   "en": "copied to Desktop: {name}  (original kept)"},
    # yardım satırları (proje)
    "help_proj_enter": {"tr": "İşlemler'de: adımı (yeniden) çalıştırır. Öneriler'de: render açar. Çıktılar'da: dosyayı oynatır.",
                        "en": "Operations: (re)runs the step. Ideas: opens render. Outputs: plays the file."},
    "help_proj_folder": {"tr": "Projenin klasörünü Finder'da açar.", "en": "Opens the project folder in Finder."},
    "help_proj_reveal": {"tr": "Çıktılar'da seçili dosyayı Finder'da seçili açar (oradan taşıyabilirsin — taşımak güvenli).",
                         "en": "Reveals the selected output in Finder (move it from there — moving is safe)."},
    "help_proj_copy": {"tr": "Çıktılar'da seçili dosyanın kopyasını masaüstüne koyar ve açar (orijinal yerinde kalır).",
                       "en": "Copies the selected output to the Desktop and opens it (original is kept)."},
    "help_proj_finish": {"tr": "Orijinal videoyu siler, çıktıları + meta zip'ler, 'done' işaretler.",
                         "en": "Deletes the source video, zips outputs + meta, marks 'done'."},
    "help_proj_back": {"tr": "Proje listesine döner.", "en": "Returns to the project list."},
    "help_proj_nav": {"tr": "Liste içinde gezinir; tab ile sekme değiştirir.",
                      "en": "Navigate the list; switch tabs with tab."},

    # proje durum özeti (_proj_status)
    "ps_starting": {"tr": "başlıyor…", "en": "starting…"},
    "ps_ready": {"tr": "analiz ✓ öneriler hazır", "en": "analysis ✓ ideas ready"},
    "ps_running": {"tr": "{stage}…", "en": "{stage}…"},
    "ps_error": {"tr": "{stage} HATA", "en": "{stage} ERROR"},
    "ps_done": {"tr": "{stage} ✓", "en": "{stage} ✓"},

    # --- modallar ---
    "new_title": {"tr": "Yeni proje", "en": "New project"},
    "new_name_ph": {"tr": "Proje adı (ör. dürtü-kontrolü)", "en": "Project name (e.g. impulse-control)"},
    "new_url_ph": {"tr": "YouTube URL", "en": "YouTube URL"},
    "new_start": {"tr": "Başlat", "en": "Start"},
    "new_need_both": {"tr": "Proje adı ve URL gerekli.", "en": "Project name and URL are required."},
    "confirm_yes": {"tr": "Evet", "en": "Yes"},
    "confirm_no": {"tr": "Hayır", "en": "No"},
    "render_title": {"tr": "Render", "en": "Render"},
    "render_duration": {"tr": "süre", "en": "length"},
    "render_hook": {"tr": "Kanca", "en": "Hook"},
    "render_coherence": {"tr": "tutarlılık", "en": "coherence"},
    "render_parts": {"tr": "parça", "en": "parts"},
    "render_note": {"tr": "not", "en": "note"},
    "render_cover": {"tr": "Açılış kapağı", "en": "Intro cover"},
    "render_captions": {"tr": "Altyazı", "en": "Captions"},
    "render_layout": {"tr": "Yüz-farkında yerleşim", "en": "Face-aware layout"},
    "render_xfade": {"tr": "Güçlü geçiş (supercut) — kapalıyken de nazik geçiş var",
                     "en": "Strong transition (supercut) — gentle by default"},
    "render_caplang_ph": {"tr": "Altyazı dili: boş=orijinal, en/es/fr/it/pt/hi",
                          "en": "Caption language: empty=original, en/es/fr/it/pt/hi"},
    "render_lang_ph": {"tr": "Dublaj dili: en, es, fr, it, pt, hi — boş=yok",
                       "en": "Dubbing language: en, es, fr, it, pt, hi — empty=none"},
    "render_need_key_dub": {"tr": "Dublaj için ANTHROPIC_API_KEY gerekli.",
                            "en": "Dubbing requires ANTHROPIC_API_KEY."},
    "render_need_key_cap": {"tr": "Altyazı çevirisi için ANTHROPIC_API_KEY gerekli.",
                            "en": "Caption translation requires ANTHROPIC_API_KEY."},
    "help_outputs_path": {"tr": "\n[dim]Çıktılar: ~/Documents/lts/<video_id>/ciktilar/[/dim]",
                          "en": "\n[dim]Outputs: ~/Documents/lts/<video_id>/ciktilar/[/dim]"},
    "help_close": {"tr": "Kapat", "en": "Close"},
}
