"""Terminal uygulaması (Textual) — long-to-shorts'u iç içe ekranlarla yönet.

`l2s ui` ile açılır. İki katman:
  • Ana ekran (HomeScreen): proje listesi — ekle / sil / aç.
  • Proje ekranı (ProjectScreen): bir projenin içi — sekmeler:
      İşlemler  (yapılan/yapılacak adımlar; her biri tekrar çalıştırılabilir)
      Öneriler  (Claude önerileri; seçip render'la)
      Çıktılar  (üretilen dosyalar; seçip aç)

Ağır işler (indirme, transkript, analiz, render, ...) AYRI bir süreçte çalışır
(bkz. app/jobs.py) — Textual'ın std-akış yönetimiyle çakışmayı önler; UI ile
yalnızca bir kuyruk üzerinden konuşulur. İlerleme/günlük o an açık ekranda gösterilir.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from rich.console import Console as _RichConsole

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button, Checkbox, DataTable, Footer, Header, Input, Label, ListItem, ListView,
    ProgressBar, RichLog, Static, TabbedContent, TabPane,
)

from . import db, i18n
from .config import ANTHROPIC_API_KEY, BASE_DIR, ensure_dirs, video_dir
from .i18n import t

# İşlem adımı → çıktı dosyası (durum ve "tekrar" için). Etiket/açıklama i18n'de:
# step_<key>_label / step_<key>_desc
_STEPS = [
    ("pipeline", None),
    ("transcribe", "transcript.json"),
    ("audio", "audio_signals.json"),
    ("fuse", "fused.json"),
    ("analyze", "recommendations.json"),
    ("supercut", "supercut.json"),
]
_NEEDS_API = {"pipeline", "analyze", "supercut"}


def _silence_pipeline_consoles() -> None:
    """Ana süreçte alt modül çıktısını sustur (çocuk süreç zaten susturuluyor)."""
    null = _RichConsole(file=open(os.devnull, "w"))
    from . import analyze, audio, dub, fuse, ingest, render, supercut, transcribe
    for m in (ingest, transcribe, audio, fuse, analyze, render, supercut, dub):
        m.console = null


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1024:.1f}{unit}"
        n /= 1024
    return f"{n:.0f}B"


# --------------------------- modallar --------------------------------------

class NewJobScreen(ModalScreen):
    """Yeni proje: ad + YouTube URL."""

    BINDINGS = [("escape", "cancel", "İptal")]

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(f"[b]{t('new_title')}[/b]", id="dtitle")
            yield Input(placeholder=t("new_name_ph"), id="name")
            yield Input(placeholder=t("new_url_ph"), id="url")
            with Horizontal(id="dbtns"):
                yield Button(t("new_start"), variant="primary", id="start")
                yield Button(t("act_cancel"), id="cancel")

    @on(Button.Pressed, "#start")
    def _start(self) -> None:
        name = self.query_one("#name", Input).value.strip()
        url = self.query_one("#url", Input).value.strip()
        if not name or not url:
            self.app.notify(t("new_need_both"), severity="warning")
            return
        self.dismiss({"name": name, "url": url})

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)


class ConfirmScreen(ModalScreen):
    """Evet/Hayır onayı (silme, bitirme gibi geri alınamaz işler için)."""

    BINDINGS = [("escape", "cancel", "İptal")]

    def action_cancel(self) -> None:
        self.dismiss(False)

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.question, id="dtitle")
            with Horizontal(id="dbtns"):
                yield Button(t("confirm_yes"), variant="error", id="yes")
                yield Button(t("confirm_no"), id="no")

    @on(Button.Pressed, "#yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no(self) -> None:
        self.dismiss(False)


class RenderScreen(ModalScreen):
    """Bir öneriyi render seçenekleriyle üret."""

    BINDINGS = [("escape", "cancel", "İptal")]

    def action_cancel(self) -> None:
        self.dismiss(None)

    def __init__(self, rec) -> None:
        super().__init__()
        self.rec = rec

    def _detail_text(self) -> str:
        """Öneri detayları: hook/açıklama + (supercut) tutarlılık, not, parça rolleri."""
        try:
            p = json.loads(self.rec["payload"] or "{}")
        except Exception:  # noqa: BLE001
            p = {}
        lines = []
        dur = (self.rec["end_sec"] - self.rec["start_sec"])
        lines.append(f"[dim]{t('render_duration')}: {dur:.0f}s[/dim]")
        if p.get("hook"):
            lines.append(f"[b]{t('render_hook')}:[/b] {p['hook']}")
        if p.get("description"):
            lines.append(f"[dim]{p['description']}[/dim]")
        if self.rec["fmt"] == "supercut":
            coh = p.get("coherence")
            spans = p.get("spans") or []
            roles = " → ".join(s.get("role", "?") for s in spans)
            if coh is not None:
                lines.append(f"[magenta]{t('render_coherence')}: {coh:.0f}[/magenta]  "
                             f"[dim]({len(spans)} {t('render_parts')})[/dim]")
            if roles:
                lines.append(f"[dim]{roles}[/dim]")
            if p.get("coherence_note"):
                lines.append(f"[dim]{t('render_note')}: {p['coherence_note']}[/dim]")
        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        r = self.rec
        with Vertical(id="dialog"):
            yield Static(f"[b]{t('render_title')}[/b]: {r['fmt']} #{r['id']}\n{r['title'] or ''}", id="dtitle")
            yield Static(self._detail_text(), id="rdetail")
            yield Checkbox(t("render_cover"), True, id="cover")
            yield Checkbox(t("render_captions"), True, id="captions")
            yield Checkbox(t("render_layout"), True, id="layout")
            yield Checkbox(t("render_xfade"), False, id="xfade")
            yield Input(placeholder=t("render_caplang_ph"), id="caplang")
            yield Input(placeholder=t("render_lang_ph"), id="lang")
            with Horizontal(id="dbtns"):
                yield Button(t("render_title"), variant="primary", id="go")
                yield Button(t("act_cancel"), id="cancel")

    @on(Button.Pressed, "#go")
    def _go(self) -> None:
        self.dismiss({
            "cover": self.query_one("#cover", Checkbox).value,
            "captions": self.query_one("#captions", Checkbox).value,
            "layout": self.query_one("#layout", Checkbox).value,
            "xfade": 0.3 if self.query_one("#xfade", Checkbox).value else 0.0,
            "caplang": self.query_one("#caplang", Input).value.strip() or None,
            "lang": self.query_one("#lang", Input).value.strip() or None,
        })

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)


class HelpScreen(ModalScreen):
    """Komut açıklamaları (bağlama göre)."""

    BINDINGS = [("escape", "dismiss", "Kapat"), ("question_mark", "dismiss", "Kapat")]

    def __init__(self, rows: list[tuple[str, str, str]], title: str) -> None:
        super().__init__()
        self.rows = rows
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(f"[b]{self.title_text}[/b]", id="dtitle")
            lines = [f"  [b cyan]{k:>6}[/b cyan]  [b]{n}[/b]\n        [dim]{d}[/dim]"
                     for k, n, d in self.rows]
            yield Static("\n".join(lines))
            yield Static(t("help_outputs_path"))
            with Horizontal(id="dbtns"):
                yield Button(t("help_close"), variant="primary", id="close")

    @on(Button.Pressed, "#close")
    def _close(self) -> None:
        self.dismiss(None)


# --------------------------- ana ekran (projeler) --------------------------

def _home_bindings():
    return [
        ("n", "new_job", t("act_new")),
        ("d", "delete", t("act_delete")),
        ("enter", "open", t("act_open")),
        # öncelikli: DataTable/ListView odaktayken bile 'c' iptali tetiklesin
        Binding("c", "cancel_job", t("act_cancel_job"), priority=True),
        ("l", "switch_lang", t("act_lang")),
        ("question_mark", "help", t("act_help")),
        ("q", "quit_app", t("act_quit")),
    ]


def _proj_bindings():
    return [
        ("escape", "back", t("act_back")),
        ("o", "open_folder", t("act_open_folder")),
        ("r", "reveal_out", t("act_reveal")),
        ("d", "copy_desktop", t("act_copy_desktop")),
        Binding("c", "cancel_job", t("act_cancel_job"), priority=True),
        ("l", "switch_lang", t("act_lang")),
        ("x", "finish", t("act_finish")),
        ("question_mark", "help", t("act_help")),
        ("q", "quit_app", t("act_quit")),
    ]


class HomeScreen(Screen):
    BINDINGS = _home_bindings()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="home-hint")
        yield DataTable(id="ptable", cursor_type="row", zebra_stripes=True)
        # adımlar + indirme yüzdesi log'a yazılır; bar log'un ALTINDA (araya girmez)
        yield RichLog(id="log", markup=True, highlight=False)
        yield ProgressBar(id="pbar", show_eta=False, total=100)
        yield Footer()

    def on_mount(self) -> None:
        tbl = self.query_one("#ptable", DataTable)
        tbl.add_columns(t("home_col_project"), t("home_col_status"),
                        t("home_col_video"), t("home_col_dur"))
        tbl.border_title = t("home_title")
        self.query_one("#pbar", ProgressBar).display = False
        self.query_one("#log", RichLog).border_title = t("home_log")
        self.refresh_data()
        if not ANTHROPIC_API_KEY:
            self.app._emit("warn", t("home_no_key"))

    def refresh_data(self) -> None:
        tbl = self.query_one("#ptable", DataTable)
        row = tbl.cursor_row
        tbl.clear()
        self._projects = list(db.list_projects())
        for p in self._projects:
            status = (t("home_status_done") if p["status"] == "done"
                      else _proj_status(p["video_id"]))
            dur = f"{p['duration_sec']:.0f}s" if p["duration_sec"] else "—"
            tbl.add_row(p["name"], status, (p["title"] or p["url"] or "")[:30], dur)
        if self._projects and 0 <= row < len(self._projects):
            tbl.move_cursor(row=row)
        hint = self.query_one("#home-hint", Static)
        hint.update(t("home_hint") if self._projects else t("home_hint_empty"))

    def _selected(self):
        t = self.query_one("#ptable", DataTable)
        if self._projects and t.cursor_row is not None and t.cursor_row < len(self._projects):
            return self._projects[t.cursor_row]
        return None

    @on(DataTable.RowSelected, "#ptable")
    def _row_selected(self) -> None:
        self.action_open()

    def action_open(self) -> None:
        row = self._selected()
        if row:
            self.app.push_screen(ProjectScreen(row["name"]))

    def action_new_job(self) -> None:
        def cb(res):
            if res:
                db.create_project(res["name"], url=res["url"])
                self.refresh_data()
                self.app._emit("info", t("home_started", name=f"[b]{res['name']}[/b]"))
                self.app.spawn(self, "pipeline", (res["name"], res["url"]))
        self.app.push_screen(NewJobScreen(), cb)

    def action_delete(self) -> None:
        row = self._selected()
        if not row:
            self.app.notify(t("home_pick_first"), severity="warning")
            return
        name = row["name"]

        def cb(ok):
            if ok:
                from .lifecycle import delete_project
                delete_project(name)
                self.refresh_data()
                self.app._emit("info", t("home_deleted", name=name))
        self.app.push_screen(ConfirmScreen(t("home_confirm_delete", name=name)), cb)

    def action_help(self) -> None:
        rows = [
            ("enter", t("act_open"), t("help_home_open")),
            ("n", t("act_new"), t("help_home_new")),
            ("d", t("act_delete"), t("help_home_delete")),
            ("↑/↓", t("nav_label"), t("help_nav_updown")),
            ("c", t("act_cancel_job"), t("help_cancel")),
            ("l", t("act_lang"), t("help_lang")),
            ("q", t("act_quit"), t("help_quit")),
            ("?", t("act_help"), t("help_help")),
        ]
        self.app.push_screen(HelpScreen(rows, t("home_help_title")))

    def action_switch_lang(self) -> None:
        self.app.switch_language()

    def action_cancel_job(self) -> None:
        self.app.request_cancel()

    def action_quit_app(self) -> None:
        self.app.exit()


# --------------------------- proje ekranı ----------------------------------

class ProjectScreen(Screen):
    BINDINGS = _proj_bindings()

    def __init__(self, name: str) -> None:
        super().__init__()
        self.pname = name
        self._recs: list = []
        self._outs: list = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("", id="pinfo")
        with TabbedContent(id="tabs"):
            with TabPane(t("proj_tab_ops"), id="tab-ops"):
                yield Static(t("proj_hint_ops"), classes="hint")
                yield ListView(id="ops")
            with TabPane(t("proj_tab_recs"), id="tab-recs"):
                yield Static(t("proj_hint_recs"), classes="hint")
                yield ListView(id="recs")
            with TabPane(t("proj_tab_out"), id="tab-out"):
                yield Static(t("proj_hint_out"), classes="hint")
                yield ListView(id="outs")
        # adımlar + indirme yüzdesi log'a yazılır; bar log'un ALTINDA (araya girmez)
        yield RichLog(id="log", markup=True, highlight=False)
        yield ProgressBar(id="pbar", show_eta=False, total=100)
        yield Footer()

    # sekme kimliği → içindeki liste (klavye odağı için)
    _TAB_LIST = {"tab-ops": "#ops", "tab-recs": "#recs", "tab-out": "#outs"}

    def on_mount(self) -> None:
        self.query_one("#pbar", ProgressBar).display = False
        self.query_one("#log", RichLog).border_title = t("proj_log")
        self.refresh_data()
        self._focus_active_list()

    def _focus_active_list(self) -> None:
        """Aktif sekmenin listesine klavye odağını ver (enter/↑↓ çalışsın)."""
        active = self.query_one("#tabs", TabbedContent).active
        lid = self._TAB_LIST.get(active)
        if lid:
            try:
                self.query_one(lid, ListView).focus()
            except Exception:  # noqa: BLE001
                pass

    @on(TabbedContent.TabActivated)
    def _on_tab(self, event: TabbedContent.TabActivated) -> None:
        self._focus_active_list()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_switch_lang(self) -> None:
        self.app.switch_language()

    def action_cancel_job(self) -> None:
        self.app.request_cancel()

    def _row(self):
        return db.get_project(self.pname)

    @staticmethod
    def _rendered_langs(vid: str | None, fmt: str, rec_id: int) -> set[str]:
        """Bu öneri için üretilmiş varyantlar: {'tr','en',...}.

        Dublaj çıktıları 'dublajlar/' klasörüne, orijinal (tr) format klasörüne
        gider → klasöre bakarak varyantı güvenle çıkarırız. Dosya adı deseni:
        '<fmt>_<id>_<lang?>_<baslik>.mp4|.m4a'.
        """
        out: set[str] = set()
        if not vid:
            return out
        root = BASE_DIR / vid / "ciktilar"
        if not root.exists():
            return out
        prefix = f"{fmt}_{rec_id}_"
        for f in root.rglob(f"{prefix}*"):
            if not f.is_file() or f.suffix not in (".mp4", ".m4a"):
                continue
            if f.parent.name == "dublajlar":
                out.add(f.name[len(prefix):].split("_")[0])   # <lang>
            else:
                out.add("tr")
        return out

    def refresh_data(self) -> None:
        row = self._row()
        if row is None:
            return
        vid = row["video_id"]
        v = db.get_video(vid) if vid else None
        title = (v["title"] if v else None) or row["url"] or "—"
        st = t("proj_status_done") if row["status"] == "done" else t("proj_status_active")
        self.query_one("#pinfo", Static).update(
            f"[dim]{t('proj_breadcrumb')}[/dim] [b]{row['name']}[/b]   {st}\n"
            f"[dim]{title}   ·   video_id: {vid or t('proj_vid_none')}[/dim]")

        vdir = video_dir(vid) if vid else None

        # İşlemler — her adım durumuyla; enter ile (yeniden) çalışır
        ops = self.query_one("#ops", ListView)
        ops.clear()
        for key, fname in _STEPS:
            label, desc = t(f"step_{key}_label"), t(f"step_{key}_desc")
            if key == "pipeline":
                badge, act = "[cyan]▶[/cyan]", t("act_run_full")
            else:
                done = bool(vid and fname and (vdir / fname).exists())
                badge = t("badge_done") if done else t("badge_pending")
                act = t("act_run_again") if done else t("act_run")
            ops.append(ListItem(Label(f"{badge}  [b]{label}[/b]   {act}\n     [dim]{desc}[/dim]")))

        # Öneriler — render öneriyi SİLMEZ; aynı öneri farklı dillerde tekrar
        # üretilebilir. Üretilen varyantlar satırda gösterilir, seçim korunur.
        recs = self.query_one("#recs", ListView)
        prev = recs.index                       # render sonrası seçim kaybolmasın
        recs.clear()
        self._recs = list(db.get_recommendations(vid)) if vid else []
        for r in self._recs:
            langs = self._rendered_langs(vid, r["fmt"], r["id"])
            done = (f"  [green]▸ {t('recs_rendered')}: {', '.join(sorted(langs))}[/green]"
                    if langs else "")
            coh = ""
            if r["fmt"] == "supercut":                     # tutarlılık rozeti (Faz 2 kapısı)
                try:
                    c = json.loads(r["payload"] or "{}").get("coherence")
                except Exception:  # noqa: BLE001
                    c = None
                if c is not None:
                    coh = f"  [magenta]{t('render_coherence')} {c:.0f}[/magenta]"
            recs.append(ListItem(Label(
                f"[cyan]{r['fmt']}[/cyan] [dim]#{r['id']}[/dim] "
                f"[yellow]{r['score']:.0f}[/yellow]  {r['title'] or ''}{coh}{done}")))
        if not self._recs:
            recs.append(ListItem(Label(t("recs_empty"))))
        elif prev is not None:
            tgt = min(prev, len(self._recs) - 1)
            self.call_after_refresh(lambda: setattr(recs, "index", tgt))

        # Çıktılar
        outs = self.query_one("#outs", ListView)
        outs.clear()
        self._outs = []
        if vid:
            cikti = BASE_DIR / vid / "ciktilar"
            if cikti.exists():
                for f in sorted(cikti.rglob("*")):
                    if f.is_file() and not f.name.startswith("."):
                        self._outs.append(f)
                        rel = f.relative_to(cikti)
                        outs.append(ListItem(Label(
                            f"[cyan]{rel.parent}/[/cyan]{f.name}  "
                            f"[dim]{_human_size(f.stat().st_size)}[/dim]")))
            zip_f = BASE_DIR / vid / f"{vid}.zip"
            if zip_f.exists():
                self._outs.append(zip_f)
                outs.append(ListItem(Label(
                    f"[magenta]{t('out_archive')}[/magenta] {zip_f.name}  "
                    f"[dim]{_human_size(zip_f.stat().st_size)}[/dim]")))
        if not self._outs:
            outs.append(ListItem(Label(t("outs_empty"))))

    # ---- sekmelere göre 'enter' davranışı ----

    @on(ListView.Selected, "#ops")
    def _run_op(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None or idx >= len(_STEPS):
            return
        key, _fname = _STEPS[idx]
        row = self._row()
        vid = row["video_id"] if row else None
        if key in _NEEDS_API and not ANTHROPIC_API_KEY:
            self.app.notify(t("proj_need_key_for", label=t(f"step_{key}_label")), severity="error")
            return
        if key == "pipeline":
            if not row["url"]:
                self.app.notify(t("proj_no_url"), severity="warning")
                return
            self.app.spawn(self, "pipeline", (self.pname, row["url"]))
        else:
            if not vid:
                self.app.notify(t("proj_need_pipeline"), severity="warning")
                return
            self.app.spawn(self, key, (vid,))

    @on(ListView.Selected, "#recs")
    def _render_rec(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None or idx >= len(self._recs):
            return
        rec = self._recs[idx]

        def cb(opts):
            if opts:
                if opts["lang"] and not ANTHROPIC_API_KEY:
                    self.app.notify(t("render_need_key_dub"), severity="error")
                    return
                if opts.get("caplang") and not opts["lang"] and not ANTHROPIC_API_KEY:
                    self.app.notify(t("render_need_key_cap"), severity="error")
                    return
                self.app.spawn(self, "render", (rec["video_id"], rec["id"], opts))
        self.app.push_screen(RenderScreen(rec), cb)

    @on(ListView.Selected, "#outs")
    def _open_out(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None or idx >= len(self._outs):
            return
        f = self._outs[idx]
        if not f.exists():                       # dışarıdan taşınmış → listeyi tazele
            self.app.notify(t("out_moved"), severity="warning")
            self.refresh_data()
            return
        subprocess.run(["open", str(f)], check=False)   # macOS: oynat
        self.app._emit("info", t("out_opened", name=f.name))

    def _selected_out(self) -> Path | None:
        """Çıktılar sekmesinde seçili dosya (yoksa None). Taşınmışsa listeyi tazeler."""
        try:
            outs = self.query_one("#outs", ListView)
        except Exception:  # noqa: BLE001
            return None
        idx = outs.index
        if idx is None or idx >= len(self._outs):
            return None
        f = self._outs[idx]
        if not f.exists():
            self.app.notify(t("out_moved"), severity="warning")
            self.refresh_data()
            return None
        return f

    # ---- eylemler ----

    def action_reveal_out(self) -> None:
        f = self._selected_out()
        if f is None:
            self.app.notify(t("out_pick_first"), severity="warning")
            return
        subprocess.run(["open", "-R", str(f)], check=False)   # Finder'da seçili göster
        self.app._emit("info", t("out_in_finder", name=f.name))

    def action_copy_desktop(self) -> None:
        f = self._selected_out()
        if f is None:
            self.app.notify(t("out_pick_first"), severity="warning")
            return
        dest = Path.home() / "Desktop" / f.name
        try:
            shutil.copy2(f, dest)
        except OSError as exc:
            self.app.notify(t("out_copy_fail", err=exc), severity="error")
            return
        subprocess.run(["open", str(dest)], check=False)      # kopyayı aç/oynat
        self.app._emit("ok", t("out_copied", name=dest.name))

    def action_open_folder(self) -> None:
        row = self._row()
        if not row or not row["video_id"]:
            self.app.notify(t("proj_video_none"), severity="warning")
            return
        subprocess.run(["open", str(BASE_DIR / row["video_id"])], check=False)

    def action_finish(self) -> None:
        row = self._row()
        if not row or not row["video_id"]:
            self.app.notify(t("proj_video_none"), severity="warning")
            return
        if row["status"] == "done":
            self.app.notify(t("proj_already_done"), severity="warning")
            return

        def cb(ok):
            if ok:
                self.app.spawn(self, "finish", (row["video_id"], self.pname))
        self.app.push_screen(ConfirmScreen(t("proj_confirm_finish", name=self.pname)), cb)

    def action_help(self) -> None:
        rows = [
            ("enter", t("act_open"), t("help_proj_enter")),
            ("o", t("act_open_folder"), t("help_proj_folder")),
            ("r", t("act_reveal"), t("help_proj_reveal")),
            ("d", t("act_copy_desktop"), t("help_proj_copy")),
            ("x", t("act_finish"), t("help_proj_finish")),
            ("esc", t("act_back"), t("help_proj_back")),
            ("↑/↓ · tab", t("nav_label"), t("help_proj_nav")),
            ("c", t("act_cancel_job"), t("help_cancel")),
            ("l", t("act_lang"), t("help_lang")),
            ("?", t("act_help"), t("help_help")),
            ("q", t("act_quit"), t("help_quit")),
        ]
        self.app.push_screen(HelpScreen(rows, t("proj_help_title", name=self.pname)))

    def action_back(self) -> None:
        self.app.pop_screen()


def _proj_status(video_id: str | None) -> str:
    if not video_id:
        return t("ps_starting")
    stages = {s["stage"]: s["status"] for s in db.get_stages(video_id)}
    if stages.get("analyze") == "done":
        return t("ps_ready")
    order = ["ingest", "transcribe", "audio", "fuse", "analyze"]
    running = next((s for s in order if stages.get(s) == "running"), None)
    if running:
        return t("ps_running", stage=running)
    err = next((s for s in order if stages.get(s) == "error"), None)
    if err:
        return t("ps_error", stage=err)
    done = [s for s in order if stages.get(s) == "done"]
    return t("ps_done", stage=done[-1]) if done else t("ps_starting")


# --------------------------- uygulama --------------------------------------

class L2SApp(App):
    TITLE = "long-to-shorts"

    CSS = """
    #home-hint { padding: 1 2; color: $text-muted; }
    #ptable { height: 1fr; border: round $primary; padding: 0 1; }
    #pinfo { padding: 1 2; height: auto; background: $boost; }
    #tabs { height: 1fr; }
    .hint { padding: 1 2 0 2; color: $text-muted; height: auto; }
    ListView { height: 1fr; padding: 0 1; }
    ListItem { padding: 0 1; }
    #pbar { margin: 0 2; height: auto; }
    #log { height: 8; border: round $accent; padding: 0 1; }
    ModalScreen { align: center middle; }
    #dialog { width: 76; height: auto; padding: 1 2; border: thick $primary; background: $surface; }
    #dtitle { padding-bottom: 1; }
    #rdetail { padding-bottom: 1; color: $text-muted; height: auto; }
    #dbtns { height: auto; margin-top: 1; }
    #dbtns Button { margin-right: 2; }
    """

    _procs: list = []          # çalışan çocuk süreçler (iptal için)

    def on_mount(self) -> None:
        ensure_dirs()
        db.init_db()
        _silence_pipeline_consoles()
        self._procs = []
        self.push_screen(HomeScreen())

    def request_cancel(self) -> None:
        """İptal öncesi onay ister (iş varsa). İş yoksa yalnızca uyarır."""
        if not any(p.is_alive() for p in self._procs):
            self.notify(i18n.t("no_active_job"), severity="warning")
            return

        def cb(ok):
            if ok:
                self.cancel_active_jobs()
        self.push_screen(ConfirmScreen(i18n.t("confirm_cancel_job")), cb)

    def cancel_active_jobs(self) -> None:
        """Çalışan işi(leri) durdurur: çocuk süreç grubunu (ffmpeg dahil) sonlandırır.

        Worker döngüsü ~1 sn içinde sürecin öldüğünü görüp kapanır ve _step_done çağrılır."""
        import os
        import signal

        procs = [p for p in self._procs if p.is_alive()]
        if not procs:
            self.notify(i18n.t("no_active_job"), severity="warning")
            return
        for p in procs:
            try:                                   # çocuk setsid ile grup lideri → tüm grubu öldür
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                p.terminate()                      # yedek: yalnız süreci sonlandır
        self._emit("warn", i18n.t("log_cancelled"))

    def switch_language(self) -> None:
        """Arayüz dilini TR ↔ EN değiştirir; binding'leri ve ekranları yeniden kurar.

        Footer etiketleri sınıf düzeyi BINDINGS'ten okunduğu için, dili değiştirip
        BINDINGS'i yeniden atıyor ve ekran yığınını taze örneklerle yeniden kuruyoruz
        (yeni örnekler güncellenmiş sınıf BINDINGS'ini okur)."""
        i18n.toggle()
        # Textual, birleştirilmiş binding'leri sınıf tanımında bir kez hesaplayıp
        # _merged_bindings (ClassVar) olarak önbelleğe alır; Footer bunu okur.
        # Dil değişince hem BINDINGS'i güncelle hem de bu önbelleği yeniden kur.
        HomeScreen.BINDINGS = _home_bindings()
        ProjectScreen.BINDINGS = _proj_bindings()
        HomeScreen._merged_bindings = HomeScreen._merge_bindings()
        ProjectScreen._merged_bindings = ProjectScreen._merge_bindings()
        # yığındaki yeri koru: bir projedeyse aynı projeye geri dön
        pname = getattr(self.screen, "pname", None)
        while len(self.screen_stack) > 1:      # push edilen ekranları kaldır (base kalır)
            self.pop_screen()
        self.push_screen(HomeScreen())
        if pname:
            self.push_screen(ProjectScreen(pname))
        self.notify(i18n.t("lang_switched"))

    # ---- o an açık ekrandaki widget'lara güvenli erişim ----

    def _w(self, wid: str, wtype):
        try:
            return self.screen.query_one(wid, wtype)
        except Exception:  # noqa: BLE001
            return None

    # ---- ORTAK GÜNLÜK BİÇİMİ: "SS:DD:ss  <simge> <mesaj>" ----
    _LOG = {
        "step":  ("▸", "cyan"),      # adım başladı
        "ok":    ("✓", "green"),     # tamamlandı
        "info":  ("·", "dim"),       # bilgi / atlandı
        "warn":  ("⚠", "yellow"),    # uyarı
        "error": ("✗", "red"),       # hata
    }

    def _emit(self, kind: str, msg: str) -> None:
        """Tüm günlük satırları buradan geçer → tek biçim (zaman + simge + renk)."""
        glyph, color = self._LOG.get(kind, self._LOG["info"])
        ts = datetime.now().strftime("%H:%M:%S")
        w = self._w("#log", RichLog)
        if w:
            w.write(f"[dim]{ts}[/dim]  [{color}]{glyph}[/{color}] {msg}")

    def _log(self, msg: str) -> None:               # geriye dönük: düz bilgi satırı
        self._emit("info", msg)

    def _step(self, label: str) -> None:
        self._emit("step", label)                   # adım GÜNLÜĞE yazılır (geçmiş kalır)
        self._dl_bucket = -1                        # indirme kilometre taşı sayacını sıfırla
        pb = self._w("#pbar", ProgressBar)
        if pb:
            pb.display = True
            pb.update(total=None)                   # belirsiz (spinner)

    def _download(self, pct: float) -> None:
        pb = self._w("#pbar", ProgressBar)
        if pb:
            pb.display = True
            pb.update(total=100, progress=pct)
        bucket = int(pct // 20)                      # her %20'de bir günlüğe yaz (spam olmasın)
        if bucket != getattr(self, "_dl_bucket", -1):
            self._dl_bucket = bucket
            self._emit("info", t("log_downloading", pct=f"{pct:.0f}"))

    def _step_done(self) -> None:
        pb = self._w("#pbar", ProgressBar)
        if pb:
            pb.display = False
        if hasattr(self.screen, "refresh_data"):
            self.screen.refresh_data()

    def spawn(self, screen, jobname: str, args: tuple) -> None:
        """Bir işi ayrı süreçte başlatır (screen bağımsız; UI o an açık ekranda güncellenir)."""
        self._run_job(jobname, tuple(args))

    @work(thread=True)
    def _run_job(self, jobname: str, args: tuple) -> None:
        import multiprocessing as mp
        from queue import Empty

        from . import jobs

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=jobs.child_entry, args=(q, jobname, list(args)), daemon=True)
        proc.start()
        self._procs.append(proc)                         # iptal için kaydet
        try:
            while True:
                try:
                    kind, payload = q.get(timeout=1.0)
                except Empty:
                    if not proc.is_alive():              # (iptalde de burası kapatır)
                        break
                    continue
                if kind == "__end__":
                    break
                if kind == "download":
                    self.call_from_thread(self._download, payload)
                elif kind == "step":
                    self.call_from_thread(self._step, payload)
                elif kind == "refresh":
                    self.call_from_thread(self._step_done)   # ekranı tazele
                elif kind in ("ok", "info", "warn", "error"):
                    self.call_from_thread(self._emit, kind, payload)
                elif kind == "log":                          # geriye dönük
                    self.call_from_thread(self._emit, "info", payload)
        finally:
            try:
                self._procs.remove(proc)
            except ValueError:
                pass
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=10)
            self.call_from_thread(self._step_done)


def main() -> None:
    # multiprocessing kaynak izleyicisini (resource_tracker) Textual akışları
    # ele geçirmeden ÖNCE başlat. Aksi halde izleyici ilk Queue()'da tembel
    # başlar; o an sys.stderr Textual'ın sarmalayıcısıdır (fileno()==-1) ve
    # spawn 'bad value(s) in fds_to_keep' ile patlar.
    import multiprocessing as mp
    from multiprocessing import resource_tracker

    mp.get_context("spawn")
    resource_tracker.ensure_running()

    L2SApp().run()


if __name__ == "__main__":
    main()
