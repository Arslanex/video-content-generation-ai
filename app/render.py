"""Faz 7 — Render (montaj): kesim + yüz-farkında dikey yerleşim (+ altyazı, + kapak).

export.py kaba kesim yapar; render.py yayınlanabilir klip üretir:
  - layout  : short'larda yüz-farkında dikey yerleşim
              (iki kişi -> alt/üst stacked; tek kişi -> yüze sıkı kırpma)
  - captions: (sonraki adım) gömülü altyazı
  - intro   : (sonraki adım) başlık/hook kapağı

Bu dosya kademeli büyür; şu an layout uygulanmış durumda.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

from . import db
from .config import output_dir, video_dir

console = Console()

_W, _H = 1080, 1920  # short hedef çözünürlüğü (9:16)
_CAP_WORDS = 3          # ekranda eş zamanlı kelime sayısı
_CAP_FONT_FRAC = 0.032  # SABİT altyazı font oranı (yükseklik bazlı) — boyut oynamasın


# ----------------------------- yüz/kişi tespiti ------------------------------
_YUNET_MODEL = os.path.join(os.path.dirname(__file__), "models",
                            "face_detection_yunet_2023mar.onnx")


def _detect_people(video_path, start: float, end: float, samples: int = 15):
    """Klip boyunca yüzleri tespit eder (YuNet), kararlı kişi konumlarına kümeler.

    Döndürür: (src_w, src_h, people) — people her biri:
      {cx, cy, w, h, count, area}  (piksel; count = kaç karede görüldü)
    Model yoksa None döner (çağıran merkez kırpmaya düşer).

    Çoklu yerleşim yalnızca aynı karede birlikte görünen yüzler için kullanılır;
    sırayla konuşan kişiler (asla aynı anda ekranda değil) tek kişi sayılır.
    """
    import cv2
    import numpy as np

    if not os.path.exists(_YUNET_MODEL):
        return None

    cap = cv2.VideoCapture(str(video_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det = cv2.FaceDetectorYN_create(_YUNET_MODEL, "", (src_w, src_h), 0.6)
    det.setInputSize((src_w, src_h))

    # (frame_idx, cx, cy, w, h) — kare bazında tutulur (eşzamanlılık için)
    tagged: list[tuple[int, float, float, float, float]] = []
    span = max(end - start, 0.1)
    frame_idx = 0
    for i in range(samples):
        t = start + span * (i + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        _, faces = det.detect(frame)
        if faces is None:
            continue
        got = False
        for f in faces:
            x, y, bw, bh = f[:4]
            if f[-1] < 0.6 or bw < src_w * 0.02:
                continue
            tagged.append((frame_idx, x + bw / 2, y + bh / 2, bw, bh))
            got = True
        if got:
            frame_idx += 1
    cap.release()

    if not tagged:
        return src_w, src_h, []

    # merkez yakınlığına göre kümele (sabit konumlu konuşmacılar)
    thr = src_w * 0.10
    clusters: list[list[tuple[int, float, float, float, float]]] = []
    for item in tagged:
        _, cx, cy, w, h = item
        for c in clusters:
            mcx = np.mean([z[1] for z in c])
            mcy = np.mean([z[2] for z in c])
            if abs(cx - mcx) < thr and abs(cy - mcy) < thr * 1.5:
                c.append(item)
                break
        else:
            clusters.append([item])

    min_support = max(2, int(0.25 * max(1, frame_idx)))
    people: list[dict] = []
    cluster_frames: list[set[int]] = []
    for c in clusters:
        if len(c) < min_support:
            continue
        arr = np.array([(z[1], z[2], z[3], z[4]) for z in c], dtype=float)
        cx, cy, w, h = (float(np.median(arr[:, j])) for j in range(4))
        people.append({"cx": cx, "cy": cy, "w": w, "h": h,
                       "count": len(c), "area": w * h})
        cluster_frames.append({z[0] for z in c})

    if not people:
        return src_w, src_h, []

    # Aynı karede en fazla kaç farklı kişi var?
    max_simultaneous = 1
    for fi in range(frame_idx):
        n = sum(1 for frames in cluster_frames if fi in frames)
        max_simultaneous = max(max_simultaneous, n)

    if max_simultaneous <= 1 and len(people) > 1:
        # Sırayla konuşan konuşmacılar — en baskın yüze odaklan
        people = [max(people, key=lambda p: p["count"] * p["area"])]

    return src_w, src_h, people


def _crop_box(fcx, fcy, fh, src_w, src_h, aspect, k):
    """Yüz merkezli, verilen en-boy oranında (genişlik/yükseklik) bir kırpma kutusu.

    k: yüz yüksekliğinin kaç katı kadar dikey alan alınacağı (baş+omuz+pay).
    Döndürür: (x, y, w, h) tamsayı, kaynak sınırları içinde.
    """
    ch = min(k * fh, src_h)
    cw = ch * aspect
    if cw > src_w:
        cw = src_w
        ch = cw / aspect
    x = fcx - cw / 2
    y = fcy - ch * 0.45          # yüz hafif üstte (baş üstü payı)
    x = max(0, min(x, src_w - cw))
    y = max(0, min(y, src_h - ch))
    return int(round(x)), int(round(y)), int(round(cw)), int(round(ch))


def _center_crop_filter(src_w: int, src_h: int) -> str:
    cw = min(int(round(src_h * _W / _H)), src_w)
    x = (src_w - cw) // 2
    return f"[0:v]crop={cw}:{src_h}:{x}:0,scale={_W}:{_H},setsar=1[vbase]"


def _layout_tiles(people: list, W: int, H: int):
    """Kişi sayısına göre tile yerleşimi: [(person, x, y, w, h)] + ad.

    1: tam ekran tek kişi | 2: alt/üst | 3: büyük(konuşan*) + 2 küçük
    4: 2x2 ızgara | 5+: büyük + en fazla 3 küçük
    (*) "büyük" = en baskın kişi (statik); dinamik aktif-konuşmacı sonraki adım.
    """
    by_x = sorted(people, key=lambda p: p["cx"])
    by_prom = sorted(people, key=lambda p: p["count"] * p["area"], reverse=True)
    n = len(people)

    if n == 1:
        return [(people[0], 0, 0, W, H)], "tek kişi"
    if n == 2:
        return ([(by_x[0], 0, 0, W, H // 2), (by_x[1], 0, H // 2, W, H // 2)],
                "iki kişi (alt/üst)")
    if n == 3:
        main = by_prom[0]
        others = [p for p in by_x if p is not main]
        top_h = 1280
        bot_h = H - top_h
        return ([(main, 0, 0, W, top_h),
                 (others[0], 0, top_h, W // 2, bot_h),
                 (others[1], W // 2, top_h, W // 2, bot_h)],
                "3 kişi (büyük + 2 küçük)")
    if n == 4:
        tw, th = W // 2, H // 2
        g = by_x
        return ([(g[0], 0, 0, tw, th), (g[1], tw, 0, tw, th),
                 (g[2], 0, th, tw, th), (g[3], tw, th, tw, th)],
                "4 kişi (2x2 ızgara)")
    # 5+: büyük + en fazla 3 küçük
    main = by_prom[0]
    rest = by_prom[1:4]
    top_h = 1280
    bot_h = H - top_h
    sw = W // len(rest)
    tiles = [(main, 0, 0, W, top_h)]
    for i, p in enumerate(rest):
        tiles.append((p, i * sw, top_h, sw, bot_h))
    return tiles, f"{n} kişi (büyük + {len(rest)} küçük)"


def _build_layout_filter(video_path, start: float, end: float, dur: float) -> tuple[str, str]:
    """Klip için kişi-sayısına duyarlı dikey yerleşim filtresi üretir (-> [vbase])."""
    res = _detect_people(video_path, start, end)
    if res is None:  # model yok -> merkez kırpma (kaynak boyutu bilinmiyor, probe)
        sw, sh = _video_dims(video_path)
        return _center_crop_filter(sw, sh), "merkez-kırpma (model yok)"
    src_w, src_h, people = res
    if not people:
        return _center_crop_filter(src_w, src_h), "merkez-kırpma (yüz yok)"

    tiles, name = _layout_tiles(people, _W, _H)

    parts = [f"color=c=black:s={_W}x{_H}:r=30:d={dur:.2f}[bg]"]
    k = len(tiles)
    parts.append("[0:v]split=" + str(k) + "".join(f"[s{i}]" for i in range(k)))
    prev = "[bg]"
    for i, (p, tx, ty, tw, th) in enumerate(tiles):
        ar = tw / th
        kf = 4.0 if ar <= 0.7 else (3.2 if ar < 1.0 else 2.7)  # portre geniş, yatay sıkı
        x, y, cw, ch = _crop_box(p["cx"], p["cy"], p["h"], src_w, src_h, ar, kf)
        parts.append(f"[s{i}]crop={cw}:{ch}:{x}:{y},scale={tw}:{th},setsar=1[t{i}]")
        out = "[vbase]" if i == k - 1 else f"[ov{i}]"
        parts.append(f"{prev}[t{i}]overlay={tx}:{ty}{out}")
        prev = out
    return ";".join(parts), name


# ----------------------- altyazı (Pillow PNG + overlay) ----------------------
# Bu ffmpeg libass/freetype olmadan derlenmiş; altyazıyı Pillow ile şeffaf PNG'ye
# çizip ffmpeg'in çekirdek 'overlay' filtresiyle bindiriyoruz.

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]


def _font(size: int, path: str | None = None):
    from PIL import ImageFont

    for p in ([path] if path else []) + _FONT_CANDIDATES:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:  # noqa: BLE001 — bozuk font dosyası → sıradakine geç
                continue
    return ImageFont.load_default()


def _caption_blocks(transcript: dict, clip_start: float, clip_end: float):
    """Klip aralığındaki segmentleri (klip-göreli zaman, metin) olarak döndürür."""
    blocks = []
    for seg in transcript.get("segments", []):
        s, e = seg["start"], seg["end"]
        if e <= clip_start or s >= clip_end:
            continue
        txt = " ".join(seg["text"].split())
        if txt:
            blocks.append((max(s, clip_start) - clip_start,
                           min(e, clip_end) - clip_start, txt))
    return blocks


def _wrap_to_width(text: str, font, max_w: int):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cur and font.getlength(trial) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _render_caption_png(text: str, png_path, play_w: int, play_h: int) -> None:
    """Şeffaf, tam kare bir altyazı PNG'si çizer (beyaz metin + siyah kontur, altta ortalı)."""
    from PIL import Image, ImageDraw

    fontsize = max(30, int(play_h * 0.034))
    stroke = max(2, fontsize // 9)
    line_h = int(fontsize * 1.25)
    margin_v = int(play_h * 0.10)

    font = _font(fontsize)
    lines = _wrap_to_width(text, font, int(play_w * 0.88))[:3]

    img = Image.new("RGBA", (play_w, play_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    total_h = line_h * len(lines)
    y = play_h - margin_v - total_h
    for line in lines:
        w = font.getlength(line)
        draw.text(
            ((play_w - w) / 2, y), line, font=font,
            fill=(255, 255, 255, 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 230),
        )
        y += line_h
    img.save(png_path)


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# Hazır kapak şablonları — brand.json -> cover.style ile seçilir.
# Her biri makul varsayılanlar koyar; cover bloğundaki tek tek anahtarlar ezer.
_COVER_PRESETS = {
    "classic": {  # mevcut görünüm (varsayılan)
        "bg": "#111316", "title_color": "#FFFFFF", "subtitle_color": "#B4B6BE",
        "accent_style": "line", "logo_pos": "top-center", "align": "center",
        "vpos": "center"},
    "minimal": {
        "bg": "#0E0E10", "title_color": "#FFFFFF", "subtitle_color": "#9A9CA4",
        "accent_style": "none", "logo_pos": "top-left", "align": "left",
        "vpos": "bottom"},
    "bold": {
        "bg": "#0B0C10", "title_color": "#FFFFFF", "subtitle_color": "#C8CAD2",
        "accent_style": "bar", "logo_pos": "top-left", "align": "left",
        "vpos": "center", "title_scale": 1.25},
    "gradient": {
        "bg": ["#10131C", "#243154"], "title_color": "#FFFFFF",
        "subtitle_color": "#C2C6D2", "accent_style": "line", "logo_pos": "top-center",
        "align": "center", "vpos": "center"},
    "photo": {  # cover.bg = arka plan görseli bekler (örn. "brand/cover.jpg")
        "bg": "#111316", "title_color": "#FFFFFF", "subtitle_color": "#E2E4EA",
        "accent_style": "bar", "logo_pos": "top-left", "align": "left",
        "vpos": "bottom", "bg_dim": 0.5},
}

_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _brand_path(rel: str):
    """brand.json'da geçen göreli yolu çözer (proje kökü veya brand/ klasörü)."""
    from .config import BRAND_DIR, ROOT
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    for cand in (ROOT / rel, BRAND_DIR / rel, BRAND_DIR / p.name):
        if cand.exists():
            return cand
    return None


def _resolve_cover(brand: dict) -> dict:
    """brand['cover'] + şablon + üst düzey accent → tüm anahtarları dolu cfg sözlüğü."""
    raw = dict(brand.get("cover") or {})
    style = raw.get("style", "classic")
    cfg = dict(_COVER_PRESETS.get(style, _COVER_PRESETS["classic"]))
    cfg.update({k: v for k, v in raw.items() if k != "style"})

    cfg["accent"] = cfg.get("accent") or brand.get("accent") or "#FFC400"
    cfg["logo"] = brand.get("logo")

    # arka plan: düz renk / [renk1,renk2] gradyan / görsel yolu
    bg = cfg.get("bg", "#111316")
    if isinstance(bg, (list, tuple)):
        cfg.update(bg_kind="gradient", bg_colors=list(bg)[:2], bg_image=None)
    elif isinstance(bg, str) and bg.lower().endswith(_IMG_EXT):
        img = _brand_path(bg)
        cfg.update(bg_kind="image" if img else "solid",
                   bg_image=img, bg_colors=["#111316"])
    else:
        cfg.update(bg_kind="solid", bg_colors=[bg], bg_image=None)

    # özel font (marka fontu)
    cfg["font_path"] = str(_brand_path(cfg["font"])) if cfg.get("font") and _brand_path(cfg["font"]) else None

    cfg.setdefault("bg_dim", 0.45)        # görsel arka plan karartma 0..1
    cfg.setdefault("bg_dir", "vertical")  # gradyan yönü: vertical|horizontal|diagonal
    cfg.setdefault("title_scale", 1.0)
    cfg.setdefault("subtitle_scale", 1.0)
    cfg.setdefault("logo_scale", 1.0)
    cfg.setdefault("logo_pos", "top-center")
    cfg.setdefault("align", "center")
    cfg.setdefault("vpos", "center")
    cfg.setdefault("accent_style", "line")
    cfg.setdefault("title_color", "#FFFFFF")
    cfg.setdefault("subtitle_color", "#B4B6BE")
    cfg.setdefault("pad", 0.08)
    return cfg


def _cover_fit(im, w: int, h: int):
    """Görseli en-boy koruyarak w×h alanı dolduracak şekilde ortadan kırpar."""
    iw, ih = im.size
    scale = max(w / iw, h / ih)
    im = im.resize((max(1, int(iw * scale)), max(1, int(ih * scale))))
    nw, nh = im.size
    left, top = (nw - w) // 2, (nh - h) // 2
    return im.crop((left, top, left + w, top + h))


def _gradient(w: int, h: int, c1, c2, direction: str = "vertical"):
    import numpy as np
    from PIL import Image

    if direction == "horizontal":
        t = np.linspace(0, 1, w)[None, :, None]
    elif direction == "diagonal":
        t = ((np.linspace(0, 1, h)[:, None] + np.linspace(0, 1, w)[None, :]) / 2)[:, :, None]
    else:
        t = np.linspace(0, 1, h)[:, None, None]
    arr = np.array(c1) * (1 - t) + np.array(c2) * t
    arr = np.broadcast_to(arr, (h, w, 3)).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _cover_background(w: int, h: int, cfg: dict):
    from PIL import Image

    if cfg["bg_kind"] == "image" and cfg["bg_image"]:
        bg = _cover_fit(Image.open(cfg["bg_image"]).convert("RGB"), w, h)
        dim = min(max(float(cfg["bg_dim"]), 0.0), 1.0)
        if dim > 0:                       # okunabilirlik için karart
            bg = Image.blend(bg, Image.new("RGB", (w, h), (0, 0, 0)), dim)
        return bg
    if cfg["bg_kind"] == "gradient":
        cols = cfg["bg_colors"]
        return _gradient(w, h, _hex_rgb(cols[0]), _hex_rgb(cols[-1]), cfg["bg_dir"])
    return Image.new("RGB", (w, h), _hex_rgb(cfg["bg_colors"][0]))


def _render_cover_png(title: str, subtitle: str, png_path, w: int, h: int,
                      cfg: dict) -> None:
    """Açılış kapağı — cfg (bkz. _resolve_cover) ile tamamen özelleştirilebilir."""
    from PIL import Image, ImageDraw

    img = _cover_background(w, h, cfg)
    draw = ImageDraw.Draw(img)
    fp = cfg["font_path"]
    pad = int(w * cfg["pad"])
    max_w = w - 2 * pad
    align, vpos = cfg["align"], cfg["vpos"]
    accent_rgb = _hex_rgb(cfg["accent"])
    title_rgb, sub_rgb = _hex_rgb(cfg["title_color"]), _hex_rgb(cfg["subtitle_color"])

    # başlık otomatik sığdırma: en fazla 4 satır olana dek fontu kademeli küçült
    # (uzun başlıkta kelime kırpılmasın)
    _MAX_TITLE_LINES = 4
    base_tsize = int(h * 0.050 * cfg["title_scale"])
    tsize = base_tsize
    while tsize > int(base_tsize * 0.6):
        if len(_wrap_to_width(title or "", _font(tsize, fp), max_w)) <= _MAX_TITLE_LINES:
            break
        tsize -= max(1, base_tsize // 24)
    ssize = int(h * 0.028 * cfg["subtitle_scale"])
    tfont, sfont = _font(tsize, fp), _font(ssize, fp)
    tlh, slh = int(tsize * 1.2), int(ssize * 1.35)

    # marka logosu (konum cfg'den)
    logo, lpos = cfg["logo"], cfg["logo_pos"]
    if logo is not None and lpos != "none":
        lg = Image.open(logo).convert("RGBA")
        lw = max(1, int(w * 0.16 * cfg["logo_scale"]))
        lh = max(1, int(lw * lg.height / lg.width))
        lg = lg.resize((lw, lh))
        lx = pad if lpos == "top-left" else (w - pad - lw if lpos == "top-right" else (w - lw) // 2)
        ly = (h - pad - lh) if lpos == "bottom-center" else int(h * 0.08)
        img.paste(lg, (lx, ly), lg)

    tlines = _wrap_to_width(title or "", tfont, max_w)[:_MAX_TITLE_LINES]
    slines = _wrap_to_width(subtitle or "", sfont, max_w)[:3] if subtitle else []
    gap = int(h * 0.045)
    total = len(tlines) * tlh + (gap + len(slines) * slh if slines else 0)
    y = int(h * 0.16) if vpos == "top" else (h - int(h * 0.10) - total if vpos == "bottom"
                                             else (h - total) // 2)

    # vurgu öğesi: çizgi (başlık üstü) | bar (sol kenar) | yok
    if cfg["accent_style"] == "line":
        if align == "left":
            x1, x2 = pad, pad + int(w * 0.18)
        else:
            x1, x2 = (w - int(w * 0.18)) // 2, (w + int(w * 0.18)) // 2
        ay = y - int(h * 0.035)
        draw.rectangle([x1, ay, x2, ay + max(4, h // 320)], fill=accent_rgb)
    elif cfg["accent_style"] == "bar":
        bw = max(6, w // 120)
        bx = pad - bw - int(w * 0.012)
        draw.rectangle([max(0, bx), y, max(bw, bx + bw), y + total], fill=accent_rgb)

    def _x(width):
        return pad if align == "left" else (w - width) / 2

    for line in tlines:
        draw.text((_x(tfont.getlength(line)), y), line, font=tfont, fill=title_rgb)
        y += tlh
    if slines:
        y += gap
        for line in slines:
            draw.text((_x(sfont.getlength(line)), y), line, font=sfont, fill=sub_rgb)
            y += slh
    img.convert("RGB").save(png_path)


def _overlay_logo(video_out, logo, w: int, h: int) -> None:
    """Üretilmiş videoya marka logosunu (sağ üst köşe) bindirir."""
    tmp = video_out.with_name(video_out.stem + ".logo.mp4")
    lh = int(h * 0.07)
    m = int(h * 0.03)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_out), "-i", str(logo),
         "-filter_complex", f"[1:v]scale=-1:{lh}[lg];[0:v][lg]overlay=W-w-{m}:{m}[v]",
         "-map", "[v]", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(tmp)],
        check=True,
    )
    tmp.replace(video_out)


def _padded_bounds(vdir, start: float, end: float,
                   start_back: float = 2.0, start_fwd: float = 3.0,
                   end_back: float = 1.0, end_fwd: float = 3.0,
                   breath: float = 0.3):
    """Klip sınırlarını en yakın TEMİZ segment sınırına yaslar (ani başlangıç/bitişi önler).

    Başı: bir pencere içinde, öncesinde duraklama olan (pause_before=true) en yakın
    segment başlangıcına yaslar; sonu: pause_after=true olan en yakın segment bitişine.
    Yaslanan sınırın hemen öncesi/sonrası sessizlik olduğundan, küçük bir `breath`
    payıyla o sessizliğe girerek nefes payı bırakır. fused.json yoksa değiştirmez.
    """
    import json

    p = vdir / "fused.json"
    if not p.exists():
        return start, end
    segs = json.loads(p.read_text(encoding="utf-8")).get("segments", [])

    ns, ne = start, end
    # başlangıç: en yakın temiz giriş noktası (ileri/geri pencere)
    cands = [s for s in segs
             if s.get("pause_before") and start - start_back <= s["start"] <= start + start_fwd]
    if cands:
        seg = min(cands, key=lambda s: abs(s["start"] - start))
        ns = max(0.0, seg["start"] - breath)  # öncesindeki sessizliğe küçük nefes payı
    # bitiş: en yakın temiz çıkış noktası
    cands = [s for s in segs
             if s.get("pause_after") and end - end_back <= s["end"] <= end + end_fwd]
    if cands:
        seg = min(cands, key=lambda s: abs(s["end"] - end))
        ne = seg["end"] + breath
    if ne - ns < 1.0:  # güvenlik: çok kısaldıysa orijinali koru
        return start, end
    return round(ns, 2), round(ne, 2)


def _clip_words(transcript: dict, clip_start: float, clip_end: float):
    """Klip aralığındaki kelimeleri (klip-göreli zaman) döndürür."""
    out = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if w["end"] <= clip_start or w["start"] >= clip_end:
                continue
            word = w["word"].strip()
            if word:
                out.append({
                    "start": max(w["start"], clip_start) - clip_start,
                    "end": min(w["end"], clip_end) - clip_start,
                    "word": word,
                })
    return out


def _transparent_png(png_path, w: int, h: int) -> None:
    from PIL import Image

    Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(png_path)


def _render_word_png(chunk: list[str], active: int, png_path, w: int, h: int) -> None:
    """Öbeği SABİT font boyutuyla çizer; konuşulan kelimenin arkası siyah kutu.

    Font boyutu hiç değişmez (sahneden sahneye oynama olmaz). Öbek bir satıra
    sığmazsa font küçültülmez, satıra bölünür.
    """
    from PIL import Image, ImageDraw

    fontsize = int(h * _CAP_FONT_FRAC)   # sabit
    font = _font(fontsize)
    sp = font.getlength(" ")
    max_w = w * 0.86

    # öbeği sabit fontla satırlara böl (genelde 3 kelime tek satır)
    lines: list[list[int]] = []
    cur: list[int] = []
    cur_w = 0.0
    for idx, word in enumerate(chunk):
        ww = font.getlength(word)
        add = ww if not cur else sp + ww
        if cur and cur_w + add > max_w:
            lines.append(cur)
            cur, cur_w = [idx], ww
        else:
            cur.append(idx)
            cur_w += add
    if cur:
        lines.append(cur)

    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * 1.18)
    pad_x = max(5, int(fontsize * 0.20))
    pad_y = max(3, int(fontsize * 0.10))
    stroke = max(2, fontsize // 11)
    margin_v = int(h * 0.12)
    y0 = h - margin_v - line_h * len(lines)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for li, line in enumerate(lines):
        lw = sum(font.getlength(chunk[i]) for i in line) + sp * (len(line) - 1)
        x = (w - lw) / 2
        y = y0 + li * line_h
        for i in line:
            word = chunk[i]
            ww = font.getlength(word)
            if i == active:
                draw.rounded_rectangle(
                    [x - pad_x, y - pad_y, x + ww + pad_x, y + ascent + descent + pad_y],
                    radius=int(fontsize * 0.22), fill=(0, 0, 0, 235),
                )
                draw.text((x, y), word, font=font, fill=(255, 255, 255, 255))
            else:
                draw.text((x, y), word, font=font, fill=(255, 255, 255, 255),
                          stroke_width=stroke, stroke_fill=(0, 0, 0, 220))
            x += ww + sp
    img.save(png_path)


def burn_word_captions(video_in, video_out, words: list[dict], w: int, h: int) -> None:
    """Bir videoya kelime-kelime (3 kelime, kutulu aktif) altyazı gömer.

    words: [{start, end, word}] (klip-göreli saniye). Sesi olduğu gibi korur.
    dub için İngilizce altyazıyı da bu üretir (TR ile aynı stil/motor).
    """
    if not words:
        shutil.copy(video_in, video_out)
        return
    with tempfile.TemporaryDirectory() as td:
        cap = Path(td)
        blank = cap / "blank.png"
        _transparent_png(blank, w, h)
        entries, cursor = [], 0.0
        for i, wd in enumerate(words):
            ci = i // _CAP_WORDS
            chunk = [x["word"] for x in words[ci * _CAP_WORDS:(ci + 1) * _CAP_WORDS]]
            png = cap / f"w{i:04d}.png"
            _render_word_png(chunk, i - ci * _CAP_WORDS, png, w, h)
            d_start = wd["start"]
            d_end = words[i + 1]["start"] if i + 1 < len(words) else wd["end"]
            if d_end - wd["end"] > 1.0:
                d_end = wd["end"] + 0.3
            if d_start > cursor + 0.02:
                entries.append((blank, d_start - cursor))
            entries.append((png, max(0.05, d_end - d_start)))
            cursor = d_end
        lines = []
        for path, dur in entries:
            lines += [f"file '{path}'", f"duration {dur:.3f}"]
        lines.append(f"file '{entries[-1][0]}'")
        caplist = cap / "list.txt"
        caplist.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_in),
             "-f", "concat", "-safe", "0", "-i", str(caplist),
             "-filter_complex",
             f"[1:v]format=rgba,scale={w}:{h}[capv];[0:v][capv]overlay=0:0[v]",
             "-map", "[v]", "-map", "0:a?",
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(video_out)],
            check=True,
        )


def _video_dims(path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


# ------------------------------- render --------------------------------------
def _parse_picks(pick: str, recs_by_fmt: dict[str, list]) -> list:
    """Seçim ifadesini çözer. Kabul edilenler:
      "14"          -> öneri ID'si (recs'te görünen #)
      "short:1"     -> formatın 1. sıradaki adayı
      "short"       -> tüm short'lar
      "all"         -> tüm öneriler
    """
    by_id = {r["id"]: r for rows in recs_by_fmt.values() for r in rows}
    chosen = []
    for token in (t.strip() for t in pick.split(",") if t.strip()):
        if token == "all":
            for rows in recs_by_fmt.values():
                chosen.extend(rows)
        elif token.isdigit():                      # ID ile seçim
            rid = int(token)
            if rid in by_id:
                chosen.append(by_id[rid])
            else:
                console.print(f"  [yellow]uyarı:[/yellow] #{rid} bulunamadı")
        elif ":" in token:
            fmt, rank = token.split(":", 1)
            rows = recs_by_fmt.get(fmt, [])
            idx = int(rank) - 1
            if 0 <= idx < len(rows):
                chosen.append(rows[idx])
            else:
                console.print(f"  [yellow]uyarı:[/yellow] {token} için aday yok")
        else:
            chosen.extend(recs_by_fmt.get(token, []))
    seen, unique = set(), []
    for r in chosen:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    return unique


def _afade(dur: float) -> str:
    fd = min(0.4, max(0.05, dur / 8))
    return f"afade=t=in:st=0:d={fd:.2f},afade=t=out:st={max(0.0, dur - fd):.2f}:d={fd:.2f}"


def _render_podcast(video_path, out_dir, r, start: float, end: float, safe: str) -> None:
    """Podcast: segmentin sesini (.m4a) + statik kapaklı audiogramı (.mp4) üretir."""
    import json

    dur = end - start
    af = _afade(dur)

    # 1) ses dosyası
    audio_out = out_dir / f"podcast_{r['id']}_{safe}.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(video_path),
         "-vn", "-af", af, "-c:a", "aac", "-b:a", "192k", str(audio_out)],
        check=True,
    )

    # 2) audiogram (1080x1080 kare, statik başlık kapağı + ses)
    from .config import load_brand
    brand = load_brand()
    payload = json.loads(r["payload"] or "{}")
    subtitle = payload.get("hook") or payload.get("description") or ""
    cover = out_dir / f".pcover_{r['id']}.png"
    _render_cover_png(r["title"] or "", subtitle, cover, 1080, 1080,
                      _resolve_cover(brand))
    gram = out_dir / f"podcast_{r['id']}_{safe}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-loop", "1", "-t", f"{dur:.2f}", "-i", str(cover),
         "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(video_path),
         "-filter_complex",
         f"[0:v]scale=1080:1080,setsar=1,fps=30,format=yuv420p[v];[1:a]{af}[a]",
         "-map", "[v]", "-map", "[a]", "-shortest",
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(gram)],
        check=True,
    )
    cover.unlink(missing_ok=True)


def render(video_id: str, picks: str, layout: bool = True, captions: bool = True,
           intro: bool = True, pad: bool = True, lang: str | None = None) -> None:
    """Seçilen önerileri render eder: dikey yerleşim + altyazı + kapak + sınır yaslama/fade.

    lang verilirse (en, es, ...): görseli üretir, sesi o dile DUBLAJLAR ve hedef-dil
    altyazısı gömer → dikey dublajlı reel. (TR altyazı kapatılır.)
    """
    import json

    if lang:                       # dublajlı reel: TR altyazı yerine dub + EN altyazı
        captions = False

    vdir = video_dir(video_id)
    video_path = vdir / "video.mp4"
    out_dir = output_dir(video_id, "renders")   # nihai klipler Masaüstünde

    if not video_path.exists():
        raise FileNotFoundError("video.mp4 yok. Önce 'l2s ingest' çalıştır.")

    transcript = None
    tpath = vdir / "transcript.json"
    if captions and tpath.exists():
        transcript = json.loads(tpath.read_text(encoding="utf-8"))
    elif captions:
        console.print("  [yellow]uyarı:[/yellow] transcript.json yok, altyazı atlanıyor")

    all_recs = db.get_recommendations(video_id)
    if not all_recs:
        raise RuntimeError("Öneri yok. Önce 'l2s analyze' çalıştır.")

    recs_by_fmt: dict[str, list] = {}
    for r in all_recs:
        recs_by_fmt.setdefault(r["fmt"], []).append(r)

    selected = _parse_picks(picks, recs_by_fmt)
    if not selected:
        raise RuntimeError(f"'{picks}' hiçbir öneriyle eşleşmedi.")

    from .config import load_brand
    brand = load_brand()

    out_dir.mkdir(exist_ok=True)
    db.set_stage(video_id, "render", "running")

    try:
        for r in selected:
            is_short = r["fmt"] == "short"
            start, end = r["start_sec"], r["end_sec"]
            if pad:
                start, end = _padded_bounds(vdir, start, end)
            safe = "".join(c if c.isalnum() or c in " -_" else "_"
                           for c in (r["title"] or "clip"))[:50].strip().replace(" ", "_")

            # podcast: ses + audiogram (ayrı yol)
            if r["fmt"] == "podcast":
                console.print(f"  render: [cyan]podcast[/cyan] {start:.0f}-{end:.0f}s  ses + audiogram")
                _render_podcast(video_path, out_dir, r, start, end, safe)
                continue

            lang_tag = f"{lang}_" if lang else ""
            out = out_dir / f"{r['fmt']}_{r['id']}_{lang_tag}{safe}.mp4"

            use_layout = is_short and layout
            extras = []

            # taban video filtresi (yerleşim) -> [vbase]
            if use_layout:
                base_fc, name = _build_layout_filter(video_path, start, end, end - start)
                out_w, out_h = _W, _H
                extras.append(f"yerleşim: {name}")
            else:
                out_w, out_h = (_W, _H) if is_short else _video_dims(video_path)
                if is_short:  # layout kapalı ama yine de dikey
                    cw = min(int(round(out_h * _W / _H)), out_w)
                    base_fc = f"[0:v]crop={cw}:{out_h}:(iw-{cw})/2:0,scale={_W}:{_H},setsar=1[vbase]"
                    out_w, out_h = _W, _H
                else:
                    base_fc = "[0:v]null[vbase]"

            # altyazı: kelime-kelime; ekranda 3-4 kelime, konuşulan kelime siyah kutulu.
            # Her kelime durumu için bir PNG; concat demuxer ile zamanlı tek bir
            # altyazı akışı üretip tek overlay ile bindiriyoruz (ffmpeg metin desteği gerekmez).
            dur = end - start
            cap_list = None
            if transcript:
                words = _clip_words(transcript, start, end)
                if words:
                    cap_dir = out_dir / f".cap_{r['id']}"
                    cap_dir.mkdir(exist_ok=True)
                    blank = cap_dir / "blank.png"
                    _transparent_png(blank, out_w, out_h)

                    entries: list[tuple] = []
                    cursor = 0.0
                    for i, w in enumerate(words):
                        ci = i // _CAP_WORDS
                        chunk = [x["word"] for x in words[ci * _CAP_WORDS:(ci + 1) * _CAP_WORDS]]
                        png = cap_dir / f"w{i:04d}.png"
                        _render_word_png(chunk, i - ci * _CAP_WORDS, png, out_w, out_h)
                        d_start = w["start"]
                        d_end = words[i + 1]["start"] if i + 1 < len(words) else w["end"]
                        if d_end - w["end"] > 1.0:        # uzun boşlukta altyazı kalmasın
                            d_end = w["end"] + 0.3
                        if d_start > cursor + 0.02:
                            entries.append((blank, d_start - cursor))
                        entries.append((png, max(0.05, d_end - d_start)))
                        cursor = d_end
                    if cursor < dur:
                        entries.append((blank, dur - cursor))

                    lines = []
                    for path, d in entries:
                        lines.append(f"file '{path}'")
                        lines.append(f"duration {d:.3f}")
                    lines.append(f"file '{entries[-1][0]}'")  # concat demuxer son-kare gereği
                    cap_list = cap_dir / "list.txt"
                    cap_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    extras.append(f"altyazı (kelime ×{len(words)})")

            if cap_list is not None:
                fc = (base_fc
                      + f";[1:v]format=rgba,scale={out_w}:{out_h}[capv]"
                      + ";[vbase][capv]overlay=0:0[v]")
            else:
                fc = base_fc.replace("[vbase]", "[v]")

            if intro:
                extras.append("kapak")
            extra_str = ("  " + " • ".join(extras)) if extras else ""
            console.print(f"  render: [cyan]{r['fmt']}[/cyan] {start:.1f}-{end:.1f}s{extra_str}")

            # ses fade-in/out (ani başlangıç/bitişi yumuşatır)
            fd = min(0.30, max(0.05, dur / 4))
            fc += (
                f";[0:a]afade=t=in:st=0:d={fd:.2f},"
                f"afade=t=out:st={max(0.0, dur - fd):.2f}:d={fd:.2f}[a]"
            )

            cmd = ["ffmpeg", "-y", "-loglevel", "error",
                   "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(video_path)]
            if cap_list is not None:
                cmd += ["-f", "concat", "-safe", "0", "-i", str(cap_list)]
            cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out)]
            subprocess.run(cmd, check=True)

            # marka logosu: SADECE içeriğe (sağ üst köşe), kapak birleştirmeden önce
            if brand["logo"] is not None:
                _overlay_logo(out, brand["logo"], out_w, out_h)

            # açılış kapağı: kapak + klibi birleştir (ikinci geçiş)
            if intro:
                payload = json.loads(r["payload"] or "{}")
                subtitle = payload.get("hook") or payload.get("description") or ""
                cover = out_dir / f".cover_{r['id']}.png"
                _render_cover_png(r["title"] or "", subtitle, cover, out_w, out_h,
                                  _resolve_cover(brand))
                tmp = out.with_name(out.stem + ".withintro.mp4")
                intro_dur = 2.2
                concat_cmd = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-loop", "1", "-t", f"{intro_dur}", "-i", str(cover),
                    "-i", str(out),
                    "-f", "lavfi", "-t", f"{intro_dur}",
                    "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-filter_complex",
                    # kart sonu fade-out + içerik başı fade-in -> yumuşak dip-to-black geçiş
                    f"[0:v]scale={out_w}:{out_h},setsar=1,fps=30,format=yuv420p,"
                    f"fade=t=out:st={max(0.0, intro_dur - 0.3):.2f}:d=0.3[ci];"
                    f"[1:v]scale={out_w}:{out_h},setsar=1,fps=30,format=yuv420p,"
                    f"fade=t=in:st=0:d=0.35[cm];"
                    f"[ci][cm]concat=n=2:v=1:a=0[v];"
                    f"[2:a][1:a]concat=n=2:v=0:a=1[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(tmp),
                ]
                subprocess.run(concat_cmd, check=True)
                tmp.replace(out)

            # dublajlı reel: sesi hedef dile değiştir + hedef-dil altyazısı göm
            if lang:
                import numpy as np
                import soundfile as sf
                from .dub import make_dub_track
                console.print(f"  dublaj ({lang})…")
                canvas, sr, cap_words, note = make_dub_track(video_id, start, end, lang)
                console.print(f"  [dim]{note}[/dim]")
                intro_off = 2.2 if intro else 0.0          # kapak süresi kadar kaydır
                full = np.concatenate(
                    [np.zeros(int(intro_off * sr), dtype=np.float32), canvas])
                dwav = out_dir / f".dubaud_{r['id']}.wav"
                sf.write(str(dwav), full, sr)
                tmpa = out.with_name(out.stem + ".dubaud.mp4")
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-i", str(dwav),
                     "-map", "0:v", "-map", "1:a", "-shortest",
                     "-c:v", "copy", "-c:a", "aac", str(tmpa)], check=True)
                tmpa.replace(out)
                dwav.unlink(missing_ok=True)
                shifted = [{"start": w["start"] + intro_off, "end": w["end"] + intro_off,
                            "word": w["word"]} for w in cap_words]
                tmpc = out.with_name(out.stem + ".encap.mp4")
                burn_word_captions(out, tmpc, shifted, out_w, out_h)
                tmpc.replace(out)

            # geçici altyazı/kapak dosyalarını temizle
            shutil.rmtree(out_dir / f".cap_{r['id']}", ignore_errors=True)
            (out_dir / f".cover_{r['id']}.png").unlink(missing_ok=True)

        db.set_stage(video_id, "render", "done", f"{len(selected)} klip")
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "render", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise

    console.print(f"  [green]✓[/green] {len(selected)} klip  •  [dim]{out_dir}[/dim]")
