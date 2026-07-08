"""Merkezi yapılandırma ve yol yönetimi.

Tüm yollar proje kökünden türetilir; bulut yok, her şey yerelde.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Proje kökü = bu dosyanın iki üstü (app/config.py -> app/ -> kök)
ROOT = Path(__file__).resolve().parent.parent

PROMPTS_DIR = ROOT / "prompts"      # Claude sistem promptları + puanlama rubrikleri
DB_PATH = ROOT / "library.sqlite"   # tek dosyalık yerel veritabanı
BRAND_DIR = ROOT / "brand"          # marka: logo.png + brand.json (vurgu rengi vb.)


def load_brand() -> dict:
    """Marka ayarlarını döndürür: {accent, channel, logo: Path|None, cover: dict}.

    brand/brand.json varsa okunur; brand/logo.png varsa logo olarak kullanılır.
    'cover' bloğu kapak tasarımını özelleştirir (bkz. render._resolve_cover).
    """
    import json

    accent = "#FFC400"
    channel = ""
    cover: dict = {}
    cfg = BRAND_DIR / "brand.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        accent = data.get("accent", accent)
        channel = data.get("channel", channel)
        cover = data.get("cover", {}) or {}
    logo = BRAND_DIR / "logo.png"
    return {"accent": accent, "channel": channel,
            "logo": logo if logo.exists() else None, "cover": cover}

# Dış servis anahtarları (ortam değişkeninden)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Konuşmacı ayrımı (pyannote) için HuggingFace tokeni — diarization'a özel
HF_TOKEN = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_TOKEN", "")

# Varsayılan model — yüksek akıl yürütme işleri için
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

# Whisper modeli (mlx-community deposundan). turbo: hız/kalite dengesi.
# Daha yüksek kalite için: mlx-community/whisper-large-v3-mlx
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")


# Her proje tek bir klasörde toplanır: <BASE_DIR>/<video_id>/
#   kaynak/    → ara/işleme dosyaları (video.mp4, transcript.json, fused.json, ...)
#   ciktilar/  → teslim edilebilir yayın dosyaları (formatlara göre alt klasörler)
#   project.json, <video_id>.zip
# Varsayılan: ~/Documents/lts  (L2S_OUTPUT_DIR ile değiştirilebilir).
BASE_DIR = Path(os.getenv("L2S_OUTPUT_DIR") or (Path.home() / "Documents" / "lts"))

# Format → ciktilar alt klasörü
_FMT_DIR = {
    "short": "shortlar",
    "episode": "bolumler",
    "podcast": "podcastlar",
    "supercut": "supercutlar",
}


def project_dir(video_id: str) -> Path:
    """Projenin kök klasörü: <BASE_DIR>/<video_id>/ (yoksa oluşturur)."""
    d = BASE_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_dir(video_id: str) -> Path:
    """Ara/işleme dosyaları klasörü: <proje>/kaynak/ (yoksa oluşturur).

    (İsim geriye dönük uyum için 'video_dir'; artık proje altında 'kaynak'.)
    """
    d = project_dir(video_id) / "kaynak"
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(video_id: str, sub: str = "") -> Path:
    """Teslim edilebilir çıktı klasörü: <proje>/ciktilar/<sub>/ (yoksa oluşturur)."""
    base = project_dir(video_id) / "ciktilar"
    d = base / sub if sub else base
    d.mkdir(parents=True, exist_ok=True)
    return d


def clip_dir(video_id: str, fmt: str, lang: str | None = None) -> Path:
    """Bir klibin gideceği çıktı alt klasörü (formata/dublaja göre)."""
    sub = "dublajlar" if lang else _FMT_DIR.get(fmt, "digerleri")
    return output_dir(video_id, sub)


def ensure_dirs() -> None:
    """Temel klasörlerin var olduğundan emin olur."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
