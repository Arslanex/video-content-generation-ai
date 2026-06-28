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

DATA_DIR = ROOT / "data"            # her video için ara dosyalar: data/<video_id>/
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


# Nihai çıktılar (klipler) kullanıcının Masaüstünde toplanır.
# L2S_OUTPUT_DIR ile değiştirilebilir.
OUTPUT_DIR = Path(os.getenv("L2S_OUTPUT_DIR") or (Path.home() / "Desktop" / "long-to-shorts"))


def video_dir(video_id: str) -> Path:
    """Bir videonun ara dosya (working) klasörünü döndürür (yoksa oluşturur)."""
    d = DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(video_id: str, sub: str = "") -> Path:
    """Nihai çıktı klasörü (Masaüstü): <Masaüstü>/long-to-shorts/<video_id>/<sub>."""
    d = OUTPUT_DIR / video_id / sub if sub else OUTPUT_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_dirs() -> None:
    """Temel klasörlerin var olduğundan emin olur."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
