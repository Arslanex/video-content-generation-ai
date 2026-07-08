"""Cümle indeksi: kelime-düzeyi zamanlama + noktalama (+ duraklama yedeği) ile
transkripti CÜMLElere böler ve klip sınırlarını cümleye yaslamak için yardımcılar sunar.

Neden: Whisper SEGMENTleri cümle değildir (ASR-parçası). Sınırları segmente yaslamak
cümleyi ortadan keser ("cümle yarıda kalıyor"). Burada kelime zamanlamasından GERÇEK
cümle sınırları çıkarılır → hiçbir klip/parça cümle ortasında başlamaz/bitmez.
"""
from __future__ import annotations

import json

_TERMINAL = "!?…."       # cümle sonu noktalamaları
_TRAIL = "\"'”’»)]"       # noktalamadan sonra gelebilen sarmalayıcılar (kırpılır)
_PAUSE_GAP = 0.6          # noktalama yoksa cümle sonu sayılacak sessizlik (sn)
_DOT_MIN_GAP = 0.20       # '.' için: bu boşluk yoksa ve sonraki kelime küçük harfliyse bölme
# yaygın kısaltmalar (nokta cümle sonu SAYILMAZ) — TR + EN
_ABBREV = {"dr", "prof", "doç", "av", "vs", "vb", "örn", "bkz", "no", "sf", "yy",
           "mö", "ms", "mr", "mrs", "st", "etc", "e.g", "i.e"}


def _ends_sentence(word: str, gap: float, next_word: str | None, is_last: bool) -> bool:
    w = word.strip().rstrip(_TRAIL)
    if not w:
        return False
    ch = w[-1]
    if ch in "!?…":
        return True
    if ch == ".":
        stem = w[:-1].lower()
        if stem in _ABBREV:                     # "Dr." "vs." → cümleyi bölme
            return False
        if is_last or gap >= _DOT_MIN_GAP:
            return True
        nw = (next_word or "").strip()          # boşluk yok: sonraki büyük harfse cümle sonu
        return bool(nw) and nw[:1].isupper()
    return False


def build_sentences(segments: list[dict]) -> list[dict]:
    """transcript segmentlerinden [{start, end, text}] cümle listesi üretir.

    Cümleler segment sınırlarını AŞABİLİR (asıl amaç bu: parçalı segmentleri birleştirip
    tam cümle elde etmek). Kelime zamanlaması yoksa segmentlerin kendisi cümle sayılır.
    """
    words = [w for s in segments for w in s.get("words", []) if (w.get("word") or "").strip()]
    if not words:
        return [{"start": s["start"], "end": s["end"], "text": (s.get("text") or "").strip()}
                for s in segments if (s.get("text") or "").strip()]

    sents: list[dict] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 0.0
        end_here = _ends_sentence(w["word"], gap, nxt["word"] if nxt else None, nxt is None) \
            or (nxt is not None and gap >= _PAUSE_GAP)
        if end_here:
            sents.append({
                "start": round(cur[0]["start"], 3),
                "end": round(cur[-1]["end"], 3),
                "text": " ".join(x["word"].strip() for x in cur).strip(),
            })
            cur = []
    if cur:
        sents.append({
            "start": round(cur[0]["start"], 3),
            "end": round(cur[-1]["end"], 3),
            "text": " ".join(x["word"].strip() for x in cur).strip(),
        })
    return sents


# --- yaslama (asla cümle ortasında başlama/bitme) --------------------------

def snap_start(x: float, sentences: list[dict]) -> float:
    """x'i bir CÜMLE başlangıcına yaslar; içindeyse o cümlenin başına çeker,
    boşluktaysa SONRAKİ cümlenin başına gider (sessizlikte başlama)."""
    if not sentences:
        return x
    if x <= sentences[0]["start"]:
        return sentences[0]["start"]
    for s in sentences:
        if s["start"] <= x <= s["end"]:
            return s["start"]
    nxt = next((s for s in sentences if s["start"] > x), None)
    return nxt["start"] if nxt else sentences[-1]["start"]


def snap_end(x: float, sentences: list[dict]) -> float:
    """x'i bir CÜMLE sonuna yaslar; içindeyse o cümleyi TAMAMLAR (uzatır),
    boşluktaysa ÖNCEKİ cümlenin sonuna gider."""
    if not sentences:
        return x
    if x >= sentences[-1]["end"]:
        return sentences[-1]["end"]
    for s in sentences:
        if s["start"] <= x <= s["end"]:
            return s["end"]
    prev = None
    for s in sentences:
        if s["end"] < x:
            prev = s
        else:
            break
    return prev["end"] if prev else sentences[0]["end"]


def ends(sentences: list[dict]) -> list[float]:
    return [s["end"] for s in sentences]


def text_between(start: float, end: float, sentences: list[dict], eps: float = 0.05) -> str:
    """[start, end] aralığına TAM sığan cümlelerin metni (tutarlılık denetimi için)."""
    return " ".join(s["text"] for s in sentences
                    if s["start"] >= start - eps and s["end"] <= end + eps).strip()


def load_sentences(vdir, fused: dict | None = None) -> list[dict]:
    """Cümle listesini getirir: fused['sentences'] varsa onu; yoksa transcript.json'dan
    kurar; ikisi de yoksa boş (çağıran eski segment-tabanlı yola düşer)."""
    if fused and fused.get("sentences"):
        return fused["sentences"]
    tp = vdir / "transcript.json"
    if tp.exists():
        tr = json.loads(tp.read_text(encoding="utf-8"))
        return build_sentences(tr.get("segments", []))
    return []
