"""Dublaj — çeviri (Claude) + Kokoro seslendirme + senkron + videoya gömme.

Yalnızca Kokoro'nun desteklediği dillerde çalışır.
per_speaker=True ise:
  - konuşmacı ayrımı (diarization),
  - her konuşmacının CİNSİYETİ (perde/pitch ile) tahmin edilir → uygun ses,
  - kelime düzeyinde hizalama: konuşmacı değişince ses de değişir.
"""
from __future__ import annotations

import json
import subprocess

from rich.console import Console

from . import db
from .config import ANTHROPIC_API_KEY, CLAUDE_MODEL, clip_dir, video_dir

console = Console()

# Desteklenen diller: kod -> (kokoro_lang_code, [ses havuzu], tam_ad)
# Ses adlarında 2. harf cinsiyettir: f=kadın, m=erkek (af_heart, am_michael...).
LANGS = {
    "en":    ("a", ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "am_echo"], "English"),
    "en-gb": ("b", ["bf_emma", "bf_isabella", "bm_george", "bm_lewis"], "English (British)"),
    "es":    ("e", ["ef_dora", "em_alex", "em_santa"], "Spanish"),
    "fr":    ("f", ["ff_siwis"], "French"),
    "it":    ("i", ["if_sara", "im_nicola"], "Italian"),
    "pt":    ("p", ["pf_dora", "pm_alex"], "Portuguese"),
    "hi":    ("h", ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"], "Hindi"),
}

_SR = 24000          # Kokoro çıkış örnekleme hızı
_F0_THRESH = 165.0   # medyan perde (Hz) bunun altı erkek, üstü kadın


def _translate(texts: list[str], lang_name: str, context: str = "") -> tuple[list[str], list[dict]]:
    """Replikleri TEK bütün senaryo gibi çevirir + özel-isim sözlüğü (glossary) çıkarır.

    Önce tüm replikler ve video bağlamı birlikte değerlendirilip özel isimlerin doğru
    yazımı belirlenir; sonra tutarlı, doğal (birebir değil) çeviri yapılır. Çeviri
    zamanlama için replik-hizalı döner. context: video meta (başlık/kanal/açıklama).
    Döndürür: (translations[aynı sırada], glossary[{source, correct}]).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    schema = {
        "type": "object",
        "properties": {
            "glossary": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"source": {"type": "string"}, "correct": {"type": "string"}},
                    "required": ["source", "correct"],
                    "additionalProperties": False,
                },
            },
            "translations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["glossary", "translations"],
        "additionalProperties": False,
    }
    ctx = f"\nVİDEO BAĞLAMI (özel isimlerin doğru yazımı için):\n{context}\n" if context else ""
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=12000,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{
            "role": "user",
            "content": (
                f"Aşağıda bir videonun konuşma replikleri (numaralı, sırayla) ve video "
                f"bağlamı var. Bunu {lang_name} diline dublaj için çevireceksin.\n\n"
                f"ADIM 1 — Özel isim sözlüğü (glossary): Tüm özel isimleri (kişi, marka, "
                f"şirket, yer, ürün) tespit et. Bağlam ve video bağlamından DOĞRU yazımlarını "
                f"belirle; konuşma tanımadan kaynaklı bariz yanlışları düzelt "
                f"(ör. 'Algo Fek'→'Algofact', 'Tahacan'→'Taha Can'). glossary: [{{source, correct}}].\n\n"
                f"ADIM 2 — Çeviri: Replikleri TEK BİR BÜTÜN konuşma gibi, tutarlı ve akıcı çevir. "
                f"BİREBİR/kelime-kelime DEĞİL; anlamı koru, hedef dilde doğal kur. Özel isimleri "
                f"ASLA çevirme; glossary'deki DOĞRU yazımıyla ve TÜM repliklerde TUTARLI kullan. "
                f"Her repliğin uzunluğunu kabaca koru (dublaj zamanlaması için).\n\n"
                f"translations: her replik için bir çeviri, AYNI SIRADA, tam {len(texts)} adet.{ctx}\n\n"
                "REPLİKLER:\n"
                + "\n".join(f"{i}. {t}" for i, t in enumerate(texts))
            ),
        }],
    )
    text = next(b.text for b in msg.content if b.type == "text")
    data = json.loads(text)
    out = data.get("translations", [])
    if len(out) < len(texts):
        out += [""] * (len(texts) - len(out))
    return out[:len(texts)], data.get("glossary", [])


def _speaker_genders(audio_path, turns, start, end) -> dict[str, str]:
    """Her konuşmacının cinsiyetini medyan perde (F0) ile tahmin eder: 'm' | 'f'."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=16000, offset=start, duration=end - start)
    genders: dict[str, str] = {}
    speakers = sorted({t["speaker"] for t in turns})
    for spk in speakers:
        chunks = []
        total = 0.0
        for t in turns:
            if t["speaker"] != spk:
                continue
            a = int(max(0, t["start"] - start) * sr)
            b = int(max(0, t["end"] - start) * sr)
            if b > a:
                chunks.append(y[a:b])
                total += (b - a) / sr
            if total > 20:          # cinsiyet için ~20 sn yeterli
                break
        if not chunks:
            genders[spk] = "m"
            continue
        seg = np.concatenate(chunks)
        try:
            f0, voiced, _ = librosa.pyin(seg, fmin=65, fmax=400, sr=sr)
            vals = f0[~np.isnan(f0)]
            med = float(np.median(vals)) if len(vals) else 0.0
        except Exception:  # noqa: BLE001
            med = 0.0
        genders[spk] = "f" if med >= _F0_THRESH else "m"
    return genders


def _clip_gender(audio_path, start, end) -> str:
    """Diarization olmadan tüm klibin cinsiyetini medyan perdeyle tahmin eder ('m'|'f')."""
    import librosa
    import numpy as np

    dur = min(end - start, 25.0)
    y, sr = librosa.load(str(audio_path), sr=16000, offset=start, duration=dur)
    try:
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        vals = f0[~np.isnan(f0)]
        med = float(np.median(vals)) if len(vals) else 0.0
    except Exception:  # noqa: BLE001
        med = 0.0
    return "f" if med >= _F0_THRESH else "m"


def _gendered_default(voices, gender) -> str:
    """Verilen cinsiyete uygun ilk sesi döndürür (yoksa ilk ses)."""
    return next((v for v in voices if v[1] == gender), voices[0])


def _assign_voices(speakers_order, genders, voices) -> dict[str, str]:
    """Konuşmacılara cinsiyetine uygun ses atar (aynı cinsiyette farklı sesler)."""
    pool = {"f": [v for v in voices if v[1] == "f"], "m": [v for v in voices if v[1] == "m"]}
    count = {"f": 0, "m": 0}
    voice_of = {}
    for spk in speakers_order:
        g = genders.get(spk, "m")
        p = pool[g] or pool["m"] or pool["f"] or voices
        voice_of[spk] = p[count[g] % len(p)]
        count[g] += 1
    return voice_of


def _utterances(words, turns, speaker_for, min_dur: float = 0.8, island_dur: float = 6.0):
    """Kelimeleri konuşmacıya göre gruplar -> [{start,end,speaker,text}] (mutlak zaman).

    Yumuşatma:
      - çok kısa (<min_dur) birimler öncekine katılır,
      - "ada": iki yanı da AYNI diğer konuşmacı olan ve <island_dur birimin etiketi
        çevreye düzeltilir (diarization'ın akış-ortası yanlış etiketlerini giderir).
    """
    def dur(u):
        return u["end"] - u["start"]

    # 1) ham gruplama
    raw = []
    last_spk = None
    for w in words:
        spk = speaker_for(turns, w["start"], w["end"]) or last_spk
        last_spk = spk
        if raw and raw[-1]["speaker"] == spk:
            raw[-1]["end"] = w["end"]
            raw[-1]["text"] += w["word"]
        else:
            raw.append({"start": w["start"], "end": w["end"], "speaker": spk, "text": w["word"]})

    # 2) çok kısa birimleri öncekine kat
    merged = []
    for u in raw:
        if merged and dur(u) < min_dur:
            merged[-1]["end"] = u["end"]
            merged[-1]["text"] += u["text"]
        else:
            merged.append(u)

    # 3) ada temizleme: <island_dur ve iki yanı aynı (farklı) konuşmacı -> etiketi düzelt
    for i in range(1, len(merged) - 1):
        if (dur(merged[i]) < island_dur
                and merged[i - 1]["speaker"] == merged[i + 1]["speaker"]
                and merged[i]["speaker"] != merged[i - 1]["speaker"]):
            merged[i]["speaker"] = merged[i - 1]["speaker"]

    # 4) ardışık aynı konuşmacıyı yeniden birleştir
    final = []
    for u in merged:
        if final and final[-1]["speaker"] == u["speaker"]:
            final[-1]["end"] = u["end"]
            final[-1]["text"] += " " + u["text"]
        else:
            final.append(u)
    for u in final:
        u["text"] = " ".join(u["text"].split())
    return [u for u in final if u["text"]]


def make_dub_track(video_id: str, start: float, end: float, lang: str = "en",
                   per_speaker: bool = True, speakers: int | None = None):
    """[start,end] için dublaj ses kanalı + altyazı kelimelerini üretir (klip-göreli).

    Döndürür: (canvas float32 numpy, _SR, cap_words[{start,end,word}], spk_note).
    Hem `dub` (yatay) hem `produce --lang` (dikey reel) bunu kullanır.
    """
    import numpy as np

    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY ayarlı değil (.env).")
    if lang not in LANGS:
        raise RuntimeError(
            f"'{lang}' desteklenmiyor. Kokoro dilleri: {', '.join(LANGS)}.\n"
            "Türkçe/diğer diller için ses klonlamalı XTTS/ElevenLabs gerekir."
        )

    lang_code, voices, lang_name = LANGS[lang]
    vdir = video_dir(video_id)
    audio_path = vdir / "audio.wav"
    transcript = json.loads((vdir / "transcript.json").read_text(encoding="utf-8"))

    words = []
    for s in transcript["segments"]:
        for w in s.get("words", []):
            if w["end"] > start and w["start"] < end and w["word"].strip():
                words.append({"start": w["start"], "end": w["end"], "word": w["word"]})
    if not words:
        raise RuntimeError("Klip aralığında kelime yok (transkripti kontrol et).")

    def _segment_units(voice):
        return [{"start": s["start"] - start, "end": s["end"] - start,
                 "text": s["text"].strip(), "voice": voice}
                for s in transcript["segments"]
                if s["end"] > start and s["start"] < end and s["text"].strip()]

    spk_note = ""
    units = None
    if per_speaker:
        from .config import HF_TOKEN
        from .diarize import diarize_range, speaker_for
        if HF_TOKEN:
            try:
                console.print("  konuşmacı ayrımı + cinsiyet tespiti…")
                turns = diarize_range(video_id, start, end, num_speakers=speakers)["turns"]
                if turns:
                    genders = _speaker_genders(audio_path, turns, start, end)
                    utts = _utterances(words, turns, speaker_for)
                    talk: dict[str, float] = {}
                    for u in utts:
                        talk[u["speaker"]] = talk.get(u["speaker"], 0.0) + (u["end"] - u["start"])
                    total = sum(talk.values()) or 1.0
                    dom = max(talk, key=talk.get)
                    nondom = total - talk[dom]
                    if len(talk) == 1 or nondom < 2.0:     # monolog / hayalet azınlık
                        dvoice = _gendered_default(voices, genders.get(dom, "m"))
                        units = _segment_units(dvoice)
                        spk_note = f"monolog ({dom},{genders.get(dom,'?')}) → {dvoice}"
                    else:
                        order = list(dict.fromkeys(u["speaker"] for u in utts))
                        voice_of = _assign_voices(order, genders, voices)
                        units = [{"start": u["start"] - start, "end": u["end"] - start,
                                  "text": u["text"], "voice": voice_of[u["speaker"]]}
                                 for u in utts]
                        spk_note = (f"{len(order)} konuşmacı → "
                                    + ", ".join(f"{s}({genders.get(s,'?')}):{voice_of[s]}" for s in order))
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [yellow]diarization atlandı:[/yellow] {exc}")
        else:
            console.print("  [yellow]HF_TOKEN yok → tek ses (klip cinsiyetine göre)[/yellow]")

    if units is None:
        try:
            cg = _clip_gender(audio_path, start, end)
        except Exception:  # noqa: BLE001
            cg = "m"
        dvoice = _gendered_default(voices, cg)
        units = _segment_units(dvoice)
        spk_note = spk_note or f"tek ses ({dvoice}, klip cinsiyeti={cg})"

    # çeviri (meta'yı bağlam ver → özel isimleri doğru yazsın/düzeltsin)
    context = ""
    meta_path = vdir / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        desc = (m.get("description") or "")[:800]
        context = f"Başlık: {m.get('title','')}\nKanal: {m.get('channel','')}\nAçıklama: {desc}"
    console.print(f"  çeviri (Claude, {len(units)} replik) → {lang_name}…")
    translations, glossary = _translate([u["text"] for u in units], lang_name, context=context)
    if glossary:
        names = ", ".join(f"{g['source']}→{g['correct']}" if g['source'] != g['correct']
                          else g['correct'] for g in glossary[:12])
        console.print(f"  [dim]özel isimler: {names}[/dim]")

    # seslendirme — her repliği gerçek zamanına sabitle; uzunsa NATIVE hızla yeniden seslendir
    console.print("  seslendirme (Kokoro)…")
    from kokoro import KPipeline
    pipe = KPipeline(lang_code=lang_code)

    def _synth_chunks(text, voice, speed=1.0):
        # Kokoro metni cümle/öbek parçalarına böler; her parçanın (metni, sesi)
        out = []
        for gs, _, a in pipe(text, voice=voice, speed=speed):
            if a is not None and len(a):
                out.append((gs, np.asarray(a, dtype=np.float32)))  # torch tensor → numpy
        return out

    clip_dur = end - start
    canvas = np.zeros(int(clip_dur * _SR) + _SR, dtype=np.float32)
    cap_words: list[dict] = []
    for i, (u, tr) in enumerate(zip(units, translations)):
        if not tr.strip():
            continue
        chunks = _synth_chunks(tr, u["voice"])
        if not chunks:
            continue
        audio = np.concatenate([a for _, a in chunks])
        target = max(0.3, u["end"] - u["start"])
        cur = len(audio) / _SR
        if cur > target * 1.10:                            # uzunsa NATIVE hızla yeniden seslendir
            fchunks = _synth_chunks(tr, u["voice"], speed=min(cur / target, 1.5))
            if fchunks:
                chunks = fchunks
                audio = np.concatenate([a for _, a in chunks])
        next_start = units[i + 1]["start"] if i + 1 < len(units) else clip_dur
        max_n = max(int(0.1 * _SR), int((next_start - u["start"]) * _SR))
        if len(audio) > max_n:                             # çakışma önle + fade-out
            audio = audio[:max_n].copy()
            f = min(int(0.05 * _SR), len(audio))
            audio[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        idx = int(u["start"] * _SR)
        seg = audio[: len(canvas) - idx]
        canvas[idx: idx + len(seg)] = seg

        # altyazı: kelimeleri PARÇA sınırlarına göre dağıt (isabetli zamanlama)
        off = 0  # birim sesi içinde örnek ofseti
        limit = len(seg)
        for gs, a in chunks:
            if off >= limit:
                break
            c0, c1 = off, min(off + len(a), limit)
            ct0 = u["start"] + c0 / _SR
            cdur = (c1 - c0) / _SR
            cw = gs.split()
            tot = sum(len(w) for w in cw) or 1
            t = ct0
            for w in cw:
                d = cdur * len(w) / tot
                cap_words.append({"start": round(t, 3), "end": round(t + d, 3), "word": w})
                t += d
            off += len(a)

    return np.clip(canvas, -1.0, 1.0), _SR, cap_words, spk_note


def dub(video_id: str, rec_id: int, lang: str = "en", per_speaker: bool = True,
        captions: bool = True, speakers: int | None = None) -> None:
    """Bir öneriyi (klip) hedef dile dublajlar -> dubs/<fmt>_<id>_<lang>.mp4 (YATAY kaynak).

    Dikey dublajlı reel için: l2s produce <id> --lang <kod>.
    """
    import soundfile as sf

    vdir = video_dir(video_id)
    video_path = vdir / "video.mp4"
    rec = next((r for r in db.get_recommendations(video_id) if r["id"] == rec_id), None)
    if rec is None:
        raise RuntimeError(f"#{rec_id} bulunamadı. 'l2s recs {video_id}' ile bak.")
    start, end = rec["start_sec"], rec["end_sec"]

    console.print(f"dublaj  [dim]{rec['fmt']} #{rec_id} → {lang}[/dim]")
    db.set_stage(video_id, "dub", "running")
    try:
        canvas, sr, cap_words, note = make_dub_track(video_id, start, end, lang,
                                                      per_speaker, speakers)
        console.print(f"  [dim]{note}[/dim]")
        dub_wav = vdir / f".dub_{rec_id}_{lang}.wav"
        sf.write(str(dub_wav), canvas, sr)

        out_dir = clip_dir(video_id, rec["fmt"], lang=lang)   # ciktilar/dublajlar/
        safe = "".join(c if c.isalnum() or c in " -_" else "_"
                       for c in (rec["title"] or "clip"))[:40].strip().replace(" ", "_")
        out = out_dir / f"{rec['fmt']}_{rec_id}_{lang}_{safe}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(video_path),
             "-i", str(dub_wav), "-map", "0:v", "-map", "1:a", "-shortest",
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out)],
            check=True,
        )
        if captions and cap_words:
            from .render import _video_dims, burn_word_captions
            console.print("  altyazı (çeviri metninden, kelime-kelime)…")
            ow, oh = _video_dims(out)
            tmp = out.with_name(out.stem + ".cap.mp4")
            burn_word_captions(out, tmp, cap_words, ow, oh)
            tmp.replace(out)
        dub_wav.unlink(missing_ok=True)
        db.set_stage(video_id, "dub", "done", f"{rec['fmt']} #{rec_id} {lang}")
    except Exception as exc:  # noqa: BLE001
        db.set_stage(video_id, "dub", "error", str(exc))
        console.print(f"  [red]hata:[/red] {exc}")
        raise
    console.print(f"  [green]✓[/green] dublaj hazır  •  [dim]{out}[/dim]")
