"""Checkpoint 4: audio + speech segments -> transcript.json (Bengali words with start/end times).

How it works
- Speech-to-text runs on Groq's free Whisper large-v3 API: the audio is compressed to ~5 MB (Opus, 16 kHz mono)
  and sent in one request; Groq returns every word with its start/end time (~50 s for a 25-minute episode).
  Heavy speech recognition goes to a free API; the lighter models (scenes, music, brands) run on our own server.
- Every word is clipped to the speech found in checkpoint 3 (Whisper sometimes stretches a word across a pause,
  or invents text over music/silence), looping or broken text is dropped, and Hindi-script letters become Bengali.
- Words are grouped into short windows that start and end in silence (used by the reports and checks).

Safety check used later:  in_word(t, transcript, margin) -> the word being spoken at time t, or None.

Usage: python -m adbreak.transcribe <video> [--from 720 --to 1020]
"""
import argparse
import json
from pathlib import Path
import os
import re
import subprocess
import time
import unicodedata
import zlib

from adbreak import speech
from adbreak.common import FFMPEG, load_json, save_json, video_path, work_dir

MAX_WINDOW_SEC = 15.0      # a window holds at most this much audio (20 s ran out of tokens on fast Bengali)
SPLIT_PAUSE_SEC = 1.0      # a pause this long always starts a new window
PAD_SEC = 0.2              # audio kept before/after the speech in a window
LOOP_COMPRESSION = 2.4     # text that zip-compresses better than this is repeating itself
CLIP_PAD = 0.1             # a word may reach this far past the edge of its speech segment
SENTENCE_END = ("।", "?", "!", ".", "॥")


def make_windows(segments, start=0.0, end=float("inf")):
    wins, cur = [], None
    for s in segments:
        if s["end"] <= start or s["start"] >= end:
            continue
        if cur and s["end"] - cur[0] <= MAX_WINDOW_SEC and s["start"] - cur[1] < SPLIT_PAUSE_SEC:
            cur[1] = s["end"]
        else:
            if cur:
                wins.append(cur)
            cur = [s["start"], s["end"]]
    if cur:
        wins.append(cur)
    out = []
    for a, b in wins:  # a single very long speech segment is cut into pieces Whisper can handle (<30 s)
        while b - a > 28:
            out.append([a, a + 25])
            a += 25
        out.append([a, b])
    return out


def compression(text):
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b))) if b else 0.0


BROKEN = re.compile("[\ufffd\u09d6\u2020\u2021\u201b]")  # replacement char (text cut mid-letter), stray marks
VOWEL_LOOP = re.compile(r"(\S{1,4})\1{2,}")  # a short chunk repeated 3+ times with no space, e.g. "\u09cb\u09af\u09bc" x4


def garbage(text):
    """Text that is broken rather than just wrong: broken characters, vowel-sign loops, or mostly symbols."""
    if not text:
        return False
    letters = sum(ch.isalpha() or "ঀ" <= ch <= "৿" for ch in text)
    symbols = sum(not (ch.isalnum() or ch.isspace() or "ঀ" <= ch <= "৿") for ch in text)
    mostly_symbols = len(text) >= 8 and symbols > 0.25 * max(1, letters + symbols)
    not_a_letter = any(unicodedata.category(ch) == "Cn" for ch in text)  # e.g. "৉", an empty slot in the Bengali block
    return bool(BROKEN.search(text) or VOWEL_LOOP.search(text)) or mostly_symbols or not_a_letter


DEVANAGARI = re.compile("[ऀ-ॣ०-ॿ]")  # Hindi-script letters; the danda "।" (U+0964/65) is shared with Bengali


def devanagari_to_bengali(text):
    """Devanagari and Bengali letters sit at the same place in their Unicode blocks (0x80 apart),
    so most letters convert one-to-one. Used only if the model writes Bengali speech in Hindi script."""
    out = []
    for ch in text:
        c = ord(ch)
        if 0x0900 <= c <= 0x097F and c not in (0x0964, 0x0965):  # keep danda/double danda (shared)
            b = chr(c + 0x80)
            out.append(b if b.isprintable() else ch)
        else:
            out.append(ch)
    return "".join(out)


GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"
GROQ_MAX_BYTES = 24 * 1024 * 1024   # free tier limit is 25 MB per file


def _compress(wav, start, end, out_path):
    """Mono 16 kHz Opus at 24 kb/s: a 26-minute episode is ~5 MB."""
    subprocess.run([str(FFMPEG), "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(wav),
                    "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", str(out_path)], check=True)
    return out_path


def _groq_request(path, log=print):
    import httpx
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env; free key: https://console.groq.com/keys)")
    for attempt in range(6):
        with open(path, "rb") as fh:
            r = httpx.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, timeout=300,
                           files={"file": (Path(path).name, fh, "audio/ogg")},
                           data={"model": GROQ_MODEL, "language": "bn", "response_format": "verbose_json",
                                 "temperature": "0", "timestamp_granularities[]": ["word", "segment"]})
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = float(r.headers.get("retry-after", 0) or 0) or 5 * (attempt + 1)
            log(f"  Groq busy ({r.status_code}), retrying in {wait:.0f} s", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Groq did not answer after several retries")


def groq_transcribe(wav, windows, sp, duration, log=print):
    """Whole audio -> Groq Whisper large-v3 (one request per <= 24 MB chunk) -> windows + words, after clean-up
    (Hindi-script letters converted, garbage/looping text dropped, words outside checkpoint 3's speech dropped)."""
    import tempfile
    # chunks of at most ~60 min, cut in a silence gap, so each compressed file stays well under 25 MB
    cuts, t = [0.0], 0.0
    while duration - t > 3600:
        target = t + 3600
        gap = min((g for g in sp["gaps"] if t + 1800 < (g["start"] + g["end"]) / 2 <= target),
                  key=lambda g: abs((g["start"] + g["end"]) / 2 - target), default=None)
        t = (gap["start"] + gap["end"]) / 2 if gap else target
        cuts.append(t)
    cuts.append(duration)
    raw_words, raw_segs = [], []
    with tempfile.TemporaryDirectory() as td:
        for k, (a, b) in enumerate(zip(cuts, cuts[1:])):
            f = _compress(wav, a, b, Path(td) / f"part{k}.ogg")
            if f.stat().st_size > GROQ_MAX_BYTES:
                raise RuntimeError("compressed audio is larger than Groq's 25 MB limit")
            log(f"  Groq: sending part {k + 1}/{len(cuts) - 1} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)
            res = _groq_request(f, log)
            raw_words += [{"start": a + w["start"], "end": a + w["end"], "word": w["word"].strip()} for w in res.get("words", [])]
            raw_segs += [{"start": a + s["start"], "end": a + s["end"], "text": s["text"].strip()} for s in res.get("segments", [])]
    # drop segments that are garbage / loops, and words outside checkpoint 3's speech (Whisper invents text in music/silence)
    bad = [s for s in raw_segs if garbage(s["text"]) or compression(s["text"]) > LOOP_COMPRESSION]
    # Whisper sometimes stretches a word across a pause (a "word" of 5-25 s). Silero knows exactly where the silence
    # is, so each word is clipped to the speech segment it overlaps most; a word with no speech under it is dropped.
    segs = sp["segments"]
    words = []
    for w in raw_words:
        if not w["word"]:
            continue
        best = max(segs, key=lambda x: min(w["end"], x["end"] + CLIP_PAD) - max(w["start"], x["start"] - CLIP_PAD), default=None)
        if best is None or min(w["end"], best["end"] + CLIP_PAD) - max(w["start"], best["start"] - CLIP_PAD) <= 0:
            continue
        start, end = max(w["start"], best["start"] - CLIP_PAD), min(w["end"], best["end"] + CLIP_PAD)
        mid = (start + end) / 2
        if any(s["start"] <= mid <= s["end"] for s in bad):
            continue
        text = devanagari_to_bengali(w["word"]) if DEVANAGARI.search(w["word"]) else w["word"]
        words.append({"start": round(start, 2), "end": round(max(end, start + 0.05), 2), "word": text})
    out_windows, out_words = [], []
    for n, (a, b) in enumerate(windows):
        ws = [w for w in words if a - PAD_SEC <= (w["start"] + w["end"]) / 2 <= b + PAD_SEC]
        for w in ws:
            w["window"] = n
            w["sentence_end"] = w["word"].endswith(SENTENCE_END)
        segs_here = [s for s in bad if s["end"] > a and s["start"] < b]
        text = " ".join(w["word"] for w in ws)
        out_words += ws
        out_windows.append({"id": n, "start": round(a, 2), "end": round(b, 2), "text": text,
                            "status": "unreliable" if segs_here and not ws else "ok",
                            "script_converted": 0, "words": len(ws), "hit_token_limit": False})
    out_words.sort(key=lambda w: w["start"])
    return out_windows, out_words


def in_word(t, transcript, margin=0.15):
    """Safety check: the word being spoken at time t (with a safety margin), or None."""
    for w in transcript["words"]:
        if w["start"] - margin <= t <= w["end"] + margin:
            return w
    return None


def main(name, start=0.0, end=None, log=print):
    video = video_path(name)
    out = work_dir(video.name)
    info = load_json(out / "info.json")
    end = info["duration_sec"] if end is None else end
    sp = speech.load(video.name)
    windows = make_windows(sp["segments"], start, end)

    t0 = time.time()
    log(f"Groq {GROQ_MODEL}: {len(windows)} speech windows", flush=True)
    wins, words = groq_transcribe(out / "audio.wav", windows, sp, info["duration_sec"], log)
    wins = [w for w in wins if w["end"] > start and w["start"] < end]
    words = [w for w in words if start <= w["start"] <= end]
    t_run = time.time() - t0

    text = "".join(w["word"] for w in words)
    bengali = len(re.findall(r"[ঀ-৿]", text))
    other = sum(1 for ch in text if ch.isalpha() and not "ঀ" <= ch <= "৿")  # Latin, Cyrillic, ...
    letters = bengali + other
    result = {
        "video": video.name,
        "model": f"groq:{GROQ_MODEL}",
        "engine": "groq",
        "range": [round(start, 2), round(end, 2)],
        "settings": {"max_window_sec": MAX_WINDOW_SEC, "split_pause_sec": SPLIT_PAUSE_SEC, "pad_sec": PAD_SEC,
                     "loop_compression": LOOP_COMPRESSION},
        "stats": {"windows": len(wins), "words": len(words),
                  "ok": sum(w["status"] == "ok" for w in wins),
                  "fixed_loop": sum(w["status"] == "fixed_loop" for w in wins),
                  "unreliable": sum(w["status"] == "unreliable" for w in wins),
                  "script_converted": sum(w["script_converted"] > 0 for w in wins),
                  "letters_converted": sum(w["script_converted"] for w in wins),
                  "bengali_letter_pct": round(100 * bengali / max(1, letters), 1),
                  "sentence_ends": sum(w["sentence_end"] for w in words),
                  "hit_token_limit": sum(w["hit_token_limit"] for w in wins)},
        "windows": wins,
        "words": words,
        "timing": {"load_sec": 0.0, "run_sec": round(t_run, 1), "gpu_peak_mb": 0},
    }
    suffix = "" if (start == 0 and end == info["duration_sec"]) else f"_{int(start)}_{int(end)}"
    save_json(out / f"transcript{suffix}.json", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="video name in assets/ (without .mp4)")
    ap.add_argument("--from", dest="start", type=float, default=0.0)
    ap.add_argument("--to", dest="end", type=float, default=None)
    a = ap.parse_args()
    r = main(a.name, a.start, a.end)
    print(json.dumps({k: v for k, v in r.items() if k not in ("windows", "words")}, ensure_ascii=False, indent=2))
