"""Checkpoint 1: video -> audio.wav (16 kHz mono) + info.json.

Usage: python -m adbreak.ingest <video>
"""
import json
import subprocess
import sys
import time

from adbreak.common import FFMPEG, FFPROBE, save_json, video_path, work_dir


def probe(video) -> dict:
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    raw = json.loads(out)
    v = next(s for s in raw["streams"] if s["codec_type"] == "video")
    a = next((s for s in raw["streams"] if s["codec_type"] == "audio"), None)
    num, den = v["r_frame_rate"].split("/")
    return {
        "file": video.name,
        "duration_sec": round(float(raw["format"]["duration"]), 3),
        "size_mb": round(int(raw["format"]["size"]) / 1e6, 1),
        "video": {"codec": v["codec_name"], "width": v["width"], "height": v["height"],
                  "fps": round(int(num) / int(den), 3)},
        "audio": None if a is None else {"codec": a["codec_name"], "sample_rate": int(a["sample_rate"]),
                                         "channels": a["channels"]},
    }


def extract_audio(video, wav) -> None:
    subprocess.run(
        [str(FFMPEG), "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )


def main(name: str) -> dict:
    video = video_path(name)
    out = work_dir(video.name)
    t0 = time.time()
    info = probe(video)
    if info["audio"] is None:
        raise RuntimeError("video has no audio track")
    wav = out / "audio.wav"
    extract_audio(video, wav)
    info["audio_wav"] = {"path": str(wav.relative_to(out.parent.parent)), "sample_rate": 16000, "channels": 1,
                         "size_mb": round(wav.stat().st_size / 1e6, 1)}
    info["ingest_seconds"] = round(time.time() - t0, 1)
    save_json(out / "info.json", info)
    return info


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=2))
