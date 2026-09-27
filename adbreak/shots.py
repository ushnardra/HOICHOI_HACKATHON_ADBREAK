"""Checkpoint 2: video -> shots.json (camera cuts) + one keyframe per shot + black/fade segments.

Shots are building blocks for scenes (checkpoint 5). They are NOT ad-break candidates.

Speed: one ffmpeg pass decodes only reference frames (about every 2nd-3rd frame, ~0.08 s apart)
at 192x108 and pipes them into NumPy. Cut detection uses the same idea as PySceneDetect's
AdaptiveDetector (HSV frame difference compared with its neighbours), but vectorised.
~10-15 s for a 26-minute episode instead of ~3 minutes.

Attempt 4 (after checking disputed cuts by eye): ratio lowered 3.0 -> 2.2 and min change 15 -> 12, so
cuts in shaky, dark handheld scenes are found; plus a flash filter, so lightning / lamp flicker
(picture gets brighter for a moment, then the same picture returns) is not counted as a cut.

Usage: python -m adbreak.shots <video>
"""
import json
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from adbreak.common import FFMPEG, save_json, video_path, work_dir

W, H = 192, 108           # decode size (also used for thumbnails)
THUMB_EVERY_SEC = 0.5     # keep one thumbnail candidate per half second
ADAPTIVE_RATIO = 2.2      # cut if a frame's change is 2.2x bigger than its neighbours' ...
MIN_CHANGE = 12.0         # ... and at least this big (0-255 scale)
NEIGHBOURS = 2            # frames on each side used as the "normal" level
MIN_SHOT_SEC = 0.5
BLACK_PIXEL = 26          # a pixel this dark (about 10% brightness) counts as black ...
BLACK_RATIO = 0.98        # ... and a frame is black when 98% of its pixels are (same rule as ffmpeg blackdetect)
MIN_BLACK_SEC = 0.2
FLASH_JUMP = 1.25         # brightness up/down by 25% or more may be a flash ...
FLASH_WINDOW = 13         # ... if within ~1 s (13 decoded frames) ...
FLASH_SAME = 0.3          # ... the picture from before the flash comes back (colour distance below this)


def read_frames(video):
    """Yield (pts_seconds_list, frames_uint8[N,H,W,3]) chunks from a single ffmpeg pass."""
    cmd = [str(FFMPEG), "-v", "info", "-nostats", "-skip_frame", "noref", "-i", str(video), "-an",
           "-vf", f"scale={W}:{H}:flags=area,showinfo", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 7)
    pts = []

    def read_log():
        for line in p.stderr:
            m = re.search(rb"pts_time:([\d.]+)", line)
            if m:
                pts.append(float(m.group(1)))

    log = threading.Thread(target=read_log, daemon=True)
    log.start()
    frame_bytes = W * H * 3
    frames = []
    while True:
        b = p.stdout.read(frame_bytes * 500)
        if not b:
            break
        frames.append(np.frombuffer(b, np.uint8).reshape(-1, H, W, 3))
    p.wait()
    log.join()
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed while reading frames")
    return pts, frames


def analyse(pts, chunks):
    """Per-frame change score, cut strength, brightness, plus sparse thumbnails."""
    n = sum(len(c) for c in chunks)
    if len(pts) != n:
        raise RuntimeError(f"timestamp count {len(pts)} != frame count {n}")
    change = np.zeros(n, np.float32)
    strength = np.zeros(n, np.float32)
    luma = np.zeros(n, np.float32)
    bright = np.zeros(n, np.float32)
    colour = np.zeros((n, 512), np.float32)  # small colour fingerprint per frame, for the flash filter
    thumbs = {}
    prev_hsv = prev_hist = None
    last_thumb = -1.0
    i = 0
    for chunk in chunks:
        for f in chunk:
            small = f[::2, ::2]
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.int16)
            hist = cv2.calcHist([hsv.astype(np.uint8)], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hsv is not None:
                change[i] = np.abs(hsv - prev_hsv).mean()
                strength[i] = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            luma[i] = (hsv[..., 2] < BLACK_PIXEL).mean()  # share of black pixels
            bright[i] = hsv[..., 2].mean()
            fp = cv2.calcHist([hsv.astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(fp, fp)
            colour[i] = fp.ravel()
            if pts[i] - last_thumb >= THUMB_EVERY_SEC:
                thumbs[i] = f.copy()
                last_thumb = pts[i]
            prev_hsv, prev_hist = hsv, hist
            i += 1
    return change, strength, luma, bright, colour, thumbs


def flash_end(i, bright, colour):
    """If the change at frame i is a flash (lightning, lamp flicker) rather than a camera cut,
    return the index of the first frame after the flash; otherwise None."""
    n = len(bright)
    same = lambda a, b: cv2.compareHist(colour[a], colour[b], cv2.HISTCMP_BHATTACHARYYA) < FLASH_SAME
    b0, b1 = bright[i - 1], bright[i]
    if b1 > b0 * FLASH_JUMP:  # got brighter: does the picture from before come back soon?
        for j in range(i + 1, min(n, i + FLASH_WINDOW)):
            if abs(bright[j] - b0) < 0.12 * b0 + 2 and same(i - 1, j):
                return j + 1
    if b0 > b1 * FLASH_JUMP:  # got darker: is this the picture from just before the flash?
        if any(abs(bright[k] - b1) < 0.12 * b1 + 2 and same(k, i) for k in range(max(0, i - FLASH_WINDOW), i - 1)):
            return i + 1
    return None


def find_cuts(pts, change, bright, colour):
    """Adaptive detector: a frame is a cut if its change stands out from its neighbours and is not a flash."""
    n = len(change)
    cuts, last, flashes, skip_until = [], 0.0, [], 0
    for i in range(1, n):
        if i < skip_until:  # still inside a flash we already found
            continue
        lo, hi = max(1, i - NEIGHBOURS), min(n, i + NEIGHBOURS + 1)
        around = np.concatenate([change[lo:i], change[i + 1:hi]])
        base = max(float(around.mean()) if len(around) else 0.0, 1e-3)
        if change[i] >= MIN_CHANGE and change[i] / base >= ADAPTIVE_RATIO and pts[i] - last >= MIN_SHOT_SEC:
            end = flash_end(i, bright, colour)
            if end is not None:
                flashes.append(round((pts[i - 1] + pts[i]) / 2, 3))
                skip_until = end
                continue
            cuts.append(i)
            last = pts[i]
    return cuts, flashes


def find_black(pts, luma):
    segs, start = [], None
    for i, v in enumerate(luma):
        if v >= BLACK_RATIO and start is None:
            start = pts[i]
        elif v < BLACK_RATIO and start is not None:
            if pts[i] - start >= MIN_BLACK_SEC:
                segs.append({"start": round(start, 3), "end": round(pts[i], 3)})
            start = None
    if start is not None and pts[-1] - start >= MIN_BLACK_SEC:
        segs.append({"start": round(start, 3), "end": round(pts[-1], 3)})
    return segs


def main(name: str) -> dict:
    video = video_path(name)
    out = work_dir(video.name)
    kf_dir = out / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    for old in kf_dir.glob("*.jpg"):
        old.unlink()
    duration = json.loads((out / "info.json").read_text(encoding="utf-8"))["duration_sec"]

    t0 = time.time()
    pts, chunks = read_frames(video)
    t_read = time.time() - t0
    change, strength, luma, bright, colour, thumbs = analyse(pts, chunks)
    del chunks
    cut_idx, flashes = find_cuts(pts, change, bright, colour)
    black = find_black(pts, luma)

    # A cut happens between the previous decoded frame and this one; use the midpoint.
    bounds = [0.0] + [round((pts[i - 1] + pts[i]) / 2, 3) for i in cut_idx] + [duration]
    thumb_idx = np.array(sorted(thumbs))
    thumb_pts = np.array([pts[i] for i in thumb_idx])
    shots = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        mid = thumb_idx[int(np.abs(thumb_pts - (s + e) / 2).argmin())]
        kf = f"keyframes/shot_{k:04d}.jpg"
        cv2.imwrite(str(out / kf), cv2.cvtColor(thumbs[mid], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        shots.append({"id": k, "start": s, "end": round(e, 3), "duration": round(e - s, 3), "keyframe": kf,
                      "cut_strength": None if k == 0 else round(float(strength[cut_idx[k - 1]]), 3),
                      "change": None if k == 0 else round(float(change[cut_idx[k - 1]]), 1)})

    durations = np.array([s["duration"] for s in shots])
    result = {
        "video": video.name,
        "method": "ffmpeg reference-frame pipe + adaptive HSV detector + flash filter (attempt 4)",
        "settings": {"decode_size": f"{W}x{H}", "adaptive_ratio": ADAPTIVE_RATIO, "min_change": MIN_CHANGE,
                     "min_shot_sec": MIN_SHOT_SEC, "black_pixel": BLACK_PIXEL, "black_ratio": BLACK_RATIO,
                     "flash_jump": FLASH_JUMP, "flash_window": FLASH_WINDOW, "flash_same": FLASH_SAME},
        "frames_analysed": len(pts),
        "shot_count": len(shots),
        "stats": {"median_shot_sec": round(float(np.median(durations)), 2),
                  "mean_shot_sec": round(float(durations.mean()), 2),
                  "shortest_sec": round(float(durations.min()), 2),
                  "longest_sec": round(float(durations.max()), 2)},
        "black_segments": black,
        "flashes_ignored": flashes,
        "shots": shots,
        "timing": {"read_sec": round(t_read, 1), "total_sec": round(time.time() - t0, 1)},
        "shots_seconds": round(time.time() - t0, 1),
    }
    save_json(out / "shots.json", result)
    return result


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k not in ("shots", "black_segments")}, indent=2))
