"""Checkpoint 5 helper: 3 sharper keyframes per shot (at 20%, 50% and 80% of the shot) at 384x216.

One ffmpeg pass (reference frames only, like checkpoint 2) keeps just the frames nearest to the wanted
times, so memory stays small. Checkpoint 2's 192x108 thumbnails are too small for the picture model.

Usage: python -m adbreak.keyframes <video>
"""
import json
import sys
import time

import cv2
import numpy as np

from adbreak.common import FFMPEG, load_json, save_json, video_path, work_dir
from adbreak.shots import read_frames  # noqa: F401  (same ffmpeg reading idea, different size)

W, H = 384, 216
POSITIONS = (0.2, 0.5, 0.8)


def positions():
    """3 pictures per shot with a GPU; 1 (the middle) on a small CPU server - SigLIP is the slow part there."""
    from adbreak.common import device
    return POSITIONS if device() == "cuda" else (0.5,)


def wanted_times(shots):
    out = []
    for s in shots:
        for k, p in enumerate(positions()):
            out.append((s["id"], k, s["start"] + p * (s["end"] - s["start"])))
    return out


def grab(video, wants):
    import re
    import subprocess
    import threading
    cmd = [str(FFMPEG), "-v", "info", "-nostats", "-skip_frame", "noref", "-i", str(video), "-an",
           "-vf", f"scale={W}:{H}:flags=area,showinfo", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 7)
    pts = []

    def read_log():
        for line in p.stderr:
            m = re.search(rb"pts_time:([\d.]+)", line)
            if m:
                pts.append(float(m.group(1)))

    th = threading.Thread(target=read_log, daemon=True)
    th.start()
    targets = sorted(wants, key=lambda x: x[2])
    best = {}  # (shot, k) -> (distance, frame)
    fb, i, j = W * H * 3, 0, 0
    while True:
        b = p.stdout.read(fb)
        if len(b) < fb:
            break
        while len(pts) <= i:  # the log line for this frame may arrive a moment later
            if not th.is_alive():
                break
            time.sleep(0.001)
        t = pts[i] if i < len(pts) else None
        i += 1
        if t is None:
            continue
        while j < len(targets) and targets[j][2] < t - 0.5:
            j += 1
        k = j
        while k < len(targets) and targets[k][2] <= t + 0.5:
            key = targets[k][:2]
            d = abs(targets[k][2] - t)
            if key not in best or d < best[key][0]:
                best[key] = (d, np.frombuffer(b, np.uint8).reshape(H, W, 3).copy())
            k += 1
    p.wait()
    th.join()
    return best


def main(name):
    video = video_path(name)
    out = work_dir(video.name)
    shots = load_json(out / "shots.json")["shots"]
    kdir = out / "keyframes384"
    kdir.mkdir(exist_ok=True)
    for old in kdir.glob("*.jpg"):
        old.unlink()
    t0 = time.time()
    best = grab(video, wanted_times(shots))
    index = {}
    for (sid, k), (_, frame) in best.items():
        f = f"keyframes384/s{sid:04d}_{k}.jpg"
        cv2.imwrite(str(out / f), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        index.setdefault(sid, []).append(f)
    result = {"video": video.name, "size": f"{W}x{H}", "positions": positions(),
              "frames": {str(s): sorted(v) for s, v in sorted(index.items())},
              "missing_shots": [s["id"] for s in shots if s["id"] not in index],
              "seconds": round(time.time() - t0, 1)}
    save_json(out / "keyframes384.json", result)
    return result


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != "frames"}, indent=2), "frames:", sum(len(v) for v in r["frames"].values()))
