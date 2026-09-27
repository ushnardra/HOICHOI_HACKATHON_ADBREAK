"""The demo web app (FastAPI): upload a video, watch it being processed, then play it with its ad breaks.

  GET  /                      home: upload a video (player/upload.html)
  GET  /watch?v=<video>       the player (player/index.html)
  POST /api/upload            upload a video -> a job in the queue (one video is processed at a time)
  GET  /api/jobs/<id>         progress of one job (step by step)
  GET  /api/jobs              recent jobs
  GET  /api/videos            videos that can be played
  GET  /vmap/<video>.xml      VMAP 1.0 manifest, built for the address the page was opened from
  GET  /debug/<video>.json    every decision in one file
  GET  /media/<video>.mp4     the episode (supports Range requests, so the player can seek)
  GET  /creatives/<id>.mp4    the synthetic ad videos

Run:  python -m adbreak.server        then open http://localhost:8000
"""
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from adbreak import export, run_all
from adbreak.common import ASSETS, FFMPEG, FFPROBE, OUTPUT, ROOT, WORK, load_json, save_json

PLAYER = ROOT / "player"
MAX_UPLOAD_MB = int(os.environ.get("ADBREAK_MAX_UPLOAD_MB", 500))
MAX_MINUTES = float(os.environ.get("ADBREAK_MAX_MINUTES", 60))
ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

app = FastAPI(title="Context-aware ad breaks - demo")
app.mount("/player", StaticFiles(directory=PLAYER), name="player")
ASSETS.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)

# ---------------- job queue (one video at a time: the free server has 2 CPU cores) ----------------
JOBS_FILE = WORK / "jobs.json"
JOBS = load_json(JOBS_FILE) if JOBS_FILE.exists() else {}
for j in JOBS.values():  # a restart interrupts running jobs
    if j["state"] in ("queued", "running"):
        j["state"], j["error"] = "error", "the server restarted; please upload again"
LOCK = threading.Lock()
Q = queue.Queue()


def _save_jobs():
    with LOCK:
        save_json(JOBS_FILE, JOBS)


def _worker():
    while True:
        jid = Q.get()
        job = JOBS[jid]
        job.update(state="running", started=time.time())
        _save_jobs()

        def progress(i, sid, label, state, secs):
            job["steps"][sid] = {"label": label, "state": state, "seconds": secs}
            job["current"] = label if state == "running" else job.get("current")
            _save_jobs()

        try:
            run_all.run(job["video"], log=lambda *a, **k: None, progress=progress)
            job.update(state="done", finished=time.time())
        except Exception as e:  # show the reason on the page, keep the server alive
            job.update(state="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        _save_jobs()
        Q.task_done()


threading.Thread(target=_worker, daemon=True).start()


def _probe(path):
    out = subprocess.run([str(FFPROBE), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise HTTPException(400, "this file is not a readable video")
    info = json.loads(out.stdout)
    streams = info.get("streams", [])
    if not any(s.get("codec_type") == "video" for s in streams):
        raise HTTPException(400, "the file has no video track")
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise HTTPException(400, "the file has no audio track (speech is needed to find natural pauses)")
    return float(info["format"].get("duration", 0)), streams


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"please upload a video file ({', '.join(sorted(ALLOWED))})")
    stem = re.sub(r"[^a-z0-9]+", "_", os.path.splitext(file.filename)[0].lower()).strip("_")[:40] or "video"
    name = f"{stem}_{uuid.uuid4().hex[:6]}"
    tmp = ASSETS / f"{name}.upload{ext}"
    size = 0
    with open(tmp, "wb") as fh:
        while chunk := await file.read(4 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                fh.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"the file is larger than {MAX_UPLOAD_MB} MB")
            fh.write(chunk)
    try:
        dur, streams = _probe(tmp)
        if dur > MAX_MINUTES * 60:
            raise HTTPException(400, f"the video is {dur / 60:.0f} min; the demo accepts up to {MAX_MINUTES:.0f} min")
        final = ASSETS / f"{name}.mp4"
        vcodec = next(s["codec_name"] for s in streams if s["codec_type"] == "video")
        if ext == ".mp4" and vcodec == "h264":
            shutil.move(tmp, final)
        else:  # make a browser-playable MP4 (copy if possible, otherwise re-encode)
            cmd = [str(FFMPEG), "-y", "-v", "error", "-i", str(tmp)]
            cmd += (["-c:v", "copy"] if vcodec == "h264" else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"])
            cmd += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(final)]
            subprocess.run(cmd, check=True)
            tmp.unlink(missing_ok=True)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "video": name, "filename": file.filename, "duration_sec": round(dur, 1),
                 "state": "queued", "created": time.time(), "steps": {}, "current": None, "error": None}
    _save_jobs()
    Q.put(jid)
    return {"job": jid, "video": name, "position": Q.qsize()}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    j = dict(JOBS[jid])
    j.pop("trace", None)
    ahead = [x for x in JOBS.values() if x["state"] == "queued" and x["created"] < j["created"]]
    j["queue_ahead"] = len(ahead) + (1 if j["state"] == "queued" and any(x["state"] == "running" for x in JOBS.values()) else 0)
    j["all_steps"] = [{"id": sid, "label": label} for sid, label, _ in run_all.STEPS]
    return j


@app.get("/api/jobs")
def jobs():
    return sorted(({k: v for k, v in j.items() if k != "trace"} for j in JOBS.values()),
                  key=lambda j: -j["created"])[:30]


# ---------------- playing ----------------
def _check_name(name):
    if not re.fullmatch(r"[a-z0-9_]+", name) or not (ASSETS / f"{name}.mp4").exists():
        raise HTTPException(404, "unknown video")
    return name


@app.get("/")
def home():
    return FileResponse(PLAYER / "upload.html")


@app.get("/upload")
def upload_page():
    return FileResponse(PLAYER / "upload.html")


@app.get("/watch")
def watch():
    return FileResponse(PLAYER / "index.html")


@app.get("/api/videos")
def videos():
    out = []
    for v in sorted(ASSETS.glob("*.mp4")):
        done = (WORK / v.stem / "match.json").exists()
        info = WORK / v.stem / "info.json"
        out.append({"name": v.stem, "processed": done,
                    "duration_sec": load_json(info)["duration_sec"] if info.exists() else None})
    return out


@app.get("/vmap/{name}.xml")
def vmap(name: str, request: Request):
    _check_name(name)
    mat = WORK / name / "match.json"
    if not mat.exists():
        raise HTTPException(404, "video not processed yet")
    brands = load_json(ROOT / "config" / "brands.json")["brands"]
    base = os.environ.get("ADBREAK_PUBLIC_URL") or str(request.base_url)
    xml = export.build_vmap(load_json(mat)["placements"], brands, base)
    return Response(xml, media_type="application/xml")


@app.get("/debug/{name}.json")
def debug(name: str):
    _check_name(name)
    if not (WORK / name / "match.json").exists():
        raise HTTPException(404, "video not processed yet")
    return JSONResponse(export.build_debug(name))


@app.get("/media/{name}.mp4")
def media(name: str):
    return FileResponse(ASSETS / f"{_check_name(name)}.mp4", media_type="video/mp4")


@app.get("/creatives/{cid}.mp4")
def creative(cid: str):
    p = OUTPUT / "creatives" / f"{cid}.mp4"
    if not re.fullmatch(r"[a-z0-9_]+", cid) or not p.exists():
        raise HTTPException(404, "unknown creative")
    return FileResponse(p, media_type="video/mp4")


@app.get("/healthz")
def healthz():
    return {"ok": True, "queued": Q.qsize(), "running": sum(j["state"] == "running" for j in JOBS.values())}


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8000))
    # behind Caddy (HTTPS): trust its X-Forwarded-Proto so links in the VMAP are https (no "Not secure" warning)
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")
