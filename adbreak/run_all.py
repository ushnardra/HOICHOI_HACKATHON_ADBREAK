"""Runs the whole pipeline (checkpoints 1-10) for one video, in order, and reports progress step by step.

Usage: python -m adbreak.run_all <video> [<video> ...]
"""
import sys
import time

from adbreak import breaks, context, export, ingest, keyframes, match, music, pacing, scenes, shots, speech, transcribe

# (id, label shown to the user, function)
STEPS = [
    ("ingest", "Reading the video and its audio", ingest.main),
    ("shots", "Finding camera cuts", shots.main),
    ("speech", "Finding when people speak", speech.main),
    ("words", "Transcribing Bengali speech", transcribe.main),
    ("keyframes", "Taking pictures of every shot", keyframes.main),
    ("scenes", "Joining shots into scenes", scenes.main),
    ("music", "Finding songs and music", music.main),
    ("context", "Understanding each scene (safety)", context.main),
    ("where", "WHERE: finding natural pauses", breaks.main),
    ("whether", "WHETHER: choosing breaks (pacing)", pacing.main),
    ("what", "WHAT: matching safe brands", match.main),
    ("export", "Writing the VMAP and debug JSON", export.main),
]


def run(name, log=print, progress=None):
    """progress(step_index, step_id, label, state, seconds) is called with state 'running' / 'done'."""
    times = {}
    for i, (sid, label, fn) in enumerate(STEPS):
        if sid == "music" and not music.ready():
            log(f"  {label:36} skipped (music model not installed)", flush=True)
            continue
        if progress:
            progress(i, sid, label, "running", None)
        t0 = time.time()
        fn(name)
        times[sid] = round(time.time() - t0, 1)
        if progress:
            progress(i, sid, label, "done", times[sid])
        log(f"  {label:36} {times[sid]:7.1f} s", flush=True)
    log(f"  {'total':36} {sum(times.values()):7.1f} s", flush=True)
    return times


if __name__ == "__main__":
    for n in sys.argv[1:]:
        print(n, flush=True)
        run(n)
