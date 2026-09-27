"""Checkpoint 8 (WHETHER): from the safe break points of checkpoint 7, choose which ones to actually use.

1. Quality gate (config/pacing.json "quality_gate"): drop weak candidates -
   WHERE score too low, sentence not finished, loud music at the break (if a quieter candidate exists),
   next to a sensitive scene (viewer comfort; brand safety is still enforced separately in checkpoint 9).
2. Pacing rules (config/pacing.json "pacing"): no break in the first / last minutes, at most N breaks
   (max_breaks_per_hour scaled to the video length), at least min_gap_sec between breaks,
   total ad time <= max_ad_load_pct of the video.
3. Selection: the combination of breaks with the highest total WHERE score that obeys every rule
   (dynamic programming, so it is the best combination, not just the first good breaks).
Every candidate gets a status and a reason.

Usage: python -m adbreak.pacing <video>
"""
import json
import math
import sys
import time

from adbreak.common import ROOT, load_json, save_json, video_path, work_dir

POLICY = ROOT / "config" / "pacing.json"


def allowed_breaks(duration, pacing):
    raw = duration / 3600 * pacing["max_breaks_per_hour"]
    rounding = pacing["rounding_for_short_videos"]
    by_rule = math.ceil(raw - 1e-9) if rounding == "ceil" else math.floor(raw + 0.5) if rounding == "nearest" else math.floor(raw)
    if pacing.get("max_breaks_per_video") is not None:  # team setting overrides the per-hour rule
        by_rule = int(pacing["max_breaks_per_video"])
    by_load = math.floor(pacing["max_ad_load_pct"] / 100 * duration / pacing["break_duration_sec"])
    return max(0, min(by_rule, by_load)), round(raw, 2), by_load


def best_combination(cands, k_max, min_gap):
    """Pick up to k_max candidates (sorted by time) with gaps >= min_gap maximising total score.
    Tie-break: more even spacing. Small exact DP."""
    n = len(cands)
    best = {(): 0.0}
    order = sorted(range(n), key=lambda i: cands[i]["t"])

    def ok(combo, i):
        return all(abs(cands[i]["t"] - cands[j]["t"]) >= min_gap for j in combo)

    combos = [()]
    for i in order:  # grow combinations; n and k_max are small (tens of candidates, a few breaks)
        new = []
        for c in combos:
            if len(c) < k_max and ok(c, i):
                new.append(c + (i,))
        combos += new
    duration = max(c["t"] for c in cands) if cands else 0

    def key(c):
        total = sum(cands[i]["gated_score"] for i in c)
        ts = sorted(cands[i]["t"] for i in c)
        gaps = [b - a for a, b in zip([0] + ts, ts + [duration])] if ts else [0]
        spread = -(max(gaps) - min(gaps)) / max(1, duration)  # more even = larger
        return (round(total, 6), spread)

    return max(combos, key=key)


def main(name):
    video = video_path(name)
    out = work_dir(video.name)
    t0 = time.time()
    pol = load_json(POLICY)
    pacing, gate = pol["pacing"], pol["quality_gate"]
    info = load_json(out / "info.json")
    dur = info["duration_sec"]
    br = load_json(out / "breaks.json")
    pct = pacing.get("short_video_buffer_pct", 100) / 100  # short clips: buffers shrink with the video
    head = min(pacing["no_break_first_sec"], pct * dur)
    tail = min(pacing["no_break_last_sec"], pct * dur)

    cands = []
    for c in br["candidates"]:
        row = {"cut": c["cut"], "accepted_by_where": c["accepted"], "status": None, "why": []}
        if not c["accepted"]:
            row.update(status="rejected (WHERE)", why=[c["rejected_because"]])
            cands.append(row)
            continue
        b = c["break"]
        row.update(t=b["t"], where_score=b["score"], loudness=b["loudness_vs_speech"], reasons=c["reasons"],
                   scene_before=c["scene_before"], scene_after=c["scene_after"])
        fails = []
        if b["t"] < head:
            fails.append(f"in the first {head:.0f} s")
        if b["t"] > dur - tail:
            fails.append(f"in the last {tail:.0f} s")
        if b["score"] < gate["min_where_score"]:
            fails.append(f"WHERE score {b['score']:.2f} < {gate['min_where_score']}")
        if gate["require_sentence_finished"] and b["parts"]["sentence"] < 1.0:
            fails.append("sentence may not be finished")
        near = c["scene_before"]["sensitive"] or c["scene_after"]["sensitive"]
        side = "before" if c["scene_before"]["sensitive"] else "after"
        if gate["avoid_breaks_next_to_sensitive"] is True and near:
            fails.append(f"next to a sensitive scene ({side} the break)")
        row["near_sensitive"] = bool(near)
        row["gate_fails"] = fails
        cands.append(row)

    # loud music: reject if any quieter candidate passes the gate, otherwise halve the score
    passing = [r for r in cands if r.get("gate_fails") == []]
    quiet_exists = any(r["loudness"] < gate["loud_music_limit"] for r in passing)
    for r in passing:
        r["gated_score"] = r["where_score"]
        if gate["avoid_breaks_next_to_sensitive"] == "prefer" and r["near_sensitive"]:
            r["gated_score"] = round(r["where_score"] * gate["sensitive_penalty"], 3)
            r["why"].append(f"next to a sensitive scene: score x {gate['sensitive_penalty']} "
                            "(brand hard block applies in checkpoint 9)")
        if r["loudness"] >= gate["loud_music_limit"]:
            if quiet_exists:
                r["gate_fails"] = [f"loud music at the break ({r['loudness'] * 100:.0f}% of speech) and quieter candidates exist"]
            else:
                r["gated_score"] = round(r["where_score"] * 0.5, 3)
                r["why"].append("loud music at the break: score halved (no quieter candidate)")
    passing = [r for r in cands if r.get("gate_fails") == []]
    for r in cands:
        if r.get("gate_fails"):
            r["status"], r["why"] = "dropped (quality / pacing window)", r["gate_fails"]

    k, raw, by_load = allowed_breaks(dur, pacing)
    chosen = best_combination(passing, k, pacing["min_gap_sec"]) if passing and k else ()
    chosen_rows = [passing[i] for i in chosen]
    chosen_t = sorted(r["t"] for r in chosen_rows)
    for r in passing:
        if r in chosen_rows:
            r["status"] = "SELECTED"
            gaps = [abs(r["t"] - t) for t in chosen_t if t != r["t"]]
            r["why"] = [f"WHERE score {r['where_score']:.2f}"] + r["why"] + r["reasons"] + \
                       [f"{min(gaps) / 60:.1f} min from the next break" if gaps else "only break"]
        else:
            close = [t for t in chosen_t if abs(t - r["t"]) < pacing["min_gap_sec"]]
            r["status"] = "not used (pacing)"
            r["why"] = [f"within {pacing['min_gap_sec'] // 60} min of a better break at {mm(close[0])}" if close
                        else f"only {k} break(s) allowed; better-scoring breaks were chosen"]
    ad_time = len(chosen_rows) * pacing["break_duration_sec"]
    result = {"video": video.name, "policy_file": str(POLICY.relative_to(ROOT)), "policy": pol,
              "allowed": {"by_breaks_per_hour": raw, "by_ad_load": by_load, "final": k},
              "buffers_sec": {"start": round(head, 1), "end": round(tail, 1)},
              "stats": {"where_candidates": sum(r["accepted_by_where"] for r in cands),
                        "passed_quality_gate": len(passing), "selected": len(chosen_rows),
                        "ad_time_sec": ad_time, "ad_load_pct": round(100 * ad_time / dur, 2),
                        "min_gap_between_selected_sec": round(min((b - a for a, b in zip(chosen_t, chosen_t[1:])), default=0), 1)},
              "selected": [{"t": r["t"], "cut": r["cut"], "where_score": r["where_score"], "gated_score": r["gated_score"],
                            "near_sensitive": r["near_sensitive"], "why": r["why"],
                            "scene_before": r["scene_before"], "scene_after": r["scene_after"]}
                           for r in sorted(chosen_rows, key=lambda r: r["t"])],
              "candidates": cands, "seconds": round(time.time() - t0, 2)}
    save_json(out / "pacing.json", result)
    return result


def mm(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k in ("allowed", "stats")}, indent=2))
    for c in r["candidates"]:
        t = c.get("t", c["cut"])
        print(f"{mm(t):8} {c['status']:34} {'; '.join(c['why'])[:120]}")
