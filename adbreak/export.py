"""Checkpoint 10: writes the deliverables for one video.

  output/<video>.vmap.xml    VMAP 1.0 manifest (IAB). One <vmap:AdBreak> per selected break, timeOffset =
                             the exact break time from checkpoint 7/8; each break carries its ad INLINE as a
                             VAST 3.0 <InLine> ad (<vmap:VASTAdData>), so no ad server is needed.
  output/<video>.debug.json  every decision in one file: video, policy, scenes + context + safety,
                             WHERE candidates (with vetoes), WHETHER choices, WHAT rankings + blocks, safety assert.

Usage: python -m adbreak.export <video> [--base-url http://localhost:8000]
"""
import argparse
import json
import sys
import time
from xml.sax.saxutils import escape

from adbreak.common import OUTPUT, ROOT, load_json, save_json, video_path, work_dir

VMAP_NS = "http://www.iab.net/videosuite/vmap"


def hms(t, ms=True):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}" if ms else f"{int(h):02d}:{int(m):02d}:{int(round(s)):02d}"


def vast_inline(ad_id, title, category, media_url, duration, why):
    return f"""<VAST version="3.0">
          <Ad id="{escape(ad_id)}">
            <InLine>
              <AdSystem>adbreak-demo</AdSystem>
              <AdTitle>{escape(title)}</AdTitle>
              <Description>{escape(category)} (synthetic demo brand)</Description>
              <Extensions><Extension type="why"><![CDATA[{why}]]></Extension></Extensions>
              <Creatives>
                <Creative id="{escape(ad_id)}-creative" sequence="1">
                  <Linear>
                    <Duration>{hms(duration, ms=False)}</Duration>
                    <MediaFiles>
                      <MediaFile delivery="progressive" type="video/mp4" width="960" height="540"><![CDATA[{media_url}]]></MediaFile>
                    </MediaFiles>
                  </Linear>
                </Creative>
              </Creatives>
            </InLine>
          </Ad>
        </VAST>"""


def build_vmap(placements, brands, base_url):
    by_id = {b["id"]: b for b in brands}
    breaks = []
    for k, p in enumerate(placements, 1):
        b = by_id.get(p["brand"], {"name": p["brand_name"], "category": "house promo", "duration_sec": 15})
        url = f"{base_url.rstrip('/')}/creatives/{p['brand']}.mp4"
        why = p["why"].replace("]]>", "] ]>")
        breaks.append(f"""  <vmap:AdBreak timeOffset="{hms(p['t'])}" breakType="linear" breakId="midroll-{k}">
    <vmap:AdSource id="midroll-{k}-ad" allowMultipleAds="false" followRedirects="true">
      <vmap:VASTAdData>
        {vast_inline(p['brand'], b['name'], b['category'], url, b.get('duration_sec', 30), why)}
      </vmap:VASTAdData>
    </vmap:AdSource>
  </vmap:AdBreak>""")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<vmap:VMAP xmlns:vmap="{VMAP_NS}" version="1.0">\n' + "\n".join(breaks) + "\n</vmap:VMAP>\n"


def build_debug(name):
    video = video_path(name)
    out = work_dir(video.name)
    info = load_json(out / "info.json")
    ctx = load_json(out / "context.json")
    br = load_json(out / "breaks.json")
    pac = load_json(out / "pacing.json")
    mat = load_json(out / "match.json")
    scenes5 = load_json(out / "scenes.json")
    return {
        "video": {"file": video.name, "duration_sec": info["duration_sec"], "resolution": f'{info["video"]["width"]}x{info["video"]["height"]}',
                  "fps": info["video"]["fps"]},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"pacing": pac["policy"], "brand_matching": load_json(ROOT / "config" / "brand_matching.json"),
                   "context_labels_file": "config/context_labels.json", "brands_file": "config/brands.json"},
        "summary": {"shots": load_json(out / "shots.json")["shot_count"], "scenes": len(ctx["scenes"]),
                    "sensitive_scenes": ctx["stats"]["sensitive_scenes"], "scene_changes": br["stats"]["scene_changes"],
                    "safe_break_points": br["stats"]["accepted"], "naive_cuts_mid_speech": br["stats"]["naive_cut_would_interrupt_speech"],
                    "selected_breaks": pac["stats"]["selected"], "ad_load_pct": pac["stats"]["ad_load_pct"],
                    "house_promos": mat["stats"]["house_promos"], "safety_assert": mat["safety_assert"]},
        "scenes": [{"id": s["id"], "start": s["start"], "end": s["end"], "activity": s["activity"], "setting": s["setting"],
                    "mood": s["mood"], "sensitive": s["sensitive"], "sensitive_reasons": s["reasons"],
                    "tags": s["tags"], "keywords": s["keywords"],
                    "boundary_score": scenes5["scenes"][s["id"]]["boundary_score"]} for s in ctx["scenes"]],
        "where_candidates": [{"scene_change_at": c["cut"], "accepted": c["accepted"],
                              "break_at": c["break"]["t"] if c["break"] else None,
                              "where_score": c["break"]["score"] if c["break"] else None,
                              "score_parts": c["break"]["parts"] if c["break"] else None,
                              "reasons": c["reasons"], "vetoes": c["rejected_because"],
                              "naive_cut_problem": c["naive_cut_problem"]} for c in br["candidates"]],
        "whether": {"allowed_breaks": pac["allowed"], "decisions": [{"t": c.get("t", c["cut"]), "status": c["status"], "why": c["why"]}
                                                                    for c in pac["candidates"]]},
        "placements": [{"break_id": f"midroll-{k}", "t": p["t"], "brand": p["brand"], "brand_name": p["brand_name"],
                        "why": p["why"], "scene_before": p["scene_before"], "scene_after": p["scene_after"],
                        "present_categories": p["present_categories"], "ranking": p["ranking"], "blocked": p["blocked"]}
                       for k, p in enumerate(mat["placements"], 1)],
        "safety_assert": mat["safety_assert"],
    }


def main(name, base_url="http://localhost:8000"):
    video = video_path(name)
    out = work_dir(video.name)
    mat = load_json(out / "match.json")
    brands = load_json(ROOT / "config" / "brands.json")["brands"]
    OUTPUT.mkdir(exist_ok=True)
    vmap = build_vmap(mat["placements"], brands, base_url)
    (OUTPUT / f"{video.stem}.vmap.xml").write_text(vmap, encoding="utf-8")
    debug = build_debug(name)
    save_json(OUTPUT / f"{video.stem}.debug.json", debug)
    return OUTPUT / f"{video.stem}.vmap.xml", OUTPUT / f"{video.stem}.debug.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="video name in assets/ (without .mp4)")
    ap.add_argument("--base-url", default="http://localhost:8000")
    a = ap.parse_args()
    v, d = main(a.name, a.base_url)
    print(v, d)
    print(v.read_text(encoding="utf-8")[:1500])
