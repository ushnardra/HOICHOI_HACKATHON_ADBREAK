"""Checkpoint 10 helper: makes a simple synthetic ad video for every brand in config/brands.json (and the
house promo), so the player has something to cut to. Clearly labelled as synthetic demo ads.

Usage: python -m adbreak.creatives [--extra-brand tests/data/brand_09_unseen.json]
"""
import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

from adbreak.common import FFMPEG, OUTPUT, ROOT, load_json

W, H, FPS = 960, 540, 25
def _font(windows_name, linux_path):
    """Windows laptop: C:/Windows/Fonts; Linux server: DejaVu (fonts-dejavu-core). Escaped for ffmpeg drawtext."""
    for p in (Path("C:/Windows/Fonts") / windows_name, Path(linux_path)):
        if p.exists():
            return str(p).replace(chr(92), "/").replace(":", chr(92) + ":")
    return ""


FONT_BOLD = _font("segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT = _font("segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
HOUSE = {"id": "house_promo", "name": "Up next on this channel", "category": "house promo",
         "target_contexts": ["more Bengali stories after the break"], "duration_sec": 15}


def colour(brand_id):
    h = hashlib.md5(brand_id.encode()).hexdigest()
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"0x{r // 2 + 30:02x}{g // 2 + 30:02x}{b // 2 + 30:02x}"  # darker colours keep white text readable


def make(brand, out_dir):
    out = out_dir / f"{brand['id']}.mp4"
    dur = brand.get("duration_sec", 30)
    with tempfile.TemporaryDirectory() as td:
        texts = {"name": brand["name"], "cat": brand["category"].upper(),
                 "tag": brand["target_contexts"][0][:1].upper() + brand["target_contexts"][0][1:], "label": "SYNTHETIC DEMO AD"}
        files = {}
        for k, v in texts.items():
            files[k] = Path(td) / f"{k}.txt"
            files[k].write_text(v, encoding="utf-8")
        esc = lambda p: str(p).replace("\\", "/").replace(":", "\\:")
        fade = f"fade=t=in:st=0:d=0.6,fade=t=out:st={dur - 0.6}:d=0.6"
        vf = (f"drawtext=fontfile='{FONT_BOLD}':textfile='{esc(files['name'])}':fontsize=64:fontcolor=white:"
              f"x=(w-text_w)/2:y=(h-text_h)/2-40,"
              f"drawtext=fontfile='{FONT}':textfile='{esc(files['cat'])}':fontsize=26:fontcolor=white@0.85:"
              f"x=(w-text_w)/2:y=(h/2)+40,"
              f"drawtext=fontfile='{FONT}':textfile='{esc(files['tag'])}':fontsize=24:fontcolor=white@0.75:"
              f"x=(w-text_w)/2:y=(h/2)+85,"
              f"drawtext=fontfile='{FONT}':textfile='{esc(files['label'])}':fontsize=16:fontcolor=white@0.6:x=20:y=h-36,"
              f"{fade}")
        cmd = [str(FFMPEG), "-y", "-v", "error",
               "-f", "lavfi", "-i", f"color=c={colour(brand['id'])}:s={W}x{H}:r={FPS}:d={dur}",
               "-f", "lavfi", "-i", f"sine=frequency=330:sample_rate=48000:duration={dur}",
               "-filter_complex", f"[0:v]{vf}[v];[1:a]volume=0.05,afade=t=in:d=0.6,afade=t=out:st={dur - 0.6}:d=0.6[a]",
               "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
    return out


def main(extra_brand=None):
    out_dir = OUTPUT / "creatives"
    out_dir.mkdir(parents=True, exist_ok=True)
    brands = load_json(ROOT / "config" / "brands.json")["brands"]
    if extra_brand:
        brands = brands + [load_json(extra_brand)["brand"]]
    made = [make(b, out_dir) for b in brands + [HOUSE]]
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-brand", default=None)
    for p in main(ap.parse_args().extra_brand):
        print(p.name, round(p.stat().st_size / 1e3), "kB")
