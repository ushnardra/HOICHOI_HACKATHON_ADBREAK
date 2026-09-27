"""Shared paths and helpers. Everything stays inside the project folder (or ADBREAK_DATA when hosted)."""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    """Read KEY=value lines from ROOT/.env (secrets such as GROQ_API_KEY). Never printed, never committed."""
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() and v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

# Where videos, intermediate files and outputs live. Hosted: set ADBREAK_DATA (e.g. /data) to keep them apart.
DATA = Path(os.environ.get("ADBREAK_DATA", ROOT))
ASSETS = DATA / "assets"
WORK = DATA / "work"
OUTPUT = DATA / "output"
CACHE = ROOT / ".cache"
MODELS = Path(os.environ.get("ADBREAK_MODELS", ROOT / "models"))

# ffmpeg: the copy inside the project (Windows laptop), otherwise the system one (Linux server).
_local_ffmpeg = ROOT / "tools" / "ffmpeg" / "bin"
FFMPEG = _local_ffmpeg / "ffmpeg.exe" if (_local_ffmpeg / "ffmpeg.exe").exists() else Path(shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = _local_ffmpeg / "ffprobe.exe" if (_local_ffmpeg / "ffprobe.exe").exists() else Path(shutil.which("ffprobe") or "ffprobe")

# Keep every model download inside the project.
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))


def device():
    """'cuda' if a GPU is available (laptop), else 'cpu' (free hosting). ADBREAK_DEVICE=cpu forces the CPU
    (used to measure how fast the free server will be)."""
    import torch
    if os.environ.get("ADBREAK_DEVICE", "").lower() == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype():
    """Half precision on the GPU (saves memory), full precision on the CPU (fp16 is slow / unsupported there)."""
    import torch
    return torch.float16 if device() == "cuda" else torch.float32


def free_gpu():
    import torch
    if device() == "cuda":
        torch.cuda.empty_cache()


def video_path(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    p = ASSETS / (name if name.endswith(".mp4") else f"{name}.mp4")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def work_dir(name: str) -> Path:
    d = WORK / Path(name).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
