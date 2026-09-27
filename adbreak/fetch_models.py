"""Downloads the models the pipeline needs into MODELS (ADBREAK_MODELS, default models/). Run once;
the Docker build does this on the server.

Bengali speech-to-text uses Groq's free Whisper API, so no speech model is stored here.

Usage: python -m adbreak.fetch_models
"""
from huggingface_hub import snapshot_download

from adbreak.common import MODELS

NEEDED = {
    "siglip-base-patch16-224": ("google/siglip-base-patch16-224",
                                ["config.json", "preprocessor_config.json", "special_tokens_map.json", "spiece.model",
                                 "tokenizer.json", "tokenizer_config.json", "model.safetensors"]),
    "ast-audioset": ("MIT/ast-finetuned-audioset-10-10-0.4593",
                     ["config.json", "preprocessor_config.json", "model.safetensors"]),
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2",
                         ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json",
                          "model.safetensors"]),
}


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    for folder, (repo, files) in NEEDED.items():
        snapshot_download(repo_id=repo, allow_patterns=files, local_dir=MODELS / folder)
        print("ok", folder)


if __name__ == "__main__":
    main()
