"""
download_judge.py

Downloads Gemma 4 E4B-it into the local model directory for offline use:
  ./models/gemma-4-E4B

The script reads HF_TOKEN from a .env file (or environment variables).

Run:
  python chart-incident-management/download_judge.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


REPO_ID = "google/gemma-4-E4B-it"
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DIR = SCRIPT_DIR / "models" / "gemma-4-E4B"


def load_env_token() -> str:
    """
    Load HF token from .env and environment variables.
    """
    # Load nearest .env if present in current working directory first.
    load_dotenv()

    # Also attempt repo-root .env (one level above this script dir).
    repo_env = SCRIPT_DIR.parent / ".env"
    if repo_env.exists():
        load_dotenv(dotenv_path=repo_env, override=False)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF token not found. Add HF_TOKEN to your .env or export HF_TOKEN in your shell."
        )
    return token


def main() -> None:
    token = load_env_token()
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} -> {LOCAL_DIR}")
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()

