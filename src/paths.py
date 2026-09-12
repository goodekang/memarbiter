from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
OUTPUT_DIR = REPO_ROOT / "outputs"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    rooted = REPO_ROOT / path
    if rooted.exists() or path.parts[0] in {"data", "checkpoints", "outputs", "configs"}:
        return rooted
    return path
