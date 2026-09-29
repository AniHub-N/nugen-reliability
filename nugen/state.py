"""IDs we've created on Nugen (documents, benchmark, alignment attempts, model).
Kept in state/state.json so every script can resume. Nothing secret in here."""
import json
import time
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "state" / "state.json"


def load() -> dict:
    if PATH.exists():
        return json.loads(PATH.read_text(encoding="utf-8"))
    return {}


def save(st: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PATH)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
