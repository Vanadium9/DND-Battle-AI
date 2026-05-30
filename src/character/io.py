"""JSON IO helpers for internal characters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from character.schema import InternalCharacter


def character_to_dict(character: InternalCharacter) -> dict[str, Any]:
    """Convert an internal character to a JSON-compatible dict."""

    return character.to_dict()


def character_from_dict(data: dict[str, Any]) -> InternalCharacter:
    """Convert JSON-compatible data to an internal character."""

    return InternalCharacter.from_mapping(data)


def load_character_json(path: str | Path) -> InternalCharacter:
    """Load one internal character JSON file."""

    character_path = Path(path)
    data = json.loads(character_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Character JSON must contain an object: {character_path}")
    return character_from_dict(data)


def save_character_json(character: InternalCharacter, path: str | Path) -> Path:
    """Save one internal character JSON file."""

    character_path = Path(path)
    character_path.parent.mkdir(parents=True, exist_ok=True)
    character_path.write_text(
        json.dumps(character_to_dict(character), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return character_path
