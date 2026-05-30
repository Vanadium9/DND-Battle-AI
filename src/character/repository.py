"""Filesystem repository for GUI-created internal characters."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from uuid import uuid4

from character.io import load_character_json, save_character_json
from character.schema import InternalCharacter
from character.validation import validate_character
from rules.registry import RulesetRegistry


DEFAULT_CHARACTER_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "characters"
)


class CharacterRepository:
    """Store InternalCharacter records as JSON files for the GUI."""

    def __init__(
        self,
        storage_dir: str | Path = DEFAULT_CHARACTER_DATA_DIR,
        registry: RulesetRegistry | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.registry = registry
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_characters(self) -> list[InternalCharacter]:
        """Return all saved characters sorted by name then id."""

        characters = [
            load_character_json(path)
            for path in sorted(self.storage_dir.glob("*.json"))
        ]
        return sorted(
            characters,
            key=lambda character: (character.name.casefold(), character.id),
        )

    def get_character(self, character_id: str) -> InternalCharacter | None:
        """Load a character by id, or return None when it does not exist."""

        path = self._path_for_id(character_id)
        if not path.exists():
            return None
        return load_character_json(path)

    def save_character(self, character: InternalCharacter) -> InternalCharacter:
        """Validate and save a character, generating an id when needed."""

        character_to_save = character
        if not character_to_save.id:
            character_to_save = character_to_save.with_id(self._generate_id())
        self.validate_character(character_to_save)
        save_character_json(character_to_save, self._path_for_id(character_to_save.id))
        return character_to_save

    def delete_character(self, character_id: str) -> bool:
        """Delete a character by id and return True when a file was removed."""

        path = self._path_for_id(character_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def duplicate_character(self, character_id: str) -> InternalCharacter:
        """Duplicate a saved character with a new id."""

        character = self.get_character(character_id)
        if character is None:
            raise KeyError(f"Character '{character_id}' not found.")
        duplicate = replace(
            character,
            id=self._generate_id(),
            name=f"{character.name} Copy",
        )
        return self.save_character(duplicate)

    def validate_character(self, character: InternalCharacter) -> None:
        """Validate a character before persistence."""

        validate_character(character, self.registry)

    def _generate_id(self) -> str:
        while True:
            candidate = uuid4().hex
            if not self._path_for_id(candidate).exists():
                return candidate

    def _path_for_id(self, character_id: str) -> Path:
        safe_id = _safe_character_id(character_id)
        return self.storage_dir / f"{safe_id}.json"


def _safe_character_id(character_id: str) -> str:
    character_id = str(character_id).strip()
    if not character_id:
        raise ValueError("character_id must not be empty")
    return re.sub(r"[^a-zA-Z0-9_-]", "_", character_id)
