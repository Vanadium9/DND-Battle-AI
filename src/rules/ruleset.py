"""Ruleset description for supported D&D-like 5e content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Ruleset:
    """Supported content surface for a combat ruleset."""

    ruleset_name: str
    supported_levels: tuple[int, ...]
    supported_classes: tuple[str, ...]
    supported_subclasses: dict[str, tuple[str, ...]]
    supported_races: tuple[str, ...]
    supported_common_actions: tuple[str, ...]
    supported_spell_levels: tuple[int, ...]
    supported_content_policy: dict[str, str]
    supported_feats: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Ruleset":
        """Build a ruleset from a YAML-compatible mapping."""

        subclasses = data.get("supported_subclasses", {})
        return cls(
            ruleset_name=str(data["ruleset_name"]),
            supported_levels=_coerce_int_sequence(data.get("supported_levels", ())),
            supported_classes=_coerce_str_tuple(data.get("supported_classes", ())),
            supported_subclasses={
                str(class_name): _coerce_str_tuple(subclass_names)
                for class_name, subclass_names in subclasses.items()
            },
            supported_races=_coerce_str_tuple(data.get("supported_races", ())),
            supported_common_actions=_coerce_str_tuple(
                data.get("supported_common_actions", ())
            ),
            supported_spell_levels=_coerce_int_sequence(
                data.get("supported_spell_levels", ())
            ),
            supported_content_policy={
                str(key): str(value)
                for key, value in data.get("supported_content_policy", {}).items()
            },
            supported_feats=_coerce_str_tuple(data.get("supported_feats", ())),
        )

    def is_supported_content(self, content_type: str, name: str | int) -> bool:
        """Return True when content is explicitly supported by this ruleset."""

        content_key = _normalize_content_type(content_type)
        if content_key == "ruleset":
            return _lookup_key(name) == _lookup_key(self.ruleset_name)
        if content_key == "level":
            return _as_int(name) in self.supported_levels
        if content_key == "class":
            return _contains_name(self.supported_classes, name)
        if content_key == "subclass":
            return self._is_supported_subclass(str(name))
        if content_key == "race":
            return _contains_name(self.supported_races, name)
        if content_key == "common_action":
            return _contains_name(self.supported_common_actions, name)
        if content_key == "spell_level":
            return _as_int(name) in self.supported_spell_levels
        if content_key == "feat":
            return _contains_name(self.supported_feats, name)
        if content_key in {"spell", "feature"}:
            return False
        return False

    def get_unsupported_reason(self, content_type: str, name: str | int) -> str:
        """Explain why content is unavailable, or return an empty string if supported."""

        if self.is_supported_content(content_type, name):
            return ""

        content_key = _normalize_content_type(content_type)
        if content_key == "ruleset":
            return (
                f"Ruleset '{name}' is not active. Active ruleset: "
                f"{self.ruleset_name}."
            )
        if content_key == "level":
            return (
                f"Level '{name}' is outside supported levels "
                f"{_format_int_range(self.supported_levels)}."
            )
        if content_key == "class":
            return _policy_reason(
                self.supported_content_policy,
                "unsupported_classes",
                f"Class '{name}' is not supported by {self.ruleset_name}.",
            )
        if content_key == "subclass":
            return self._unsupported_subclass_reason(str(name))
        if content_key == "race":
            return _policy_reason(
                self.supported_content_policy,
                "unsupported_races",
                f"Race '{name}' is not supported by {self.ruleset_name}.",
            )
        if content_key == "common_action":
            return (
                f"Common action '{name}' is not supported by {self.ruleset_name}. "
                f"Supported actions: {', '.join(self.supported_common_actions)}."
            )
        if content_key == "spell_level":
            return _policy_reason(
                self.supported_content_policy,
                "unsupported_spells",
                (
                    f"Spell level '{name}' is outside supported spell levels "
                    f"{_format_int_range(self.supported_spell_levels)}."
                ),
            )
        if content_key == "spell":
            return _policy_reason(
                self.supported_content_policy,
                "unsupported_spells",
                f"Spell '{name}' is not explicitly supported by {self.ruleset_name}.",
            )
        if content_key == "feat":
            return (
                f"Feat '{name}' is not supported by {self.ruleset_name}. "
                f"Supported feats: {_format_supported_names(self.supported_feats)}."
            )
        if content_key == "feature":
            return _policy_reason(
                self.supported_content_policy,
                "unsupported_features",
                f"Feature '{name}' is not explicitly supported by {self.ruleset_name}.",
            )
        return f"Unknown content type '{content_type}' for ruleset {self.ruleset_name}."

    def _is_supported_subclass(self, name: str) -> bool:
        class_name, subclass_name = _split_subclass_name(name)
        if class_name is not None:
            for supported_class, supported_subclasses in self.supported_subclasses.items():
                if _lookup_key(supported_class) == _lookup_key(class_name):
                    return _contains_name(supported_subclasses, subclass_name)
            return False

        return any(
            _contains_name(supported_subclasses, subclass_name)
            for supported_subclasses in self.supported_subclasses.values()
        )

    def _unsupported_subclass_reason(self, name: str) -> str:
        class_name, subclass_name = _split_subclass_name(name)
        if class_name is not None:
            for supported_class, supported_subclasses in self.supported_subclasses.items():
                if _lookup_key(supported_class) == _lookup_key(class_name):
                    return (
                        f"Subclass '{subclass_name}' is not supported for "
                        f"{supported_class}. Supported subclasses: "
                        f"{', '.join(supported_subclasses)}."
                    )
            return (
                f"Class '{class_name}' has no supported subclasses in "
                f"{self.ruleset_name}."
            )

        supported = [
            f"{class_name}: {subclass_name}"
            for class_name, subclasses in self.supported_subclasses.items()
            for subclass_name in subclasses
        ]
        return (
            f"Subclass '{name}' is not supported by {self.ruleset_name}. "
            f"Supported subclasses: {', '.join(supported)}."
        )


def _normalize_content_type(content_type: str) -> str:
    aliases = {
        "action": "common_action",
        "actions": "common_action",
        "class": "class",
        "classes": "class",
        "commonaction": "common_action",
        "commonactions": "common_action",
        "common_action": "common_action",
        "common_actions": "common_action",
        "feature": "feature",
        "features": "feature",
        "feat": "feat",
        "feats": "feat",
        "level": "level",
        "levels": "level",
        "race": "race",
        "races": "race",
        "ruleset": "ruleset",
        "ruleset_name": "ruleset",
        "spell": "spell",
        "spells": "spell",
        "spelllevel": "spell_level",
        "spelllevels": "spell_level",
        "spell_level": "spell_level",
        "spell_levels": "spell_level",
        "subclass": "subclass",
        "subclasses": "subclass",
        "supported_class": "class",
        "supported_classes": "class",
        "supported_common_action": "common_action",
        "supported_common_actions": "common_action",
        "supported_feat": "feat",
        "supported_feats": "feat",
        "supported_level": "level",
        "supported_levels": "level",
        "supported_race": "race",
        "supported_races": "race",
        "supported_spell_level": "spell_level",
        "supported_spell_levels": "spell_level",
        "supported_subclass": "subclass",
        "supported_subclasses": "subclass",
    }
    normalized = str(content_type).strip().lower().replace("-", "_").replace(" ", "_")
    return aliases.get(normalized, aliases.get(_lookup_key(normalized), normalized))


def _split_subclass_name(name: str) -> tuple[str | None, str]:
    for separator in (":", "/", "|"):
        if separator in name:
            class_name, subclass_name = name.split(separator, 1)
            return class_name.strip(), subclass_name.strip()
    return None, name.strip()


def _coerce_int_sequence(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        stripped = value.strip()
        if "-" in stripped:
            start_text, end_text = stripped.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            return tuple(range(start, end + step, step))
        return (int(stripped),)
    return tuple(int(item) for item in value)


def _coerce_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _contains_name(supported_names: tuple[str, ...], name: str | int) -> bool:
    name_key = _lookup_key(name)
    return any(_lookup_key(supported_name) == name_key for supported_name in supported_names)


def _lookup_key(value: str | int) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _as_int(value: str | int) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _policy_reason(
    supported_content_policy: dict[str, str],
    policy_key: str,
    fallback: str,
) -> str:
    policy = supported_content_policy.get(policy_key)
    if policy is None:
        return fallback
    return f"{fallback} Policy: {policy}."


def _format_int_range(values: tuple[int, ...]) -> str:
    if not values:
        return "none"
    sorted_values = sorted(values)
    if sorted_values == list(range(sorted_values[0], sorted_values[-1] + 1)):
        return f"{sorted_values[0]}-{sorted_values[-1]}"
    return ", ".join(str(value) for value in sorted_values)


def _format_supported_names(values: tuple[str, ...]) -> str:
    if not values:
        return "none"
    return ", ".join(values)
