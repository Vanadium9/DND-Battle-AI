"""Validation for internally stored GUI characters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from character.schema import DEFAULT_CHARACTER_STATS, InternalCharacter
from combat.damage import coerce_damage_type
from rules.progression import get_proficiency_bonus
from rules.registry import RulesetRegistry, get_registry


@dataclass(frozen=True)
class ValidationIssue:
    """One character validation issue."""

    field: str
    message: str


class CharacterValidationError(ValueError):
    """Raised when an InternalCharacter cannot be saved."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        message = "; ".join(
            f"{issue.field}: {issue.message}" for issue in self.issues
        )
        super().__init__(message or "Character validation failed")


def validate_character(
    character: InternalCharacter,
    registry: RulesetRegistry | None = None,
) -> None:
    """Validate a character against core invariants and the active ruleset."""

    ruleset_registry = registry or get_registry()
    issues: list[ValidationIssue] = []

    _validate_required_text(issues, "id", character.id)
    _validate_required_text(issues, "name", character.name)
    _validate_supported(
        issues,
        ruleset_registry,
        "level",
        character.level,
        field="level",
    )
    _validate_supported(
        issues,
        ruleset_registry,
        "class",
        character.class_name,
        field="class_name",
    )
    if character.subclass_name:
        _validate_supported(
            issues,
            ruleset_registry,
            "subclass",
            f"{character.class_name}:{character.subclass_name}",
            field="subclass_name",
        )
    _validate_supported(
        issues,
        ruleset_registry,
        "race",
        character.race_name,
        field="race_name",
    )
    _validate_stats(issues, character)
    _validate_positive(issues, "hp", character.hp)
    _validate_positive(issues, "ac", character.ac)
    if character.speed < 0:
        issues.append(ValidationIssue("speed", "must be non-negative"))

    expected_proficiency = get_proficiency_bonus(character.level)
    if character.proficiency_bonus != expected_proficiency:
        issues.append(
            ValidationIssue(
                "proficiency_bonus",
                (
                    f"must be {expected_proficiency} for level "
                    f"{character.level}"
                ),
            )
        )

    _validate_feats(issues, ruleset_registry, character)
    _validate_spell_slots(issues, ruleset_registry, character)
    _validate_non_negative_mapping(issues, "resources", character.resources)
    _validate_inventory(issues, character)
    _validate_damage_types(issues, "resistances", character.resistances)
    _validate_damage_types(issues, "immunities", character.immunities)
    _validate_damage_types(issues, "vulnerabilities", character.vulnerabilities)

    if issues:
        raise CharacterValidationError(issues)


def _validate_required_text(
    issues: list[ValidationIssue],
    field: str,
    value: str,
) -> None:
    if not value.strip():
        issues.append(ValidationIssue(field, "must not be empty"))


def _validate_supported(
    issues: list[ValidationIssue],
    registry: RulesetRegistry,
    content_type: str,
    value: str | int,
    *,
    field: str,
) -> None:
    if registry.is_supported_content(content_type, value):
        return
    issues.append(
        ValidationIssue(
            field,
            registry.get_unsupported_reason(content_type, value),
        )
    )


def _validate_stats(
    issues: list[ValidationIssue],
    character: InternalCharacter,
) -> None:
    missing = sorted(set(DEFAULT_CHARACTER_STATS) - set(character.stats))
    if missing:
        issues.append(
            ValidationIssue("stats", f"missing abilities: {', '.join(missing)}")
        )
    for ability, value in character.stats.items():
        if ability not in DEFAULT_CHARACTER_STATS:
            issues.append(ValidationIssue(f"stats.{ability}", "unknown ability"))
            continue
        if value <= 0:
            issues.append(ValidationIssue(f"stats.{ability}", "must be positive"))


def _validate_positive(
    issues: list[ValidationIssue],
    field: str,
    value: int,
) -> None:
    if value <= 0:
        issues.append(ValidationIssue(field, "must be positive"))


def _validate_feats(
    issues: list[ValidationIssue],
    registry: RulesetRegistry,
    character: InternalCharacter,
) -> None:
    for feat in character.feats:
        if registry.is_supported_content("feat", feat):
            continue
        issues.append(
            ValidationIssue(
                "feats",
                registry.get_unsupported_reason("feat", feat),
            )
        )


def _validate_spell_slots(
    issues: list[ValidationIssue],
    registry: RulesetRegistry,
    character: InternalCharacter,
) -> None:
    for level, count in character.spell_slots.items():
        if int(count) < 0:
            issues.append(ValidationIssue(f"spell_slots.{level}", "must be non-negative"))
        try:
            level_value = int(level)
        except (TypeError, ValueError):
            issues.append(ValidationIssue(f"spell_slots.{level}", "invalid spell level"))
            continue
        if registry.is_supported_content("spell_level", level_value):
            continue
        issues.append(
            ValidationIssue(
                f"spell_slots.{level}",
                registry.get_unsupported_reason("spell_level", level_value),
            )
        )
    for spell in character.spells:
        if not isinstance(spell, dict) or "level" not in spell:
            continue
        try:
            level = int(spell["level"])
        except (TypeError, ValueError):
            issues.append(ValidationIssue("spells", "invalid spell level"))
            continue
        if registry.is_supported_content("spell_level", level):
            continue
        issues.append(
            ValidationIssue(
                "spells",
                registry.get_unsupported_reason("spell_level", level),
            )
        )


def _validate_non_negative_mapping(
    issues: list[ValidationIssue],
    field: str,
    values: dict[str, int],
) -> None:
    for name, value in values.items():
        if int(value) < 0:
            issues.append(ValidationIssue(f"{field}.{name}", "must be non-negative"))


def _validate_inventory(
    issues: list[ValidationIssue],
    character: InternalCharacter,
) -> None:
    for index, item in enumerate(character.inventory):
        quantity = int(item.get("quantity", 0))
        if quantity < 0:
            issues.append(
                ValidationIssue(f"inventory.{index}.quantity", "must be non-negative")
            )


def _validate_damage_types(
    issues: list[ValidationIssue],
    field: str,
    values: tuple[str, ...],
) -> None:
    for value in values:
        if coerce_damage_type(value) is not None:
            continue
        issues.append(ValidationIssue(field, f"unknown damage type '{value}'"))
