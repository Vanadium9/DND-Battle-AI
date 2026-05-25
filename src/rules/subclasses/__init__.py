"""Data-driven subclass definitions for the supported ruleset."""

from __future__ import annotations

from dataclasses import dataclass, field

from combat.class_features import FeatureDefinition


@dataclass(frozen=True)
class SubclassDefinition:
    """Progression table for one subclass."""

    name: str
    parent_class: str
    level_features: dict[int, tuple[FeatureDefinition, ...]] = field(
        default_factory=dict
    )

    def features_for_level(self, level: int) -> tuple[FeatureDefinition, ...]:
        """Return subclass features available up to the requested level."""

        normalized_level = int(level)
        return tuple(
            feature
            for feature_level, features in sorted(self.level_features.items())
            if feature_level <= normalized_level
            for feature in features
        )


def _subclass_key(parent_class: str, subclass_name: str) -> str:
    return f"{_lookup_key(parent_class)}:{_lookup_key(subclass_name)}"


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


from rules.subclasses.champion import CHAMPION_DEFINITION  # noqa: E402
from rules.subclasses.life_domain import LIFE_DOMAIN_DEFINITION  # noqa: E402
from rules.subclasses.evocation import EVOCATION_DEFINITION  # noqa: E402


SUBCLASS_DEFINITIONS: dict[str, SubclassDefinition] = {
    _subclass_key(CHAMPION_DEFINITION.parent_class, CHAMPION_DEFINITION.name): (
        CHAMPION_DEFINITION
    ),
    _subclass_key(LIFE_DOMAIN_DEFINITION.parent_class, LIFE_DOMAIN_DEFINITION.name): (
        LIFE_DOMAIN_DEFINITION
    ),
    _subclass_key(EVOCATION_DEFINITION.parent_class, EVOCATION_DEFINITION.name): (
        EVOCATION_DEFINITION
    ),
}


def get_subclass_definition(
    parent_class: str | None,
    subclass_name: str | None,
) -> SubclassDefinition | None:
    """Return a subclass definition by parent class and subclass name."""

    if parent_class is None or subclass_name is None:
        return None
    return SUBCLASS_DEFINITIONS.get(_subclass_key(parent_class, subclass_name))


def get_subclasses_for_class(parent_class: str | None) -> tuple[SubclassDefinition, ...]:
    """Return all known subclasses for one parent class."""

    if parent_class is None:
        return ()
    parent_key = _lookup_key(parent_class)
    return tuple(
        definition
        for definition in SUBCLASS_DEFINITIONS.values()
        if _lookup_key(definition.parent_class) == parent_key
    )
