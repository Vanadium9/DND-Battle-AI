"""Class features, feature definitions and per-combat resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FIGHTING_STYLE_ARCHERY = "Archery"
FIGHTING_STYLE_DEFENSE = "Defense"
FIGHTING_STYLE_GREAT_WEAPON_FIGHTING = "Great Weapon Fighting"


@dataclass
class Resource:
    """A finite class or creature resource."""

    name: str
    max_uses: int
    uses_remaining: int | None = None

    def __post_init__(self) -> None:
        if self.max_uses < 0:
            raise ValueError("max_uses must be non-negative")
        if self.uses_remaining is None:
            self.uses_remaining = self.max_uses

    @property
    def available(self) -> bool:
        return (self.uses_remaining or 0) > 0

    def spend(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self.uses_remaining = max(0, (self.uses_remaining or 0) - amount)

    def reset(self) -> None:
        self.uses_remaining = self.max_uses


@dataclass(frozen=True)
class FeatureDefinition:
    """Data-driven class or subclass feature definition."""

    name: str
    level: int = 1
    action_cost: str | None = None
    resource_cost: str | None = None
    passive_hooks: tuple[str, ...] = ()
    active_action: str | None = None
    description: str = ""
    implemented: bool = False
    resource_name: str | None = None
    required_level: int | None = None

    def __post_init__(self) -> None:
        normalized_level = int(self.required_level or self.level)
        resource_name = self.resource_name
        resource_cost = self.resource_cost
        if resource_name is None and isinstance(resource_cost, str):
            resource_name = resource_cost
        if resource_cost is None and resource_name is not None:
            resource_cost = resource_name

        object.__setattr__(self, "level", normalized_level)
        object.__setattr__(self, "required_level", normalized_level)
        object.__setattr__(self, "resource_name", resource_name)
        object.__setattr__(self, "resource_cost", resource_cost)
        object.__setattr__(self, "passive_hooks", tuple(self.passive_hooks))

    @property
    def not_implemented(self) -> bool:
        return not self.implemented

    def is_available(self, resources: dict[str, Resource]) -> bool:
        if self.resource_name is None:
            return True
        resource = resources.get(self.resource_name)
        return resource is not None and resource.available


class ClassFeature(FeatureDefinition):
    """Backward-compatible name for data-driven feature definitions."""


def reset_resources(resources: dict[str, Resource]) -> None:
    """Reset all tracked per-combat resources."""

    for resource in resources.values():
        resource.reset()


def feature_resource_name(feature: Any) -> str | None:
    """Return the resource name consumed or tracked by a feature."""

    resource_name = getattr(feature, "resource_name", None)
    if resource_name is not None:
        return str(resource_name)
    resource_cost = getattr(feature, "resource_cost", None)
    if isinstance(resource_cost, str):
        return resource_cost
    return None


def is_feature_implemented(feature: Any) -> bool:
    """Return True only for features implemented in combat logic."""

    return bool(getattr(feature, "implemented", False))


def implemented_class_features(character: Any) -> tuple[FeatureDefinition, ...]:
    """Return implemented class features stored on a character."""

    return tuple(
        feature
        for feature in getattr(character, "class_features", ())
        if is_feature_implemented(feature)
    )


def available_implemented_class_features(character: Any) -> tuple[FeatureDefinition, ...]:
    """Return implemented class features whose resource requirements are available."""

    resources = getattr(character, "resources", {})
    return tuple(
        feature
        for feature in implemented_class_features(character)
        if feature.is_available(resources)
    )


def implemented_feature_active_actions(
    character: Any,
    action_cost: str | None = None,
) -> tuple[str, ...]:
    """Return active actions exposed by implemented available class features."""

    actions: list[str] = []
    for feature in available_implemented_class_features(character):
        if feature.active_action is None:
            continue
        if action_cost is not None and feature.action_cost != action_cost:
            continue
        actions.append(feature.active_action)
    return tuple(actions)


def character_has_class_feature(character: Any, feature_name: str) -> bool:
    """Return True if the character has an implemented feature by name."""

    feature_key = _lookup_key(feature_name)
    return any(
        _lookup_key(feature.name) == feature_key
        for feature in implemented_class_features(character)
    )


def available_feature_for_active_action(
    character: Any,
    active_action: str,
    action_cost: str | None = None,
) -> FeatureDefinition | None:
    """Return the available implemented feature exposing an active action."""

    action_key = _lookup_key(active_action)
    for feature in available_implemented_class_features(character):
        if feature.active_action is None:
            continue
        if _lookup_key(feature.active_action) != action_key:
            continue
        if action_cost is not None and feature.action_cost != action_cost:
            continue
        return feature
    return None


def can_use_feature_action(
    character: Any,
    active_action: str,
    action_cost: str | None = None,
) -> bool:
    """Return True if an active class feature is currently available."""

    return available_feature_for_active_action(
        character,
        active_action,
        action_cost,
    ) is not None


def spend_feature_resource(character: Any, active_action: str) -> bool:
    """Spend the resource attached to an active feature if one exists."""

    feature = available_feature_for_active_action(character, active_action)
    if feature is None:
        return False
    resource_name = feature_resource_name(feature)
    if resource_name is None:
        return True
    resource = getattr(character, "resources", {}).get(resource_name)
    if resource is None or not resource.available:
        return False
    resource.spend()
    return True


def fighting_style(character: Any) -> str | None:
    """Return the selected fighting style if the character has Fighting Style."""

    if not character_has_class_feature(character, "Fighting Style"):
        return None
    style = getattr(character, "fighting_style", None)
    if style is None:
        return None
    normalized = _lookup_key(style)
    for option in (
        FIGHTING_STYLE_ARCHERY,
        FIGHTING_STYLE_DEFENSE,
        FIGHTING_STYLE_GREAT_WEAPON_FIGHTING,
    ):
        if _lookup_key(option) == normalized:
            return option
    return None


def archery_attack_bonus(character: Any, weapon: Any) -> int:
    """Return the Archery fighting style attack bonus for ranged weapons."""

    if fighting_style(character) != FIGHTING_STYLE_ARCHERY:
        return 0
    if int(getattr(weapon, "range", 0)) <= 1:
        return 0
    return 2


def apply_defense_fighting_style(character: Any) -> None:
    """Apply Defense fighting style AC bonus once when armor is worn."""

    if fighting_style(character) != FIGHTING_STYLE_DEFENSE:
        return
    if not bool(getattr(character, "wearing_armor", False)):
        return
    if bool(getattr(character, "_defense_fighting_style_applied", False)):
        return
    character.ac += 1
    character._defense_fighting_style_applied = True


def should_use_great_weapon_fighting(character: Any, weapon: Any) -> bool:
    """Return True when low weapon damage dice should be rerolled once."""

    if fighting_style(character) != FIGHTING_STYLE_GREAT_WEAPON_FIGHTING:
        return False
    return bool(
        getattr(weapon, "two_handed", False)
        or getattr(weapon, "heavy", False)
        or getattr(weapon, "versatile_two_handed", False)
    )


def critical_hit_threshold(character: Any) -> int:
    """Return the natural d20 value that starts a critical hit."""

    if character_has_class_feature(character, "Improved Critical"):
        return 19
    return 21


def weapon_attack_count_for_attack_action(character: Any) -> int:
    """Return weapon attacks made inside one common Attack action."""

    if character_has_class_feature(character, "Extra Attack"):
        return 2
    return 1


def _lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
