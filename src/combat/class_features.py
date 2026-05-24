"""Class features, feature definitions and per-combat resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
