"""Class features and per-combat resources."""

from __future__ import annotations

from dataclasses import dataclass


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
class ClassFeature:
    """A class feature definition separated from common combat actions."""

    name: str
    description: str = ""
    resource_name: str | None = None
    required_level: int = 1

    def is_available(self, resources: dict[str, Resource]) -> bool:
        if self.resource_name is None:
            return True
        resource = resources.get(self.resource_name)
        return resource is not None and resource.available


def reset_resources(resources: dict[str, Resource]) -> None:
    """Reset all tracked per-combat resources."""

    for resource in resources.values():
        resource.reset()
