"""Ruleset registry and default supported-content checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rules.ruleset import Ruleset


DEFAULT_RULESET_NAME = "srd5e_minimal_2014"
DEFAULT_RULESET_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "ruleset_srd5e_minimal.yaml"
)


@dataclass
class RulesetRegistry:
    """Registry of known rulesets."""

    rulesets: dict[str, Ruleset] = field(default_factory=dict)
    active_ruleset_name: str = DEFAULT_RULESET_NAME

    def register(self, ruleset: Ruleset) -> None:
        self.rulesets[ruleset.ruleset_name] = ruleset

    def get(self, ruleset_name: str | None = None) -> Ruleset:
        name = ruleset_name or self.active_ruleset_name
        try:
            return self.rulesets[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.rulesets)) or "none"
            raise KeyError(
                f"Ruleset '{name}' is not registered. Available rulesets: {available}."
            ) from exc

    def is_supported_content(
        self,
        content_type: str,
        name: str | int,
        ruleset_name: str | None = None,
    ) -> bool:
        return self.get(ruleset_name).is_supported_content(content_type, name)

    def get_unsupported_reason(
        self,
        content_type: str,
        name: str | int,
        ruleset_name: str | None = None,
    ) -> str:
        return self.get(ruleset_name).get_unsupported_reason(content_type, name)


def load_ruleset_from_yaml(path: str | Path) -> Ruleset:
    """Load a ruleset definition from a YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    return Ruleset.from_mapping(data)


def create_default_registry(
    config_path: str | Path = DEFAULT_RULESET_CONFIG_PATH,
) -> RulesetRegistry:
    registry = RulesetRegistry()
    registry.register(load_ruleset_from_yaml(config_path))
    return registry


_default_registry: RulesetRegistry | None = None


def get_registry() -> RulesetRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_registry()
    return _default_registry


def get_active_ruleset() -> Ruleset:
    return get_registry().get()


def is_supported_content(content_type: str, name: str | int) -> bool:
    return get_registry().is_supported_content(content_type, name)


def get_unsupported_reason(content_type: str, name: str | int) -> str:
    return get_registry().get_unsupported_reason(content_type, name)
