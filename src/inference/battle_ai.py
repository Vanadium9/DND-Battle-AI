"""High-level battle AI service used by the desktop UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents import (
    AggressiveMeleeAgent,
    CoverAwareRangedAgent,
    RandomLegalAgent,
    RangedKitingAgent,
    RuleBasedAgent,
    SimpleCasterAgent,
    SimpleHealerAgent,
)
from combat import CombatAction, CombatState
from inference.action_selector import (
    ActionSelectionError,
    select_action_with_policy,
    select_fallback_action,
)
from inference.policy_loader import LoadedPolicy, load_policy_checkpoint


FALLBACK_AGENT_TYPES: dict[str, type[RuleBasedAgent]] = {
    "random_legal": RandomLegalAgent,
    "rule_based": RuleBasedAgent,
    "aggressive_melee": AggressiveMeleeAgent,
    "ranged_kiting": RangedKitingAgent,
    "simple_healer": SimpleHealerAgent,
    "simple_caster": SimpleCasterAgent,
    "cover_aware_ranged": CoverAwareRangedAgent,
}


@dataclass
class BattleAISettings:
    """User-facing inference settings for GUI configuration."""

    checkpoint_path: str = ""
    model_type: str = "mlp"
    fallback_agent: str = "random_legal"


class BattleAIService:
    """Load a trained policy and select combat actions for GUI battles."""

    def __init__(
        self,
        *,
        fallback_agent: str = "random_legal",
        seed: int | None = None,
    ) -> None:
        self._loaded_policy: LoadedPolicy | None = None
        self._settings = BattleAISettings(fallback_agent=fallback_agent)
        self._seed = seed
        self._fallback_agent = self._create_fallback_agent(fallback_agent)

    @property
    def settings(self) -> BattleAISettings:
        return BattleAISettings(
            checkpoint_path=self._settings.checkpoint_path,
            model_type=self._settings.model_type,
            fallback_agent=self._settings.fallback_agent,
        )

    def configure(
        self,
        *,
        checkpoint_path: str | None = None,
        model_type: str | None = None,
        fallback_agent: str | None = None,
    ) -> None:
        """Update settings without loading a checkpoint."""

        if checkpoint_path is not None:
            self._settings.checkpoint_path = checkpoint_path
        if model_type is not None:
            self._settings.model_type = _normalize_model_type(model_type)
        if fallback_agent is not None:
            self.set_fallback_agent(fallback_agent)

    def load_checkpoint(self, path: str | Path, model_type: str) -> None:
        """Load a trained checkpoint into the service."""

        loaded_policy = load_policy_checkpoint(path, model_type)
        self._loaded_policy = loaded_policy
        self._settings.checkpoint_path = str(path)
        self._settings.model_type = loaded_policy.model_type

    def unload_checkpoint(self) -> None:
        """Return to fallback inference."""

        self._loaded_policy = None

    def is_model_loaded(self) -> bool:
        return self._loaded_policy is not None

    def select_action(self, combat_state: CombatState, actor_id: int) -> CombatAction:
        """Select one concrete action for the current actor."""

        if self._loaded_policy is None:
            return select_fallback_action(
                self._fallback_agent,
                combat_state,
                actor_id,
                deterministic=False,
            )
        try:
            return select_action_with_policy(
                self._loaded_policy,
                combat_state,
                actor_id,
                deterministic=True,
            )
        except ActionSelectionError:
            return select_fallback_action(
                self._fallback_agent,
                combat_state,
                actor_id,
                deterministic=False,
            )

    def get_policy_name(self) -> str:
        if self._loaded_policy is not None:
            return self._loaded_policy.policy_name
        return f"Fallback: {self._settings.fallback_agent}"

    def set_fallback_agent(self, fallback_agent: str) -> None:
        self._fallback_agent = self._create_fallback_agent(fallback_agent)
        self._settings.fallback_agent = fallback_agent

    def _create_fallback_agent(self, fallback_agent: str) -> RuleBasedAgent:
        normalized = str(fallback_agent).strip().lower()
        agent_type = FALLBACK_AGENT_TYPES.get(normalized)
        if agent_type is None:
            options = ", ".join(sorted(FALLBACK_AGENT_TYPES))
            raise ValueError(f"Unknown fallback_agent '{fallback_agent}'. Supported: {options}")
        return agent_type(seed=self._seed)


def _normalize_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    if normalized not in {"mlp", "gnn"}:
        raise ValueError("model_type must be 'mlp' or 'gnn'")
    return normalized


__all__ = [
    "BattleAIService",
    "BattleAISettings",
    "FALLBACK_AGENT_TYPES",
]
