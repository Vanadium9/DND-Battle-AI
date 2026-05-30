"""GUI service for configuring trained-policy inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from combat import CombatAction, CombatState
from inference import BattleAIService, CheckpointLoadError
from inference.battle_ai import FALLBACK_AGENT_TYPES


DEFAULT_SETTINGS_PATH = Path("data") / "settings" / "model_service.json"
MODEL_TYPES = ("mlp", "gnn")


@dataclass
class ModelServiceSettings:
    """Persisted GUI settings for inference."""

    checkpoint_path: str = ""
    model_type: str = "mlp"
    fallback_agent: str = "random_legal"


class ModelService:
    """Small facade between PySide widgets and BattleAIService."""

    def __init__(
        self,
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
        battle_ai: BattleAIService | None = None,
    ) -> None:
        self.settings_path = Path(settings_path)
        self.battle_ai = battle_ai or BattleAIService()
        self.settings = self.load_settings()
        self.last_error: str = ""
        self.apply_settings(load_checkpoint=False)

    def available_model_types(self) -> tuple[str, ...]:
        return MODEL_TYPES

    def available_fallback_agents(self) -> tuple[str, ...]:
        return tuple(sorted(FALLBACK_AGENT_TYPES))

    def load_checkpoint(self, path: str, model_type: str) -> None:
        """Load checkpoint and persist the selected settings."""

        try:
            self.battle_ai.load_checkpoint(path, model_type)
        except CheckpointLoadError as error:
            self.last_error = str(error)
            raise
        self.last_error = ""
        self.settings.checkpoint_path = path
        self.settings.model_type = model_type
        self.save_settings()

    def select_action(self, combat_state: CombatState, actor_id: int) -> CombatAction:
        return self.battle_ai.select_action(combat_state, actor_id)

    def is_model_loaded(self) -> bool:
        return self.battle_ai.is_model_loaded()

    def get_policy_name(self) -> str:
        return self.battle_ai.get_policy_name()

    def set_settings(
        self,
        *,
        checkpoint_path: str | None = None,
        model_type: str | None = None,
        fallback_agent: str | None = None,
    ) -> None:
        if checkpoint_path is not None:
            self.settings.checkpoint_path = checkpoint_path
        if model_type is not None:
            self.settings.model_type = _normalize_model_type(model_type)
        if fallback_agent is not None:
            self.settings.fallback_agent = _normalize_fallback_agent(fallback_agent)
            self.battle_ai.set_fallback_agent(self.settings.fallback_agent)
        self.battle_ai.configure(
            checkpoint_path=self.settings.checkpoint_path,
            model_type=self.settings.model_type,
            fallback_agent=self.settings.fallback_agent,
        )
        self.save_settings()

    def apply_settings(self, *, load_checkpoint: bool = True) -> None:
        self.battle_ai.configure(
            checkpoint_path=self.settings.checkpoint_path,
            model_type=self.settings.model_type,
            fallback_agent=self.settings.fallback_agent,
        )
        if load_checkpoint and self.settings.checkpoint_path:
            self.load_checkpoint(self.settings.checkpoint_path, self.settings.model_type)

    def load_settings(self) -> ModelServiceSettings:
        if not self.settings_path.exists():
            return ModelServiceSettings()
        try:
            raw = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ModelServiceSettings()
        if not isinstance(raw, dict):
            return ModelServiceSettings()
        return _settings_from_mapping(raw)

    def save_settings(self) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(
            json.dumps(asdict(self.settings), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _settings_from_mapping(raw: dict[str, Any]) -> ModelServiceSettings:
    model_type = str(raw.get("model_type", "mlp"))
    fallback_agent = str(raw.get("fallback_agent", "random_legal"))
    return ModelServiceSettings(
        checkpoint_path=str(raw.get("checkpoint_path", "")),
        model_type=_normalize_model_type(model_type),
        fallback_agent=_normalize_fallback_agent(fallback_agent),
    )


def _normalize_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    return normalized if normalized in MODEL_TYPES else "mlp"


def _normalize_fallback_agent(fallback_agent: str) -> str:
    normalized = str(fallback_agent).strip().lower()
    return normalized if normalized in FALLBACK_AGENT_TYPES else "random_legal"
