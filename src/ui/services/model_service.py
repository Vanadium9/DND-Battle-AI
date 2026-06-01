"""GUI service for configuring trained-policy inference."""

from __future__ import annotations

from pathlib import Path

from combat import CombatAction, CombatState
from inference import BattleAIService, CheckpointLoadError
from ui.settings import (
    DEFAULT_SETTINGS_PATH,
    FALLBACK_AGENT_LABELS,
    GuiSettings,
    SUPPORTED_FALLBACK_AGENTS,
    SUPPORTED_MODEL_TYPES,
    SettingsLoadError,
    load_gui_settings,
    normalize_animation_speed,
    normalize_autobattle_delay,
    normalize_fallback_agent,
    normalize_model_type,
    normalize_optional_seed,
    save_gui_settings,
)


ModelServiceSettings = GuiSettings
_UNSET = object()


class ModelService:
    """Small facade between PySide widgets and BattleAIService."""

    def __init__(
        self,
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
        battle_ai: BattleAIService | None = None,
    ) -> None:
        self.settings_path = Path(settings_path)
        self.battle_ai = battle_ai or BattleAIService()
        self.last_error: str = ""
        self.settings = self.load_settings()
        self.apply_settings(load_checkpoint=True)

    def available_model_types(self) -> tuple[str, ...]:
        return SUPPORTED_MODEL_TYPES

    def available_fallback_agents(self) -> tuple[str, ...]:
        return SUPPORTED_FALLBACK_AGENTS

    def fallback_agent_label(self, fallback_agent: str) -> str:
        return FALLBACK_AGENT_LABELS.get(fallback_agent, fallback_agent)

    def load_checkpoint(self, path: str, model_type: str) -> None:
        """Load checkpoint and persist the selected settings."""

        try:
            self.battle_ai.load_checkpoint(path, normalize_model_type(model_type))
        except CheckpointLoadError as error:
            self.last_error = str(error)
            raise
        self.last_error = ""
        self.settings.checkpoint_path = path
        self.settings.model_type = self.battle_ai.settings.model_type
        self.save_settings()

    def check_model(self) -> str:
        """Try to load the configured checkpoint and return a user-facing status."""

        checkpoint_path = self.settings.checkpoint_path.strip()
        if not checkpoint_path:
            self.battle_ai.unload_checkpoint()
            return "Checkpoint не задан. GUI будет использовать fallback agent."
        if not Path(checkpoint_path).expanduser().exists():
            message = f"Checkpoint не найден: {checkpoint_path}"
            self.last_error = message
            raise CheckpointLoadError(message)
        self.load_checkpoint(checkpoint_path, self.settings.model_type)
        return f"Модель загружена: {self.get_policy_name()}"

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
        animations_enabled: bool | None = None,
        animation_speed: int | None = None,
        autobattle_delay: int | None = None,
        character_dir: str | None = None,
        replay_dir: str | None = None,
        map_dir: str | None = None,
        random_battle_seed: object = _UNSET,
    ) -> None:
        if checkpoint_path is not None:
            self.settings.checkpoint_path = checkpoint_path
        if model_type is not None:
            self.settings.model_type = normalize_model_type(model_type)
        if fallback_agent is not None:
            self.settings.fallback_agent = normalize_fallback_agent(fallback_agent)
            self.battle_ai.set_fallback_agent(self.settings.fallback_agent)
        if animations_enabled is not None:
            self.settings.animations_enabled = bool(animations_enabled)
        if animation_speed is not None:
            self.settings.animation_speed = normalize_animation_speed(animation_speed)
        if autobattle_delay is not None:
            self.settings.autobattle_delay = normalize_autobattle_delay(autobattle_delay)
        if character_dir is not None:
            self.settings.character_dir = str(character_dir).strip() or self.settings.character_dir
        if replay_dir is not None:
            self.settings.replay_dir = str(replay_dir).strip() or self.settings.replay_dir
        if map_dir is not None:
            self.settings.map_dir = str(map_dir).strip() or self.settings.map_dir
        if random_battle_seed is not _UNSET:
            self.settings.random_battle_seed = normalize_optional_seed(random_battle_seed)
        self.battle_ai.configure(
            checkpoint_path=self.settings.checkpoint_path,
            model_type=self.settings.model_type,
            fallback_agent=self.settings.fallback_agent,
        )
        self.last_error = ""
        self.save_settings()

    def apply_settings(self, *, load_checkpoint: bool = True) -> None:
        self.battle_ai.configure(
            checkpoint_path=self.settings.checkpoint_path,
            model_type=self.settings.model_type,
            fallback_agent=self.settings.fallback_agent,
        )
        if load_checkpoint and self.settings.checkpoint_path:
            previous_error = self.last_error
            try:
                self.load_checkpoint(self.settings.checkpoint_path, self.settings.model_type)
            except CheckpointLoadError:
                self.battle_ai.unload_checkpoint()
                if previous_error:
                    self.last_error = previous_error
            else:
                if previous_error:
                    self.last_error = previous_error

    def load_settings(self) -> ModelServiceSettings:
        try:
            return load_gui_settings(self.settings_path)
        except SettingsLoadError as error:
            self.last_error = str(error)
            return ModelServiceSettings()

    def save_settings(self) -> None:
        save_gui_settings(self.settings, self.settings_path)
