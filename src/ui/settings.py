"""Persistent desktop GUI settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from ui.animations import DEFAULT_ANIMATION_SPEED_MS, normalize_animation_speed


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "data" / "settings.json"
DEFAULT_CHARACTER_DIR = "data/characters"
DEFAULT_REPLAY_DIR = "replays"
DEFAULT_MAP_DIR = "maps"
DEFAULT_CHECKPOINT_PATH = "checkpoints/gnn_ppo_actor_critic.pt"
FIXED_MODEL_TYPE = "gnn"
FIXED_FALLBACK_AGENT = "rule_based"
SUPPORTED_MODEL_TYPES = ("mlp", "gnn")
SUPPORTED_FALLBACK_AGENTS = ("random_legal", "aggressive_melee", "rule_based")
FALLBACK_AGENT_LABELS = {
    "random_legal": "Random legal",
    "aggressive_melee": "Aggressive melee",
    "rule_based": "Rule-based",
}


class SettingsLoadError(RuntimeError):
    """Raised when the persisted GUI settings file cannot be parsed."""


@dataclass
class GuiSettings:
    """Settings persisted in data/settings.json."""

    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    model_type: str = FIXED_MODEL_TYPE
    fallback_agent: str = FIXED_FALLBACK_AGENT
    animation_speed: int = DEFAULT_ANIMATION_SPEED_MS
    animations_enabled: bool = True
    autobattle_delay: int = DEFAULT_ANIMATION_SPEED_MS
    character_dir: str = DEFAULT_CHARACTER_DIR
    replay_dir: str = DEFAULT_REPLAY_DIR
    map_dir: str = DEFAULT_MAP_DIR
    random_battle_seed: int | None = None


def load_gui_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> GuiSettings:
    """Load GUI settings or return defaults when the file does not exist."""

    settings_path = Path(path)
    if not settings_path.exists():
        return GuiSettings()
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SettingsLoadError(f"Повреждённый файл настроек: {settings_path}") from error
    except OSError as error:
        raise SettingsLoadError(f"Не удалось прочитать файл настроек: {settings_path}") from error
    if not isinstance(raw, dict):
        raise SettingsLoadError(f"Повреждённый файл настроек: {settings_path}")
    return settings_from_mapping(raw)


def save_gui_settings(
    settings: GuiSettings,
    path: str | Path = DEFAULT_SETTINGS_PATH,
) -> None:
    """Persist GUI settings to JSON."""

    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings_to_dict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def settings_to_dict(settings: GuiSettings) -> dict[str, Any]:
    """Return a JSON-ready mapping for settings."""

    return asdict(settings)


def settings_from_mapping(raw: dict[str, Any]) -> GuiSettings:
    """Build normalized settings from a loosely typed mapping."""

    return GuiSettings(
        checkpoint_path=_non_empty_path(raw.get("checkpoint_path"), DEFAULT_CHECKPOINT_PATH),
        model_type=FIXED_MODEL_TYPE,
        fallback_agent=FIXED_FALLBACK_AGENT,
        animation_speed=normalize_animation_speed(
            raw.get("animation_speed", DEFAULT_ANIMATION_SPEED_MS)
        ),
        animations_enabled=_as_bool(raw.get("animations_enabled", True)),
        autobattle_delay=normalize_autobattle_delay(
            raw.get("autobattle_delay", DEFAULT_ANIMATION_SPEED_MS)
        ),
        character_dir=_non_empty_path(raw.get("character_dir"), DEFAULT_CHARACTER_DIR),
        replay_dir=_non_empty_path(raw.get("replay_dir"), DEFAULT_REPLAY_DIR),
        map_dir=_non_empty_path(raw.get("map_dir"), DEFAULT_MAP_DIR),
        random_battle_seed=normalize_optional_seed(raw.get("random_battle_seed")),
    )


def normalize_model_type(model_type: object) -> str:
    """Normalize a model type to a supported UI value."""

    normalized = str(model_type).strip().lower()
    return normalized if normalized in SUPPORTED_MODEL_TYPES else FIXED_MODEL_TYPE


def normalize_fallback_agent(fallback_agent: object) -> str:
    """Normalize a fallback agent key or label to a supported UI value."""

    normalized = str(fallback_agent).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "randomlegal": "random_legal",
        "random_legal_agent": "random_legal",
        "aggressivemelee": "aggressive_melee",
        "rulebased": "rule_based",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in SUPPORTED_FALLBACK_AGENTS else FIXED_FALLBACK_AGENT


def normalize_autobattle_delay(value: object) -> int:
    """Clamp autobattle delay to the same readable range as animations."""

    return normalize_animation_speed(value)


def normalize_optional_seed(value: object) -> int | None:
    """Normalize an optional random battle seed."""

    if value in (None, ""):
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return None
    if seed < 0:
        return None
    return min(seed, 2_147_483_647)


def _non_empty_path(value: object, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)
