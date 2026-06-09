import json
from pathlib import Path
from uuid import uuid4

import pytest

from inference import CheckpointLoadError
from ui.services import ModelService
from ui.settings import (
    DEFAULT_CHECKPOINT_PATH,
    GuiSettings,
    SettingsLoadError,
    load_gui_settings,
    save_gui_settings,
)


def test_gui_settings_save_and_load_full_payload() -> None:
    path = _settings_path("full")
    settings = GuiSettings(
        checkpoint_path="checkpoints/policy.pt",
        model_type="gnn",
        fallback_agent="aggressive_melee",
        animation_speed=900,
        animations_enabled=False,
        autobattle_delay=1200,
        character_dir="data/custom_characters",
        map_dir="data/custom_maps",
        random_battle_seed=77,
    )

    save_gui_settings(settings, path)
    loaded = load_gui_settings(path)

    assert loaded.checkpoint_path == settings.checkpoint_path
    assert loaded.model_type == "gnn"
    assert loaded.fallback_agent == "rule_based"
    assert loaded.animation_speed == settings.animation_speed
    assert loaded.animations_enabled is False
    assert loaded.autobattle_delay == settings.autobattle_delay
    assert loaded.character_dir == settings.character_dir
    assert loaded.map_dir == settings.map_dir
    assert loaded.random_battle_seed == settings.random_battle_seed


def test_gui_settings_reports_corrupted_file() -> None:
    path = _settings_path("broken")
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(SettingsLoadError, match="Повреждённый файл настроек"):
        load_gui_settings(path)


def test_gui_settings_migrates_legacy_checkpoint_to_current_policy() -> None:
    path = _settings_path("legacy_checkpoint")
    path.write_text(
        json.dumps({"checkpoint_path": "checkpoints/gnn_ppo_actor_critic.pt"}),
        encoding="utf-8",
    )

    loaded = load_gui_settings(path)

    assert loaded.checkpoint_path == DEFAULT_CHECKPOINT_PATH


def test_model_service_falls_back_after_corrupted_settings() -> None:
    path = _settings_path("service_broken")
    path.write_text("{not-json", encoding="utf-8")

    service = ModelService(settings_path=path)

    assert service.settings == GuiSettings()
    assert "Повреждённый файл настроек" in service.last_error


def test_model_service_persists_gui_paths_and_autobattle_delay() -> None:
    path = _settings_path("service_paths")
    service = ModelService(settings_path=path)

    service.set_settings(
        fallback_agent="rule-based",
        animation_speed=9999,
        autobattle_delay=300,
        character_dir="data/test_characters",
        map_dir="data/test_maps",
        random_battle_seed=123,
    )
    reloaded = ModelService(settings_path=path)

    assert reloaded.settings.fallback_agent == "rule_based"
    assert reloaded.settings.animation_speed == 1500
    assert reloaded.settings.autobattle_delay == 300
    assert reloaded.settings.character_dir == "data/test_characters"
    assert reloaded.settings.map_dir == "data/test_maps"
    assert reloaded.settings.random_battle_seed == 123


def test_model_service_check_model_reports_missing_checkpoint() -> None:
    path = _settings_path("missing_checkpoint")
    settings = GuiSettings(checkpoint_path="checkpoints/no_such_checkpoint.pt")
    path.write_text(json.dumps(settings.__dict__), encoding="utf-8")
    service = ModelService(settings_path=path)

    with pytest.raises(CheckpointLoadError, match="Файл модели не найден"):
        service.check_model()


def _settings_path(prefix: str) -> Path:
    directory = Path("checkpoints") / "test_ui_settings"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{uuid4().hex}.json"
