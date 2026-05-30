import os
from pathlib import Path
from uuid import uuid4

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from character import CharacterRepository, InternalCharacter
from ui.app import create_app
from ui.main_window import MainWindow
from ui.navigation import NAVIGATION_ITEMS
from ui.screens import BattleScreen
from ui.services import BattleSetupRequest, BattleSetupService, ModelService


def test_main_window_registers_navigation_screens() -> None:
    app = create_app(["test_gui_smoke"])
    window = MainWindow()

    assert window.windowTitle() == "D&D Battle AI"
    assert set(window._screen_indexes) == {item.key for item in NAVIGATION_ITEMS}

    window.set_screen("replays")
    assert window._stack.currentIndex() == window._screen_indexes["replays"]

    window.close()
    app.processEvents()


def test_character_list_screen_loads_repository_characters() -> None:
    repository = CharacterRepository(
        Path("checkpoints") / f"test_gui_characters_{uuid4().hex}"
    )
    saved = repository.save_character(_valid_character())
    app = create_app(["test_gui_character_list"])
    window = MainWindow(character_repository=repository)

    window.set_screen("characters")
    screen = window._stack.widget(window._screen_indexes["characters"])

    assert len(screen._cards) == 1
    assert screen._cards[0].character.id == saved.id

    window.close()
    app.processEvents()


def test_battle_screen_accepts_setup_result_and_steps_ai() -> None:
    app = create_app(["test_gui_battle_screen"])
    settings_path = Path("checkpoints") / f"test_gui_model_settings_{uuid4().hex}.json"
    model_service = ModelService(settings_path=settings_path)
    setup_result = BattleSetupService().create_random_battle(
        BattleSetupRequest(
            party_preset="fighter_level_1",
            enemy_group="goblin_patrol",
            controller_mode="ai_all",
            seed=11,
        )
    )
    screen = BattleScreen(model_service)

    screen.set_battle(setup_result)
    screen._next_step()

    assert screen._environment is setup_result.environment
    assert screen._map_widget._environment is setup_result.environment
    assert screen._replay is not None
    assert len(screen._replay.steps) == 1
    assert "Выбранное действие:" in "\n".join(screen._environment.action_log)

    screen.close()
    app.processEvents()


def _valid_character() -> InternalCharacter:
    return InternalCharacter(
        id="gui-hero",
        name="GUI Hero",
        class_name="Fighter",
        subclass_name="Champion",
        level=3,
        experience=900,
        race_name="Human",
        role="MELEE_DAMAGE",
        stats={
            "str": 16,
            "dex": 12,
            "con": 14,
            "int": 10,
            "wis": 10,
            "cha": 10,
        },
        hp=28,
        ac=16,
        speed=30,
        proficiency_bonus=2,
        weapons=({"name": "Longsword", "damage": "1d8"},),
        armor={"name": "Chain Mail", "ac": 16},
        class_features=("Fighting Style", "Second Wind"),
        subclass_features=("Improved Critical",),
        race_traits={"skill_proficiencies": ["Athletics"]},
    )
