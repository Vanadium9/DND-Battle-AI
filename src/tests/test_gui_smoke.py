import os
from pathlib import Path
from uuid import uuid4

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from character import CharacterRepository, InternalCharacter
from combat import CombatEnvironment, FighterLevel1Basic, GoblinMelee, GridMap, Position
from ui.app import create_app
from ui.main_window import MainWindow
from ui.navigation import NAVIGATION_ITEMS
from ui.screens import BattleScreen, CharacterBuilderScreen
from ui.services import BattleSetupRequest, BattleSetupResult, BattleSetupService, ModelService
from ui.widgets.inventory_editor import InventoryEditor
from ui.widgets.stat_editor import StatEditor


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


def test_builder_preview_and_save_generate_character_id() -> None:
    repository = CharacterRepository(
        Path("checkpoints") / f"test_gui_builder_characters_{uuid4().hex}"
    )
    app = create_app(["test_gui_character_builder"])
    screen = CharacterBuilderScreen(repository)

    screen.name_edit.setText("New Hero")
    _check_weapon(screen, "Longsword")
    screen.update_review()

    assert not hasattr(screen, "sheet_summary")
    assert "id:" not in screen.review_text.toPlainText().lower()
    assert screen.save_button.isEnabled()

    screen._save()
    saved_characters = repository.list_characters()

    assert len(saved_characters) == 1
    assert saved_characters[0].id
    assert saved_characters[0].name == "New Hero"

    screen.close()
    app.processEvents()


def test_editor_widgets_emit_zero_argument_signals() -> None:
    app = create_app(["test_gui_editor_widgets"])
    stat_editor = StatEditor()
    stat_emissions: list[bool] = []
    stat_editor.stats_changed.connect(lambda: stat_emissions.append(True))

    stat_editor._spin_boxes["str"].setValue(12)
    stat_editor.set_proficiency_context(
        proficiency_bonus=2,
        saving_throw_proficiencies=("str", "con"),
    )

    inventory_editor = InventoryEditor()
    inventory_emissions: list[bool] = []
    inventory_editor.inventory_changed.connect(lambda: inventory_emissions.append(True))
    first_item_spin = next(iter(inventory_editor._spin_boxes.values()))
    first_item_spin.setValue(1)

    assert stat_emissions
    assert inventory_emissions
    assert stat_editor._save_labels["str"].text() == "+3"
    assert stat_editor._save_labels["dex"].text() == "-"

    stat_editor.close()
    inventory_editor.close()
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


def test_battle_screen_manual_player_can_end_turn() -> None:
    app = create_app(["test_gui_manual_battle_screen"])
    settings_path = Path("checkpoints") / f"test_gui_model_settings_{uuid4().hex}.json"
    model_service = ModelService(settings_path=settings_path)
    environment = CombatEnvironment(
        characters=[
            FighterLevel1Basic(Position(0, 0)),
            GoblinMelee(Position(1, 0)),
        ],
        grid_map=GridMap(width=5, height=5),
        use_initiative=False,
        log_to_console=False,
    )
    setup_result = BattleSetupResult(
        environment=environment,
        party_names=("Fighter Level 1 Basic",),
        enemy_names=("Goblin",),
        map_name="open_field",
        difficulty="easy",
        controller_mode="manual_players_ai_enemies",
        seed=None,
        summary="Manual test battle",
    )
    screen = BattleScreen(model_service)

    screen.set_battle(setup_result)
    assert screen._manual_plan is not None
    assert not screen._next_button.isEnabled()

    end_turn = screen._manual_plan.groups["End Turn"][0]
    screen._select_manual_option(end_turn)

    assert screen._environment is environment
    assert screen._environment.combat_state.active_actor_id == 1
    assert screen._replay is not None
    assert len(screen._replay.steps) == 1

    screen.close()
    app.processEvents()


def test_battle_screen_click_prefers_alive_target_over_corpse() -> None:
    app = create_app(["test_gui_battle_screen_corpse_target"])
    settings_path = Path("checkpoints") / f"test_gui_model_settings_{uuid4().hex}.json"
    model_service = ModelService(settings_path=settings_path)
    corpse = GoblinMelee(Position(1, 0))
    living = GoblinMelee(Position(1, 0))
    corpse.hp = 0
    environment = CombatEnvironment(
        characters=[
            FighterLevel1Basic(Position(0, 0)),
            corpse,
            living,
        ],
        grid_map=GridMap(width=5, height=5),
        use_initiative=False,
        log_to_console=False,
    )
    setup_result = BattleSetupResult(
        environment=environment,
        party_names=("Fighter Level 1 Basic",),
        enemy_names=("Dead Goblin", "Living Goblin"),
        map_name="open_field",
        difficulty="easy",
        controller_mode="manual_players_ai_enemies",
        seed=None,
        summary="Corpse target test battle",
    )
    screen = BattleScreen(model_service)

    screen.set_battle(setup_result)

    assert screen._creature_id_at(Position(1, 0), allowed_target_ids=(1, 2)) == 2

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


def _check_weapon(screen: CharacterBuilderScreen, weapon_name: str) -> None:
    for row in range(screen.weapon_list.count()):
        item = screen.weapon_list.item(row)
        if item.text() == weapon_name or item.data(Qt.ItemDataRole.UserRole) == weapon_name:
            item.setCheckState(Qt.CheckState.Checked)
            return
    raise AssertionError(f"Weapon option is missing: {weapon_name}")
