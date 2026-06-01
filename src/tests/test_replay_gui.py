import json
import os
from pathlib import Path
from uuid import uuid4

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from combat.replay import load_replay_file, replay_step_to_state, replay_summary
from ui.app import create_app
from ui.screens import ReplayListScreen, ReplayViewerScreen


def test_replay_summary_and_state_reconstruction() -> None:
    path = Path("checkpoints") / f"test_replay_gui_{uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sample_replay(), ensure_ascii=False), encoding="utf-8")

    payload = load_replay_file(path)
    summary = replay_summary(path, payload)
    state = replay_step_to_state(payload["steps"][0])

    assert summary.display_name == "Hero vs Goblin"
    assert summary.winner == "players"
    assert summary.round_count == 1
    assert summary.participants == ("Hero", "Goblin")
    assert state.grid_map is not None
    assert state.grid_map.width == 3
    assert state.characters[0].name == "Hero"
    assert state.characters[1].hp == 0


def test_replay_list_screen_lists_replay_files() -> None:
    app = create_app(["test_replay_list_screen"])
    replay_dir = Path("checkpoints") / f"test_replay_list_{uuid4().hex}"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / "sample_replay.json"
    replay_path.write_text(json.dumps(_sample_replay(), ensure_ascii=False), encoding="utf-8")

    screen = ReplayListScreen(replay_dir)
    screen._table.setCurrentCell(0, 0)

    assert screen._table.rowCount() == 1
    assert screen.selected_path() == replay_path

    screen.close()
    app.processEvents()


def test_replay_viewer_loads_replay_without_environment_step() -> None:
    app = create_app(["test_replay_viewer_screen"])
    replay_path = Path("checkpoints") / f"test_replay_viewer_{uuid4().hex}.json"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(json.dumps(_sample_replay(), ensure_ascii=False), encoding="utf-8")
    screen = ReplayViewerScreen()

    screen.load_replay(replay_path)

    assert screen._payload is not None
    assert screen._map_widget._environment is not None
    assert "Шаг 1/1" in screen._step_label.text()
    assert "Hero attacks Goblin" in screen._last_action_text.toPlainText()

    screen.close()
    app.processEvents()


def _sample_replay() -> dict[str, object]:
    return {
        "format": "BattleReplay",
        "version": 1,
        "metadata": {"summary": "Hero vs Goblin"},
        "winner": "players",
        "xp_gained": 0,
        "steps": [
            {
                "round": 1,
                "turn_index": 0,
                "initiative_order": [
                    {"id": 0, "name": "Hero"},
                    {"id": 1, "name": "Goblin"},
                ],
                "actor": {"id": 0, "name": "Hero"},
                "actor_team": "players",
                "positions": {
                    "0": {"name": "Hero", "team": "players", "x": 0, "y": 0},
                    "1": {"name": "Goblin", "team": "enemies", "x": 2, "y": 1},
                },
                "hp_values": {
                    "0": {
                        "name": "Hero",
                        "team": "players",
                        "hp": 10,
                        "max_hp": 10,
                        "ac": 15,
                        "alive": True,
                    },
                    "1": {
                        "name": "Goblin",
                        "team": "enemies",
                        "hp": 0,
                        "max_hp": 7,
                        "ac": 13,
                        "alive": False,
                    },
                },
                "conditions": {
                    "0": {"name": "Hero", "conditions": [], "flags": {}},
                    "1": {"name": "Goblin", "conditions": [], "flags": {}},
                },
                "resources": {
                    "0": {
                        "name": "Hero",
                        "class_resources": {"Second Wind": 1},
                        "spell_slots": {},
                        "spell_slots_remaining": {},
                        "inventory": [],
                        "action_economy": {
                            "action_available": False,
                            "bonus_action_available": True,
                            "reaction_available": True,
                            "movement_remaining": 3,
                        },
                    },
                    "1": {
                        "name": "Goblin",
                        "class_resources": {},
                        "spell_slots": {},
                        "spell_slots_remaining": {},
                        "inventory": [],
                        "action_economy": {
                            "action_available": True,
                            "bonus_action_available": True,
                            "reaction_available": True,
                            "movement_remaining": 3,
                        },
                    },
                },
                "action": {"type": "AttackAction", "actor_id": 0, "target_id": 1},
                "action_category": "main_action",
                "targets": [
                    {"type": "creature", "id": 1, "name": "Goblin", "team": "enemies"}
                ],
                "dice_rolls": [{"die": "d20", "rolls": [20]}],
                "damage": [{"target": "Goblin", "amount": 7}],
                "healing": [],
                "spell_slots_spent": [],
                "items_spent": [],
                "deaths": [{"id": 1, "name": "Goblin", "team": "enemies"}],
                "reward": 1.0,
                "reward_breakdown": {},
                "success": True,
                "description": "Hero attacks Goblin.",
                "map_metadata": {
                    "width": 3,
                    "height": 2,
                    "terrain_counts": {},
                    "terrain_snapshot": [
                        ["normal", "blocked", "difficult_terrain"],
                        ["low_cover", "high_cover", "normal"],
                    ],
                },
                "cover_line_of_sight": None,
                "winner": "players",
                "xp_gained": {"total": 0, "by_character": {}},
            }
        ],
    }
