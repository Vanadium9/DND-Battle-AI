import json
from pathlib import Path

from scripts.view_replay_console import load_replay, render_ascii_map, render_step


def test_console_replay_viewer_loads_replay() -> None:
    replay_path = Path("checkpoints") / "test_console_viewer_replay.json"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps(_sample_replay(), ensure_ascii=False),
        encoding="utf-8",
    )

    replay = load_replay(replay_path)
    rendered = render_step(replay, 0)
    ascii_map = render_ascii_map(replay["steps"][0])

    assert replay["format"] == "BattleReplay"
    assert "BattleReplay step 1/1" in rendered
    assert "Initiative: 0:Hero -> 1:Goblin" in rendered
    assert "Hero -> AttackAction" in rendered
    assert "#" in ascii_map
    assert "~" in ascii_map
    assert "=" in ascii_map
    assert "^" in ascii_map


def _sample_replay() -> dict[str, object]:
    return {
        "format": "BattleReplay",
        "version": 1,
        "metadata": {},
        "winner": None,
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
                    "0": {"name": "Hero", "x": 0, "y": 0},
                    "1": {"name": "Goblin", "x": 2, "y": 1},
                },
                "hp_values": {
                    "0": {"name": "Hero", "hp": 10, "max_hp": 10, "alive": True},
                    "1": {"name": "Goblin", "hp": 4, "max_hp": 7, "alive": True},
                },
                "conditions": {
                    "0": {"name": "Hero", "conditions": [], "flags": {}},
                    "1": {"name": "Goblin", "conditions": [], "flags": {}},
                },
                "resources": {
                    "0": {
                        "name": "Hero",
                        "class_resources": {"Second Wind": 1},
                        "spell_slots": {"1": 2},
                        "spell_slots_remaining": {"1": 1},
                        "inventory": [{"name": "Potion of Healing", "quantity": 1}],
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
                "dice_rolls": [{"die": "d20", "rolls": [15]}],
                "damage": [{"target": "Goblin", "amount": 3}],
                "healing": [],
                "spell_slots_spent": [],
                "items_spent": [],
                "deaths": [],
                "reward": 0.3,
                "reward_breakdown": {"damage_dealt": 0.3},
                "success": True,
                "description": "Hero attacks Goblin.",
                "map_metadata": {
                    "width": 3,
                    "height": 2,
                    "terrain_counts": {
                        "normal": 2,
                        "blocked": 1,
                        "difficult_terrain": 1,
                        "low_cover": 1,
                        "high_cover": 1,
                    },
                    "terrain_snapshot": [
                        ["normal", "blocked", "difficult_terrain"],
                        ["low_cover", "high_cover", "normal"],
                    ],
                },
                "cover_line_of_sight": {
                    "line_of_sight": True,
                    "cover": "no_cover",
                },
                "winner": None,
                "xp_gained": {"total": 0, "by_character": {}},
            }
        ],
    }
