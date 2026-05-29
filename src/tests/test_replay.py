import json
from pathlib import Path

from combat import (
    AttackAction,
    BattleReplay,
    Character,
    CombatEnvironment,
    GridMap,
    Position,
    Stats,
    Team,
    WeaponAttack,
)


def test_battle_replay_saves_json() -> None:
    replay, _ = _record_attack_replay()
    replay_path = Path("checkpoints") / "test_battle_replay.json"

    saved_path = replay.save(replay_path)
    data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert data["format"] == "BattleReplay"
    assert data["version"] == 1
    assert len(data["steps"]) == 1


def test_battle_replay_step_contains_required_fields() -> None:
    _, step = _record_attack_replay()

    required_fields = {
        "round",
        "turn_index",
        "initiative_order",
        "actor",
        "actor_team",
        "positions",
        "hp_values",
        "conditions",
        "resources",
        "action",
        "action_category",
        "targets",
        "dice_rolls",
        "damage",
        "healing",
        "spell_slots_spent",
        "items_spent",
        "deaths",
        "reward_breakdown",
        "map_metadata",
        "cover_line_of_sight",
        "winner",
        "xp_gained",
    }

    assert required_fields.issubset(step)
    assert step["action"]["type"] == "AttackAction"
    assert step["action_category"] == "main_action"
    assert step["damage"][0]["target"] == "Goblin"
    assert step["deaths"][0]["name"] == "Goblin"
    assert step["winner"] == "players"
    assert step["xp_gained"]["total"] == 50
    assert step["cover_line_of_sight"]["line_of_sight"] is True


def _record_attack_replay() -> tuple[BattleReplay, dict[str, object]]:
    sword = WeaponAttack(name="Sword", range=1, damage=4, attack_bonus=20)
    hero = Character(
        name="Hero",
        hp=10,
        max_hp=10,
        ac=14,
        position=Position(0, 0),
        speed=3,
        stats=Stats(str=16),
        team=Team.PLAYERS,
        weapons=[sword],
    )
    goblin = Character(
        name="Goblin",
        hp=4,
        max_hp=4,
        ac=12,
        position=Position(1, 0),
        speed=3,
        stats=Stats(),
        team=Team.ENEMIES,
        xp_value=50,
    )
    environment = CombatEnvironment(
        characters=[hero, goblin],
        grid_map=GridMap(width=3, height=3),
        use_initiative=False,
        log_to_console=False,
    )
    replay = BattleReplay(metadata={"test": True})
    action = AttackAction(actor_id=0, target_id=1, weapon=sword)

    before = replay.snapshot_state(environment.combat_state)
    result = environment.step(action)
    step = replay.record_step(before, environment.combat_state, action, result)

    assert result.success
    return replay, step
