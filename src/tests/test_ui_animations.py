from pathlib import Path
from uuid import uuid4

from combat import (
    ActionResult,
    AttackAction,
    Character,
    CombatState,
    MoveAction,
    Position,
    Stats,
    Team,
    WeaponAttack,
)
from ui.animations import (
    BattleAnimationKind,
    build_battle_animations,
    normalize_animation_speed,
    snapshot_battle_state,
)
from ui.services import ModelService


def test_battle_animations_include_movement_delta() -> None:
    state = _state()
    before = snapshot_battle_state(state)
    state.characters[0].position = Position(1, 0)

    animations = build_battle_animations(
        before,
        state,
        MoveAction(actor_id=0, destination=Position(1, 0)),
        ActionResult(True, "moved"),
    )

    movement = [item for item in animations if item.kind is BattleAnimationKind.MOVEMENT]
    assert len(movement) == 1
    assert movement[0].start == Position(0, 0)
    assert movement[0].end == Position(1, 0)


def test_battle_animations_include_attack_and_damage_delta() -> None:
    state = _state()
    before = snapshot_battle_state(state)
    state.characters[1].hp = 4
    weapon = state.characters[0].weapons[0]

    animations = build_battle_animations(
        before,
        state,
        AttackAction(actor_id=0, target_id=1, weapon=weapon),
        ActionResult(True, "hit for 6 damage"),
    )

    kinds = {item.kind for item in animations}
    assert BattleAnimationKind.MELEE_ATTACK in kinds
    assert BattleAnimationKind.DAMAGE in kinds


def test_animation_settings_are_persisted() -> None:
    settings_path = Path("checkpoints") / f"test_ui_animation_settings_{uuid4().hex}.json"
    service = ModelService(settings_path=settings_path)

    service.set_settings(animations_enabled=False, animation_speed=900)
    reloaded = ModelService(settings_path=settings_path)

    assert reloaded.settings.animations_enabled is False
    assert reloaded.settings.animation_speed == 900
    assert normalize_animation_speed(9999) == 1500


def _state() -> CombatState:
    hero = Character(
        name="Hero",
        hp=10,
        max_hp=10,
        ac=14,
        position=Position(0, 0),
        speed=30,
        stats=Stats(str=16, dex=12, con=14),
        team=Team.PLAYERS,
        weapons=[WeaponAttack(name="Sword", range=1, damage="1d6")],
    )
    enemy = Character(
        name="Goblin",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(1, 0),
        speed=30,
        stats=Stats(str=10, dex=14, con=10),
        team=Team.ENEMIES,
        weapons=[WeaponAttack(name="Dagger", range=1, damage="1d4")],
    )
    return CombatState(characters=[hero, enemy])
