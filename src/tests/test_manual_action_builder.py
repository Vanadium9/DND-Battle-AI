import pytest

from combat import (
    AttackAction,
    CastSpellAction,
    Character,
    CombatState,
    GridMap,
    MoveAction,
    Position,
    Stats,
    Team,
    WeaponAttack,
)
from combat.monsters import GoblinMelee
from combat.presets import WizardEvoker
from ui.services import ManualActionBuilder, ManualTargetMode


def test_manual_builder_exposes_legal_movement_attack_and_end_turn() -> None:
    state = _duel_state()
    builder = ManualActionBuilder()

    plan = builder.build_plan(state, actor_id=0)

    assert "Movement" in plan.groups
    assert "Attack Abilities" in plan.groups
    assert "End Turn" in plan.groups

    move_option = plan.groups["Movement"][0]
    move_action = builder.build_action(
        state,
        0,
        move_option,
        target_cell=move_option.target_cells[0],
    )
    assert isinstance(move_action, MoveAction)
    assert move_action.is_valid(state)

    attack_option = next(
        option
        for option in plan.groups["Attack Abilities"]
        if option.label.startswith("Attack:")
    )
    assert attack_option.target_mode is ManualTargetMode.CREATURE
    assert attack_option.target_ids == (1,)
    attack_action = builder.build_action(
        state,
        0,
        attack_option,
        target_id=1,
    )
    assert isinstance(attack_action, AttackAction)
    assert attack_action.is_valid(state)


def test_manual_builder_rejects_illegal_manual_target() -> None:
    state = _duel_state()
    builder = ManualActionBuilder()
    plan = builder.build_plan(state, actor_id=0)
    attack_option = next(
        option
        for option in plan.groups["Attack Abilities"]
        if option.label.startswith("Attack:")
    )

    with pytest.raises(ValueError):
        builder.build_action(state, 0, attack_option, target_id=0)


def test_manual_builder_exposes_spell_slot_and_aoe_cell_options() -> None:
    wizard = WizardEvoker(Position(0, 0))
    goblin = GoblinMelee(Position(3, 0))
    state = CombatState(
        characters=[wizard, goblin],
        grid_map=GridMap(width=8, height=8),
        initiative_order=[0, 1],
    )
    builder = ManualActionBuilder()

    plan = builder.build_plan(state, actor_id=0)
    fireball_option = next(
        option
        for option in plan.groups["Spells"]
        if "Fireball" in option.label
    )

    assert fireball_option.slot_level == 3
    assert fireball_option.target_mode is ManualTargetMode.CELL
    action = builder.build_action(
        state,
        0,
        fireball_option,
        target_cell=fireball_option.target_cells[0],
    )
    assert isinstance(action, CastSpellAction)
    assert action.cast_level == 3
    assert action.is_valid(state)


def test_manual_builder_does_not_expose_unowned_class_features_as_spells() -> None:
    wizard = WizardEvoker(Position(0, 0))
    goblin = GoblinMelee(Position(3, 0))
    state = CombatState(
        characters=[wizard, goblin],
        grid_map=GridMap(width=8, height=8),
        initiative_order=[0, 1],
    )
    builder = ManualActionBuilder()

    plan = builder.build_plan(state, actor_id=0)
    spell_labels = [option.label for option in plan.groups["Spells"]]

    assert spell_labels
    assert not any("Second Wind" in label for label in spell_labels)
    assert not any("Action Surge" in label for label in spell_labels)
    assert any("Fireball" in label for label in spell_labels)


def _duel_state() -> CombatState:
    hero = Character(
        name="Hero",
        hp=12,
        max_hp=12,
        ac=15,
        position=Position(0, 0),
        speed=3,
        stats=Stats(str=16, dex=12, con=14),
        team=Team.PLAYERS,
        weapons=[
            WeaponAttack(
                name="Longsword",
                range=1,
                damage="1d8",
                ability_score="str",
                damage_ability_score="str",
            )
        ],
    )
    goblin = Character(
        name="Goblin",
        hp=7,
        max_hp=7,
        ac=13,
        position=Position(1, 0),
        speed=3,
        stats=Stats(str=8, dex=14, con=10),
        team=Team.ENEMIES,
        weapons=[WeaponAttack(name="Scimitar", range=1, damage="1d6")],
    )
    return CombatState(
        characters=[hero, goblin],
        grid_map=GridMap(width=5, height=5),
        initiative_order=[0, 1],
    )
