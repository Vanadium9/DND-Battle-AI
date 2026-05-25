import pytest

from agents import MainActionType, build_action_masks
from combat import (
    CastSpellAction,
    Character,
    CombatState,
    GridMap,
    Position,
    Stats,
    Team,
    available_castable_spells,
    build_character,
    supported_spell_options,
)


def make_enemy() -> Character:
    return Character(
        name="Enemy",
        hp=30,
        max_hp=30,
        ac=12,
        position=Position(1, 0),
        speed=3,
        stats=Stats(),
        team=Team.ENEMIES,
    )


def test_wizard_level_five_has_spell_slots_through_level_three() -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        subclass_name="School of Evocation",
        level=5,
        stats=Stats(int=18),
    )

    assert wizard.spell_slots == {1: 4, 2: 3, 3: 2}
    assert wizard.spell_slots_remaining == {1: 4, 2: 3, 3: 2}
    assert wizard.spellcasting_ability == "int"
    assert wizard.spell_save_dc == 15
    assert wizard.spell_attack_bonus == 7


def test_cleric_level_three_has_spell_slots_through_level_two() -> None:
    cleric = build_character(
        name="Cleric",
        class_name="Cleric",
        subclass_name="Life Domain",
        level=3,
        stats=Stats(wis=16),
    )

    assert cleric.spell_slots == {1: 4, 2: 2}
    assert cleric.spell_slots_remaining == {1: 4, 2: 2}
    assert cleric.spellcasting_ability == "wis"
    assert cleric.spell_save_dc == 13
    assert cleric.spell_attack_bonus == 5


def test_cantrip_does_not_spend_spell_slot(monkeypatch) -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        level=1,
        cantrips=("Fire Bolt",),
        prepared_spells=("Magic Missile",),
    )
    enemy = make_enemy()
    state = CombatState(characters=[wizard, enemy], grid_map=GridMap(width=4, height=4))
    fire_bolt = next(spell for spell in wizard.cantrips if spell.name == "Fire Bolt")
    before_slots = dict(wizard.spell_slots_remaining)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 5)

    result = CastSpellAction(actor_id=0, spell=fire_bolt, target_id=1).execute(state)

    assert result.success
    assert wizard.spell_slots_remaining == before_slots
    assert enemy.hp == 25


def test_level_three_spell_cannot_be_cast_without_level_three_slot() -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        level=5,
        prepared_spells=("Fireball",),
    )
    enemy = make_enemy()
    state = CombatState(characters=[wizard, enemy], grid_map=GridMap(width=8, height=8))
    fireball = next(spell for spell in wizard.prepared_spells if spell.name == "Fireball")
    wizard.spell_slots_remaining[3] = 0

    action = CastSpellAction(actor_id=0, spell=fireball, target_id=1)
    masks = build_action_masks(state, actor_id=0)
    castable_names = {spell.name for spell in available_castable_spells(wizard)}

    assert action.is_valid(state) is False
    assert "Fireball" not in castable_names
    assert masks["main_action_type"][MainActionType.CAST_SPELL]


def test_builder_rejects_spell_level_above_available_level() -> None:
    with pytest.raises(ValueError, match="not supported"):
        build_character(
            name="Wizard",
            class_name="Wizard",
            level=1,
            prepared_spells=("Fireball",),
        )


def test_action_masks_show_cantrips_and_prepared_spells_only() -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        level=5,
        known_spells=("Magic Missile", "Fireball"),
        prepared_spells=("Magic Missile",),
        cantrips=("Fire Bolt",),
    )
    state = CombatState(characters=[wizard, make_enemy()], grid_map=GridMap(width=8, height=8))

    castable_names = {spell.name for spell in available_castable_spells(wizard)}
    visible_options = [
        index
        for index, allowed in enumerate(build_action_masks(state, actor_id=0)["option_index"])
        if allowed
    ]

    assert {spell.name for spell in supported_spell_options("Wizard", level=5)}
    assert castable_names == {"Fire Bolt", "Magic Missile"}
    assert visible_options[:2] == [0, 1]
