import pytest

from agents import ActionCategory, build_action_masks
from combat import (
    CastSpellAction,
    Character,
    CombatState,
    GridMap,
    Position,
    Stats,
    Team,
    WizardEvoker,
    build_character,
    supported_spell_options,
)
from combat.spellcasting import available_castable_spells


def make_enemy(position: Position) -> Character:
    return Character(
        name="Enemy",
        hp=40,
        max_hp=40,
        ac=12,
        position=position,
        speed=3,
        stats=Stats(dex=8),
        team=Team.ENEMIES,
    )


def make_ally(position: Position) -> Character:
    return Character(
        name="Ally",
        hp=24,
        max_hp=24,
        ac=14,
        position=position,
        speed=3,
        stats=Stats(dex=10),
        team=Team.PLAYERS,
    )


def spell_by_name(character: Character, spell_name: str):
    spells = [*character.cantrips, *character.prepared_spells]
    return next(spell for spell in spells if spell.name == spell_name)


def test_wizard_level_one_cannot_prepare_fireball() -> None:
    level_one_spells = {spell.name for spell in supported_spell_options("Wizard", level=1)}

    assert "Fireball" not in level_one_spells
    with pytest.raises(ValueError, match="not supported"):
        build_character(
            name="Wizard",
            class_name="Wizard",
            level=1,
            prepared_spells=("Fireball",),
        )


def test_wizard_level_five_can_cast_fireball_with_level_three_slot(monkeypatch) -> None:
    wizard = WizardEvoker(Position(0, 0))
    enemy = make_enemy(Position(3, 0))
    state = CombatState(
        characters=[wizard, enemy],
        grid_map=GridMap(width=6, height=4),
    )
    fireball = spell_by_name(wizard, "Fireball")
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    result = CastSpellAction(actor_id=0, spell=fireball, target_id=1).execute(state)

    assert result.success
    assert wizard.spell_slots_remaining[3] == wizard.spell_slots[3] - 1
    assert enemy.hp < enemy.max_hp


def test_shield_spends_reaction_and_temporary_ac() -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        level=1,
        known_spells=("Shield",),
        prepared_spells=("Shield",),
    )
    state = CombatState(characters=[wizard], grid_map=GridMap(width=3, height=3))
    shield = spell_by_name(wizard, "Shield")
    starting_ac = wizard.ac
    starting_slots = wizard.spell_slots_remaining[1]

    result = CastSpellAction(actor_id=0, spell=shield).execute(state)

    assert result.success
    assert wizard.action_economy.reaction_available is False
    assert wizard.spell_slots_remaining[1] == starting_slots - 1
    assert wizard.ac == starting_ac + 5
    assert build_action_masks(state, actor_id=0)["action_category"][
        ActionCategory.REACTION
    ].item() is False

    state.reset_turn_resources(actor_id=0)

    assert wizard.ac == starting_ac
    assert wizard.action_economy.reaction_available is True


@pytest.mark.parametrize(
    ("spell_name", "enemy_position", "ally_position"),
    [
        ("Fireball", Position(3, 0), Position(2, 0)),
        ("Burning Hands", Position(1, 0), Position(1, 1)),
    ],
)
def test_sculpt_spells_protects_allies_from_evocation_aoe(
    monkeypatch,
    spell_name: str,
    enemy_position: Position,
    ally_position: Position,
) -> None:
    wizard = WizardEvoker(Position(0, 0))
    ally = make_ally(ally_position)
    enemy = make_enemy(enemy_position)
    state = CombatState(
        characters=[wizard, ally, enemy],
        grid_map=GridMap(width=6, height=4),
    )
    spell = spell_by_name(wizard, spell_name)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    result = CastSpellAction(actor_id=0, spell=spell, target_id=2).execute(state)

    assert result.success
    assert ally.hp == ally.max_hp
    assert enemy.hp < enemy.max_hp


def test_arcane_recovery_resets_wizard_slots_between_combats() -> None:
    wizard = WizardEvoker()
    wizard.spell_slots_remaining[1] = 0
    wizard.spell_slots_remaining[3] = 0

    wizard.reset_combat_resources()

    assert wizard.spell_slots_remaining == wizard.spell_slots


def test_wizard_evoker_preset_uses_quarterstaff_common_attack() -> None:
    wizard = WizardEvoker()
    castable_names = {spell.name for spell in available_castable_spells(wizard)}

    assert wizard.level == 5
    assert wizard.subclass_name == "School of Evocation"
    assert wizard.weapons[0].name == "Quarterstaff"
    assert {"Fire Bolt", "Shield", "Burning Hands", "Fireball"}.issubset(castable_names)
