import pytest

from character import CharacterRaceSchema, CharacterSchema
from combat import (
    AttackAction,
    Character,
    CombatState,
    GridMap,
    Position,
    Stats,
    Team,
    WeaponAttack,
    apply_race_traits,
    has_damage_resistance,
    use_halfling_lucky,
)
from rules import get_race_definition, is_supported_race


def make_character(
    name: str = "Hero",
    race_name: str | None = None,
    speed: int = 3,
    weapon: WeaponAttack | None = None,
) -> Character:
    return Character(
        name=name,
        hp=20,
        max_hp=20,
        ac=12,
        position=Position(0, 0),
        speed=speed,
        stats=Stats(),
        team=Team.PLAYERS,
        race_name=race_name,
        weapons=[] if weapon is None else [weapon],
    )


def test_race_bonuses_and_traits_are_applied() -> None:
    character = make_character(race_name="Human")

    apply_race_traits(character)

    assert character.race_name == "Human"
    assert character.race_traits is not None
    assert character.stats.str == 11
    assert character.stats.dex == 11
    assert character.stats.con == 11
    assert character.stats.int == 11
    assert character.stats.wis == 11
    assert character.stats.cha == 11
    assert character.race_traits.ability_score_bonuses["str"] == 1


def test_speed_comes_from_race_or_explicit_character_override() -> None:
    dwarf = make_character(race_name="Dwarf", speed=3)
    elf = make_character(race_name="Elf", speed=3)

    apply_race_traits(dwarf)
    apply_race_traits(elf, override_speed=4)

    assert dwarf.speed == 2
    assert dwarf.action_economy.movement_remaining == 2
    assert elf.speed == 4
    assert elf.action_economy.movement_remaining == 4


def test_racial_weapon_proficiency_marks_matching_weapons_proficient() -> None:
    longbow = WeaponAttack(name="Longbow", range=6, damage=4, proficient=False)
    character = make_character(race_name="Elf", weapon=longbow)

    apply_race_traits(character)

    assert longbow.proficient is True


def test_racial_damage_resistance_reduces_weapon_damage() -> None:
    attacker_weapon = WeaponAttack(
        name="Venom Blade",
        range=1,
        damage=10,
        attack_bonus=20,
        damage_type="poison",
    )
    attacker = make_character(name="Attacker", weapon=attacker_weapon)
    attacker.team = Team.ENEMIES
    defender = make_character(name="Defender", race_name="Dwarf")
    defender.position = Position(1, 0)
    apply_race_traits(defender)
    state = CombatState(
        characters=[attacker, defender],
        grid_map=GridMap(width=3, height=3),
    )

    result = AttackAction(actor_id=0, target_id=1, weapon=attacker_weapon).execute(state)

    assert result.success
    assert has_damage_resistance(defender, "poison")
    assert defender.hp == 15
    assert "for 5 damage" in result.description


def test_darkvision_is_stored_but_not_used_by_combat_targeting() -> None:
    elf = make_character(race_name="Elf")

    apply_race_traits(elf)

    assert elf.race_traits is not None
    assert elf.race_traits.darkvision_range == 6


def test_halfling_lucky_feature_flag_can_reroll_natural_one() -> None:
    halfling = make_character(race_name="Halfling")
    apply_race_traits(halfling)

    assert halfling.race_traits is not None
    assert halfling.race_traits.halfling_lucky_enabled is True
    assert use_halfling_lucky(halfling, roll=1, reroll=12) == 12
    assert use_halfling_lucky(halfling, roll=2, reroll=12) == 2


def test_custom_race_fallback_is_import_only_and_warns() -> None:
    assert not is_supported_race("Aasimar")
    with pytest.raises(ValueError):
        get_race_definition("Aasimar")

    with pytest.warns(UserWarning):
        schema = CharacterRaceSchema.from_imported_name("Aasimar")

    assert schema.race_name == "CustomRace"


def test_character_schema_includes_race_metadata() -> None:
    character = make_character(race_name="Dwarf")
    apply_race_traits(character)

    schema = CharacterSchema.from_character(character)

    assert schema.race.race_name == "Dwarf"
    assert schema.race.size == "Medium"
    assert schema.race.speed == 2
    assert "poison" in schema.race.damage_resistances
