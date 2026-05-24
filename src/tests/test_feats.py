import pytest

from character import CharacterSchema
from combat import (
    AbilityScoreImprovement,
    Character,
    Position,
    Stats,
    Team,
    add_feat,
    apply_ability_score_improvement,
    get_active_combat_hooks,
    get_supported_feats_for_builder,
    on_ability_check,
    on_attack_roll,
    on_damage_roll,
    on_saving_throw,
    on_turn_end,
    on_turn_start,
)
from combat.checks import AbilityCheckResult
from rules import (
    FeatDefinition,
    get_feat_definition,
    is_supported_content,
    validate_feat_prerequisites,
)


def make_character(
    *,
    level: int = 4,
    stats: Stats | None = None,
) -> Character:
    return Character(
        name="Hero",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(0, 0),
        speed=3,
        stats=stats or Stats(),
        team=Team.PLAYERS,
        class_name="Fighter",
        level=level,
    )


def test_asi_changes_stats_and_is_stored() -> None:
    character = make_character(stats=Stats(str=10, dex=12))

    apply_ability_score_improvement(character, {"str": 2})

    assert character.stats.str == 12
    assert character.stats.dex == 12
    assert character.ability_score_improvements == [
        AbilityScoreImprovement({"str": 2})
    ]


def test_asi_allows_plus_one_to_two_stats() -> None:
    character = make_character(stats=Stats(str=10, dex=12))

    apply_ability_score_improvement(character, {"str": 1, "dex": 1})

    assert character.stats.str == 11
    assert character.stats.dex == 13


def test_asi_respects_stat_cap_when_cap_is_set() -> None:
    character = make_character(stats=Stats(str=19))

    with pytest.raises(ValueError, match="above cap"):
        apply_ability_score_improvement(character, {"str": 2}, stat_cap=20)

    assert character.stats.str == 19
    assert character.ability_score_improvements == []


def test_feat_prerequisites_are_checked() -> None:
    grappler = get_feat_definition("Grappler")
    assert grappler is not None

    assert not validate_feat_prerequisites(
        make_character(level=4, stats=Stats(str=12)),
        grappler,
    )
    assert not validate_feat_prerequisites(
        make_character(level=3, stats=Stats(str=16)),
        grappler,
    )
    assert validate_feat_prerequisites(
        make_character(level=4, stats=Stats(str=13)),
        grappler,
    )


def test_builder_shows_only_ruleset_supported_implemented_feats() -> None:
    character = make_character(level=4, stats=Stats(str=18))

    available_names = {feat.name for feat in get_supported_feats_for_builder(character)}

    assert "Ability Score Improvement" in available_names
    assert "Grappler" not in available_names
    assert is_supported_content("feat", "Ability Score Improvement")
    assert not is_supported_content("feat", "Grappler")
    with pytest.raises(ValueError, match="not supported"):
        add_feat(character, "Grappler")


def test_unimplemented_feat_does_not_affect_combat_hooks() -> None:
    character = make_character()
    character.feats.append(
        FeatDefinition(
            name="Prototype Feat",
            combat_hooks={
                "on_attack_roll": ("prototype_attack",),
                "on_damage_roll": ("prototype_damage",),
                "on_saving_throw": ("prototype_save",),
                "on_ability_check": ("prototype_check",),
                "on_turn_start": ("prototype_start",),
                "on_turn_end": ("prototype_end",),
            },
            implemented=False,
        )
    )
    ability_check = AbilityCheckResult(
        character_name="Hero",
        check_name="Athletics",
        ability="str",
        rolls=(10,),
        kept_roll=10,
        ability_modifier=0,
        proficiency_bonus=2,
        total=12,
    )

    assert get_active_combat_hooks(character) == {}
    assert on_attack_roll(character, 7) == 7
    assert on_damage_roll(character, 9) == 9
    assert on_saving_throw(character, 15) == 15
    assert on_ability_check(character, ability_check) is ability_check
    assert on_turn_start(character) is character
    assert on_turn_end(character) is character


def test_character_schema_includes_feats_and_asi() -> None:
    character = make_character()
    character.feats.append("Grappler")
    apply_ability_score_improvement(character, {"wis": 1, "cha": 1})

    schema = CharacterSchema.from_character(character)

    assert schema.feats[0].name == "Grappler"
    assert schema.ability_score_improvements[0].bonuses == {"wis": 1, "cha": 1}
