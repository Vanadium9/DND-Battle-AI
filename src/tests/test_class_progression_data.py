import pytest

from agents import (
    ACTOR_CLASS_FEATURE_SIZE,
    ACTOR_FEATURE_SIZE,
    build_action_masks,
    encode_observation,
)
from agents.action_space import ActionCategory
from combat import (
    Character,
    ClassFeature,
    CombatState,
    GridMap,
    Position,
    Stats,
    Team,
    build_character,
    supported_class_options,
    supported_subclass_options,
    validate_class_selection,
)
from rules import (
    ClassDefinition,
    FeatureDefinition,
    SubclassDefinition,
    build_class_features,
    get_class_definition,
)


def make_enemy() -> Character:
    return Character(
        name="Enemy",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(1, 0),
        speed=3,
        stats=Stats(),
        team=Team.ENEMIES,
    )


def test_fighter_level_one_gets_only_level_one_features() -> None:
    fighter = build_character(name="Fighter", class_name="Fighter", level=1)

    assert {feature.name for feature in fighter.class_features} == {
        "Fighting Style",
        "Second Wind",
    }
    assert all(feature.level <= 1 for feature in fighter.class_features)
    assert all(isinstance(feature, FeatureDefinition) for feature in fighter.class_features)
    assert all(isinstance(feature, ClassFeature) for feature in fighter.class_features)


def test_fighter_level_three_gets_subclass_features() -> None:
    fighter = build_character(
        name="Champion",
        class_name="Fighter",
        subclass_name="Champion",
        level=3,
    )

    assert {feature.name for feature in fighter.class_features} == {
        "Fighting Style",
        "Second Wind",
        "Action Surge",
        "Martial Archetype: Champion",
        "Improved Critical",
    }


def test_wizard_level_five_gets_class_features_through_level_five() -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        subclass_name="School of Evocation",
        level=5,
    )

    feature_names = {feature.name for feature in wizard.class_features}
    assert {
        "Spellcasting",
        "Arcane Recovery",
        "Arcane Tradition: School of Evocation",
        "Evocation Savant",
        "Sculpt Spells",
        "2nd-level Spells",
        "Ability Score Improvement",
        "3rd-level Spells",
    } == feature_names
    assert wizard.spell_slots == {1: 4, 2: 3, 3: 2}


def test_unsupported_feature_does_not_enter_action_mask() -> None:
    actor = Character(
        name="Actor",
        hp=10,
        max_hp=10,
        ac=12,
        position=Position(0, 0),
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
        common_actions=["end_turn"],
        class_features=[
            ClassFeature(
                name="Prototype Bonus Feature",
                level=1,
                action_cost="bonus_action",
                active_action="prototype_bonus",
                implemented=False,
            )
        ],
    )
    state = CombatState(
        characters=[actor, make_enemy()],
        grid_map=GridMap(width=3, height=3),
    )

    masks = build_action_masks(state, actor_id=0)

    assert masks["action_category"][ActionCategory.BONUS_ACTION].item() is False


def test_character_builder_filters_classes_and_subclasses_by_ruleset() -> None:
    class_names = {definition.name for definition in supported_class_options()}
    fighter_subclasses = {
        definition.name
        for definition in supported_subclass_options("Fighter", level=3)
    }

    assert all(isinstance(definition, ClassDefinition) for definition in supported_class_options())
    assert all(
        isinstance(definition, SubclassDefinition)
        for definition in supported_subclass_options("Fighter", level=3)
    )
    assert class_names == {"Fighter", "Cleric", "Wizard"}
    assert fighter_subclasses == {"Champion"}
    assert supported_subclass_options("Fighter", level=1) == ()
    with pytest.raises(ValueError, match="not supported|rejected during import"):
        validate_class_selection("Rogue")
    with pytest.raises(ValueError, match="chooses a subclass at level 3"):
        validate_class_selection("Fighter", "Champion", level=1)


def test_observation_encodes_implemented_class_feature_signals() -> None:
    wizard = build_character(name="Wizard", class_name="Wizard", level=5)
    state = CombatState(
        characters=[wizard, make_enemy()],
        grid_map=GridMap(width=3, height=3),
    )

    observation = encode_observation(state, actor_id=0)
    feature_block = observation[
        ACTOR_FEATURE_SIZE - ACTOR_CLASS_FEATURE_SIZE : ACTOR_FEATURE_SIZE
    ]

    assert feature_block[0] == 6
    assert feature_block[1] == 6
    assert feature_block[2] == 1
    assert feature_block[3] == 1


def test_class_definitions_are_data_driven_progression_tables() -> None:
    fighter_definition = get_class_definition("Fighter")

    assert fighter_definition is not None
    assert fighter_definition.hit_die == 10
    assert fighter_definition.subclass_level == 3
    assert 1 in fighter_definition.level_features
    assert build_class_features("Fighter", 1, "Champion")[0].name == "Fighting Style"
