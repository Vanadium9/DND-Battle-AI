from rules import (
    DEFAULT_RULESET_NAME,
    get_active_ruleset,
    get_unsupported_reason,
    is_supported_content,
)


def test_default_ruleset_loads_minimal_srd5e_surface() -> None:
    ruleset = get_active_ruleset()

    assert ruleset.ruleset_name == DEFAULT_RULESET_NAME
    assert ruleset.supported_levels == (1, 2, 3, 4, 5)
    assert ruleset.supported_classes == ("Fighter", "Cleric", "Wizard")
    assert ruleset.supported_subclasses["Fighter"] == ("Champion",)
    assert ruleset.supported_races == ("Human", "Dwarf", "Elf", "Halfling")
    assert ruleset.supported_spell_levels == (0, 1, 2, 3)
    assert ruleset.supported_feats == ("Ability Score Improvement",)


def test_supported_content_checks_are_case_and_style_insensitive() -> None:
    assert is_supported_content("class", "fighter")
    assert is_supported_content("subclass", "Fighter: Champion")
    assert is_supported_content("subclass", "school_of_evocation")
    assert is_supported_content("race", "HALFLING")
    assert is_supported_content("common_action", "cast_spell")
    assert is_supported_content("common_action", "EndTurn")
    assert is_supported_content("level", "5")
    assert is_supported_content("spell_level", 3)
    assert is_supported_content("feat", "ability-score-improvement")
    assert is_supported_content("supported_classes", "Wizard")
    assert is_supported_content("supported_spell_levels", "0")


def test_unsupported_reasons_include_content_policy() -> None:
    assert not is_supported_content("class", "Rogue")
    assert "rejected during import" in get_unsupported_reason("class", "Rogue")

    assert not is_supported_content("race", "Dragonborn")
    assert "CustomRace" in get_unsupported_reason("race", "Dragonborn")

    assert not is_supported_content("spell", "Fireball")
    assert "marked as unavailable" in get_unsupported_reason("spell", "Fireball")

    assert not is_supported_content("feature", "Sneak Attack")
    assert "saved as notes" in get_unsupported_reason("feature", "Sneak Attack")


def test_unsupported_actions_and_levels_are_explained() -> None:
    assert not is_supported_content("common_action", "ImprovisedAction")
    assert "Supported actions" in get_unsupported_reason(
        "common_action",
        "ImprovisedAction",
    )

    assert not is_supported_content("level", 6)
    assert "1-5" in get_unsupported_reason("level", 6)

    assert not is_supported_content("spell_level", 4)
    assert "0-3" in get_unsupported_reason("spell_level", 4)
