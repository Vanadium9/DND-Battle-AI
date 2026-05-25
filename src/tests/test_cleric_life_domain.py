from agents import ActionCategory, MainActionType, build_action_masks, decode_action
from combat import (
    CastSpellAction,
    ChannelDivinityPreserveLifeAction,
    Character,
    ClericLifeSupport,
    CombatState,
    GridMap,
    Position,
    Stats,
    Team,
    WeaponAttack,
    build_character,
    supported_spell_options,
)


def make_ally(hp: int = 5, max_hp: int = 20) -> Character:
    return Character(
        name="Ally",
        hp=hp,
        max_hp=max_hp,
        ac=12,
        position=Position(1, 0),
        speed=3,
        stats=Stats(),
        team=Team.PLAYERS,
    )


def make_enemy() -> Character:
    return Character(
        name="Enemy",
        hp=20,
        max_hp=20,
        ac=12,
        position=Position(2, 0),
        speed=3,
        stats=Stats(dex=8),
        team=Team.ENEMIES,
    )


def spell_by_name(character: Character, name: str):
    for spell in [*character.cantrips, *character.prepared_spells]:
        if spell.name == name:
            return spell
    raise AssertionError(f"Missing spell: {name}")


def test_healing_word_spends_bonus_action_and_slot(monkeypatch) -> None:
    cleric = build_character(
        name="Cleric",
        class_name="Cleric",
        subclass_name="Life Domain",
        level=1,
        prepared_spells=("Healing Word",),
    )
    ally = make_ally()
    state = CombatState(characters=[cleric, ally], grid_map=GridMap(width=4, height=4))
    spell = spell_by_name(cleric, "Healing Word")
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 4)

    result = CastSpellAction(actor_id=0, spell=spell, target_id=1).execute(state)

    assert result.success
    assert cleric.action_economy.bonus_action_available is False
    assert cleric.action_economy.action_available is True
    assert cleric.spell_slots_remaining[1] == cleric.spell_slots[1] - 1
    assert ally.hp == 9


def test_cure_wounds_spends_action_and_slot(monkeypatch) -> None:
    cleric = build_character(
        name="Cleric",
        class_name="Cleric",
        subclass_name="Life Domain",
        level=1,
        prepared_spells=("Cure Wounds",),
    )
    ally = make_ally()
    state = CombatState(characters=[cleric, ally], grid_map=GridMap(width=4, height=4))
    spell = spell_by_name(cleric, "Cure Wounds")
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 6)

    result = CastSpellAction(actor_id=0, spell=spell, target_id=1).execute(state)

    assert result.success
    assert cleric.action_economy.action_available is False
    assert cleric.action_economy.bonus_action_available is True
    assert cleric.spell_slots_remaining[1] == cleric.spell_slots[1] - 1
    assert ally.hp == 11


def test_channel_divinity_preserve_life_spends_resource() -> None:
    cleric = build_character(
        name="Cleric",
        class_name="Cleric",
        subclass_name="Life Domain",
        level=2,
    )
    ally = make_ally(hp=1, max_hp=20)
    state = CombatState(characters=[cleric, ally], grid_map=GridMap(width=4, height=4))

    masks = build_action_masks(state, actor_id=0)
    decoded = decode_action(
        ActionCategory.CLASS_FEATURE,
        MainActionType.ATTACK,
        target_index=1,
        move_index=0,
        option_index=0,
        state=state,
        actor_id=0,
    )
    result = decoded.execute(state)

    assert isinstance(decoded, ChannelDivinityPreserveLifeAction)
    assert masks["action_category"][ActionCategory.CLASS_FEATURE]
    assert result.success
    assert cleric.resources["channel_divinity"].uses_remaining == 0
    assert cleric.action_economy.action_available is False
    assert ally.hp == 10


def test_cleric_level_one_has_no_channel_divinity() -> None:
    cleric = build_character(
        name="Cleric",
        class_name="Cleric",
        subclass_name="Life Domain",
        level=1,
    )
    state = CombatState(characters=[cleric, make_ally()], grid_map=GridMap(width=4, height=4))

    feature_names = {feature.name for feature in cleric.class_features}
    masks = build_action_masks(state, actor_id=0)

    assert "Channel Divinity" not in feature_names
    assert "channel_divinity" not in cleric.resources
    assert not masks["action_category"][ActionCategory.CLASS_FEATURE]


def test_cleric_life_support_preset_uses_common_mace_attack() -> None:
    cleric = ClericLifeSupport()
    mace = cleric.weapons[0]

    assert cleric.level == 5
    assert cleric.subclass_name == "Life Domain"
    assert cleric.spellcasting_ability == "wis"
    assert cleric.spell_slots == {1: 4, 2: 3, 3: 2}
    assert isinstance(mace, WeaponAttack)
    assert mace.name == "Mace"
    assert mace.ability_score == "str"


def test_builder_shows_only_supported_cleric_spells() -> None:
    spell_names = {spell.name for spell in supported_spell_options("Cleric", level=5)}

    assert {
        "Sacred Flame",
        "Cure Wounds",
        "Healing Word",
        "Guiding Bolt",
        "Bless",
    }.issubset(spell_names)
    assert "Spiritual Weapon" not in spell_names
    assert "Revivify" not in spell_names


def test_healing_spells_are_masked_without_wounded_allies() -> None:
    cleric = build_character(
        name="Cleric",
        class_name="Cleric",
        subclass_name="Life Domain",
        level=1,
        prepared_spells=("Cure Wounds", "Healing Word"),
    )
    cleric.cantrips = []
    cleric.abilities = [spell for spell in cleric.abilities if spell in cleric.prepared_spells]
    ally = make_ally(hp=20, max_hp=20)
    state = CombatState(characters=[cleric, ally], grid_map=GridMap(width=4, height=4))

    masks = build_action_masks(state, actor_id=0)

    assert not masks["action_category"][ActionCategory.BONUS_ACTION]
    assert not masks["main_action_type"][MainActionType.CAST_SPELL]
