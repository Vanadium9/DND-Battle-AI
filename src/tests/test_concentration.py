import logging

from combat import (
    AttackAction,
    CastSpellAction,
    Character,
    CombatState,
    GridMap,
    Position,
    SpellAbility,
    Stats,
    Team,
    WeaponAttack,
    build_character,
    handle_concentration_damage,
)


def make_cleric() -> Character:
    return build_character(
        name="Cleric",
        class_name="Cleric",
        subclass_name="Life Domain",
        level=1,
        stats=Stats(con=10, wis=16),
        prepared_spells=("Bless",),
    )


def spell_by_name(character: Character, spell_name: str) -> SpellAbility:
    return next(
        spell
        for spell in [*character.cantrips, *character.prepared_spells]
        if spell.name == spell_name
    )


def test_concentration_spell_sets_active_concentration_spell(caplog) -> None:
    cleric = make_cleric()
    state = CombatState(characters=[cleric], grid_map=GridMap(width=3, height=3))
    bless = spell_by_name(cleric, "Bless")

    with caplog.at_level(logging.INFO, logger="combat.conditions"):
        result = CastSpellAction(actor_id=0, spell=bless).execute(state)

    assert result.success
    assert cleric.active_concentration_spell is bless
    assert "starts concentration on Bless" in caplog.text


def test_new_concentration_spell_replaces_old(caplog) -> None:
    cleric = make_cleric()
    state = CombatState(characters=[cleric], grid_map=GridMap(width=3, height=3))
    bless = spell_by_name(cleric, "Bless")
    protection = SpellAbility(
        name="Protection",
        spell_level=0,
        action_cost="action",
        target_type="self",
        concentration=True,
    )
    cleric.cantrips.append(protection)
    cleric.abilities.append(protection)

    CastSpellAction(actor_id=0, spell=bless).execute(state)
    cleric.action_economy.action_available = True
    with caplog.at_level(logging.INFO, logger="combat.conditions"):
        result = CastSpellAction(actor_id=0, spell=protection).execute(state)

    assert result.success
    assert cleric.active_concentration_spell is protection
    assert "replaces concentration on Bless with Protection" in caplog.text


def test_damage_rolls_concentration_con_save(monkeypatch) -> None:
    cleric = make_cleric()
    bless = spell_by_name(cleric, "Bless")
    cleric.active_concentration_spell = bless
    monkeypatch.setattr("combat.conditions.random.randint", lambda _low, _high: 20)

    result = handle_concentration_damage(cleric, 20)

    assert result is not None
    assert result.dc == 10
    assert result.d20_roll == 20
    assert result.total == 20
    assert result.success is True
    assert cleric.active_concentration_spell is bless


def test_failed_con_save_drops_concentration_from_damage(monkeypatch, caplog) -> None:
    cleric = make_cleric()
    bless = spell_by_name(cleric, "Bless")
    cleric.active_concentration_spell = bless
    attacker = Character(
        name="Orc",
        hp=20,
        max_hp=20,
        ac=12,
        position=Position(1, 0),
        speed=3,
        stats=Stats(str=16),
        team=Team.ENEMIES,
        weapons=[
            WeaponAttack(
                name="Axe",
                range=1,
                damage=20,
                attack_bonus=20,
                ability_score="str",
                damage_ability_score=None,
            )
        ],
    )
    state = CombatState(
        characters=[attacker, cleric],
        grid_map=GridMap(width=3, height=3),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 20)
    monkeypatch.setattr("combat.conditions.random.randint", lambda _low, _high: 1)

    with caplog.at_level(logging.INFO, logger="combat.conditions"):
        result = AttackAction(actor_id=0, target_id=1).execute(state)

    assert result.success
    assert cleric.active_concentration_spell is None
    assert "loses concentration on Bless" in caplog.text
