import logging

from agents import MainActionType, build_action_masks
from combat import (
    AOE_DIRECTIONS,
    AoEDirection,
    AoEShape,
    AoETargeting,
    CastSpellAction,
    Character,
    CombatItem,
    CombatState,
    GridMap,
    Position,
    Stats,
    Team,
    UseObjectAction,
    WizardEvoker,
    build_character,
    calculate_combat_reward,
    positions_for_aoe,
    snapshot_combat_state,
)


def move_index(position: Position, width: int = 6) -> int:
    return position.y * width + position.x


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
        hp=30,
        max_hp=30,
        ac=12,
        position=position,
        speed=3,
        stats=Stats(dex=8),
        team=Team.PLAYERS,
    )


def spell_by_name(character: Character, spell_name: str):
    return next(
        spell
        for spell in [*character.cantrips, *character.prepared_spells]
        if spell.name == spell_name
    )


def test_radius_cone_and_line_positions() -> None:
    radius = positions_for_aoe(
        AoETargeting(
            shape=AoEShape.RADIUS,
            origin=Position(0, 0),
            size=1,
            target_cell=Position(2, 2),
        )
    )
    cone = positions_for_aoe(
        AoETargeting(
            shape=AoEShape.CONE,
            origin=Position(2, 2),
            size=2,
            direction=AoEDirection.RIGHT,
        )
    )
    line = positions_for_aoe(
        AoETargeting(
            shape=AoEShape.LINE,
            origin=Position(2, 2),
            size=3,
            direction=AoEDirection.UP,
        )
    )

    assert radius == {
        Position(2, 2),
        Position(1, 2),
        Position(3, 2),
        Position(2, 1),
        Position(2, 3),
    }
    assert {Position(3, 2), Position(4, 2), Position(4, 1), Position(4, 3)}.issubset(cone)
    assert line == {Position(2, 1), Position(2, 0), Position(2, -1)}


def test_fireball_target_cell_applies_friendly_fire(monkeypatch, caplog) -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        level=5,
        prepared_spells=("Fireball",),
    )
    ally = make_ally(Position(2, 0))
    enemy = make_enemy(Position(3, 0))
    state = CombatState(
        characters=[wizard, ally, enemy],
        grid_map=GridMap(width=6, height=4),
    )
    fireball = spell_by_name(wizard, "Fireball")
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    with caplog.at_level(logging.INFO, logger="combat.aoe"):
        result = CastSpellAction(
            actor_id=0,
            spell=fireball,
            target_cell=Position(3, 0),
        ).execute(state)

    assert result.success
    assert enemy.hp < enemy.max_hp
    assert ally.hp < ally.max_hp
    assert "affected targets" in caplog.text
    assert "Enemy" in caplog.text
    assert "Ally" in caplog.text


def test_sculpt_spells_excludes_ally_from_fireball_target_cell(monkeypatch) -> None:
    wizard = WizardEvoker(Position(0, 0))
    ally = make_ally(Position(2, 0))
    enemy = make_enemy(Position(3, 0))
    state = CombatState(
        characters=[wizard, ally, enemy],
        grid_map=GridMap(width=6, height=4),
    )
    fireball = spell_by_name(wizard, "Fireball")
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    result = CastSpellAction(
        actor_id=0,
        spell=fireball,
        target_cell=Position(3, 0),
    ).execute(state)

    assert result.success
    assert ally.hp == ally.max_hp
    assert enemy.hp < enemy.max_hp


def test_burning_hands_uses_directional_cone(monkeypatch) -> None:
    wizard = WizardEvoker(Position(0, 0))
    enemy = make_enemy(Position(1, 0))
    outside_enemy = make_enemy(Position(0, 2))
    state = CombatState(
        characters=[wizard, enemy, outside_enemy],
        grid_map=GridMap(width=6, height=4),
    )
    burning_hands = spell_by_name(wizard, "Burning Hands")
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    result = CastSpellAction(
        actor_id=0,
        spell=burning_hands,
        direction=AoEDirection.RIGHT,
    ).execute(state)

    assert result.success
    assert enemy.hp < enemy.max_hp
    assert outside_enemy.hp == outside_enemy.max_hp


def test_aoe_item_damages_all_affected_creatures(monkeypatch) -> None:
    actor = Character(
        name="Alchemist",
        hp=20,
        max_hp=20,
        ac=12,
        position=Position(0, 0),
        speed=3,
        stats=Stats(dex=14),
        team=Team.PLAYERS,
    )
    bomb = CombatItem(
        name="Bomb",
        range=4,
        damage=6,
        damage_type="fire",
        area_shape=AoEShape.RADIUS,
        area_size=1,
    )
    actor.items = [bomb]
    ally = make_ally(Position(2, 0))
    enemy = make_enemy(Position(3, 0))
    state = CombatState(
        characters=[actor, ally, enemy],
        grid_map=GridMap(width=6, height=4),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    result = UseObjectAction(
        actor_id=0,
        item=bomb,
        target_cell=Position(3, 0),
    ).execute(state)

    assert result.success
    assert enemy.hp == enemy.max_hp - 6
    assert ally.hp == ally.max_hp - 6


def test_action_masks_expose_target_cells_and_directions() -> None:
    wizard = WizardEvoker(Position(0, 0))
    enemy = make_enemy(Position(1, 0))
    state = CombatState(
        characters=[wizard, enemy],
        grid_map=GridMap(width=6, height=4),
    )

    masks = build_action_masks(state, actor_id=0)

    assert masks["main_action_type"][MainActionType.CAST_SPELL]
    assert masks["target_cell_index"][move_index(Position(1, 0))]
    assert masks["direction_index"][AOE_DIRECTIONS.index(AoEDirection.RIGHT)]


def test_friendly_fire_reward_penalty(monkeypatch) -> None:
    wizard = build_character(
        name="Wizard",
        class_name="Wizard",
        level=5,
        prepared_spells=("Fireball",),
    )
    ally = make_ally(Position(2, 0))
    enemy = make_enemy(Position(3, 0))
    state = CombatState(
        characters=[wizard, ally, enemy],
        grid_map=GridMap(width=6, height=4),
    )
    fireball = spell_by_name(wizard, "Fireball")
    action = CastSpellAction(actor_id=0, spell=fireball, target_cell=Position(3, 0))
    before = snapshot_combat_state(state)
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 1)

    result = action.execute(state)
    reward = calculate_combat_reward(
        before,
        snapshot_combat_state(state),
        actor_team=Team.PLAYERS,
        action=action,
        action_result=result,
    )

    assert result.success
    assert reward.breakdown["friendly_fire"] < 0
