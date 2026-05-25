from agents import MainActionType, build_action_masks
from combat import (
    CastSpellAction,
    Character,
    CombatState,
    DashAction,
    DisengageAction,
    DodgeAction,
    GrappleAction,
    GridMap,
    HideAction,
    PotionOfHealing,
    Position,
    ReadyAction,
    SearchAction,
    ShoveAction,
    SpellAbility,
    StabilizeAction,
    Stats,
    Team,
    TerrainType,
)


def make_character(
    name: str,
    position: Position,
    team: Team,
    hp: int = 10,
    stats: Stats | None = None,
) -> Character:
    return Character(
        name=name,
        hp=hp,
        max_hp=10,
        ac=12,
        position=position,
        speed=3,
        stats=stats or Stats(),
        team=team,
    )


def test_dash_disengage_dodge_hide_search_and_ready_spend_action(monkeypatch) -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    state = CombatState(characters=[hero], grid_map=GridMap(width=4, height=4))

    result = DashAction(actor_id=0).execute(state)
    assert result.success
    assert hero.action_economy.movement_remaining == 6
    assert hero.action_economy.action_available is False

    hero.action_economy.action_available = True
    result = DisengageAction(actor_id=0).execute(state)
    assert result.success
    assert hero.disengaged_until_end_of_turn is True

    hero.action_economy.action_available = True
    result = DodgeAction(actor_id=0).execute(state)
    assert result.success
    assert hero.dodging_until_start_of_next_turn is True

    hero.action_economy.action_available = True
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)
    result = HideAction(actor_id=0).execute(state)
    assert result.success
    assert hero.hidden is True
    assert "Stealth:" in result.description
    assert "d20=10" in result.description

    hero.action_economy.action_available = True
    result = SearchAction(actor_id=0).execute(state)
    assert result.success
    assert "Searches" in result.description
    assert "Perception:" in result.description

    hero.action_economy.action_available = True
    result = ReadyAction(
        actor_id=0,
        prepared_action="attack",
        trigger_description="enemy approaches",
    ).execute(state)
    assert result.success
    assert hero.prepared_action == "attack"
    assert hero.trigger_description == "enemy approaches"


def test_cast_spell_invokes_spell_ability_and_spends_action() -> None:
    spell = SpellAbility(name="Fire Bolt", range=3, damage="3")
    caster = make_character("Caster", Position(0, 0), Team.PLAYERS)
    caster.abilities.append(spell)
    target = make_character("Target", Position(2, 0), Team.ENEMIES)
    state = CombatState(characters=[caster, target], grid_map=GridMap(width=4, height=4))

    result = CastSpellAction(actor_id=0, spell=spell, target_id=1).execute(state)

    assert result.success
    assert target.hp == 7
    assert caster.action_economy.action_available is False


def test_cast_spell_is_masked_until_spell_system_exists() -> None:
    spell = SpellAbility(name="Fire Bolt", range=3, damage="3")
    caster = make_character("Caster", Position(0, 0), Team.PLAYERS)
    caster.abilities.append(spell)
    target = make_character("Target", Position(2, 0), Team.ENEMIES)
    state = CombatState(characters=[caster, target], grid_map=GridMap(width=4, height=4))

    masks = build_action_masks(state, actor_id=0)

    assert not masks["main_action_type"][MainActionType.CAST_SPELL]


def test_grapple_shove_and_stabilize_use_action_economy(monkeypatch) -> None:
    hero = make_character(
        "Hero",
        Position(0, 0),
        Team.PLAYERS,
        stats=Stats(str=18, dex=10, wis=10),
    )
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES, stats=Stats(str=8, dex=8))
    ally = make_character("Ally", Position(0, 1), Team.PLAYERS, hp=0)
    state = CombatState(
        characters=[hero, enemy, ally],
        grid_map=GridMap(width=4, height=4),
    )

    rolls = iter([20, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))
    result = GrappleAction(actor_id=0, target_id=1).execute(state)
    assert result.success
    assert enemy.grappled_by == 0
    assert hero.action_economy.action_available is False
    assert "Athletics:" in result.description
    assert "best of" in result.description

    hero.action_economy.action_available = True
    rolls = iter([20, 1, 1])
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: next(rolls))
    result = ShoveAction(actor_id=0, target_id=1).execute(state)
    assert result.success
    assert enemy.prone is True
    assert hero.action_economy.action_available is False
    assert "Athletics:" in result.description

    hero.action_economy.action_available = True
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 8)
    result = StabilizeAction(actor_id=0, target_id=2).execute(state)
    assert result.success
    assert ally.stable is True
    assert hero.action_economy.action_available is False
    assert "Medicine:" in result.description


def test_hide_can_use_passive_perception_dc(monkeypatch) -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, stats=Stats(dex=10))
    enemy = make_character("Observer", Position(1, 0), Team.ENEMIES, stats=Stats(wis=18))
    state = CombatState(
        characters=[hero, enemy],
        grid_map=GridMap(
            width=4,
            height=4,
            terrain_grid=[
                [TerrainType.LOW_COVER, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
            ],
        ),
    )
    monkeypatch.setattr("combat.common_actions.random.randint", lambda _low, _high: 10)

    result = HideAction(actor_id=0, dc=None, observer_id=1).execute(state)

    assert result.success
    assert hero.hidden is False
    assert "Observer passive Perception 16" in result.description


def test_action_masks_include_common_action_resources() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    hero.hp = 5
    hero.inventory = [PotionOfHealing()]
    ally = make_character("Ally", Position(0, 1), Team.PLAYERS, hp=0)
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    state = CombatState(
        characters=[hero, ally, enemy],
        grid_map=GridMap(
            width=4,
            height=4,
            terrain_grid=[
                [TerrainType.LOW_COVER, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
                [TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL, TerrainType.NORMAL],
            ],
        ),
    )

    masks = build_action_masks(state, actor_id=0)

    assert masks["main_action_type"][MainActionType.DASH]
    assert masks["main_action_type"][MainActionType.DISENGAGE]
    assert masks["main_action_type"][MainActionType.DODGE]
    assert masks["main_action_type"][MainActionType.HELP]
    assert masks["main_action_type"][MainActionType.HIDE]
    assert masks["main_action_type"][MainActionType.SEARCH]
    assert masks["main_action_type"][MainActionType.USE_OBJECT]
    assert masks["main_action_type"][MainActionType.READY]
    assert masks["main_action_type"][MainActionType.GRAPPLE]
    assert masks["main_action_type"][MainActionType.SHOVE]
    assert masks["main_action_type"][MainActionType.STABILIZE]
    assert masks["main_action_type"][MainActionType.IMPROVISED]
    assert not masks["main_action_type"][MainActionType.CAST_SPELL]

    hero.action_economy.action_available = False
    masks = build_action_masks(state, actor_id=0)
    assert not masks["main_action_type"][MainActionType.DASH]
    assert not masks["main_action_type"][MainActionType.GRAPPLE]
