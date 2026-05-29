from agents import (
    AggressiveMeleeAgent,
    CoverAwareRangedAgent,
    RandomLegalAgent,
    RangedKitingAgent,
    SimpleCasterAgent,
    SimpleHealerAgent,
    build_action_masks,
    decode_action,
)
from combat import (
    CastSpellAction,
    ClericLifeSupport,
    CombatState,
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    GridMap,
    MoveAction,
    Position,
    TerrainType,
    WizardEvoker,
)


def _decode_agent_action(agent, state: CombatState, actor_id: int = 0):
    actor = state.character_at(actor_id)
    masks = build_action_masks(state, actor_id)
    output = agent.act(
        None,
        masks,
        state=state,
        actor_id=actor_id,
        actor=actor,
    )
    return decode_action(
        int(output["action_category"].item()),
        int(output["main_action_type"].item()),
        int(output["target_index"].item()),
        int(output["move_index"].item()),
        int(output["option_index"].item()),
        state,
        actor_id,
    )


def test_each_baseline_agent_selects_legal_action() -> None:
    cases = [
        (
            AggressiveMeleeAgent(),
            CombatState(
                characters=[
                    FighterChampionGreatsword(Position(0, 0)),
                    Goblin(Position(1, 0)),
                ],
                grid_map=GridMap(width=8, height=8),
            ),
        ),
        (
            RangedKitingAgent(),
            CombatState(
                characters=[FighterArcher(Position(0, 0)), Goblin(Position(3, 0))],
                grid_map=GridMap(width=8, height=8),
            ),
        ),
        (
            SimpleHealerAgent(),
            _healer_state(),
        ),
        (
            SimpleCasterAgent(),
            CombatState(
                characters=[WizardEvoker(Position(0, 0)), Goblin(Position(3, 0))],
                grid_map=GridMap(width=8, height=8),
            ),
        ),
        (
            CoverAwareRangedAgent(),
            _cover_state(),
        ),
        (
            RandomLegalAgent(seed=7),
            CombatState(
                characters=[FighterArcher(Position(0, 0)), Goblin(Position(2, 0))],
                grid_map=GridMap(width=8, height=8),
            ),
        ),
    ]

    for agent, state in cases:
        action = _decode_agent_action(agent, state)
        assert action.is_valid(state)


def test_simple_healer_uses_healing_spell_for_low_hp_ally() -> None:
    state = _healer_state()

    action = _decode_agent_action(SimpleHealerAgent(), state)

    assert isinstance(action, CastSpellAction)
    assert action.target_id == 1
    assert action.spell is not None
    assert action.spell.healing is not None


def test_ranged_kiting_agent_does_not_move_into_melee_when_escape_available() -> None:
    state = CombatState(
        characters=[FighterArcher(Position(0, 0)), Goblin(Position(1, 0))],
        grid_map=GridMap(width=8, height=8),
    )

    action = _decode_agent_action(RangedKitingAgent(), state)

    assert isinstance(action, MoveAction)
    assert state.grid_map.manhattan_distance(action.destination, Position(1, 0)) > 1


def _healer_state() -> CombatState:
    cleric = ClericLifeSupport(Position(0, 0))
    ally = FighterChampionGreatsword(Position(1, 0))
    ally.hp = 8
    return CombatState(
        characters=[
            cleric,
            ally,
            Goblin(Position(4, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
    )


def _cover_state() -> CombatState:
    return CombatState(
        characters=[
            FighterArcher(Position(0, 0)),
            Goblin(Position(5, 0)),
        ],
        grid_map=GridMap(
            width=8,
            height=8,
            terrain_grid=[
                [TerrainType.NORMAL, TerrainType.LOW_COVER, *([TerrainType.NORMAL] * 6)],
                *([[TerrainType.NORMAL] * 8] * 7),
            ],
        ),
    )
