import torch

from agents import (
    OBSERVATION_SIZE,
    ActionCategory,
    MainActionType,
    PPOActorCritic,
    build_action_masks,
    encode_observation,
)
from combat import (
    AttackAction,
    Character,
    CombatEnvironment,
    CombatState,
    EndTurnAction,
    FighterArcher,
    Goblin,
    GridMap,
    MoveAction,
    Position,
    Stats,
    Team,
    WeaponAttack,
)


def make_character(
    name: str,
    position: Position,
    team: Team,
    hp: int = 10,
    weapon: WeaponAttack | None = None,
) -> Character:
    return Character(
        name=name,
        hp=hp,
        max_hp=10,
        ac=12,
        position=position,
        speed=2,
        stats=Stats(),
        team=team,
        abilities=[] if weapon is None else [weapon],
    )


def make_duel(damage: int = 4) -> tuple[Character, Character, CombatState, WeaponAttack]:
    weapon = WeaponAttack(name="Sword", range=1, damage=damage, attack_bonus=20)
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS, weapon=weapon)
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    state = CombatState(
        characters=[hero, enemy],
        grid_map=GridMap(width=4, height=4),
    )
    return hero, enemy, state, weapon


def test_move_inside_map_rejects_out_of_bounds_and_spends_movement() -> None:
    hero = make_character("Hero", Position(0, 0), Team.PLAYERS)
    state = CombatState(characters=[hero], grid_map=GridMap(width=3, height=3))

    assert MoveAction(actor_id=0, destination=Position(-1, 0)).is_valid(state) is False

    result = MoveAction(actor_id=0, destination=Position(1, 1)).execute(state)

    assert result.success
    assert hero.position == Position(1, 1)
    assert hero.action_economy.movement_remaining == 0


def test_attack_in_range_spends_action_and_dead_targets_are_invalid() -> None:
    hero, enemy, state, weapon = make_duel()
    dead_enemy = make_character("Dead", Position(1, 0), Team.ENEMIES, hp=0)
    dead_state = CombatState(
        characters=[hero, dead_enemy],
        grid_map=GridMap(width=4, height=4),
    )

    assert AttackAction(actor_id=0, target_id=1, weapon=weapon).is_valid(dead_state) is False

    action = AttackAction(actor_id=0, target_id=1, weapon=weapon)
    result = action.execute(state)

    assert result.success
    assert enemy.hp == 6
    assert hero.action_economy.action_available is False
    assert action.is_valid(state) is False


def test_environment_ends_combat_when_last_enemy_dies() -> None:
    hero, _, _, weapon = make_duel(damage=10)
    enemy = make_character("Enemy", Position(1, 0), Team.ENEMIES)
    environment = CombatEnvironment(
        characters=[hero, enemy],
        grid_map=GridMap(width=4, height=4),
        use_initiative=False,
        log_to_console=False,
    )

    result = environment.step(AttackAction(actor_id=0, target_id=1, weapon=weapon))

    assert result.success
    assert environment.is_done()
    assert environment.get_winner() is Team.PLAYERS


def test_action_masks_reflect_valid_actions_and_resources() -> None:
    state = CombatState(
        characters=[
            FighterArcher(Position(0, 0)),
            Goblin(Position(1, 0)),
        ],
        grid_map=GridMap(width=8, height=8),
    )

    masks = build_action_masks(state, actor_id=0)

    assert masks["action_category"][ActionCategory.MOVEMENT]
    assert masks["main_action_type"][MainActionType.ATTACK]
    assert masks["action_category"][ActionCategory.END_TURN]
    assert not masks["action_category"][ActionCategory.BONUS_ACTION]
    assert not masks["action_category"][ActionCategory.REACTION]
    assert masks["target_index"][1]

    state.characters[0].action_economy.action_available = False
    state.characters[0].action_economy.movement_remaining = 0
    masks = build_action_masks(state, actor_id=0)

    assert not masks["main_action_type"][MainActionType.ATTACK]
    assert not masks["action_category"][ActionCategory.MOVEMENT]


def test_encode_observation_has_fixed_size() -> None:
    state = CombatState(
        characters=[FighterArcher(Position(0, 0)), Goblin(Position(1, 0))],
        grid_map=GridMap(width=8, height=8),
    )

    observation = encode_observation(state, actor_id=0)

    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == torch.float32


def test_ppo_model_returns_unmasked_action() -> None:
    state = CombatState(
        characters=[FighterArcher(Position(0, 0)), Goblin(Position(1, 0))],
        grid_map=GridMap(width=8, height=8),
    )
    model = PPOActorCritic(target_count=6, move_count=64, hidden_sizes=(32,))
    observation = encode_observation(state, actor_id=0)
    masks = build_action_masks(state, actor_id=0)

    action = model.act(observation, masks, deterministic=True)
    action_category = int(action["action_category"].item())
    main_action_type = int(action["main_action_type"].item())

    assert masks["action_category"][action_category]
    if (
        action_category == int(ActionCategory.MAIN_ACTION)
        and main_action_type == int(MainActionType.ATTACK)
    ):
        assert masks["main_action_type"][main_action_type]
        assert masks["target_index"][int(action["target_index"].item())]
        assert masks["option_index"][int(action["option_index"].item())]
    if action_category == int(ActionCategory.MOVEMENT):
        assert masks["move_index"][int(action["move_index"].item())]


def test_new_turn_resets_action_and_bonus_but_bonus_is_unused() -> None:
    hero, _, state, _ = make_duel()
    state.turn_index = 1
    hero.action_economy.action_available = False
    hero.action_economy.bonus_action_available = False
    hero.action_economy.movement_remaining = 0

    result = EndTurnAction(actor_id=1).execute(state)
    masks = build_action_masks(state, actor_id=0)

    assert result.success
    assert state.active_character is hero
    assert hero.action_economy.action_available is True
    assert hero.action_economy.bonus_action_available is True
    assert hero.action_economy.movement_remaining == hero.speed
    assert not masks["action_category"][ActionCategory.BONUS_ACTION]
