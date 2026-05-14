from agents import (
    ACTION_CATEGORY_COUNT,
    CHARACTER_FEATURE_SIZE,
    DEFAULT_MOVE_COUNT,
    DEFAULT_OPTION_COUNT,
    DEFAULT_TARGET_COUNT,
    MAIN_ACTION_TYPE_COUNT,
    MAX_NEARBY_CHARACTERS,
    OBSERVATION_SIZE,
    ActionCategory,
    BaseAgent,
    MainActionType,
    PPOActorCritic,
    RandomAgent,
    build_action_masks,
    decode_action,
    encode_observation,
)
from combat import (
    ActionEconomy,
    ActionResult,
    Ability,
    AttackAction,
    CastSpellAction,
    Character,
    CombatAction,
    CombatEnvironment,
    CombatState,
    ClassFeature,
    Condition,
    DashAction,
    DisengageAction,
    DodgeAction,
    EndTurnAction,
    Enemy,
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    GrappleAction,
    GridMap,
    HelpAction,
    HideAction,
    ImprovisedAction,
    EncounterGenerator,
    MoveAction,
    Orc,
    Position,
    ReadyAction,
    RewardConfig,
    Resource,
    SearchAction,
    ShoveAction,
    create_test_encounter,
    calculate_combat_reward,
    reset_turn_resources,
    SpellAbility,
    StabilizeAction,
    CombatReward,
    CombatRewardSnapshot,
    opposing_team,
    snapshot_combat_state,
    Team,
    UseObjectAction,
    WeaponAttack,
)
from configs import CombatConfig, PPOConfig, TrainingConfig
from training import EpisodeStats, PPOTrainer, RolloutBuffer, Trainer


def test_project_imports() -> None:
    assert BaseAgent is not None
    assert RandomAgent is not None
    assert ACTION_CATEGORY_COUNT is not None
    assert MAIN_ACTION_TYPE_COUNT is not None
    assert ActionCategory is not None
    assert MainActionType is not None
    assert build_action_masks is not None
    assert CHARACTER_FEATURE_SIZE is not None
    assert DEFAULT_MOVE_COUNT is not None
    assert DEFAULT_OPTION_COUNT is not None
    assert DEFAULT_TARGET_COUNT is not None
    assert decode_action is not None
    assert MAX_NEARBY_CHARACTERS is not None
    assert OBSERVATION_SIZE is not None
    assert PPOActorCritic is not None
    assert encode_observation is not None
    assert ActionEconomy is not None
    assert ActionResult is not None
    assert AttackAction is not None
    assert CastSpellAction is not None
    assert CombatAction is not None
    assert CombatEnvironment is not None
    assert CombatReward is not None
    assert CombatRewardSnapshot is not None
    assert CombatState is not None
    assert EndTurnAction is not None
    assert Character is not None
    assert ClassFeature is not None
    assert DashAction is not None
    assert DisengageAction is not None
    assert DodgeAction is not None
    assert Enemy is not None
    assert FighterArcher is not None
    assert FighterChampionGreatsword is not None
    assert Goblin is not None
    assert GrappleAction is not None
    assert HelpAction is not None
    assert HideAction is not None
    assert ImprovisedAction is not None
    assert EncounterGenerator is not None
    assert Ability is not None
    assert MoveAction is not None
    assert Orc is not None
    assert ReadyAction is not None
    assert WeaponAttack is not None
    assert RewardConfig is not None
    assert Resource is not None
    assert SearchAction is not None
    assert ShoveAction is not None
    assert SpellAbility is not None
    assert StabilizeAction is not None
    assert Condition is not None
    assert GridMap is not None
    assert Position is not None
    assert calculate_combat_reward is not None
    assert create_test_encounter is not None
    assert opposing_team is not None
    assert reset_turn_resources is not None
    assert snapshot_combat_state is not None
    assert Team is not None
    assert UseObjectAction is not None
    assert CombatConfig is not None
    assert PPOConfig is not None
    assert TrainingConfig is not None
    assert EpisodeStats is not None
    assert PPOTrainer is not None
    assert RolloutBuffer is not None
    assert Trainer is not None
