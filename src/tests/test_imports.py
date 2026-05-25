from agents import (
    ACTION_CATEGORY_COUNT,
    ACTOR_CLASS_FEATURE_SIZE,
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
    ActionSurgeAction,
    Ability,
    AbilityScoreImprovement,
    AttackAction,
    CastSpellAction,
    Character,
    CombatAction,
    CombatEnvironment,
    CombatState,
    ClassFeature,
    CharacterBuildRequest,
    Condition,
    DashAction,
    DisengageAction,
    DodgeAction,
    EndTurnAction,
    Enemy,
    FighterArcher,
    FighterChampionArcher,
    FighterChampionGreatsword,
    FighterLevel1Basic,
    Goblin,
    GrappleAction,
    GridMap,
    HelpAction,
    HideAction,
    ImprovisedAction,
    InitiativeCheckResult,
    InitiativeResult,
    InitiativeRoll,
    EncounterGenerator,
    FeatureDefinition,
    MoveAction,
    Orc,
    Position,
    ReadyAction,
    RewardConfig,
    Resource,
    SearchAction,
    SecondWindAction,
    ShoveAction,
    create_test_encounter,
    calculate_combat_reward,
    RaceTraits,
    add_feat,
    archery_attack_bonus,
    available_implemented_class_features,
    apply_defense_fighting_style,
    apply_race_traits,
    XP_THRESHOLDS,
    apply_ability_score_improvement,
    apply_combat_hook,
    apply_level_four_choice,
    apply_level_up,
    can_choose_asi_or_feat,
    can_level_up,
    can_use_feature_action,
    character_has_class_feature,
    character_has_feat,
    build_character,
    feature_resource_name,
    fighting_style,
    get_active_combat_hooks,
    get_active_feat_definitions,
    get_level_for_xp,
    get_proficiency_bonus,
    get_supported_feats_for_builder,
    has_damage_resistance,
    implemented_class_features,
    implemented_feature_active_actions,
    is_feature_implemented,
    on_attack_roll,
    on_damage_roll,
    roll_initiative_check,
    roll_initiative_order,
    reset_turn_resources,
    SpellAbility,
    StabilizeAction,
    CombatReward,
    CombatRewardSnapshot,
    opposing_team,
    snapshot_combat_state,
    sync_character_progression,
    should_use_great_weapon_fighting,
    spend_feature_resource,
    Team,
    use_halfling_lucky,
    UseObjectAction,
    WeaponAttack,
    weapon_is_racially_proficient,
    weapon_attack_count_for_attack_action,
    supported_class_options,
    supported_subclass_options,
    validate_class_selection,
)
from character import (
    AbilityScoreImprovementSchema,
    CharacterFeatSchema,
    CharacterProgressionSchema,
    CharacterRaceSchema,
    CharacterSchema,
)
from configs import CombatConfig, PPOConfig, TrainingConfig
from rules import (
    DEFAULT_RULESET_NAME,
    ABILITY_SCORE_IMPROVEMENT_NAME,
    COMBAT_HOOK_NAMES,
    CLASS_DEFINITIONS,
    FEAT_DEFINITIONS,
    FIGHTER_DEFINITION,
    FIGHTING_STYLE_OPTIONS,
    GRAPPLER_NAME,
    CHAMPION_DEFINITION,
    ClassDefinition,
    FeatDefinition,
    MAX_SUPPORTED_LEVEL,
    MIN_SUPPORTED_LEVEL,
    RaceDefinition,
    Ruleset,
    RulesetRegistry,
    SUBCLASS_DEFINITIONS,
    SubclassDefinition,
    class_uses_spellcasting,
    get_race_definition,
    get_active_ruleset,
    get_class_definition,
    get_feat_definition,
    get_subclass_definition,
    get_supported_class_definitions,
    get_supported_feat_definitions,
    get_supported_subclass_definitions,
    get_unsupported_reason,
    is_feat_supported,
    is_supported_content,
    is_supported_race,
    spell_slots_for_class_level,
)
from training import EpisodeStats, PPOTrainer, RolloutBuffer, Trainer


def test_project_imports() -> None:
    assert BaseAgent is not None
    assert RandomAgent is not None
    assert AbilityScoreImprovement is not None
    assert ACTION_CATEGORY_COUNT is not None
    assert ACTOR_CLASS_FEATURE_SIZE is not None
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
    assert ActionSurgeAction is not None
    assert AttackAction is not None
    assert CastSpellAction is not None
    assert CombatAction is not None
    assert CombatEnvironment is not None
    assert CombatReward is not None
    assert CombatRewardSnapshot is not None
    assert CombatState is not None
    assert EndTurnAction is not None
    assert Character is not None
    assert CharacterBuildRequest is not None
    assert ClassFeature is not None
    assert DashAction is not None
    assert DisengageAction is not None
    assert DodgeAction is not None
    assert Enemy is not None
    assert FighterArcher is not None
    assert FighterChampionArcher is not None
    assert FighterChampionGreatsword is not None
    assert FighterLevel1Basic is not None
    assert Goblin is not None
    assert GrappleAction is not None
    assert HelpAction is not None
    assert HideAction is not None
    assert ImprovisedAction is not None
    assert InitiativeCheckResult is not None
    assert InitiativeResult is not None
    assert InitiativeRoll is not None
    assert EncounterGenerator is not None
    assert FeatureDefinition is not None
    assert Ability is not None
    assert MoveAction is not None
    assert Orc is not None
    assert ReadyAction is not None
    assert WeaponAttack is not None
    assert RewardConfig is not None
    assert Resource is not None
    assert SearchAction is not None
    assert SecondWindAction is not None
    assert ShoveAction is not None
    assert SpellAbility is not None
    assert StabilizeAction is not None
    assert Condition is not None
    assert GridMap is not None
    assert Position is not None
    assert calculate_combat_reward is not None
    assert create_test_encounter is not None
    assert RaceTraits is not None
    assert add_feat is not None
    assert archery_attack_bonus is not None
    assert available_implemented_class_features is not None
    assert apply_defense_fighting_style is not None
    assert apply_race_traits is not None
    assert XP_THRESHOLDS is not None
    assert apply_ability_score_improvement is not None
    assert apply_combat_hook is not None
    assert apply_level_four_choice is not None
    assert apply_level_up is not None
    assert can_choose_asi_or_feat is not None
    assert can_level_up is not None
    assert can_use_feature_action is not None
    assert character_has_class_feature is not None
    assert character_has_feat is not None
    assert build_character is not None
    assert feature_resource_name is not None
    assert fighting_style is not None
    assert get_active_combat_hooks is not None
    assert get_active_feat_definitions is not None
    assert get_level_for_xp is not None
    assert get_proficiency_bonus is not None
    assert get_supported_feats_for_builder is not None
    assert has_damage_resistance is not None
    assert implemented_class_features is not None
    assert implemented_feature_active_actions is not None
    assert is_feature_implemented is not None
    assert on_attack_roll is not None
    assert on_damage_roll is not None
    assert roll_initiative_check is not None
    assert roll_initiative_order is not None
    assert opposing_team is not None
    assert reset_turn_resources is not None
    assert snapshot_combat_state is not None
    assert sync_character_progression is not None
    assert should_use_great_weapon_fighting is not None
    assert spend_feature_resource is not None
    assert Team is not None
    assert use_halfling_lucky is not None
    assert UseObjectAction is not None
    assert weapon_is_racially_proficient is not None
    assert weapon_attack_count_for_attack_action is not None
    assert supported_class_options is not None
    assert supported_subclass_options is not None
    assert validate_class_selection is not None
    assert CombatConfig is not None
    assert PPOConfig is not None
    assert TrainingConfig is not None
    assert AbilityScoreImprovementSchema is not None
    assert CharacterFeatSchema is not None
    assert CharacterProgressionSchema is not None
    assert CharacterRaceSchema is not None
    assert CharacterSchema is not None
    assert DEFAULT_RULESET_NAME is not None
    assert ABILITY_SCORE_IMPROVEMENT_NAME is not None
    assert COMBAT_HOOK_NAMES is not None
    assert CLASS_DEFINITIONS is not None
    assert FEAT_DEFINITIONS is not None
    assert FIGHTER_DEFINITION is not None
    assert FIGHTING_STYLE_OPTIONS is not None
    assert GRAPPLER_NAME is not None
    assert CHAMPION_DEFINITION is not None
    assert ClassDefinition is not None
    assert FeatDefinition is not None
    assert MAX_SUPPORTED_LEVEL is not None
    assert MIN_SUPPORTED_LEVEL is not None
    assert RaceDefinition is not None
    assert Ruleset is not None
    assert RulesetRegistry is not None
    assert SUBCLASS_DEFINITIONS is not None
    assert SubclassDefinition is not None
    assert class_uses_spellcasting is not None
    assert get_race_definition is not None
    assert get_active_ruleset is not None
    assert get_class_definition is not None
    assert get_feat_definition is not None
    assert get_subclass_definition is not None
    assert get_supported_class_definitions is not None
    assert get_supported_feat_definitions is not None
    assert get_supported_subclass_definitions is not None
    assert get_unsupported_reason is not None
    assert is_feat_supported is not None
    assert is_supported_content is not None
    assert is_supported_race is not None
    assert spell_slots_for_class_level is not None
    assert EpisodeStats is not None
    assert PPOTrainer is not None
    assert RolloutBuffer is not None
    assert Trainer is not None
