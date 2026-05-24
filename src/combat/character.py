"""Character model compatibility exports."""

from combat.models import (
    Character,
    CombatState,
    Condition,
    Enemy,
    Position,
    Stats,
    Team,
)
from rules.progression import (
    XP_THRESHOLDS,
    apply_level_up,
    can_level_up,
    get_level_for_xp,
    get_proficiency_bonus,
    sync_character_progression,
)
from combat.features import (
    AbilityScoreImprovement,
    add_feat,
    apply_ability_score_improvement,
    apply_level_four_choice,
    can_choose_asi_or_feat,
    character_has_feat,
    get_supported_feats_for_builder,
)
from combat.character_builder import (
    CharacterBuildRequest,
    build_character,
    supported_class_options,
    supported_subclass_options,
    validate_class_selection,
)
from combat.race_traits import (
    RaceTraits,
    apply_race_traits,
    has_damage_resistance,
    weapon_is_racially_proficient,
)

__all__ = [
    "Character",
    "CharacterBuildRequest",
    "CombatState",
    "Condition",
    "Enemy",
    "Position",
    "RaceTraits",
    "Stats",
    "Team",
    "XP_THRESHOLDS",
    "AbilityScoreImprovement",
    "add_feat",
    "apply_ability_score_improvement",
    "apply_level_four_choice",
    "apply_level_up",
    "apply_race_traits",
    "build_character",
    "can_choose_asi_or_feat",
    "can_level_up",
    "character_has_feat",
    "get_level_for_xp",
    "get_proficiency_bonus",
    "get_supported_feats_for_builder",
    "has_damage_resistance",
    "supported_class_options",
    "supported_subclass_options",
    "sync_character_progression",
    "validate_class_selection",
    "weapon_is_racially_proficient",
]
