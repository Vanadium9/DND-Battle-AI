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
from combat.race_traits import (
    RaceTraits,
    apply_race_traits,
    has_damage_resistance,
    weapon_is_racially_proficient,
)

__all__ = [
    "Character",
    "CombatState",
    "Condition",
    "Enemy",
    "Position",
    "RaceTraits",
    "Stats",
    "Team",
    "XP_THRESHOLDS",
    "apply_level_up",
    "apply_race_traits",
    "can_level_up",
    "get_level_for_xp",
    "get_proficiency_bonus",
    "has_damage_resistance",
    "sync_character_progression",
    "weapon_is_racially_proficient",
]
