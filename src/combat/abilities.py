"""Combat abilities and weapon definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from combat.damage import DamageType

if TYPE_CHECKING:
    from combat.models import Character, Stats


@dataclass
class Ability:
    """A generic non-class combat ability."""

    name: str
    description: str = ""
    range: int = 0
    cooldown: int = 0
    remaining_cooldown: int = 0

    @property
    def available(self) -> bool:
        return self.remaining_cooldown <= 0


@dataclass
class WeaponAttack(Ability):
    """A weapon attack owned by a creature, not by a class."""

    range: int = 1
    damage: int | str = "1d6"
    attack_bonus: int = 0
    ability_score: str = "str"
    damage_ability_score: str | None = None
    damage_bonus: int = 0
    damage_type: DamageType | str | None = None
    proficient: bool = True
    two_handed: bool = False
    heavy: bool = False

    def attack_modifier(self, attacker: Character) -> int:
        proficiency = attacker.proficiency_bonus if self.proficient else 0
        return (
            ability_modifier(attacker.stats, self.ability_score)
            + proficiency
            + self.attack_bonus
        )

    def damage_modifier(self, attacker: Character) -> int:
        if self.damage_ability_score is None:
            ability_bonus = 0
        else:
            ability_bonus = ability_modifier(attacker.stats, self.damage_ability_score)
        return ability_bonus + self.damage_bonus


@dataclass
class SpellAbility(Ability):
    """A simple spell ability."""

    spell_level: int = 0
    casting_level: int | None = None
    action_cost: str = "action"
    target_type: str = "enemy"
    damage: int | str | None = None
    healing: int | str | None = None
    damage_type: DamageType | str | None = None
    save_dc: int | None = None
    save_ability: str | None = None
    save_half_damage: bool = False
    concentration: bool = False
    upcast_damage_per_level: str | int | None = None
    upcast_healing_per_level: str | int | None = None
    school: str | None = None
    area_shape: object | None = None
    area_size: int = 0
    ac_bonus: int = 0
    duration: str | None = None


def ability_modifier(stats: Stats, ability_score: str) -> int:
    """Return a D&D-style modifier for one ability score."""

    normalized_name = ability_score.lower()
    if normalized_name not in {"str", "dex", "con", "int", "wis", "cha"}:
        raise ValueError(f"Unknown ability score: {ability_score}")
    value = getattr(stats, normalized_name)
    return (value - 10) // 2
