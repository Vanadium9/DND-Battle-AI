"""Reusable creature entities for tests and demos."""

from __future__ import annotations

from combat.abilities import WeaponAttack
from combat.damage import DamageType
from combat.models import Enemy, Position, Stats


def FireElementalSimple(position: Position | None = None) -> Enemy:
    """Create a simplified fire elemental with basic damage defenses."""

    return Enemy(
        name="Fire Elemental Simple",
        hp=45,
        max_hp=45,
        ac=13,
        position=position or Position(3, 0),
        speed=4,
        stats=Stats(str=10, dex=16, con=14, int=6, wis=10, cha=7),
        immunities={DamageType.FIRE},
        resistances={
            DamageType.SLASHING,
            DamageType.PIERCING,
            DamageType.BLUDGEONING,
        },
        weapons=[
            WeaponAttack(
                name="Fire Touch",
                description="Simple fire damage attack.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type=DamageType.FIRE,
            )
        ],
    )
