"""Basic enemy presets built from common combat actions."""

from __future__ import annotations

from combat.abilities import WeaponAttack
from combat.damage import DamageType
from combat.models import Enemy, Position, Stats


def GoblinMelee(position: Position | None = None) -> Enemy:
    """Create a basic melee skirmisher."""

    return Enemy(
        name="Goblin Melee",
        hp=7,
        max_hp=7,
        ac=13,
        position=position or Position(5, 1),
        speed=3,
        stats=Stats(str=8, dex=14, con=10, int=10, wis=8, cha=8),
        proficiency_bonus=2,
        challenge_rating=0.25,
        xp_value=50,
        role="melee_skirmisher",
        weapons=[
            WeaponAttack(
                name="Scimitar",
                description="Light melee attack.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type=DamageType.SLASHING,
            )
        ],
    )


def GoblinArcher(position: Position | None = None) -> Enemy:
    """Create a basic ranged skirmisher."""

    return Enemy(
        name="Goblin Archer",
        hp=7,
        max_hp=7,
        ac=13,
        position=position or Position(5, 2),
        speed=3,
        stats=Stats(str=8, dex=14, con=10, int=10, wis=8, cha=8),
        proficiency_bonus=2,
        challenge_rating=0.25,
        xp_value=50,
        role="ranged_skirmisher",
        weapons=[
            WeaponAttack(
                name="Shortbow",
                description="Short-range bow attack.",
                range=5,
                damage="1d6",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type=DamageType.PIERCING,
            ),
            WeaponAttack(
                name="Dagger",
                description="Backup melee attack.",
                range=1,
                damage="1d4",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type=DamageType.PIERCING,
            ),
        ],
    )


def OrcWarrior(position: Position | None = None) -> Enemy:
    """Create a durable melee brute."""

    return Enemy(
        name="Orc Warrior",
        hp=18,
        max_hp=18,
        ac=13,
        position=position or Position(5, 3),
        speed=3,
        stats=Stats(str=16, dex=12, con=16, int=8, wis=11, cha=10),
        proficiency_bonus=2,
        challenge_rating=0.5,
        xp_value=100,
        role="brute",
        weapons=[
            WeaponAttack(
                name="Greataxe",
                description="Heavy melee attack.",
                range=1,
                damage="1d12",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
                damage_type=DamageType.SLASHING,
                two_handed=True,
                heavy=True,
            )
        ],
    )


def SkeletonArcher(position: Position | None = None) -> Enemy:
    """Create an undead ranged attacker with poison immunity."""

    return Enemy(
        name="Skeleton Archer",
        hp=13,
        max_hp=13,
        ac=13,
        position=position or Position(5, 0),
        speed=3,
        stats=Stats(str=10, dex=14, con=15, int=6, wis=8, cha=5),
        proficiency_bonus=2,
        challenge_rating=0.25,
        xp_value=50,
        role="ranged_undead",
        immunities={DamageType.POISON},
        weapons=[
            WeaponAttack(
                name="Shortbow",
                description="Undead archer ranged attack.",
                range=6,
                damage="1d6",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type=DamageType.PIERCING,
            ),
            WeaponAttack(
                name="Shortsword",
                description="Backup melee attack.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type=DamageType.PIERCING,
            ),
        ],
    )


def Bandit(position: Position | None = None) -> Enemy:
    """Create a simple humanoid opponent."""

    return Enemy(
        name="Bandit",
        hp=11,
        max_hp=11,
        ac=12,
        position=position or Position(4, 2),
        speed=3,
        stats=Stats(str=11, dex=12, con=12, int=10, wis=10, cha=10),
        proficiency_bonus=2,
        challenge_rating=0.125,
        xp_value=25,
        role="melee_humanoid",
        weapons=[
            WeaponAttack(
                name="Scimitar",
                description="Curved melee blade.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type=DamageType.SLASHING,
            ),
            WeaponAttack(
                name="Light Crossbow",
                description="Simple ranged weapon.",
                range=6,
                damage="1d8",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type=DamageType.PIERCING,
            ),
        ],
    )


def Wolf(position: Position | None = None) -> Enemy:
    """Create a fast melee attacker."""

    return Enemy(
        name="Wolf",
        hp=11,
        max_hp=11,
        ac=13,
        position=position or Position(4, 1),
        speed=4,
        stats=Stats(str=12, dex=15, con=12, int=3, wis=12, cha=6),
        proficiency_bonus=2,
        challenge_rating=0.25,
        xp_value=50,
        role="fast_melee",
        weapons=[
            WeaponAttack(
                name="Bite",
                description="Fast melee bite attack.",
                range=1,
                damage="2d4",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
                damage_type=DamageType.PIERCING,
            )
        ],
    )


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
        proficiency_bonus=3,
        challenge_rating=5,
        xp_value=1800,
        role="elemental_striker",
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


BASIC_MONSTER_PRESETS = (
    GoblinMelee,
    GoblinArcher,
    OrcWarrior,
    SkeletonArcher,
    Bandit,
    Wolf,
    FireElementalSimple,
)


__all__ = [
    "BASIC_MONSTER_PRESETS",
    "Bandit",
    "FireElementalSimple",
    "GoblinArcher",
    "GoblinMelee",
    "OrcWarrior",
    "SkeletonArcher",
    "Wolf",
]
