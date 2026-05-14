"""Reusable combat presets for tests and demos."""

from __future__ import annotations

from combat.class_features import ClassFeature, Resource
from combat.map import GridMap
from combat.models import (
    Character,
    CombatState,
    Enemy,
    Position,
    Stats,
    Team,
    WeaponAttack,
)


def FighterChampionGreatsword(position: Position | None = None) -> Character:
    """Create a strong melee fighter with a greatsword."""

    return Character(
        name="Fighter Champion Greatsword",
        hp=34,
        max_hp=34,
        ac=16,
        position=position or Position(0, 1),
        speed=3,
        stats=Stats(str=18, dex=12, con=16, int=10, wis=11, cha=10),
        team=Team.PLAYERS,
        class_name="Fighter",
        level=3,
        proficiency_bonus=2,
        weapons=[
            WeaponAttack(
                name="Greatsword",
                description="Heavy two-handed melee attack.",
                range=1,
                damage="2d6",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
            )
        ],
        class_features=_fighter_features(),
        resources=_fighter_resources(),
    )


def FighterArcher(position: Position | None = None) -> Character:
    """Create a dexterous ranged fighter."""

    return Character(
        name="Fighter Archer",
        hp=26,
        max_hp=26,
        ac=15,
        position=position or Position(0, 3),
        speed=3,
        stats=Stats(str=10, dex=18, con=14, int=10, wis=12, cha=10),
        team=Team.PLAYERS,
        class_name="Fighter",
        level=3,
        proficiency_bonus=2,
        weapons=[
            WeaponAttack(
                name="Longbow",
                description="Long-range bow attack.",
                range=6,
                damage="1d8",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
            )
        ],
        class_features=_fighter_features(),
        resources=_fighter_resources(),
    )


def Goblin(position: Position | None = None) -> Enemy:
    """Create a weak enemy with melee and ranged options."""

    return Enemy(
        name="Goblin",
        hp=7,
        max_hp=7,
        ac=13,
        position=position or Position(5, 1),
        speed=3,
        stats=Stats(str=8, dex=14, con=10, int=10, wis=8, cha=8),
        proficiency_bonus=2,
        weapons=[
            WeaponAttack(
                name="Scimitar",
                description="Light melee attack.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
            ),
            WeaponAttack(
                name="Shortbow",
                description="Weak ranged attack.",
                range=5,
                damage="1d6",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
            ),
        ],
    )


def Orc(position: Position | None = None) -> Enemy:
    """Create a tougher enemy with more HP and damage."""

    return Enemy(
        name="Orc",
        hp=18,
        max_hp=18,
        ac=13,
        position=position or Position(5, 3),
        speed=3,
        stats=Stats(str=16, dex=12, con=16, int=8, wis=11, cha=10),
        proficiency_bonus=2,
        weapons=[
            WeaponAttack(
                name="Greataxe",
                description="Heavy melee attack.",
                range=1,
                damage="1d12",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
            )
        ],
    )


def create_test_encounter() -> CombatState:
    """Create a small players-vs-enemies test encounter."""

    return CombatState(
        characters=[
            FighterChampionGreatsword(Position(0, 1)),
            FighterArcher(Position(0, 3)),
            Goblin(Position(5, 1)),
            Orc(Position(5, 3)),
        ],
        grid_map=GridMap(width=6, height=5),
    )


def _fighter_features() -> list[ClassFeature]:
    return [
        ClassFeature(
            name="Action Surge",
            description="Class feature resource for a future extra action implementation.",
            resource_name="action_surge",
            required_level=2,
        ),
        ClassFeature(
            name="Second Wind",
            description="Class feature resource for a future self-heal implementation.",
            resource_name="second_wind",
            required_level=1,
        ),
    ]


def _fighter_resources() -> dict[str, Resource]:
    return {
        "action_surge": Resource(name="action_surge", max_uses=1),
        "second_wind": Resource(name="second_wind", max_uses=1),
    }
