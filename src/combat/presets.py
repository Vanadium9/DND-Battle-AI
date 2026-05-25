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
from combat.spellcasting import resolve_spell_list
from rules.progression import build_class_features, build_class_resources


def FighterChampionGreatsword(position: Position | None = None) -> Character:
    """Create a level 5 Champion fighter with Great Weapon Fighting."""

    return Character(
        name="Fighter Champion Greatsword",
        hp=49,
        max_hp=49,
        ac=16,
        position=position or Position(0, 1),
        speed=3,
        stats=Stats(str=18, dex=12, con=16, int=10, wis=11, cha=10),
        team=Team.PLAYERS,
        class_name="Fighter",
        subclass_name="Champion",
        level=5,
        experience=6500,
        proficiency_bonus=3,
        fighting_style="Great Weapon Fighting",
        wearing_armor=True,
        weapons=[
            WeaponAttack(
                name="Greatsword",
                description="Heavy two-handed melee attack.",
                range=1,
                damage="2d6",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
                damage_type="slashing",
                two_handed=True,
                heavy=True,
            )
        ],
        class_features=_fighter_features("Champion", 5),
        resources=_fighter_resources("Champion", 5),
    )


def FighterChampionArcher(position: Position | None = None) -> Character:
    """Create a level 5 Champion fighter with Archery."""

    return Character(
        name="Fighter Champion Archer",
        hp=44,
        max_hp=44,
        ac=15,
        position=position or Position(0, 3),
        speed=3,
        stats=Stats(str=10, dex=18, con=14, int=10, wis=12, cha=10),
        team=Team.PLAYERS,
        class_name="Fighter",
        subclass_name="Champion",
        level=5,
        experience=6500,
        proficiency_bonus=3,
        fighting_style="Archery",
        wearing_armor=True,
        weapons=[
            WeaponAttack(
                name="Longbow",
                description="Long-range bow attack.",
                range=6,
                damage="1d8",
                attack_bonus=0,
                ability_score="dex",
                damage_ability_score="dex",
                damage_type="piercing",
                two_handed=True,
            )
        ],
        class_features=_fighter_features("Champion", 5),
        resources=_fighter_resources("Champion", 5),
    )


def FighterArcher(position: Position | None = None) -> Character:
    """Backward-compatible alias for the level 5 Champion archer preset."""

    archer = FighterChampionArcher(position)
    archer.name = "Fighter Archer"
    return archer


def FighterLevel1Basic(position: Position | None = None) -> Character:
    """Create a level 1 Fighter with Defense and a longsword."""

    return Character(
        name="Fighter Level 1 Basic",
        hp=13,
        max_hp=13,
        ac=16,
        position=position or Position(0, 1),
        speed=3,
        stats=Stats(str=16, dex=12, con=16, int=10, wis=10, cha=10),
        team=Team.PLAYERS,
        class_name="Fighter",
        level=1,
        experience=0,
        proficiency_bonus=2,
        fighting_style="Defense",
        wearing_armor=True,
        weapons=[
            WeaponAttack(
                name="Longsword",
                description="One-handed martial melee attack.",
                range=1,
                damage="1d8",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
                damage_type="slashing",
            )
        ],
        class_features=_fighter_features(None, 1),
        resources=_fighter_resources(None, 1),
    )


def ClericLifeSupport(position: Position | None = None) -> Character:
    """Create a level 5 Life Domain Cleric support caster."""

    return Character(
        name="Cleric Life Support",
        hp=38,
        max_hp=38,
        ac=16,
        position=position or Position(0, 0),
        speed=3,
        stats=Stats(str=12, dex=10, con=14, int=10, wis=18, cha=12),
        team=Team.PLAYERS,
        class_name="Cleric",
        subclass_name="Life Domain",
        level=5,
        experience=6500,
        proficiency_bonus=3,
        wearing_armor=True,
        weapons=[
            WeaponAttack(
                name="Mace",
                description="Simple melee weapon attack.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
                damage_type="bludgeoning",
            )
        ],
        cantrips=resolve_spell_list(("Sacred Flame", "Spare the Dying")),
        prepared_spells=resolve_spell_list(
            ("Cure Wounds", "Healing Word", "Guiding Bolt", "Bless")
        ),
        class_features=_cleric_features("Life Domain", 5),
        resources=_cleric_resources("Life Domain", 5),
    )


def WizardEvoker(position: Position | None = None) -> Character:
    """Create a level 5 School of Evocation Wizard."""

    wizard_spells = (
        "Magic Missile",
        "Shield",
        "Burning Hands",
        "Scorching Ray",
        "Fireball",
    )
    return Character(
        name="Wizard Evoker",
        hp=32,
        max_hp=32,
        ac=12,
        position=position or Position(0, 2),
        speed=3,
        stats=Stats(str=8, dex=14, con=14, int=18, wis=12, cha=10),
        team=Team.PLAYERS,
        class_name="Wizard",
        subclass_name="School of Evocation",
        level=5,
        experience=6500,
        proficiency_bonus=3,
        weapons=[
            WeaponAttack(
                name="Quarterstaff",
                description="Simple melee weapon attack.",
                range=1,
                damage="1d6",
                attack_bonus=0,
                ability_score="str",
                damage_ability_score="str",
                damage_type="bludgeoning",
            )
        ],
        cantrips=resolve_spell_list(("Fire Bolt", "Ray of Frost")),
        known_spells=resolve_spell_list((*wizard_spells, "Fire Bolt", "Ray of Frost")),
        prepared_spells=resolve_spell_list(wizard_spells),
        class_features=_wizard_features("School of Evocation", 5),
        resources=_wizard_resources("School of Evocation", 5),
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


def _fighter_features(subclass_name: str | None = None, level: int = 3) -> list[ClassFeature]:
    return build_class_features("Fighter", level, subclass_name)


def _fighter_resources(
    subclass_name: str | None = None,
    level: int = 3,
) -> dict[str, Resource]:
    return build_class_resources(_fighter_features(subclass_name, level))


def _cleric_features(
    subclass_name: str | None = None,
    level: int = 5,
) -> list[ClassFeature]:
    return build_class_features("Cleric", level, subclass_name)


def _cleric_resources(
    subclass_name: str | None = None,
    level: int = 5,
) -> dict[str, Resource]:
    return build_class_resources(_cleric_features(subclass_name, level))


def _wizard_features(
    subclass_name: str | None = None,
    level: int = 5,
) -> list[ClassFeature]:
    return build_class_features("Wizard", level, subclass_name)


def _wizard_resources(
    subclass_name: str | None = None,
    level: int = 5,
) -> dict[str, Resource]:
    return build_class_resources(_wizard_features(subclass_name, level))
