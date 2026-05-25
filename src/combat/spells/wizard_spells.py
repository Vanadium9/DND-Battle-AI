"""Supported Wizard spell definitions for the MVP ruleset."""

from __future__ import annotations

from combat.aoe import AoEShape
from combat.damage import DamageType
from combat.spellcasting import SpellDefinition


WIZARD_SPELLS: tuple[SpellDefinition, ...] = (
    SpellDefinition(
        name="Fire Bolt",
        spell_level=0,
        classes=("Wizard",),
        range=6,
        action_cost="action",
        target_type="enemy",
        damage="1d10",
        damage_type=DamageType.FIRE,
        school="evocation",
    ),
    SpellDefinition(
        name="Ray of Frost",
        spell_level=0,
        classes=("Wizard",),
        range=6,
        action_cost="action",
        target_type="enemy",
        damage="1d8",
        damage_type=DamageType.COLD,
        school="evocation",
    ),
    SpellDefinition(
        name="Magic Missile",
        spell_level=1,
        classes=("Wizard",),
        range=6,
        action_cost="action",
        target_type="enemy",
        damage="1d4+1",
        damage_type=DamageType.FORCE,
        upcast_damage_per_level="1d4+1",
        school="evocation",
    ),
    SpellDefinition(
        name="Shield",
        spell_level=1,
        classes=("Wizard",),
        range=0,
        action_cost="reaction",
        target_type="self",
        ac_bonus=5,
        duration="until_start_of_next_turn",
        school="abjuration",
    ),
    SpellDefinition(
        name="Burning Hands",
        spell_level=1,
        classes=("Wizard",),
        range=2,
        action_cost="action",
        target_type="enemy",
        damage="3d6",
        damage_type=DamageType.FIRE,
        save_ability="dex",
        save_half_damage=True,
        upcast_damage_per_level="1d6",
        area_shape=AoEShape.CONE,
        area_size=2,
        school="evocation",
    ),
    SpellDefinition(
        name="Scorching Ray",
        spell_level=2,
        classes=("Wizard",),
        range=6,
        action_cost="action",
        target_type="enemy",
        damage="2d6",
        damage_type=DamageType.FIRE,
        upcast_damage_per_level="2d6",
        school="evocation",
    ),
    SpellDefinition(
        name="Fireball",
        spell_level=3,
        classes=("Wizard",),
        range=6,
        action_cost="action",
        target_type="enemy",
        damage="8d6",
        damage_type=DamageType.FIRE,
        save_ability="dex",
        save_half_damage=True,
        upcast_damage_per_level="1d6",
        area_shape=AoEShape.RADIUS,
        area_size=2,
        school="evocation",
    ),
)

WIZARD_DEFAULT_CANTRIPS: tuple[str, ...] = ("Fire Bolt", "Ray of Frost")
WIZARD_DEFAULT_PREPARED_SPELLS: tuple[str, ...] = (
    "Magic Missile",
    "Shield",
    "Burning Hands",
    "Scorching Ray",
    "Fireball",
)
