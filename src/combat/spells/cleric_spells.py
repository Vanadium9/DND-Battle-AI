"""Supported Cleric spell definitions for the MVP ruleset."""

from __future__ import annotations

from combat.damage import DamageType
from combat.spellcasting import SpellDefinition


CLERIC_SPELLS: tuple[SpellDefinition, ...] = (
    SpellDefinition(
        name="Sacred Flame",
        spell_level=0,
        classes=("Cleric",),
        range=6,
        action_cost="action",
        target_type="enemy",
        damage="1d8",
        damage_type=DamageType.RADIANT,
        save_ability="dex",
    ),
    SpellDefinition(
        name="Spare the Dying",
        spell_level=0,
        classes=("Cleric",),
        range=1,
        action_cost="action",
        target_type="ally",
    ),
    SpellDefinition(
        name="Cure Wounds",
        spell_level=1,
        classes=("Cleric",),
        range=1,
        action_cost="action",
        target_type="self_or_ally",
        healing="1d8",
        upcast_healing_per_level="1d8",
    ),
    SpellDefinition(
        name="Healing Word",
        spell_level=1,
        classes=("Cleric",),
        range=6,
        action_cost="bonus_action",
        target_type="self_or_ally",
        healing="1d4",
        upcast_healing_per_level="1d4",
    ),
    SpellDefinition(
        name="Guiding Bolt",
        spell_level=1,
        classes=("Cleric",),
        range=6,
        action_cost="action",
        target_type="enemy",
        damage="4d6",
        damage_type=DamageType.RADIANT,
        upcast_damage_per_level="1d6",
    ),
    SpellDefinition(
        name="Bless",
        spell_level=1,
        classes=("Cleric",),
        range=6,
        action_cost="action",
        target_type="self_or_ally",
        concentration=True,
    ),
    SpellDefinition(
        name="Spiritual Weapon",
        spell_level=2,
        classes=("Cleric",),
        range=6,
        action_cost="bonus_action",
        target_type="enemy",
        implemented=False,
    ),
    SpellDefinition(
        name="Revivify",
        spell_level=3,
        classes=("Cleric",),
        range=1,
        action_cost="action",
        target_type="ally",
        implemented=False,
    ),
)

CLERIC_DEFAULT_CANTRIPS: tuple[str, ...] = ("Sacred Flame", "Spare the Dying")
CLERIC_DEFAULT_PREPARED_SPELLS: tuple[str, ...] = (
    "Cure Wounds",
    "Healing Word",
    "Guiding Bolt",
    "Bless",
)
