"""Cleric class progression for levels 1-5."""

from __future__ import annotations

from combat.class_features import ClassFeature
from rules.classes import ClassDefinition
from rules.spellcasting_progression import FULL_CASTER, FULL_CASTER_SPELL_SLOTS


CLERIC_DEFINITION = ClassDefinition(
    name="Cleric",
    hit_die=8,
    primary_abilities=("wis", "con", "str"),
    saving_throw_proficiencies=("wis", "cha"),
    armor_proficiencies=("light", "medium", "shield"),
    weapon_proficiencies=("simple",),
    skill_choices=("history", "insight", "medicine", "persuasion", "religion"),
    spellcasting_progression=FULL_CASTER_SPELL_SLOTS,
    spellcasting_type=FULL_CASTER,
    spellcasting_ability="wis",
    subclass_level=1,
    level_features={
        1: (
            ClassFeature(
                name="Spellcasting",
                level=1,
                passive_hooks=("on_spellcasting",),
                description="Cleric spellcasting through prepared spells.",
                implemented=True,
            ),
            ClassFeature(
                name="Divine Domain: Life Domain",
                level=1,
                passive_hooks=("on_subclass_selection",),
                description="Life Domain subclass selection.",
                implemented=True,
            ),
        ),
        2: (
            ClassFeature(
                name="Channel Divinity",
                level=2,
                resource_cost="channel_divinity",
                description="Cleric Channel Divinity resource.",
                implemented=True,
            ),
            ClassFeature(
                name="Turn Undead",
                level=2,
                action_cost="action",
                resource_cost="channel_divinity",
                active_action="turn_undead",
                description="Saved as a rule note until undead creature tags exist.",
                implemented=False,
            ),
        ),
        3: (
            ClassFeature(
                name="2nd-level Spells",
                level=3,
                passive_hooks=("on_spell_slots",),
                description="Cleric spell slot progression note.",
                implemented=True,
            ),
        ),
        4: (
            ClassFeature(
                name="Ability Score Improvement",
                level=4,
                passive_hooks=("on_level_up",),
                description="Level 4 cleric progression choice.",
                implemented=True,
            ),
        ),
        5: (
            ClassFeature(
                name="3rd-level Spells",
                level=5,
                passive_hooks=("on_spell_slots",),
                description="Cleric spell slot progression note.",
                implemented=True,
            ),
        ),
    },
)
