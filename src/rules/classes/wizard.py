"""Wizard class progression for levels 1-5."""

from __future__ import annotations

from combat.class_features import ClassFeature
from rules.classes import ClassDefinition
from rules.spellcasting_progression import FULL_CASTER, FULL_CASTER_SPELL_SLOTS


WIZARD_DEFINITION = ClassDefinition(
    name="Wizard",
    hit_die=6,
    primary_abilities=("int", "con", "dex"),
    saving_throw_proficiencies=("int", "wis"),
    armor_proficiencies=(),
    weapon_proficiencies=("dagger", "dart", "sling", "quarterstaff", "light_crossbow"),
    skill_choices=("arcana", "history", "insight", "investigation", "medicine", "religion"),
    spellcasting_progression=FULL_CASTER_SPELL_SLOTS,
    spellcasting_type=FULL_CASTER,
    spellcasting_ability="int",
    subclass_level=2,
    level_features={
        1: (
            ClassFeature(
                name="Spellcasting",
                level=1,
                passive_hooks=("on_spellcasting",),
                description="Wizard spellbook and prepared spellcasting.",
                implemented=True,
            ),
            ClassFeature(
                name="Arcane Recovery",
                level=1,
                resource_cost="arcane_recovery",
                passive_hooks=("on_short_rest", "on_combat_reset"),
                description="Restore part of spent spell slots between encounters.",
                implemented=True,
            ),
        ),
        2: (
            ClassFeature(
                name="Arcane Tradition: School of Evocation",
                level=2,
                passive_hooks=("on_subclass_selection",),
                description="School of Evocation subclass selection.",
                implemented=True,
            ),
        ),
        3: (
            ClassFeature(
                name="2nd-level Spells",
                level=3,
                passive_hooks=("on_spell_slots",),
                description="Wizard spell slot progression note.",
                implemented=True,
            ),
        ),
        4: (
            ClassFeature(
                name="Ability Score Improvement",
                level=4,
                passive_hooks=("on_level_up",),
                description="Level 4 wizard progression choice.",
                implemented=True,
            ),
        ),
        5: (
            ClassFeature(
                name="3rd-level Spells",
                level=5,
                passive_hooks=("on_spell_slots",),
                description="Wizard spell slot progression note.",
                implemented=True,
            ),
        ),
    },
)
