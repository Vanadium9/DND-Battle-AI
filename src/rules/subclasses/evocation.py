"""School of Evocation wizard subclass progression."""

from __future__ import annotations

from combat.class_features import ClassFeature
from rules.subclasses import SubclassDefinition


EVOCATION_DEFINITION = SubclassDefinition(
    name="School of Evocation",
    parent_class="Wizard",
    level_features={
        2: (
            ClassFeature(
                name="Evocation Savant",
                level=2,
                description="School of Evocation feature note for spellbook costs.",
                implemented=False,
            ),
            ClassFeature(
                name="Sculpt Spells",
                level=2,
                passive_hooks=("on_spell_area_targeting",),
                description="Exclude allies from supported Evocation AoE spells.",
                implemented=True,
            ),
        ),
    },
)
