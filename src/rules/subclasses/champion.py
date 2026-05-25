"""Champion fighter subclass progression."""

from __future__ import annotations

from combat.class_features import ClassFeature
from rules.subclasses import SubclassDefinition


CHAMPION_DEFINITION = SubclassDefinition(
    name="Champion",
    parent_class="Fighter",
    level_features={
        3: (
            ClassFeature(
                name="Improved Critical",
                level=3,
                passive_hooks=("on_attack_roll",),
                description="Weapon attacks score a critical hit on a natural 19 or 20.",
                implemented=True,
            ),
        ),
    },
)
