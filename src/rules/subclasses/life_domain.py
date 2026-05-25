"""Life Domain cleric subclass progression."""

from __future__ import annotations

from combat.class_features import ClassFeature
from rules.subclasses import SubclassDefinition


LIFE_DOMAIN_DEFINITION = SubclassDefinition(
    name="Life Domain",
    parent_class="Cleric",
    level_features={
        1: (
            ClassFeature(
                name="Disciple of Life",
                level=1,
                passive_hooks=("on_healing_roll",),
                description="Life Domain healing feature note for future hooks.",
                implemented=False,
            ),
        ),
        2: (
            ClassFeature(
                name="Channel Divinity: Preserve Life",
                level=2,
                action_cost="action",
                resource_cost="channel_divinity",
                active_action="preserve_life",
                description="Spend Channel Divinity to heal a wounded ally.",
                implemented=True,
            ),
        ),
    },
)
