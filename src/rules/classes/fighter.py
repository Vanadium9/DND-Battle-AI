"""Fighter class progression for levels 1-5."""

from __future__ import annotations

from combat.class_features import ClassFeature
from rules.classes import ClassDefinition


FIGHTING_STYLE_ARCHERY = "Archery"
FIGHTING_STYLE_DEFENSE = "Defense"
FIGHTING_STYLE_GREAT_WEAPON_FIGHTING = "Great Weapon Fighting"
FIGHTING_STYLE_OPTIONS: tuple[str, ...] = (
    FIGHTING_STYLE_ARCHERY,
    FIGHTING_STYLE_DEFENSE,
    FIGHTING_STYLE_GREAT_WEAPON_FIGHTING,
)


FIGHTER_DEFINITION = ClassDefinition(
    name="Fighter",
    hit_die=10,
    primary_abilities=("str", "dex", "con"),
    saving_throw_proficiencies=("str", "con"),
    armor_proficiencies=("light", "medium", "heavy", "shield"),
    weapon_proficiencies=("simple", "martial"),
    skill_choices=(
        "acrobatics",
        "animal_handling",
        "athletics",
        "history",
        "insight",
        "intimidation",
        "perception",
        "survival",
    ),
    subclass_level=3,
    level_features={
        1: (
            ClassFeature(
                name="Fighting Style",
                level=1,
                passive_hooks=("on_attack_roll", "on_damage_roll", "on_ac_calculation"),
                description="Fighter combat style selected during character creation.",
                implemented=True,
            ),
            ClassFeature(
                name="Second Wind",
                level=1,
                action_cost="bonus_action",
                resource_cost="second_wind",
                active_action="second_wind",
                description="Recover 1d10 + fighter level hit points once per combat.",
                implemented=True,
            ),
        ),
        2: (
            ClassFeature(
                name="Action Surge",
                level=2,
                resource_cost="action_surge",
                active_action="action_surge",
                description="Restore the main action once per combat.",
                implemented=True,
            ),
        ),
        3: (
            ClassFeature(
                name="Martial Archetype: Champion",
                level=3,
                passive_hooks=("on_subclass_selection",),
                description="Fighter subclass selection for Champion.",
                implemented=True,
            ),
        ),
        4: (
            ClassFeature(
                name="Ability Score Improvement",
                level=4,
                passive_hooks=("on_level_up",),
                description="Level 4 fighter progression choice.",
                implemented=True,
            ),
        ),
        5: (
            ClassFeature(
                name="Extra Attack",
                level=5,
                passive_hooks=("on_attack_action",),
                description="Make two weapon attacks inside one Attack action.",
                implemented=True,
            ),
        ),
    },
)
