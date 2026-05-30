"""Serializable character schema focused on progression metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from rules.progression import get_level_for_xp, get_proficiency_bonus
from rules.feats import FeatDefinition, get_feat_definition
from rules.races import get_race_definition
from combat.items import (
    ItemDefinition,
    item_damage,
    item_damage_type,
    item_healing,
    item_stabilizes,
    normalize_action_cost,
    normalize_target_type,
)


DEFAULT_CHARACTER_STATS: dict[str, int] = {
    "str": 10,
    "dex": 10,
    "con": 10,
    "int": 10,
    "wis": 10,
    "cha": 10,
}


@dataclass(frozen=True)
class CharacterProgressionSchema:
    """Level, XP and class metadata for importing or exporting a character."""

    level: int = 1
    experience: int = 0
    proficiency_bonus: int = 2
    class_name: str | None = None
    subclass_name: str | None = None

    @classmethod
    def from_xp(
        cls,
        experience: int,
        class_name: str | None = None,
        subclass_name: str | None = None,
    ) -> "CharacterProgressionSchema":
        level = get_level_for_xp(experience)
        return cls(
            level=level,
            experience=max(0, int(experience)),
            proficiency_bonus=get_proficiency_bonus(level),
            class_name=class_name,
            subclass_name=subclass_name,
        )


@dataclass(frozen=True)
class CharacterRaceSchema:
    """Race metadata for importer/exporter code."""

    race_name: str | None = None
    size: str = "Medium"
    speed: int | None = None
    darkvision_range: int | None = None
    skill_proficiencies: tuple[str, ...] = ()
    weapon_proficiencies: tuple[str, ...] = ()
    saving_throw_advantages: tuple[str, ...] = ()
    damage_resistances: tuple[str, ...] = ()
    special_traits: tuple[str, ...] = ()

    @classmethod
    def from_imported_name(cls, race_name: str | None) -> "CharacterRaceSchema":
        definition = get_race_definition(race_name, allow_custom_fallback=True)
        return cls(
            race_name=definition.name,
            size=definition.size,
            speed=definition.speed,
            darkvision_range=definition.darkvision_range,
            skill_proficiencies=tuple(definition.skill_proficiencies),
            weapon_proficiencies=tuple(definition.weapon_proficiencies),
            saving_throw_advantages=tuple(definition.saving_throw_advantages),
            damage_resistances=tuple(definition.damage_resistances),
            special_traits=tuple(definition.special_traits),
        )


@dataclass(frozen=True)
class CharacterFeatSchema:
    """Feat metadata exposed to importer/exporter and builder code."""

    name: str
    prerequisites: dict[str, Any]
    stat_bonuses: dict[str, int]
    passive_effects: tuple[str, ...]
    active_effects: tuple[str, ...]
    combat_hooks: dict[str, tuple[str, ...]]
    implemented: bool = False

    @classmethod
    def from_value(cls, feat: object) -> "CharacterFeatSchema":
        definition = _coerce_feat_definition(feat)
        if definition is not None:
            return cls(
                name=definition.name,
                prerequisites=dict(definition.prerequisites),
                stat_bonuses=dict(definition.stat_bonuses),
                passive_effects=tuple(definition.passive_effects),
                active_effects=tuple(definition.active_effects),
                combat_hooks={
                    hook_name: tuple(effect_names)
                    for hook_name, effect_names in definition.combat_hooks.items()
                },
                implemented=definition.implemented,
            )
        return cls(
            name=str(getattr(feat, "name", feat)),
            prerequisites={},
            stat_bonuses={},
            passive_effects=(),
            active_effects=(),
            combat_hooks={},
            implemented=False,
        )


@dataclass(frozen=True)
class AbilityScoreImprovementSchema:
    """Serialized ASI selection."""

    bonuses: dict[str, int]
    source: str = "Ability Score Improvement"

    @classmethod
    def from_value(cls, asi: object) -> "AbilityScoreImprovementSchema":
        bonuses = getattr(asi, "bonuses", asi)
        if not isinstance(bonuses, Mapping):
            bonuses = {}
        return cls(
            bonuses={str(ability): int(value) for ability, value in bonuses.items()},
            source=str(getattr(asi, "source", "Ability Score Improvement")),
        )


@dataclass(frozen=True)
class CharacterInventoryItemSchema:
    """Serialized inventory item metadata."""

    name: str
    item_type: str
    quantity: int
    action_cost: str
    target_type: str
    range: int
    consumable: bool
    implemented: bool
    healing: str | int | None = None
    damage: str | int | None = None
    damage_type: str | None = None
    stabilize: bool = False

    @classmethod
    def from_value(cls, item: object) -> "CharacterInventoryItemSchema":
        if isinstance(item, ItemDefinition):
            damage_type = item_damage_type(item)
            return cls(
                name=item.name,
                item_type=item.item_type,
                quantity=int(item.quantity),
                action_cost=normalize_action_cost(item.action_cost).name,
                target_type=normalize_target_type(item.target_type).name,
                range=int(item.range),
                consumable=bool(item.consumable),
                implemented=bool(item.implemented),
                healing=item_healing(item),
                damage=item_damage(item),
                damage_type=str(getattr(damage_type, "value", damage_type))
                if damage_type is not None
                else None,
                stabilize=item_stabilizes(item),
            )
        return cls(
            name=str(getattr(item, "name", item)),
            item_type=str(getattr(item, "item_type", "unknown")),
            quantity=int(getattr(item, "quantity", 0)),
            action_cost=str(getattr(item, "action_cost", "")),
            target_type=str(getattr(item, "target_type", "")),
            range=int(getattr(item, "range", 0)),
            consumable=bool(getattr(item, "consumable", False)),
            implemented=bool(getattr(item, "implemented", False)),
        )


@dataclass(frozen=True)
class CharacterSchema:
    """Minimal character schema used by importer-facing code."""

    name: str
    progression: CharacterProgressionSchema = CharacterProgressionSchema()
    race: CharacterRaceSchema = CharacterRaceSchema()
    feats: tuple[CharacterFeatSchema, ...] = ()
    ability_score_improvements: tuple[AbilityScoreImprovementSchema, ...] = ()
    inventory: tuple[CharacterInventoryItemSchema, ...] = ()

    @classmethod
    def from_character(cls, character: object) -> "CharacterSchema":
        level = int(getattr(character, "level", 1))
        race_traits = getattr(character, "race_traits", None)
        return cls(
            name=str(getattr(character, "name", "")),
            progression=CharacterProgressionSchema(
                level=level,
                experience=int(getattr(character, "experience", 0)),
                proficiency_bonus=int(
                    getattr(character, "proficiency_bonus", get_proficiency_bonus(level))
                ),
                class_name=getattr(character, "class_name", None),
                subclass_name=getattr(character, "subclass_name", None),
            ),
            race=CharacterRaceSchema(
                race_name=getattr(character, "race_name", None),
                size=str(getattr(character, "size", "Medium")),
                speed=getattr(race_traits, "speed", None),
                darkvision_range=getattr(race_traits, "darkvision_range", None),
                skill_proficiencies=tuple(
                    getattr(race_traits, "skill_proficiencies", ())
                ),
                weapon_proficiencies=tuple(
                    getattr(race_traits, "weapon_proficiencies", ())
                ),
                saving_throw_advantages=tuple(
                    getattr(race_traits, "saving_throw_advantages", ())
                ),
                damage_resistances=tuple(
                    getattr(race_traits, "damage_resistances", ())
                ),
                special_traits=tuple(getattr(race_traits, "special_traits", ())),
            ),
            feats=tuple(
                CharacterFeatSchema.from_value(feat)
                for feat in getattr(character, "feats", ())
            ),
            ability_score_improvements=tuple(
                AbilityScoreImprovementSchema.from_value(asi)
                for asi in getattr(character, "ability_score_improvements", ())
            ),
            inventory=tuple(
                CharacterInventoryItemSchema.from_value(item)
                for item in getattr(character, "inventory", ())
            ),
        )


def _coerce_feat_definition(feat: object) -> FeatDefinition | None:
    if isinstance(feat, FeatDefinition):
        return feat
    if isinstance(feat, str):
        return get_feat_definition(feat)
    name = getattr(feat, "name", None)
    if isinstance(name, str):
        return get_feat_definition(name)
    return None


@dataclass(frozen=True)
class InternalCharacter:
    """Internal GUI-facing character format persisted by CharacterRepository."""

    id: str = ""
    name: str = ""
    class_name: str = ""
    subclass_name: str | None = None
    level: int = 1
    experience: int = 0
    race_name: str = ""
    role: str = "combatant"
    stats: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CHARACTER_STATS))
    hp: int = 1
    ac: int = 10
    speed: int = 30
    proficiency_bonus: int = 2
    weapons: tuple[dict[str, Any], ...] = ()
    armor: dict[str, Any] = field(default_factory=dict)
    class_features: tuple[str, ...] = ()
    subclass_features: tuple[str, ...] = ()
    race_traits: dict[str, Any] = field(default_factory=dict)
    feats: tuple[str, ...] = ()
    spells: tuple[dict[str, Any] | str, ...] = ()
    prepared_spells: tuple[str, ...] = ()
    spell_slots: dict[str, int] = field(default_factory=dict)
    spell_save_dc: int = 0
    spell_attack_bonus: int = 0
    resources: dict[str, int] = field(default_factory=dict)
    inventory: tuple[dict[str, Any], ...] = ()
    resistances: tuple[str, ...] = ()
    immunities: tuple[str, ...] = ()
    vulnerabilities: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "InternalCharacter":
        """Build an InternalCharacter from JSON-compatible mapping data."""

        stats = dict(DEFAULT_CHARACTER_STATS)
        stats.update(
            {
                str(key): int(value)
                for key, value in _mapping(data.get("stats")).items()
                if str(key) in DEFAULT_CHARACTER_STATS
            }
        )
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            class_name=str(data.get("class_name", "")),
            subclass_name=_optional_string(data.get("subclass_name")),
            level=int(data.get("level", 1)),
            experience=int(data.get("experience", 0)),
            race_name=str(data.get("race_name", "")),
            role=str(data.get("role", "combatant")),
            stats=stats,
            hp=int(data.get("hp", 1)),
            ac=int(data.get("ac", 10)),
            speed=int(data.get("speed", 30)),
            proficiency_bonus=int(data.get("proficiency_bonus", 2)),
            weapons=_tuple_of_mappings(data.get("weapons")),
            armor=dict(_mapping(data.get("armor"))),
            class_features=_tuple_of_strings(data.get("class_features")),
            subclass_features=_tuple_of_strings(data.get("subclass_features")),
            race_traits=dict(_mapping(data.get("race_traits"))),
            feats=_tuple_of_strings(data.get("feats")),
            spells=_tuple_of_spell_values(data.get("spells")),
            prepared_spells=_tuple_of_strings(data.get("prepared_spells")),
            spell_slots={
                str(level): int(count)
                for level, count in _mapping(data.get("spell_slots")).items()
            },
            spell_save_dc=int(data.get("spell_save_dc", 0)),
            spell_attack_bonus=int(data.get("spell_attack_bonus", 0)),
            resources={
                str(name): int(count)
                for name, count in _mapping(data.get("resources")).items()
            },
            inventory=_tuple_of_mappings(data.get("inventory")),
            resistances=_tuple_of_strings(data.get("resistances")),
            immunities=_tuple_of_strings(data.get("immunities")),
            vulnerabilities=_tuple_of_strings(data.get("vulnerabilities")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "id": self.id,
            "name": self.name,
            "class_name": self.class_name,
            "subclass_name": self.subclass_name,
            "level": self.level,
            "experience": self.experience,
            "race_name": self.race_name,
            "role": self.role,
            "stats": dict(self.stats),
            "hp": self.hp,
            "ac": self.ac,
            "speed": self.speed,
            "proficiency_bonus": self.proficiency_bonus,
            "weapons": [dict(weapon) for weapon in self.weapons],
            "armor": dict(self.armor),
            "class_features": list(self.class_features),
            "subclass_features": list(self.subclass_features),
            "race_traits": dict(self.race_traits),
            "feats": list(self.feats),
            "spells": [
                dict(spell) if isinstance(spell, Mapping) else str(spell)
                for spell in self.spells
            ],
            "prepared_spells": list(self.prepared_spells),
            "spell_slots": dict(self.spell_slots),
            "spell_save_dc": self.spell_save_dc,
            "spell_attack_bonus": self.spell_attack_bonus,
            "resources": dict(self.resources),
            "inventory": [dict(item) for item in self.inventory],
            "resistances": list(self.resistances),
            "immunities": list(self.immunities),
            "vulnerabilities": list(self.vulnerabilities),
        }

    def with_id(self, character_id: str) -> "InternalCharacter":
        """Return a copy with a repository-assigned id."""

        data = self.to_dict()
        data["id"] = character_id
        return InternalCharacter.from_mapping(data)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _tuple_of_mappings(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _tuple_of_strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value)


def _tuple_of_spell_values(value: object) -> tuple[dict[str, Any] | str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        dict(item) if isinstance(item, Mapping) else str(item)
        for item in value
    )
