"""Shared entity feature helpers for observation encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from combat.cover import CoverType
from combat.damage import DamageType, coerce_damage_type
from combat.inventory import available_inventory_items, get_supported_item_definitions
from combat.items import item_damage_type
from combat.models import Character, CombatState, Position, Team
from combat.spellcasting import SUPPORTED_SPELLS
from combat.terrain import TerrainType
from rules.feats import ABILITY_SCORE_IMPROVEMENT_NAME, GRAPPLER_NAME


CLASS_ID_NAMES: tuple[str, ...] = ("Fighter", "Cleric", "Wizard")
SUBCLASS_ID_NAMES: tuple[str, ...] = (
    "Champion",
    "Life Domain",
    "School of Evocation",
)
RACE_ID_NAMES: tuple[str, ...] = ("Human", "Dwarf", "Elf", "Halfling", "CustomRace")
FEAT_FLAG_NAMES: tuple[str, ...] = (
    ABILITY_SCORE_IMPROVEMENT_NAME,
    GRAPPLER_NAME,
)
PREPARED_SPELL_FLAG_NAMES: tuple[str, ...] = tuple(
    definition.name for definition in SUPPORTED_SPELLS if definition.implemented
)
INVENTORY_ITEM_FLAG_NAMES: tuple[str, ...] = tuple(
    definition.name for definition in get_supported_item_definitions()
)
COMBAT_ROLE_NAMES: tuple[str, ...] = (
    "MELEE_DAMAGE",
    "RANGED_DAMAGE",
    "TANK",
    "SUPPORT",
    "CASTER",
    "BRUTE_ENEMY",
    "SKIRMISHER_ENEMY",
)
MONSTER_ROLE_ID_NAMES = COMBAT_ROLE_NAMES
ROLE_NAME_ALIASES: dict[str, str] = {
    "meleeskirmisher": "SKIRMISHER_ENEMY",
    "rangedskirmisher": "SKIRMISHER_ENEMY",
    "rangedundead": "SKIRMISHER_ENEMY",
    "fastranged": "SKIRMISHER_ENEMY",
    "fastmelee": "SKIRMISHER_ENEMY",
    "meleehumanoid": "SKIRMISHER_ENEMY",
    "skirmisherenemy": "SKIRMISHER_ENEMY",
    "brute": "BRUTE_ENEMY",
    "elementalstriker": "BRUTE_ENEMY",
    "bruteenemies": "BRUTE_ENEMY",
    "bruteenemy": "BRUTE_ENEMY",
    "meleedamage": "MELEE_DAMAGE",
    "rangeddamage": "RANGED_DAMAGE",
    "tank": "TANK",
    "support": "SUPPORT",
    "caster": "CASTER",
}
CONDITION_FLAG_NAMES: tuple[str, ...] = (
    "prone",
    "grappled",
    "hidden",
    "dodging",
    "disengaged",
    "incapacitated",
    "unconscious",
    "stable",
)
TERRAIN_FEATURE_TYPES: tuple[TerrainType, ...] = tuple(TerrainType)
DAMAGE_PROFILE_TYPES: tuple[DamageType, ...] = tuple(DamageType)
LOCAL_MAP_RADIUS = 2
LOCAL_MAP_WIDTH = LOCAL_MAP_RADIUS * 2 + 1
LOCAL_MAP_CELL_COUNT = LOCAL_MAP_WIDTH * LOCAL_MAP_WIDTH
MAP_CELL_FEATURE_SIZE = len(TERRAIN_FEATURE_TYPES) + 5
MAP_FEATURE_SIZE = LOCAL_MAP_CELL_COUNT * MAP_CELL_FEATURE_SIZE


@dataclass(frozen=True)
class EntityObservation:
    """Structured observation tensors for entity-based policies."""

    actor_features: torch.Tensor
    entities_features: torch.Tensor
    map_features: torch.Tensor
    global_features: torch.Tensor
    entity_mask: torch.Tensor


def lookup_id(value: object, names: tuple[str, ...]) -> float:
    """Return a stable one-based id, or 0 for unknown content."""

    if value is None:
        return 0.0
    value_key = lookup_key(value)
    for index, name in enumerate(names, start=1):
        if lookup_key(name) == value_key:
            return float(index)
    return 0.0


def class_id(character: Character) -> float:
    return lookup_id(getattr(character, "class_name", None), CLASS_ID_NAMES)


def subclass_id(character: Character) -> float:
    return lookup_id(getattr(character, "subclass_name", None), SUBCLASS_ID_NAMES)


def race_id(character: Character) -> float:
    return lookup_id(getattr(character, "race_name", None), RACE_ID_NAMES)


def role_id(character: Character) -> float:
    return lookup_id(combat_role_name(character), COMBAT_ROLE_NAMES)


def combat_role_name(character: Character) -> str:
    """Infer a coarse combat role from explicit role, team and capabilities."""

    explicit_role = ROLE_NAME_ALIASES.get(lookup_key(getattr(character, "role", "")))
    if explicit_role is not None:
        return explicit_role

    if character.team is Team.ENEMIES:
        if _is_enemy_brute(character):
            return "BRUTE_ENEMY"
        return "SKIRMISHER_ENEMY"

    class_key = lookup_key(getattr(character, "class_name", ""))
    if class_key == "cleric" or _has_healing_spell(character):
        return "SUPPORT"
    if class_key == "wizard" or _has_damaging_spell(character):
        return "CASTER"
    if _has_ranged_weapon(character) and not _has_melee_weapon(character):
        return "RANGED_DAMAGE"
    if int(getattr(character, "ac", 0)) >= 18:
        return "TANK"
    return "MELEE_DAMAGE"


def normalized_level(character: Character) -> float:
    return clamp01(float(getattr(character, "level", 1)) / 5.0)


def normalized_proficiency_bonus(character: Character) -> float:
    return clamp01(float(getattr(character, "proficiency_bonus", 0)) / 6.0)


def normalized_challenge_rating(character: Character) -> float:
    return clamp01(_cr_to_float(getattr(character, "challenge_rating", 0)) / 5.0)


def normalized_xp_value(character: Character) -> float:
    return clamp01(float(getattr(character, "xp_value", 0)) / 1800.0)


def feat_flags(character: Character) -> list[float]:
    feat_names = {lookup_key(getattr(feat, "name", feat)) for feat in getattr(character, "feats", ())}
    return [float(lookup_key(name) in feat_names) for name in FEAT_FLAG_NAMES]


def prepared_spell_flags(character: Character) -> list[float]:
    spell_names = {
        lookup_key(getattr(spell, "name", spell))
        for spell in (
            *tuple(getattr(character, "cantrips", ()) or ()),
            *tuple(getattr(character, "prepared_spells", ()) or ()),
        )
    }
    return [float(lookup_key(name) in spell_names) for name in PREPARED_SPELL_FLAG_NAMES]


def inventory_usable_item_flags(character: Character) -> list[float]:
    item_names = {
        lookup_key(getattr(item, "name", item))
        for item in available_inventory_items(character)
    }
    return [float(lookup_key(name) in item_names) for name in INVENTORY_ITEM_FLAG_NAMES]


def class_resource_flags(character: Character) -> list[float]:
    return [
        resource_available(character, "second_wind"),
        resource_available(character, "action_surge"),
        resource_available(character, "channel_divinity"),
        resource_available(character, "arcane_recovery"),
    ]


def resource_available(character: Character, resource_name: str) -> float:
    resource = getattr(character, "resources", {}).get(resource_name)
    return float(resource is not None and resource.available)


def spell_slot_features(character: Character) -> list[float]:
    remaining = getattr(character, "spell_slots_remaining", {}) or {}
    maximum = getattr(character, "spell_slots", {}) or {}
    features: list[float] = []
    for level in (1, 2, 3):
        features.append(float(remaining.get(level, 0)))
        features.append(float(maximum.get(level, 0)))
    return features


def condition_flags(character: Character) -> list[float]:
    named_conditions = {
        lookup_key(getattr(condition, "name", condition))
        for condition in getattr(character, "conditions", ())
    }
    values = {
        "prone": bool(getattr(character, "prone", False)),
        "grappled": bool(getattr(character, "grappled", False)),
        "hidden": bool(getattr(character, "hidden", False)),
        "dodging": bool(getattr(character, "dodging_until_start_of_next_turn", False)),
        "disengaged": bool(getattr(character, "disengaged_until_end_of_turn", False)),
        "incapacitated": bool(getattr(character, "is_incapacitated", False)),
        "unconscious": "unconscious" in named_conditions,
        "stable": bool(getattr(character, "stable", False)),
    }
    return [float(values[name]) for name in CONDITION_FLAG_NAMES]


def active_concentration_flag(character: Character) -> float:
    return float(getattr(character, "active_concentration_spell", None) is not None)


def terrain_around_features(state: CombatState, actor: Character) -> list[float]:
    positions = (
        Position(actor.position.x, actor.position.y - 1),
        Position(actor.position.x + 1, actor.position.y),
        Position(actor.position.x, actor.position.y + 1),
        Position(actor.position.x - 1, actor.position.y),
    )
    features: list[float] = []
    for position in positions:
        terrain_type = terrain_at(state, position)
        features.extend(float(terrain_type is candidate) for candidate in TERRAIN_FEATURE_TYPES)
    return features


def terrain_at(state: CombatState, position: Position) -> TerrainType:
    grid_map = state.grid_map
    if grid_map is None or not grid_map.in_bounds(position):
        return TerrainType.BLOCKED
    return grid_map.terrain_at(position)


def visible_enemies_count(state: CombatState, actor: Character) -> float:
    return float(
        sum(
            1
            for character in state.characters
            if character is not actor
            and character.team != actor.team
            and character.is_alive
            and not getattr(character, "hidden", False)
            and has_line_of_sight(state, actor.position, character.position)
            and cover_between(state, actor.position, character.position) is not CoverType.FULL_COVER
        )
    )


def current_cover_status(state: CombatState, actor: Character) -> float:
    enemies = [
        character
        for character in state.characters
        if character is not actor and character.team != actor.team and character.is_alive
    ]
    if not enemies:
        return cover_value(CoverType.NO_COVER)
    return max(
        cover_value(cover_between(state, enemy.position, actor.position))
        for enemy in enemies
    )


def has_line_of_sight(state: CombatState, origin: Position, target: Position) -> bool:
    grid_map = state.grid_map
    if grid_map is None:
        return True
    for method_name in ("has_line_of_sight", "line_of_sight"):
        method = getattr(grid_map, method_name, None)
        if callable(method):
            return bool(method(origin, target))
    return True


def cover_between(state: CombatState, origin: Position, target: Position) -> CoverType:
    grid_map = state.grid_map
    if grid_map is None:
        return CoverType.NO_COVER
    return grid_map.get_cover_between(origin, target)


def cover_value(cover: CoverType) -> float:
    return float(
        {
            CoverType.NO_COVER: 0,
            CoverType.HALF_COVER: 1,
            CoverType.THREE_QUARTERS_COVER: 2,
            CoverType.FULL_COVER: 3,
        }[cover]
    )


def reachable_by_actor(state: CombatState, actor: Character, target: Character) -> float:
    grid_map = state.grid_map
    movement_remaining = int(getattr(actor.action_economy, "movement_remaining", 0))
    if distance(actor.position, target.position, state) <= 1:
        return 1.0
    if grid_map is None:
        return float(distance(actor.position, target.position, state) <= movement_remaining + 1)
    reachable = grid_map.movement_costs_from(
        actor.position,
        movement_remaining,
        state.characters,
    )
    adjacent_cells = (
        Position(target.position.x + 1, target.position.y),
        Position(target.position.x - 1, target.position.y),
        Position(target.position.x, target.position.y + 1),
        Position(target.position.x, target.position.y - 1),
    )
    return float(any(position in reachable for position in adjacent_cells))


def distance(first: Position, second: Position, state: CombatState) -> int:
    if state.grid_map is not None:
        return state.grid_map.manhattan_distance(first, second)
    return abs(first.x - second.x) + abs(first.y - second.y)


def threat_estimate(character: Character, actor: Character, state: CombatState) -> float:
    hp_ratio = character.hp / character.max_hp if character.max_hp > 0 else 0.0
    damage = _best_damage_estimate(character)
    proximity = 1.0 / max(1.0, float(distance(actor.position, character.position, state)))
    raw = hp_ratio * 0.25 + min(damage / 30.0, 1.0) * 0.55 + proximity * 0.20
    return clamp01(raw)


def available_damage_type_flags(character: Character) -> list[float]:
    damage_types: set[DamageType] = set()
    for weapon in getattr(character, "available_weapons", ()):
        damage_type = coerce_damage_type(getattr(weapon, "damage_type", None))
        if damage_type is not None:
            damage_types.add(damage_type)
    for spell in (
        *tuple(getattr(character, "cantrips", ()) or ()),
        *tuple(getattr(character, "prepared_spells", ()) or ()),
    ):
        damage_type = coerce_damage_type(getattr(spell, "damage_type", None))
        if damage_type is not None:
            damage_types.add(damage_type)
    for item in available_inventory_items(character):
        damage_type = coerce_damage_type(item_damage_type(item))
        if damage_type is not None:
            damage_types.add(damage_type)
    return damage_type_flags(damage_types)


def damage_type_flags(damage_types: Iterable[object]) -> list[float]:
    normalized = {
        damage_type
        for value in damage_types
        for damage_type in (coerce_damage_type(value),)
        if damage_type is not None
    }
    return [float(damage_type in normalized) for damage_type in DAMAGE_PROFILE_TYPES]


def global_feature_values(state: CombatState, actor_id: int) -> list[float]:
    actor = state.character_at(actor_id)
    if actor is None:
        return [0.0] * 7
    allies_alive = sum(
        1 for character in state.characters if character.team == actor.team and character.is_alive
    )
    enemies_alive = sum(
        1 for character in state.characters if character.team != actor.team and character.is_alive
    )
    grid_map = state.grid_map
    width = grid_map.width if grid_map is not None else 0
    height = grid_map.height if grid_map is not None else 0
    return [
        float(getattr(state, "round_number", 1)),
        initiative_position(state, actor_id),
        float(allies_alive),
        float(enemies_alive),
        encounter_difficulty_estimate(state, actor),
        clamp01(float(width) / 20.0),
        clamp01(float(height) / 20.0),
    ]


def initiative_position(state: CombatState, actor_id: int) -> float:
    order = tuple(getattr(state, "initiative_order", ()) or ())
    if not order:
        return 0.0
    if actor_id not in order:
        return 0.0
    if len(order) == 1:
        return 0.0
    return float(order.index(actor_id)) / float(len(order) - 1)


def encounter_difficulty_estimate(state: CombatState, actor: Character) -> float:
    enemy_xp = sum(
        max(0, int(getattr(character, "xp_value", 0)))
        for character in state.characters
        if character.team != actor.team
    )
    party_level_budget = sum(
        max(1, int(getattr(character, "level", 1))) * 200
        for character in state.characters
        if character.team == actor.team
    )
    return clamp01(enemy_xp / max(1.0, float(party_level_budget)))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lookup_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _cr_to_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _best_damage_estimate(character: Character) -> float:
    weapon_damage = [
        _numeric_damage_estimate(getattr(weapon, "damage", 0))
        for weapon in getattr(character, "available_weapons", ())
    ]
    spell_damage = [
        _numeric_damage_estimate(getattr(spell, "damage", 0))
        for spell in (
            *tuple(getattr(character, "cantrips", ()) or ()),
            *tuple(getattr(character, "prepared_spells", ()) or ()),
        )
    ]
    item_damage = [
        _numeric_damage_estimate(
            getattr(item.effect, "damage", 0)
            if item.effect is not None
            else getattr(item, "damage", 0)
        )
        for item in available_inventory_items(character)
    ]
    return max((*weapon_damage, *spell_damage, *item_damage, 0.0))


def _is_enemy_brute(character: Character) -> bool:
    return (
        int(getattr(character, "max_hp", 0)) >= 18
        or int(getattr(getattr(character, "stats", None), "str", 10)) >= 16
        or _cr_to_float(getattr(character, "challenge_rating", 0)) >= 1.0
    )


def _has_melee_weapon(character: Character) -> bool:
    return any(int(getattr(weapon, "range", 1)) <= 1 for weapon in getattr(character, "weapons", ()))


def _has_ranged_weapon(character: Character) -> bool:
    return any(int(getattr(weapon, "range", 1)) > 1 for weapon in getattr(character, "weapons", ()))


def _has_healing_spell(character: Character) -> bool:
    return any(
        getattr(spell, "healing", None) is not None
        for spell in (
            *tuple(getattr(character, "cantrips", ()) or ()),
            *tuple(getattr(character, "prepared_spells", ()) or ()),
            *tuple(getattr(character, "known_spells", ()) or ()),
        )
    )


def _has_damaging_spell(character: Character) -> bool:
    return any(
        getattr(spell, "damage", None) is not None
        for spell in (
            *tuple(getattr(character, "cantrips", ()) or ()),
            *tuple(getattr(character, "prepared_spells", ()) or ()),
            *tuple(getattr(character, "known_spells", ()) or ()),
        )
    )


def _numeric_damage_estimate(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if "d" not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    dice, _, modifier = text.partition("+")
    count_text, _, sides_text = dice.partition("d")
    try:
        count = int(count_text or "1")
        sides = int(sides_text)
        average = count * (sides + 1) / 2.0
        if modifier:
            average += int(modifier)
        return average
    except ValueError:
        return 0.0
