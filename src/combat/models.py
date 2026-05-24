"""Basic combat data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from combat.action_economy import ActionEconomy, reset_turn_resources
from combat.abilities import Ability, SpellAbility, WeaponAttack
from combat.class_features import ClassFeature, Resource, reset_resources

if TYPE_CHECKING:
    from combat.map import GridMap
    from combat.race_traits import RaceTraits


class Team(Enum):
    """Combat sides."""

    PLAYERS = "players"
    ENEMIES = "enemies"


@dataclass(frozen=True)
class Position:
    """A position on a tactical grid."""

    x: int = 0
    y: int = 0


@dataclass
class Stats:
    """Core character stats."""

    str: int = 10
    dex: int = 10
    con: int = 10
    int: int = 10
    wis: int = 10
    cha: int = 10


@dataclass
class Condition:
    """A simple status effect."""

    name: str
    duration_rounds: int | None = None
    description: str = ""


@dataclass
class Character:
    """A combat participant."""

    name: str
    hp: int
    max_hp: int
    ac: int
    position: Position
    speed: int
    stats: Stats
    team: Team
    class_name: str | None = None
    subclass_name: str | None = None
    level: int = 1
    experience: int = 0
    proficiency_bonus: int = 2
    race_name: str | None = None
    race_traits: RaceTraits | None = None
    size: str = "Medium"
    weapons: list[WeaponAttack] = field(default_factory=list)
    common_actions: list[str] = field(
        default_factory=lambda: [
            "move",
            "attack",
            "cast_spell",
            "dash",
            "disengage",
            "dodge",
            "help",
            "hide",
            "search",
            "use_object",
            "ready",
            "grapple",
            "shove",
            "stabilize",
            "improvised_action",
            "opportunity_attack",
            "end_turn",
        ]
    )
    class_features: list[ClassFeature] = field(default_factory=list)
    feats: list[object] = field(default_factory=list)
    ability_score_improvements: list[object] = field(default_factory=list)
    resources: dict[str, Resource] = field(default_factory=dict)
    abilities: list[Ability] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    action_economy: ActionEconomy = field(default_factory=ActionEconomy)
    stable: bool = False

    def __post_init__(self) -> None:
        self._migrate_legacy_weapon_abilities()
        self._sync_legacy_abilities()
        self._ensure_progression_features()
        self._ensure_feature_resources()
        reset_turn_resources(self)

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def is_dead(self) -> bool:
        return not self.is_alive

    @property
    def is_incapacitated(self) -> bool:
        return any(
            condition.name.strip().casefold() in {"incapacitated", "unconscious"}
            for condition in self.conditions
        )

    @property
    def can_take_turn(self) -> bool:
        return self.is_alive and not self.is_incapacitated

    @property
    def alive(self) -> bool:
        return self.is_alive

    @property
    def dead(self) -> bool:
        return self.is_dead

    @property
    def available_weapons(self) -> list[WeaponAttack]:
        if self.is_dead:
            return []
        return [weapon for weapon in self.weapons if weapon.available]

    @property
    def available_abilities(self) -> list[Ability]:
        if self.is_dead:
            return []
        available_abilities = [
            ability for ability in self.abilities if ability.available
        ]
        for weapon in self.available_weapons:
            if weapon not in available_abilities:
                available_abilities.append(weapon)
        return available_abilities

    def reset_combat_resources(self) -> None:
        reset_resources(self.resources)

    def reset_start_of_turn_state(self) -> None:
        self.action_economy.dodging_until_start_of_next_turn = False
        self.action_economy.advantage_on_next_check = False

    def reset_end_of_turn_state(self) -> None:
        self.action_economy.end_turn()

    @property
    def disengaged_until_end_of_turn(self) -> bool:
        return self.action_economy.disengaged_until_end_of_turn

    @disengaged_until_end_of_turn.setter
    def disengaged_until_end_of_turn(self, value: bool) -> None:
        self.action_economy.disengaged_until_end_of_turn = value

    @property
    def dodging_until_start_of_next_turn(self) -> bool:
        return self.action_economy.dodging_until_start_of_next_turn

    @dodging_until_start_of_next_turn.setter
    def dodging_until_start_of_next_turn(self, value: bool) -> None:
        self.action_economy.dodging_until_start_of_next_turn = value

    @property
    def hidden(self) -> bool:
        return self.action_economy.hidden

    @hidden.setter
    def hidden(self, value: bool) -> None:
        self.action_economy.hidden = value

    @property
    def prone(self) -> bool:
        return self.action_economy.prone

    @prone.setter
    def prone(self, value: bool) -> None:
        self.action_economy.prone = value

    @property
    def grappled(self) -> bool:
        return self.action_economy.grappled

    @grappled.setter
    def grappled(self, value: bool) -> None:
        self.action_economy.grappled = value
        if value:
            self.action_economy.movement_remaining = 0

    @property
    def grappled_by(self) -> int | None:
        return self.action_economy.grappled_by

    @grappled_by.setter
    def grappled_by(self, value: int | None) -> None:
        self.action_economy.grappled_by = value
        self.action_economy.grappled = value is not None
        if value is not None:
            self.action_economy.movement_remaining = 0

    @property
    def grappling_target_id(self) -> int | None:
        return self.action_economy.grappling_target_id

    @grappling_target_id.setter
    def grappling_target_id(self, value: int | None) -> None:
        self.action_economy.grappling_target_id = value

    @property
    def helped_target_id(self) -> int | None:
        return self.action_economy.helped_target_id

    @helped_target_id.setter
    def helped_target_id(self, value: int | None) -> None:
        self.action_economy.helped_target_id = value

    @property
    def help_against_target_id(self) -> int | None:
        return self.action_economy.help_against_target_id

    @help_against_target_id.setter
    def help_against_target_id(self, value: int | None) -> None:
        self.action_economy.help_against_target_id = value

    @property
    def help_attack_target_id(self) -> int | None:
        return self.action_economy.help_against_target_id

    @help_attack_target_id.setter
    def help_attack_target_id(self, value: int | None) -> None:
        self.action_economy.help_against_target_id = value

    @property
    def prepared_action(self) -> str | None:
        return self.action_economy.prepared_action

    @prepared_action.setter
    def prepared_action(self, value: str | None) -> None:
        self.action_economy.prepared_action = value

    @property
    def trigger_description(self) -> str | None:
        return self.action_economy.trigger_description

    @trigger_description.setter
    def trigger_description(self, value: str | None) -> None:
        self.action_economy.trigger_description = value

    @property
    def advantage_on_next_check(self) -> bool:
        return self.action_economy.advantage_on_next_check

    @advantage_on_next_check.setter
    def advantage_on_next_check(self, value: bool) -> None:
        self.action_economy.advantage_on_next_check = value

    def _migrate_legacy_weapon_abilities(self) -> None:
        for ability in self.abilities:
            if isinstance(ability, WeaponAttack) and ability not in self.weapons:
                self.weapons.append(ability)

    def _sync_legacy_abilities(self) -> None:
        for weapon in self.weapons:
            if weapon not in self.abilities:
                self.abilities.append(weapon)

    def _ensure_feature_resources(self) -> None:
        from combat.class_features import feature_resource_name

        for feature in self.class_features:
            resource_name = feature_resource_name(feature)
            if resource_name is None:
                continue
            if resource_name not in self.resources:
                self.resources[resource_name] = Resource(
                    name=resource_name,
                    max_uses=1,
                )

    def _ensure_progression_features(self) -> None:
        if self.class_name is None or self.class_features:
            return

        from rules.progression import (
            build_class_features,
            build_class_resources,
            is_spellcaster,
            spell_slots_for_level,
        )

        self.class_features = build_class_features(
            self.class_name,
            self.level,
            self.subclass_name,
        )
        self.resources = build_class_resources(self.class_features, self.resources)
        if is_spellcaster(self):
            spell_slots = spell_slots_for_level(self.level)
            self.spellcasting = True
            self.spell_slots = dict(spell_slots)
            self.spell_slots_remaining = dict(spell_slots)


@dataclass
class Enemy(Character):
    """A combat participant on the enemies team."""

    team: Team = field(default=Team.ENEMIES, init=False)


@dataclass
class CombatState:
    """Current combat snapshot."""

    characters: list[Character] = field(default_factory=list)
    grid_map: GridMap | None = None
    round_number: int = 1
    turn_index: int = 0
    initiative_order: list[int] = field(default_factory=list)
    current_turn_index: int = 0
    initiative_rolls: dict[int, int] = field(default_factory=dict)
    initiative_totals: dict[int, int] = field(default_factory=dict)
    initiative_dex_modifiers: dict[int, int] = field(default_factory=dict)
    initiative_tie_breakers: dict[int, float] = field(default_factory=dict)
    skipped_turn_actor_ids: list[int] = field(default_factory=list)

    @property
    def active_actor_id(self) -> int | None:
        if not self.characters:
            return None
        if self.initiative_order:
            order_index = self.current_turn_index % len(self.initiative_order)
            actor_id = self.initiative_order[order_index]
            if actor_id < 0 or actor_id >= len(self.characters):
                return None
            return actor_id
        return self.turn_index % len(self.characters)

    @property
    def active_character(self) -> Character | None:
        actor_id = self.active_actor_id
        if actor_id is None:
            return None
        return self.character_at(actor_id)

    @property
    def living_characters(self) -> list[Character]:
        return [character for character in self.characters if character.is_alive]

    def characters_for_team(self, team: Team) -> list[Character]:
        return [character for character in self.characters if character.team == team]

    def character_at(self, character_id: int) -> Character | None:
        if character_id < 0 or character_id >= len(self.characters):
            return None
        return self.characters[character_id]

    def reset_turn_resources(self, actor_id: int | None = None) -> Character | None:
        actor = self.active_character if actor_id is None else self.character_at(actor_id)
        if actor is None:
            return None
        actor.reset_start_of_turn_state()
        reset_turn_resources(actor)
        return actor

    def reset_combat_resources(self) -> None:
        for character in self.characters:
            character.reset_combat_resources()

    def advance_turn(self) -> Character | None:
        if not self.characters:
            return None

        order = self.initiative_order or list(range(len(self.characters)))
        if not order:
            return None

        self.skipped_turn_actor_ids = []
        current_index = self.current_turn_index % len(order)
        if not self.initiative_order:
            current_index = self.turn_index % len(self.characters)
        wrapped = False
        for offset in range(1, len(order) + 1):
            next_index = (current_index + offset) % len(order)
            if next_index == 0:
                wrapped = True
            character_id = order[next_index]
            character = self.character_at(character_id)
            if character is None:
                continue
            if character.can_take_turn:
                self.current_turn_index = next_index
                self.turn_index = character_id
                if wrapped:
                    self.round_number += 1
                return self.reset_turn_resources(character_id)
            self.skipped_turn_actor_ids.append(character_id)

        return None
