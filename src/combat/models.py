"""Basic combat data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from combat.map import GridMap


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
class Ability:
    """A generic combat ability."""

    name: str
    description: str = ""
    range: int = 0
    cooldown: int = 0
    remaining_cooldown: int = 0

    @property
    def available(self) -> bool:
        return self.remaining_cooldown <= 0


@dataclass
class WeaponAttack(Ability):
    """A simple weapon-based attack."""

    range: int = 1
    damage: int | str = "1d6"
    attack_bonus: int = 0


@dataclass
class SpellAbility(Ability):
    """A simple spell ability."""

    spell_level: int = 0
    damage: str | None = None
    save_dc: int | None = None


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
    abilities: list[Ability] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def is_dead(self) -> bool:
        return not self.is_alive

    @property
    def alive(self) -> bool:
        return self.is_alive

    @property
    def dead(self) -> bool:
        return self.is_dead

    @property
    def available_abilities(self) -> list[Ability]:
        if self.is_dead:
            return []
        return [ability for ability in self.abilities if ability.available]


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

    @property
    def active_character(self) -> Character | None:
        if not self.characters:
            return None
        return self.characters[self.turn_index % len(self.characters)]

    @property
    def living_characters(self) -> list[Character]:
        return [character for character in self.characters if character.is_alive]

    def characters_for_team(self, team: Team) -> list[Character]:
        return [character for character in self.characters if character.team == team]
