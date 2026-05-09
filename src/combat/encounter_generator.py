"""Random encounter generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import random

from combat.environment import CombatEnvironment
from combat.map import GridMap
from combat.models import Character, CombatState, Position
from combat.presets import (
    FighterArcher,
    FighterChampionGreatsword,
    Goblin,
    Orc,
)


CharacterFactory = Callable[[Position], Character]


@dataclass
class EncounterGenerator:
    """Generate simple random combat encounters."""

    seed: int | None = None
    width: int = 8
    height: int = 8
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def reseed(self, seed: int | None) -> None:
        self.seed = seed
        self._random.seed(seed)

    def generate_state(self) -> CombatState:
        player_count = self._random.randint(1, 2)
        enemy_count = self._random.randint(1, 4)

        player_factories = self._random.sample(
            [FighterChampionGreatsword, FighterArcher],
            k=player_count,
        )
        enemy_factories = [
            self._random.choice([Goblin, Orc])
            for _ in range(enemy_count)
        ]

        player_positions = self._random_positions(
            player_count,
            x_values=range(0, self.width // 2),
        )
        enemy_positions = self._random_positions(
            enemy_count,
            x_values=range(self.width // 2, self.width),
        )

        characters = [
            factory(position)
            for factory, position in zip(player_factories, player_positions)
        ]
        characters.extend(
            factory(position)
            for factory, position in zip(enemy_factories, enemy_positions)
        )

        return CombatState(
            characters=characters,
            grid_map=GridMap(width=self.width, height=self.height),
        )

    def generate_environment(
        self,
        use_initiative: bool = False,
        log_to_console: bool = True,
    ) -> CombatEnvironment:
        combat_state = self.generate_state()
        return CombatEnvironment(
            characters=combat_state.characters,
            grid_map=combat_state.grid_map,
            use_initiative=use_initiative,
            log_to_console=log_to_console,
        )

    def generate(
        self,
        as_environment: bool = False,
        use_initiative: bool = False,
        log_to_console: bool = True,
    ) -> CombatState | CombatEnvironment:
        if as_environment:
            return self.generate_environment(
                use_initiative=use_initiative,
                log_to_console=log_to_console,
            )
        return self.generate_state()

    def _random_positions(
        self,
        count: int,
        x_values: range,
    ) -> list[Position]:
        positions = [
            Position(x, y)
            for x in x_values
            for y in range(self.height)
        ]
        self._random.shuffle(positions)
        return positions[:count]
