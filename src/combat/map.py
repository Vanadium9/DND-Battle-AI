"""Grid map helpers for tactical combat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from combat.models import Character, Position


@dataclass(frozen=True)
class GridMap:
    """A rectangular tactical grid."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be greater than zero")
        if self.height <= 0:
            raise ValueError("height must be greater than zero")

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    @staticmethod
    def manhattan_distance(first: Position, second: Position) -> int:
        return abs(first.x - second.x) + abs(first.y - second.y)

    def neighbors(self, position: Position) -> list[Position]:
        candidates = [
            Position(position.x + 1, position.y),
            Position(position.x - 1, position.y),
            Position(position.x, position.y + 1),
            Position(position.x, position.y - 1),
        ]
        return [candidate for candidate in candidates if self.in_bounds(candidate)]

    def is_occupied(
        self,
        position: Position,
        characters: Iterable[Character],
    ) -> bool:
        return any(
            character.is_alive and character.position == position
            for character in characters
        )

    def movement_cells(
        self,
        start: Position,
        speed: int,
        characters: Iterable[Character] | None = None,
    ) -> set[Position]:
        if speed < 0 or not self.in_bounds(start):
            return set()

        occupied_positions = set()
        if characters is not None:
            occupied_positions = {
                character.position
                for character in characters
                if character.is_alive and character.position != start
            }

        cells = set()
        for x in range(self.width):
            for y in range(self.height):
                position = Position(x, y)
                if self.manhattan_distance(start, position) <= speed:
                    cells.add(position)

        return cells - occupied_positions
