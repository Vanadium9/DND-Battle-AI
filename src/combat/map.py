"""Grid map helpers for tactical combat."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Iterable

from combat.cover import CoverType, get_cover_between
from combat.line_of_sight import line_of_sight
from combat.models import Character, Position
from combat.terrain import (
    TerrainType,
    coerce_terrain_type,
    is_walkable_terrain,
    terrain_movement_cost,
)


@dataclass(frozen=True)
class GridMap:
    """A rectangular tactical grid."""

    width: int
    height: int
    terrain_grid: tuple[tuple[TerrainType, ...], ...] | None = None
    _neighbor_cost_cache: dict[tuple[int, int], tuple[tuple[Position, int], ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _line_of_sight_cache: dict[tuple[int, int, int, int], bool] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _cover_cache: dict[tuple[int, int, int, int], CoverType] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be greater than zero")
        if self.height <= 0:
            raise ValueError("height must be greater than zero")
        object.__setattr__(self, "terrain_grid", self._normalize_terrain_grid())

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    @staticmethod
    def manhattan_distance(first: Position, second: Position) -> int:
        return abs(first.x - second.x) + abs(first.y - second.y)

    def terrain_at(self, position: Position) -> TerrainType:
        """Return terrain type at a map position."""

        if not self.in_bounds(position):
            return TerrainType.BLOCKED
        if self.terrain_grid is None:
            return TerrainType.NORMAL
        return self.terrain_grid[position.y][position.x]

    def is_blocked(self, position: Position) -> bool:
        """Return True when a position is outside the map or blocked."""

        return not self.in_bounds(position) or self.terrain_at(position) is TerrainType.BLOCKED

    def is_walkable(self, position: Position) -> bool:
        """Return True when terrain may be entered."""

        return self.in_bounds(position) and is_walkable_terrain(self.terrain_at(position))

    def movement_cost(self, position: Position) -> int | None:
        """Return movement cost for entering a position, or None if blocked."""

        if not self.in_bounds(position):
            return None
        return terrain_movement_cost(self.terrain_at(position))

    def neighbors(self, position: Position) -> list[Position]:
        """Return walkable cardinal neighbors."""

        candidates = [
            Position(position.x + 1, position.y),
            Position(position.x - 1, position.y),
            Position(position.x, position.y + 1),
            Position(position.x, position.y - 1),
        ]
        return [candidate for candidate in candidates if self.is_walkable(candidate)]

    def neighbor_costs(self, position: Position) -> list[tuple[Position, int]]:
        """Return walkable cardinal neighbors with terrain movement costs."""

        key = (position.x, position.y)
        cached = self._neighbor_cost_cache.get(key)
        if cached is not None:
            return list(cached)

        result: list[tuple[Position, int]] = []
        for neighbor in self.neighbors(position):
            movement_cost = self.movement_cost(neighbor)
            if movement_cost is not None:
                result.append((neighbor, movement_cost))
        self._neighbor_cost_cache[key] = tuple(result)
        return result

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
        """Return cells reachable within movement budget."""

        return set(
            self.movement_costs_from(start, speed, characters).keys()
        )

    def movement_costs_from(
        self,
        start: Position,
        speed: int,
        characters: Iterable[Character] | None = None,
    ) -> dict[Position, int]:
        """Return reachable cells and cheapest path costs from start."""

        if speed < 0 or not self.in_bounds(start) or not self.is_walkable(start):
            return {}

        occupied_positions = self._occupied_positions(characters, start)
        distances: dict[Position, int] = {start: 0}
        heap: list[tuple[int, int, int, Position]] = [(0, start.x, start.y, start)]

        while heap:
            current_cost, _, _, current = heapq.heappop(heap)
            if current_cost != distances[current]:
                continue
            for neighbor, step_cost in self.neighbor_costs(current):
                next_cost = current_cost + step_cost
                if next_cost > speed:
                    continue
                if next_cost < distances.get(neighbor, 10**9):
                    distances[neighbor] = next_cost
                    heapq.heappush(heap, (next_cost, neighbor.x, neighbor.y, neighbor))

        return {
            position: cost
            for position, cost in distances.items()
            if position not in occupied_positions
        }

    def path_movement_cost(
        self,
        start: Position,
        destination: Position,
        characters: Iterable[Character] | None = None,
    ) -> int | None:
        """Return cheapest movement cost to a destination, or None if unreachable."""

        if not self.in_bounds(destination) or not self.is_walkable(destination):
            return None
        occupied_positions = self._occupied_positions(characters, start)
        if destination in occupied_positions:
            return None

        distances: dict[Position, int] = {start: 0}
        heap: list[tuple[int, int, int, Position]] = [(0, start.x, start.y, start)]
        while heap:
            current_cost, _, _, current = heapq.heappop(heap)
            if current == destination:
                return current_cost
            if current_cost != distances[current]:
                continue
            for neighbor, step_cost in self.neighbor_costs(current):
                next_cost = current_cost + step_cost
                if next_cost < distances.get(neighbor, 10**9):
                    distances[neighbor] = next_cost
                    heapq.heappush(heap, (next_cost, neighbor.x, neighbor.y, neighbor))

        return None

    def line_of_sight(self, start: Position, end: Position) -> bool:
        """Return True when sight is not blocked by BLOCKED/HIGH_COVER terrain."""

        if not self.in_bounds(start) or not self.in_bounds(end):
            return False
        key = (start.x, start.y, end.x, end.y)
        cached = self._line_of_sight_cache.get(key)
        if cached is not None:
            return cached
        visible = line_of_sight(start, end, self.terrain_at)
        self._line_of_sight_cache[key] = visible
        self._line_of_sight_cache[(end.x, end.y, start.x, start.y)] = visible
        return visible

    def has_line_of_sight(self, start: Position, end: Position) -> bool:
        """Compatibility alias used by action space and common actions."""

        return self.line_of_sight(start, end)

    def get_cover_between(self, attacker_pos: Position, target_pos: Position) -> CoverType:
        """Return cover between attacker and target positions."""

        if not self.in_bounds(attacker_pos) or not self.in_bounds(target_pos):
            return CoverType.FULL_COVER
        key = (attacker_pos.x, attacker_pos.y, target_pos.x, target_pos.y)
        cached = self._cover_cache.get(key)
        if cached is not None:
            return cached
        cover = get_cover_between(attacker_pos, target_pos, self.terrain_at)
        self._cover_cache[key] = cover
        return cover

    def _normalize_terrain_grid(self) -> tuple[tuple[TerrainType, ...], ...]:
        if self.terrain_grid is None:
            return tuple(
                tuple(TerrainType.NORMAL for _ in range(self.width))
                for _ in range(self.height)
            )
        if len(self.terrain_grid) != self.height:
            raise ValueError("terrain_grid height must match map height")

        rows: list[tuple[TerrainType, ...]] = []
        for row in self.terrain_grid:
            if len(row) != self.width:
                raise ValueError("terrain_grid width must match map width")
            rows.append(tuple(coerce_terrain_type(cell) for cell in row))
        return tuple(rows)

    def _occupied_positions(
        self,
        characters: Iterable[Character] | None,
        start: Position,
    ) -> set[Position]:
        if characters is None:
            return set()
        return {
            character.position
            for character in characters
            if character.is_alive and character.position != start
        }
