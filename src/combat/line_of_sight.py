"""Grid raycast helpers for line-of-sight checks."""

from __future__ import annotations

from collections.abc import Callable

from combat.models import Position
from combat.terrain import TerrainType, blocks_line_of_sight


def bresenham_line(start: Position, end: Position) -> tuple[Position, ...]:
    """Return grid cells touched by an integer line from start to end."""

    x0, y0 = start.x, start.y
    x1, y1 = end.x, end.y
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy

    cells: list[Position] = []
    while True:
        cells.append(Position(x0, y0))
        if x0 == x1 and y0 == y1:
            break
        doubled_error = 2 * error
        if doubled_error > -dy:
            error -= dy
            x0 += sx
        if doubled_error < dx:
            error += dx
            y0 += sy
    return tuple(cells)


def line_of_sight(
    start: Position,
    end: Position,
    terrain_at: Callable[[Position], TerrainType],
) -> bool:
    """Return True if no intermediate terrain blocks sight between two cells."""

    cells = bresenham_line(start, end)
    if len(cells) <= 2:
        return True
    return not any(blocks_line_of_sight(terrain_at(cell)) for cell in cells[1:-1])
