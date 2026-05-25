"""Grid-based area-of-effect targeting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Iterable

from combat.models import Character, Position


logger = logging.getLogger(__name__)


class AoEShape(str, Enum):
    """Supported area shapes."""

    RADIUS = "radius"
    CONE = "cone"
    LINE = "line"


class AoEDirection(str, Enum):
    """Cardinal directions used by simplified cone and line AoE."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


AOE_DIRECTIONS: tuple[AoEDirection, ...] = (
    AoEDirection.UP,
    AoEDirection.DOWN,
    AoEDirection.LEFT,
    AoEDirection.RIGHT,
)


@dataclass(frozen=True)
class AoETargeting:
    """Resolved AoE placement on a grid."""

    shape: AoEShape
    origin: Position
    size: int
    target_cell: Position | None = None
    direction: AoEDirection | None = None
    radius_metric: str = "manhattan"

    @property
    def source_position(self) -> Position:
        """Return the position cover should be measured from."""

        if self.shape is AoEShape.RADIUS and self.target_cell is not None:
            return self.target_cell
        return self.origin


def coerce_aoe_shape(value: object) -> AoEShape | None:
    """Return an AoEShape from an enum/string value."""

    if value is None:
        return None
    if isinstance(value, AoEShape):
        return value
    normalized = str(value).strip().casefold()
    for shape in AoEShape:
        if normalized in {shape.value, shape.name.casefold()}:
            return shape
    return None


def coerce_aoe_direction(value: object) -> AoEDirection | None:
    """Return an AoEDirection from an enum/string/index value."""

    if value is None:
        return None
    if isinstance(value, AoEDirection):
        return value
    if isinstance(value, int):
        if 0 <= value < len(AOE_DIRECTIONS):
            return AOE_DIRECTIONS[value]
        return None
    normalized = str(value).strip().casefold()
    for direction in AOE_DIRECTIONS:
        if normalized in {direction.value, direction.name.casefold()}:
            return direction
    return None


def direction_to_index(direction: AoEDirection | str | int | None) -> int | None:
    """Return the stable action-space index for a direction."""

    coerced = coerce_aoe_direction(direction)
    if coerced is None:
        return None
    return AOE_DIRECTIONS.index(coerced)


def direction_from_positions(origin: Position, target: Position) -> AoEDirection:
    """Resolve the dominant cardinal direction from origin to target."""

    dx = target.x - origin.x
    dy = target.y - origin.y
    if abs(dx) >= abs(dy):
        return AoEDirection.RIGHT if dx >= 0 else AoEDirection.LEFT
    return AoEDirection.DOWN if dy >= 0 else AoEDirection.UP


def positions_for_aoe(targeting: AoETargeting) -> set[Position]:
    """Return all grid cells covered by an AoE template."""

    size = max(0, int(targeting.size))
    if size <= 0:
        return set()
    if targeting.shape is AoEShape.RADIUS:
        center = targeting.target_cell or targeting.origin
        return _radius_positions(center, size, targeting.radius_metric)
    if targeting.shape is AoEShape.CONE:
        if targeting.direction is None:
            return set()
        return _cone_positions(targeting.origin, size, targeting.direction)
    if targeting.shape is AoEShape.LINE:
        if targeting.direction is None:
            return set()
        return _line_positions(targeting.origin, size, targeting.direction)
    return set()


def affected_creatures(
    creatures: Iterable[Character],
    positions: set[Position],
    *,
    include_dead: bool = False,
) -> list[Character]:
    """Return creatures whose current cell is inside the AoE."""

    return [
        creature
        for creature in creatures
        if creature.position in positions and (include_dead or creature.is_alive)
    ]


def log_affected_targets(
    source_name: str,
    targeting: AoETargeting,
    targets: Iterable[Any],
) -> None:
    """Log the final AoE target list for combat debugging."""

    target_names = [str(getattr(target, "name", target)) for target in targets]
    logger.info(
        "%s AoE %s affected targets: %s",
        source_name,
        targeting.shape.name,
        ", ".join(target_names) if target_names else "none",
    )


def _radius_positions(center: Position, radius: int, metric: str) -> set[Position]:
    positions: set[Position] = set()
    normalized_metric = metric.strip().casefold()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if normalized_metric == "chebyshev":
                inside = max(abs(dx), abs(dy)) <= radius
            else:
                inside = abs(dx) + abs(dy) <= radius
            if inside:
                positions.add(Position(center.x + dx, center.y + dy))
    return positions


def _cone_positions(
    origin: Position,
    size: int,
    direction: AoEDirection,
) -> set[Position]:
    positions: set[Position] = set()
    for forward in range(1, size + 1):
        for lateral in range(-forward, forward + 1):
            positions.add(_offset_position(origin, direction, forward, lateral))
    return positions


def _line_positions(
    origin: Position,
    size: int,
    direction: AoEDirection,
) -> set[Position]:
    return {
        _offset_position(origin, direction, forward, 0)
        for forward in range(1, size + 1)
    }


def _offset_position(
    origin: Position,
    direction: AoEDirection,
    forward: int,
    lateral: int,
) -> Position:
    if direction is AoEDirection.UP:
        return Position(origin.x + lateral, origin.y - forward)
    if direction is AoEDirection.DOWN:
        return Position(origin.x + lateral, origin.y + forward)
    if direction is AoEDirection.LEFT:
        return Position(origin.x - forward, origin.y + lateral)
    return Position(origin.x + forward, origin.y + lateral)
