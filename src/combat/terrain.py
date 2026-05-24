"""Terrain types and movement costs for tactical maps."""

from __future__ import annotations

from enum import Enum


class TerrainType(Enum):
    """Supported terrain cell types."""

    NORMAL = "normal"
    DIFFICULT_TERRAIN = "difficult_terrain"
    BLOCKED = "blocked"
    LOW_COVER = "low_cover"
    HIGH_COVER = "high_cover"


MOVEMENT_COSTS: dict[TerrainType, int | None] = {
    TerrainType.NORMAL: 1,
    TerrainType.DIFFICULT_TERRAIN: 2,
    TerrainType.BLOCKED: None,
    TerrainType.LOW_COVER: 1,
    TerrainType.HIGH_COVER: 1,
}

LOS_BLOCKING_TERRAIN: frozenset[TerrainType] = frozenset(
    {
        TerrainType.BLOCKED,
        TerrainType.HIGH_COVER,
    }
)


def coerce_terrain_type(value: TerrainType | str | None) -> TerrainType:
    """Normalize enum/string terrain values."""

    if value is None:
        return TerrainType.NORMAL
    if isinstance(value, TerrainType):
        return value
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    for terrain_type in TerrainType:
        if terrain_type.name.lower() == normalized or terrain_type.value == normalized:
            return terrain_type
    raise ValueError(f"Unknown terrain type: {value}")


def terrain_movement_cost(terrain_type: TerrainType | str | None) -> int | None:
    """Return movement cost for entering a terrain cell, or None if blocked."""

    return MOVEMENT_COSTS[coerce_terrain_type(terrain_type)]


def is_walkable_terrain(terrain_type: TerrainType | str | None) -> bool:
    """Return True when a terrain cell may be entered."""

    return terrain_movement_cost(terrain_type) is not None


def blocks_line_of_sight(terrain_type: TerrainType | str | None) -> bool:
    """Return True when terrain blocks line of sight."""

    return coerce_terrain_type(terrain_type) in LOS_BLOCKING_TERRAIN
