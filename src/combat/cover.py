"""Cover rules for tactical grid attacks and saves."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from combat.line_of_sight import bresenham_line
from combat.models import Position
from combat.terrain import TerrainType, coerce_terrain_type


class CoverType(Enum):
    """D&D-like cover categories."""

    NO_COVER = "no_cover"
    HALF_COVER = "half_cover"
    THREE_QUARTERS_COVER = "three_quarters_cover"
    FULL_COVER = "full_cover"

    @property
    def ac_bonus(self) -> int:
        return {
            CoverType.NO_COVER: 0,
            CoverType.HALF_COVER: 2,
            CoverType.THREE_QUARTERS_COVER: 5,
            CoverType.FULL_COVER: 0,
        }[self]

    @property
    def dex_save_bonus(self) -> int:
        return self.ac_bonus

    @property
    def blocks_targeting(self) -> bool:
        return self is CoverType.FULL_COVER


def get_cover_between(
    attacker_pos: Position,
    target_pos: Position,
    terrain_at: Callable[[Position], TerrainType],
) -> CoverType:
    """Return the best cover on a grid line from attacker to target."""

    cells = bresenham_line(attacker_pos, target_pos)
    if len(cells) <= 1:
        return CoverType.NO_COVER

    intermediate_cover = [
        coerce_terrain_type(terrain_at(cell))
        for cell in cells[1:-1]
    ]
    target_terrain = coerce_terrain_type(terrain_at(target_pos))

    if any(
        terrain in {TerrainType.BLOCKED, TerrainType.HIGH_COVER}
        for terrain in intermediate_cover
    ):
        return CoverType.FULL_COVER
    if target_terrain is TerrainType.BLOCKED:
        return CoverType.FULL_COVER
    if target_terrain is TerrainType.HIGH_COVER:
        return CoverType.THREE_QUARTERS_COVER

    low_cover_count = sum(
        1
        for terrain in intermediate_cover
        if terrain is TerrainType.LOW_COVER
    )
    if target_terrain is TerrainType.LOW_COVER:
        low_cover_count += 1
    if low_cover_count >= 2:
        return CoverType.THREE_QUARTERS_COVER
    if low_cover_count == 1:
        return CoverType.HALF_COVER
    return CoverType.NO_COVER


def apply_cover_to_ac(base_ac: int, cover: CoverType) -> int:
    """Return AC after cover bonus. Full cover should be handled before targeting."""

    return int(base_ac) + cover.ac_bonus


def apply_cover_to_dex_save(total: int, cover: CoverType) -> int:
    """Return a DEX saving throw total after cover bonus."""

    return int(total) + cover.dex_save_bonus
