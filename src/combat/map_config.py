"""JSON-backed tactical map configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from combat.map import GridMap
from combat.models import Position
from combat.terrain import TerrainType, coerce_terrain_type


DEFAULT_MAP_CONFIG_DIR = Path(__file__).resolve().parents[2] / "maps"


class MapConfigValidationError(ValueError):
    """Raised when a map JSON file is structurally invalid."""


@dataclass(frozen=True)
class MapSpawnZones:
    """Spawn cells for both combat sides."""

    players: tuple[Position, ...]
    enemies: tuple[Position, ...]


@dataclass(frozen=True)
class MapConfig:
    """Validated map definition loaded from JSON."""

    name: str
    width: int
    height: int
    terrain_grid: tuple[tuple[TerrainType, ...], ...]
    spawn_zones: MapSpawnZones
    source_path: Path | None = None

    def to_grid_map(self) -> GridMap:
        """Create the combat GridMap represented by this config."""

        return GridMap(
            width=self.width,
            height=self.height,
            terrain_grid=self.terrain_grid,
        )

    def terrain_at(self, position: Position) -> TerrainType:
        """Return terrain at a map position."""

        if not self.in_bounds(position):
            return TerrainType.BLOCKED
        return self.terrain_grid[position.y][position.x]

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height


def load_map_config(path: str | Path) -> MapConfig:
    """Load and validate one map JSON file."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MapConfigValidationError(f"Invalid map JSON: {config_path}") from error
    except OSError as error:
        raise MapConfigValidationError(f"Cannot read map JSON: {config_path}") from error
    return map_config_from_mapping(raw, source_path=config_path)


def map_config_from_mapping(
    raw: dict[str, Any],
    *,
    source_path: str | Path | None = None,
) -> MapConfig:
    """Validate and normalize a raw map mapping."""

    if not isinstance(raw, dict):
        raise MapConfigValidationError("Map config must be a JSON object.")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise MapConfigValidationError("Map config must define name.")
    width = _positive_int(raw.get("width"), "width")
    height = _positive_int(raw.get("height"), "height")
    terrain_grid = _parse_terrain_grid(raw.get("terrain_grid"), width, height)
    spawn_zones = _parse_spawn_zones(raw.get("spawn_zones"), width, height, terrain_grid)
    return MapConfig(
        name=name,
        width=width,
        height=height,
        terrain_grid=terrain_grid,
        spawn_zones=spawn_zones,
        source_path=Path(source_path) if source_path is not None else None,
    )


def load_map_config_by_name(
    name: str,
    map_dir: str | Path = DEFAULT_MAP_CONFIG_DIR,
) -> MapConfig:
    """Load a map by file stem or display name from a map directory."""

    normalized = normalize_map_key(name)
    map_path = Path(map_dir) / f"{normalized}.json"
    if map_path.exists():
        return load_map_config(map_path)
    for candidate in iter_map_config_paths(map_dir):
        config = load_map_config(candidate)
        if normalize_map_key(config.name) == normalized:
            return config
    raise MapConfigValidationError(f"Unknown map config: {name}")


def list_map_configs(map_dir: str | Path = DEFAULT_MAP_CONFIG_DIR) -> dict[str, MapConfig]:
    """Return valid map configs keyed by file stem."""

    configs: dict[str, MapConfig] = {}
    for path in iter_map_config_paths(map_dir):
        config = load_map_config(path)
        configs[normalize_map_key(path.stem)] = config
    return configs


def map_options(map_dir: str | Path = DEFAULT_MAP_CONFIG_DIR) -> dict[str, str]:
    """Return GUI-friendly map options from map JSON files."""

    return {
        key: config.name
        for key, config in sorted(list_map_configs(map_dir).items())
    }


def iter_map_config_paths(map_dir: str | Path = DEFAULT_MAP_CONFIG_DIR) -> tuple[Path, ...]:
    directory = Path(map_dir)
    if not directory.exists():
        return ()
    return tuple(sorted(path for path in directory.glob("*.json") if path.is_file()))


def normalize_map_key(value: object) -> str:
    """Normalize map names to stable file-style keys."""

    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_terrain_grid(
    raw_grid: object,
    width: int,
    height: int,
) -> tuple[tuple[TerrainType, ...], ...]:
    if not isinstance(raw_grid, list):
        raise MapConfigValidationError("terrain_grid must be a list of rows.")
    if len(raw_grid) != height:
        raise MapConfigValidationError("terrain_grid height must match map height.")
    rows: list[tuple[TerrainType, ...]] = []
    for row_index, raw_row in enumerate(raw_grid):
        if not isinstance(raw_row, list):
            raise MapConfigValidationError(f"terrain_grid row {row_index} must be a list.")
        if len(raw_row) != width:
            raise MapConfigValidationError("terrain_grid width must match map width.")
        try:
            rows.append(tuple(coerce_terrain_type(cell) for cell in raw_row))
        except ValueError as error:
            raise MapConfigValidationError(str(error)) from error
    return tuple(rows)


def _parse_spawn_zones(
    raw_zones: object,
    width: int,
    height: int,
    terrain_grid: tuple[tuple[TerrainType, ...], ...],
) -> MapSpawnZones:
    if not isinstance(raw_zones, dict):
        raise MapConfigValidationError("spawn_zones must be an object.")
    players = _parse_spawn_list(raw_zones.get("players"), "players")
    enemies = _parse_spawn_list(raw_zones.get("enemies"), "enemies")
    if not players:
        raise MapConfigValidationError("spawn_zones.players must not be empty.")
    if not enemies:
        raise MapConfigValidationError("spawn_zones.enemies must not be empty.")
    for label, positions in (("players", players), ("enemies", enemies)):
        _validate_spawn_positions(label, positions, width, height, terrain_grid)
    return MapSpawnZones(players=players, enemies=enemies)


def _parse_spawn_list(raw_positions: object, label: str) -> tuple[Position, ...]:
    if not isinstance(raw_positions, list):
        raise MapConfigValidationError(f"spawn_zones.{label} must be a list.")
    return tuple(_parse_position(raw_position, label) for raw_position in raw_positions)


def _parse_position(raw_position: object, label: str) -> Position:
    if isinstance(raw_position, dict):
        return Position(_positive_or_zero(raw_position.get("x"), f"{label}.x"), _positive_or_zero(raw_position.get("y"), f"{label}.y"))
    if isinstance(raw_position, list) and len(raw_position) == 2:
        return Position(_positive_or_zero(raw_position[0], f"{label}.x"), _positive_or_zero(raw_position[1], f"{label}.y"))
    raise MapConfigValidationError(
        f"spawn_zones.{label} positions must be [x, y] lists or objects."
    )


def _validate_spawn_positions(
    label: str,
    positions: Iterable[Position],
    width: int,
    height: int,
    terrain_grid: tuple[tuple[TerrainType, ...], ...],
) -> None:
    seen: set[Position] = set()
    for position in positions:
        if position in seen:
            raise MapConfigValidationError(f"Duplicate {label} spawn cell: {position.x},{position.y}")
        seen.add(position)
        if not (0 <= position.x < width and 0 <= position.y < height):
            raise MapConfigValidationError(
                f"{label} spawn cell is outside map bounds: {position.x},{position.y}"
            )
        if terrain_grid[position.y][position.x] is TerrainType.BLOCKED:
            raise MapConfigValidationError(
                f"{label} spawn cell is blocked: {position.x},{position.y}"
            )


def _positive_int(value: object, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise MapConfigValidationError(f"{label} must be an integer.") from error
    if result <= 0:
        raise MapConfigValidationError(f"{label} must be greater than zero.")
    return result


def _positive_or_zero(value: object, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise MapConfigValidationError(f"{label} must be an integer.") from error
    if result < 0:
        raise MapConfigValidationError(f"{label} must not be negative.")
    return result
