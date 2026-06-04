"""JSON-backed tactical map configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
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


def save_map_config(config: MapConfig, path: str | Path) -> Path:
    """Save a validated map config to JSON."""

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(map_config_to_mapping(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_path


def map_config_to_mapping(config: MapConfig) -> dict[str, object]:
    """Return a JSON-ready map mapping."""

    return {
        "name": config.name,
        "width": config.width,
        "height": config.height,
        "terrain_grid": [
            [cell.name for cell in row]
            for row in config.terrain_grid
        ],
        "spawn_zones": {
            "players": [[position.x, position.y] for position in config.spawn_zones.players],
            "enemies": [[position.x, position.y] for position in config.spawn_zones.enemies],
        },
    }


def generate_random_map_config(
    *,
    name: str,
    width: int = 8,
    height: int = 6,
    seed: int | None = None,
    template: str = "balanced",
) -> MapConfig:
    """Generate a simple valid tactical map for GUI use."""

    rng = random.Random(seed)
    safe_width = max(4, min(16, int(width)))
    safe_height = max(4, min(12, int(height)))
    terrain_grid = [
        [TerrainType.NORMAL for _x in range(safe_width)]
        for _y in range(safe_height)
    ]
    blocked_probability, difficult_probability, low_cover_probability, high_cover_probability = (
        _template_probabilities(template)
    )
    for y in range(safe_height):
        for x in range(safe_width):
            if x in {0, safe_width - 1}:
                continue
            roll = rng.random()
            if roll < blocked_probability:
                terrain_grid[y][x] = TerrainType.BLOCKED
            elif roll < blocked_probability + difficult_probability:
                terrain_grid[y][x] = TerrainType.DIFFICULT_TERRAIN
            elif roll < blocked_probability + difficult_probability + low_cover_probability:
                terrain_grid[y][x] = TerrainType.LOW_COVER
            elif roll < blocked_probability + difficult_probability + low_cover_probability + high_cover_probability:
                terrain_grid[y][x] = TerrainType.HIGH_COVER

    player_spawns = _spawn_column_positions(0, safe_height)
    enemy_spawns = _spawn_column_positions(safe_width - 1, safe_height)
    for position in (*player_spawns, *enemy_spawns):
        terrain_grid[position.y][position.x] = TerrainType.NORMAL

    return map_config_from_mapping(
        {
            "name": name,
            "width": safe_width,
            "height": safe_height,
            "terrain_grid": [
                [cell.name for cell in row]
                for row in terrain_grid
            ],
            "spawn_zones": {
                "players": [[position.x, position.y] for position in player_spawns],
                "enemies": [[position.x, position.y] for position in enemy_spawns],
            },
        }
    )


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


def _template_probabilities(template: str) -> tuple[float, float, float, float]:
    key = normalize_map_key(template)
    if key == "open":
        return (0.02, 0.04, 0.04, 0.02)
    if key == "cover":
        return (0.04, 0.04, 0.16, 0.06)
    if key == "terrain":
        return (0.03, 0.22, 0.06, 0.03)
    if key == "obstacles":
        return (0.12, 0.06, 0.08, 0.08)
    return (0.06, 0.10, 0.10, 0.04)


def _spawn_column_positions(x: int, height: int) -> tuple[Position, ...]:
    middle = height // 2
    candidates = [middle, middle - 1, middle + 1, middle - 2, middle + 2]
    positions = []
    seen = set()
    for y in candidates:
        if 0 <= y < height and y not in seen:
            positions.append(Position(x, y))
            seen.add(y)
    return tuple(positions[:4])


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
