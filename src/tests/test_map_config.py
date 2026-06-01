import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from combat import (
    MapConfigValidationError,
    Position,
    TerrainType,
    load_map_config,
)


def test_map_config_loads_from_json() -> None:
    config = load_map_config(Path("maps") / "cover_arena.json")

    assert config.name == "cover_arena"
    assert config.width == 8
    assert config.height == 5
    assert config.terrain_at(Position(5, 0)) is TerrainType.LOW_COVER
    assert config.spawn_zones.players[0] == Position(0, 1)


def test_invalid_map_config_raises_validation_error() -> None:
    path = _map_path("invalid_blocked_spawn")
    path.write_text(
        json.dumps(
            {
                "name": "invalid",
                "width": 2,
                "height": 2,
                "terrain_grid": [
                    ["BLOCKED", "NORMAL"],
                    ["NORMAL", "NORMAL"],
                ],
                "spawn_zones": {
                    "players": [[0, 0]],
                    "enemies": [[1, 1]],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MapConfigValidationError, match="blocked"):
        load_map_config(path)


def test_map_preview_widget_receives_config() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from ui.app import create_app
    from ui.widgets import MapPreviewWidget, terrain_preview_color

    app = create_app(["test_map_preview_widget"])
    config = load_map_config(Path("maps") / "open_field.json")
    widget = MapPreviewWidget()

    widget.set_map_config(config)

    assert widget.map_config() is config
    assert terrain_preview_color(TerrainType.NORMAL).name() == "#b7e68a"

    widget.close()
    app.processEvents()


def _map_path(prefix: str) -> Path:
    directory = Path("checkpoints") / "test_map_config"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{uuid4().hex}.json"
