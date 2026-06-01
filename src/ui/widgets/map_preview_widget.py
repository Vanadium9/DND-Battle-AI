"""Compact map preview widget for setup screens."""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from combat import MapConfig, Position, TerrainType
from ui.widgets.battle_map_widget import terrain_qcolor


class MapPreviewWidget(QWidget):
    """Draw a read-only terrain grid and configured spawn zones."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._map_config: MapConfig | None = None
        self._error_text = ""
        self.setMinimumSize(260, 190)

    def set_map_config(self, map_config: MapConfig | None) -> None:
        self._map_config = map_config
        self._error_text = ""
        self.update()

    def set_error(self, message: str) -> None:
        self._map_config = None
        self._error_text = message
        self.update()

    def map_config(self) -> MapConfig | None:
        return self._map_config

    def sizeHint(self) -> QSize:
        return QSize(360, 240)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#eef3f6"))
        if self._map_config is None:
            painter.setPen(QColor("#52616f"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._error_text or "Карта не выбрана",
            )
            return

        self._paint_grid(painter)
        self._paint_spawn_zones(painter)

    def _paint_grid(self, painter: QPainter) -> None:
        if self._map_config is None:
            return
        for y in range(self._map_config.height):
            for x in range(self._map_config.width):
                position = Position(x, y)
                cell_rect = self._cell_rect(position)
                painter.fillRect(cell_rect, terrain_qcolor(self._map_config.terrain_at(position)))
                painter.setPen(QPen(QColor("#8fa2ad"), 1))
                painter.drawRect(cell_rect)

    def _paint_spawn_zones(self, painter: QPainter) -> None:
        if self._map_config is None:
            return
        self._paint_spawn_cells(
            painter,
            self._map_config.spawn_zones.players,
            QColor(47, 128, 237, 118),
            "P",
        )
        self._paint_spawn_cells(
            painter,
            self._map_config.spawn_zones.enemies,
            QColor(214, 69, 69, 118),
            "E",
        )

    def _paint_spawn_cells(
        self,
        painter: QPainter,
        positions: tuple[Position, ...],
        color: QColor,
        label: str,
    ) -> None:
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        for position in positions:
            cell_rect = self._cell_rect(position).adjusted(2, 2, -2, -2)
            painter.fillRect(cell_rect, color)
            painter.setPen(QColor("#17202a"))
            painter.drawText(cell_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _cell_rect(self, position: Position) -> QRectF:
        board_rect, cell_size = self._board_geometry()
        return QRectF(
            board_rect.left() + position.x * cell_size,
            board_rect.top() + position.y * cell_size,
            cell_size,
            cell_size,
        )

    def _board_geometry(self) -> tuple[QRectF, float]:
        if self._map_config is None:
            return QRectF(), 1.0
        width = max(1, self._map_config.width)
        height = max(1, self._map_config.height)
        margin = 12
        available_width = max(1, self.width() - margin * 2)
        available_height = max(1, self.height() - margin * 2)
        cell_size = min(available_width / width, available_height / height)
        board_width = cell_size * width
        board_height = cell_size * height
        return (
            QRectF(
                (self.width() - board_width) / 2,
                (self.height() - board_height) / 2,
                board_width,
                board_height,
            ),
            cell_size,
        )


def terrain_preview_color(terrain: TerrainType) -> QColor:
    """Public helper for tests and legend widgets."""

    return terrain_qcolor(terrain)
