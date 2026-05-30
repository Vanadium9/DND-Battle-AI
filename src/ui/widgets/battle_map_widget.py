"""Painted tactical battle map widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

from agents import build_action_masks
from combat import CombatEnvironment, Position, Team, TerrainType


class BattleMapWidget(QWidget):
    """Draw a grid map, terrain, action highlights and creature tokens."""

    creature_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._environment: CombatEnvironment | None = None
        self._selected_creature_id: int | None = None
        self.setMinimumSize(420, 320)
        self.setMouseTracking(True)

    def set_environment(self, environment: CombatEnvironment | None) -> None:
        self._environment = environment
        self.update()

    def set_selected_creature_id(self, creature_id: int | None) -> None:
        self._selected_creature_id = creature_id
        self.update()

    def selected_creature_id(self) -> int | None:
        return self._selected_creature_id

    def sizeHint(self) -> QSize:
        return QSize(640, 440)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._environment is None:
            return
        position = self._cell_at_point(event.position().toPoint())
        if position is None:
            return
        for creature_id, creature in enumerate(self._environment.combat_state.characters):
            if creature.position == position:
                self._selected_creature_id = creature_id
                self.creature_selected.emit(creature_id)
                self.update()
                return

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#eef3f6"))
        if self._environment is None or self._environment.combat_state.grid_map is None:
            painter.setPen(QColor("#52616f"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Бой не запущен")
            return

        self._paint_grid(painter)
        self._paint_highlights(painter)
        self._paint_tokens(painter)

    def _paint_grid(self, painter: QPainter) -> None:
        state = self._environment.combat_state
        grid_map = state.grid_map
        if grid_map is None:
            return
        for y in range(grid_map.height):
            for x in range(grid_map.width):
                position = Position(x, y)
                cell_rect = self._cell_rect(position)
                painter.fillRect(cell_rect, _terrain_color(grid_map.terrain_at(position)))
                painter.setPen(QPen(QColor("#8fa2ad"), 1))
                painter.drawRect(cell_rect)

    def _paint_highlights(self, painter: QPainter) -> None:
        movement, targets = self._highlight_positions()
        for position in movement:
            painter.fillRect(self._cell_rect(position), QColor(71, 126, 247, 72))
        for position in targets:
            painter.fillRect(self._cell_rect(position), QColor(220, 53, 69, 86))

    def _paint_tokens(self, painter: QPainter) -> None:
        state = self._environment.combat_state
        active_actor_id = state.active_actor_id
        for creature_id, creature in enumerate(state.characters):
            cell_rect = self._cell_rect(creature.position)
            token_rect = cell_rect.adjusted(
                cell_rect.width() * 0.17,
                cell_rect.height() * 0.16,
                -cell_rect.width() * 0.17,
                -cell_rect.height() * 0.22,
            )
            color = _token_color(creature_id, creature.team, creature.is_alive)
            painter.setBrush(color)
            if creature_id == active_actor_id:
                painter.setPen(QPen(QColor("#f2c94c"), 4))
            elif creature_id == self._selected_creature_id:
                painter.setPen(QPen(QColor("#111827"), 3))
            else:
                painter.setPen(QPen(QColor("#ffffff"), 2))

            portrait = _portrait_pixmap(creature)
            if portrait is not None and creature.is_alive:
                path = QPainterPath()
                path.addEllipse(token_rect)
                painter.save()
                painter.setClipPath(path)
                painter.drawPixmap(token_rect.toRect(), portrait)
                painter.restore()
                painter.drawEllipse(token_rect)
            else:
                painter.drawEllipse(token_rect)
                painter.setPen(QColor("#ffffff") if creature.is_alive else QColor("#20242a"))
                painter.setFont(QFont("Segoe UI", max(8, int(token_rect.height() * 0.34)), QFont.Weight.Bold))
                painter.drawText(token_rect, Qt.AlignmentFlag.AlignCenter, _initials(creature.name))

            if not creature.is_alive:
                painter.setPen(QPen(QColor("#20242a"), 3))
                painter.drawLine(token_rect.topLeft(), token_rect.bottomRight())
                painter.drawLine(token_rect.bottomLeft(), token_rect.topRight())

            self._paint_hp_label(painter, creature.hp, creature.max_hp, cell_rect)

    def _paint_hp_label(
        self,
        painter: QPainter,
        hp: int,
        max_hp: int,
        cell_rect: QRectF,
    ) -> None:
        hp_text = f"{max(0, hp)}/{max_hp}"
        painter.setFont(QFont("Segoe UI", max(7, int(cell_rect.height() * 0.15))))
        metrics = QFontMetrics(painter.font())
        label_width = metrics.horizontalAdvance(hp_text) + 10
        label_rect = QRectF(
            cell_rect.center().x() - label_width / 2,
            cell_rect.bottom() - metrics.height() - 3,
            label_width,
            metrics.height() + 2,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 214))
        painter.drawRoundedRect(label_rect, 4, 4)
        painter.setPen(QColor("#17202a"))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, hp_text)

    def _highlight_positions(self) -> tuple[set[Position], set[Position]]:
        state = self._environment.combat_state
        grid_map = state.grid_map
        actor_id = state.active_actor_id
        if grid_map is None or actor_id is None:
            return set(), set()
        try:
            masks = build_action_masks(state, actor_id)
        except ValueError:
            return set(), set()

        movement = {
            Position(index % grid_map.width, index // grid_map.width)
            for index, allowed in enumerate(masks.get("move_index", []))
            if bool(allowed) and index < grid_map.width * grid_map.height
        }
        target_positions = {
            state.characters[index].position
            for index, allowed in enumerate(masks.get("target_index", []))
            if bool(allowed) and index < len(state.characters)
        }
        target_positions.update(
            Position(index % grid_map.width, index // grid_map.width)
            for index, allowed in enumerate(masks.get("target_cell_index", []))
            if bool(allowed) and index < grid_map.width * grid_map.height
        )
        return movement, target_positions

    def _cell_at_point(self, point: QPoint) -> Position | None:
        if self._environment is None or self._environment.combat_state.grid_map is None:
            return None
        grid_map = self._environment.combat_state.grid_map
        board_rect, cell_size = self._board_geometry()
        if not board_rect.contains(point):
            return None
        x = int((point.x() - board_rect.left()) // cell_size)
        y = int((point.y() - board_rect.top()) // cell_size)
        position = Position(x, y)
        return position if grid_map.in_bounds(position) else None

    def _cell_rect(self, position: Position) -> QRectF:
        board_rect, cell_size = self._board_geometry()
        return QRectF(
            board_rect.left() + position.x * cell_size,
            board_rect.top() + position.y * cell_size,
            cell_size,
            cell_size,
        )

    def _board_geometry(self) -> tuple[QRectF, float]:
        grid_map = self._environment.combat_state.grid_map
        width = max(1, grid_map.width)
        height = max(1, grid_map.height)
        margin = 18
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


def _terrain_color(terrain: TerrainType) -> QColor:
    return {
        TerrainType.NORMAL: QColor("#b7e68a"),
        TerrainType.BLOCKED: QColor("#8b5a2b"),
        TerrainType.DIFFICULT_TERRAIN: QColor("#c9d957"),
        TerrainType.LOW_COVER: QColor("#d9dee5"),
        TerrainType.HIGH_COVER: QColor("#5f6670"),
    }[terrain]


def _token_color(creature_id: int, team: Team, alive: bool) -> QColor:
    if not alive:
        return QColor("#6b7280")
    if team is Team.PLAYERS:
        return QColor("#2f80ed") if creature_id % 2 == 0 else QColor("#27ae60")
    return QColor("#d64545") if creature_id % 2 == 0 else QColor("#f2994a")


def _portrait_pixmap(creature: object) -> QPixmap | None:
    for attribute in ("portrait", "portrait_path", "icon", "icon_path"):
        path_value = getattr(creature, attribute, None)
        if not path_value:
            continue
        path = Path(str(path_value))
        if path.exists():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                return pixmap
    return None


def _initials(name: str) -> str:
    parts = [part for part in str(name).replace("-", " ").split() if part]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return "".join(part[0].upper() for part in parts[:2])
