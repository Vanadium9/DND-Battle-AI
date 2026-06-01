"""Read-only BattleReplay viewer screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from combat import CombatState
from combat.replay import ReplayLoadError, load_replay_file, replay_step_to_state
from ui.widgets import BattleMapWidget
from ui.widgets.screen import ScreenFrame


class _ReplayEnvironmentView:
    """Small adapter so BattleMapWidget can paint a replay CombatState."""

    def __init__(self, combat_state: CombatState) -> None:
        self.combat_state = combat_state


class ReplayViewerScreen(QWidget):
    """Navigate a saved BattleReplay without running combat or inference."""

    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._path: Path | None = None
        self._payload: dict[str, Any] | None = None
        self._step_index = 0
        self._selected_creature_id: int | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(750)
        self._timer.timeout.connect(self.next_step)

        self._summary_label = QLabel("Replay не выбран.")
        self._step_label = QLabel("")
        self._map_widget = BattleMapWidget()
        self._initiative_list = QListWidget()
        self._status_text = QPlainTextEdit()
        self._status_text.setReadOnly(True)
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._last_action_text = QPlainTextEdit()
        self._last_action_text.setReadOnly(True)

        self._back_button = QPushButton("Назад")
        self._start_button = QPushButton("В начало")
        self._prev_button = QPushButton("Предыдущий шаг")
        self._next_button = QPushButton("Следующий шаг")
        self._play_button = QPushButton("Autoplay")
        self._pause_button = QPushButton("Pause")
        self._end_button = QPushButton("В конец")

        self._build_layout()
        self._connect_signals()
        self.refresh()

    def load_replay(self, path: str | Path) -> None:
        replay_path = Path(path)
        self._summary_label.setText(f"Открытие replay: {replay_path.name}...")
        QApplication.processEvents()
        try:
            payload = load_replay_file(replay_path)
        except ReplayLoadError as error:
            QMessageBox.warning(self, "Ошибка replay", str(error))
            self._summary_label.setText("Replay не выбран.")
            return
        self._timer.stop()
        self._path = replay_path
        self._payload = payload
        self._step_index = 0
        self._selected_creature_id = None
        self.refresh()

    def refresh(self) -> None:
        if self._payload is None:
            self._summary_label.setText("Replay не выбран.")
            self._step_label.setText("")
            self._map_widget.set_environment(None)
            self._initiative_list.clear()
            self._status_text.setPlainText("")
            self._log_text.setPlainText("")
            self._last_action_text.setPlainText("")
            self._set_buttons_enabled(False)
            return

        steps = self._steps()
        if not steps:
            self._summary_label.setText(f"{self._path.name if self._path else 'Replay'}: нет шагов.")
            self._set_buttons_enabled(False)
            return

        self._step_index = max(0, min(self._step_index, len(steps) - 1))
        step = steps[self._step_index]
        state = replay_step_to_state(step)
        actor_id = int((step.get("actor") or {}).get("id", 0) or 0)
        if self._selected_creature_id is None:
            self._selected_creature_id = actor_id

        self._summary_label.setText(self._summary_text(step))
        self._step_label.setText(f"Шаг {self._step_index + 1}/{len(steps)}")
        self._map_widget.set_environment(_ReplayEnvironmentView(state))
        self._map_widget.set_selected_creature_id(self._selected_creature_id)
        self._map_widget.set_manual_click_mode(False)
        self._map_widget.set_manual_highlights(movement=set(), targets=set())
        self._refresh_initiative(step)
        self._refresh_status(step)
        self._refresh_log()
        self._refresh_last_action(step)
        self._set_buttons_enabled(True)

    def next_step(self) -> None:
        if self._payload is None:
            return
        if self._step_index >= len(self._steps()) - 1:
            self.pause()
            return
        self._step_index += 1
        self.refresh()

    def previous_step(self) -> None:
        if self._payload is None:
            return
        self._step_index = max(0, self._step_index - 1)
        self.refresh()

    def autoplay(self) -> None:
        if self._payload is None or not self._steps():
            return
        self._timer.start()
        self._set_buttons_enabled(True)

    def pause(self) -> None:
        self._timer.stop()
        self._set_buttons_enabled(self._payload is not None and bool(self._steps()))

    def go_to_start(self) -> None:
        self.pause()
        self._step_index = 0
        self.refresh()

    def go_to_end(self) -> None:
        if self._payload is None:
            return
        self.pause()
        self._step_index = max(0, len(self._steps()) - 1)
        self.refresh()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "Replay viewer",
            "Read-only просмотр BattleReplay JSON без запуска CombatEnvironment.",
        )
        self._summary_label.setWordWrap(True)
        self._step_label.setObjectName("battleActiveLabel")
        frame.content_layout.addWidget(self._summary_label)
        frame.content_layout.addWidget(self._step_label)
        frame.content_layout.addWidget(self._control_bar())

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._map_widget)
        splitter.addWidget(self._side_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        frame.content_layout.addWidget(splitter, stretch=1)
        frame.content_layout.addWidget(self._log_text, stretch=1)
        layout.addWidget(frame)

    def _control_bar(self) -> QWidget:
        bar = QFrame()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for button in (
            self._back_button,
            self._start_button,
            self._prev_button,
            self._next_button,
            self._play_button,
            self._pause_button,
            self._end_button,
        ):
            layout.addWidget(button)
        layout.addStretch(1)
        return bar

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(QLabel("Инициатива"))
        layout.addWidget(self._initiative_list, stretch=1)
        layout.addWidget(QLabel("HP / resources"))
        layout.addWidget(self._status_text, stretch=2)
        layout.addWidget(QLabel("Последнее действие"))
        layout.addWidget(self._last_action_text, stretch=2)
        return panel

    def _connect_signals(self) -> None:
        self._back_button.clicked.connect(self._back)
        self._start_button.clicked.connect(self.go_to_start)
        self._prev_button.clicked.connect(self.previous_step)
        self._next_button.clicked.connect(self.next_step)
        self._play_button.clicked.connect(self.autoplay)
        self._pause_button.clicked.connect(self.pause)
        self._end_button.clicked.connect(self.go_to_end)
        self._initiative_list.itemClicked.connect(self._select_initiative_item)
        self._map_widget.creature_selected.connect(self._select_creature)

    def _back(self) -> None:
        self.pause()
        self.back_requested.emit()

    def _steps(self) -> list[dict[str, Any]]:
        if self._payload is None:
            return []
        return [step for step in self._payload.get("steps", []) if isinstance(step, dict)]

    def _summary_text(self, step: dict[str, Any]) -> str:
        metadata = (self._payload or {}).get("metadata") or {}
        title = metadata.get("summary") or metadata.get("name") or (self._path.stem if self._path else "Replay")
        winner = (self._payload or {}).get("winner") or step.get("winner") or "-"
        return f"{title} | winner: {winner} | file: {self._path.name if self._path else '-'}"

    def _refresh_initiative(self, step: dict[str, Any]) -> None:
        self._initiative_list.clear()
        actor_id = int((step.get("actor") or {}).get("id", -1) or -1)
        for item in step.get("initiative_order") or []:
            if not isinstance(item, dict):
                continue
            creature_id = int(item.get("id", -1))
            marker = "▶ " if creature_id == actor_id else "  "
            row = QListWidgetItem(f"{marker}{creature_id}: {item.get('name') or '?'}")
            row.setData(256, creature_id)
            self._initiative_list.addItem(row)

    def _refresh_status(self, step: dict[str, Any]) -> None:
        hp_values = step.get("hp_values") or {}
        resources = step.get("resources") or {}
        lines = []
        for creature_id, hp in sorted(hp_values.items(), key=lambda item: _sort_key(item[0])):
            resource = resources.get(str(creature_id), {})
            selected = "*" if int(creature_id) == self._selected_creature_id else " "
            lines.append(
                (
                    f"{selected}[{creature_id}] {hp.get('name', 'unknown')} "
                    f"HP {hp.get('hp', '?')}/{hp.get('max_hp', '?')} "
                    f"AC {hp.get('ac', '?')} | {_resource_summary(resource)}"
                )
            )
        self._status_text.setPlainText("\n".join(lines))

    def _refresh_log(self) -> None:
        lines = []
        for index, step in enumerate(self._steps()[: self._step_index + 1], start=1):
            lines.append(f"{index}. {step.get('description', '')}")
        self._log_text.setPlainText("\n".join(lines))
        self._log_text.verticalScrollBar().setValue(self._log_text.verticalScrollBar().maximum())

    def _refresh_last_action(self, step: dict[str, Any]) -> None:
        action = step.get("action") or {}
        actor = step.get("actor") or {}
        lines = [
            f"Actor: {actor.get('name', '?')}",
            f"Action: {action.get('type', '?')} ({step.get('action_category', '?')})",
            f"Targets: {_targets_text(step.get('targets') or [])}",
            f"Damage: {_list_text(step.get('damage') or [])}",
            f"Healing: {_list_text(step.get('healing') or [])}",
            f"Dice: {_list_text(step.get('dice_rolls') or [])}",
            f"Resources: slots={_list_text(step.get('spell_slots_spent') or [])}, items={_list_text(step.get('items_spent') or [])}",
            f"Description: {step.get('description', '')}",
        ]
        self._last_action_text.setPlainText("\n".join(lines))

    def _set_buttons_enabled(self, enabled: bool) -> None:
        steps = self._steps()
        has_previous = enabled and self._step_index > 0
        has_next = enabled and self._step_index < len(steps) - 1
        self._start_button.setEnabled(has_previous)
        self._prev_button.setEnabled(has_previous)
        self._next_button.setEnabled(has_next)
        self._play_button.setEnabled(has_next and not self._timer.isActive())
        self._pause_button.setEnabled(self._timer.isActive())
        self._end_button.setEnabled(has_next)
        self._back_button.setEnabled(True)

    def _select_initiative_item(self, item: QListWidgetItem) -> None:
        self._select_creature(int(item.data(256)))

    def _select_creature(self, creature_id: int) -> None:
        self._selected_creature_id = creature_id
        self.refresh()


def _resource_summary(resource: dict[str, Any]) -> str:
    action_economy = resource.get("action_economy") or {}
    class_resources = resource.get("class_resources") or {}
    slots = resource.get("spell_slots") or {}
    slots_remaining = resource.get("spell_slots_remaining") or {}
    inventory = resource.get("inventory") or []
    action_text = (
        f"a={_bool_text(action_economy.get('action_available'))} "
        f"b={_bool_text(action_economy.get('bonus_action_available'))} "
        f"r={_bool_text(action_economy.get('reaction_available'))} "
        f"move={action_economy.get('movement_remaining', 0)}"
    )
    resources_text = ", ".join(f"{key}={value}" for key, value in class_resources.items()) or "-"
    slot_text = ", ".join(
        f"L{level} {slots_remaining.get(level, 0)}/{slots.get(level, 0)}"
        for level in sorted(set(slots) | set(slots_remaining), key=int)
    ) or "-"
    item_text = ", ".join(
        f"{item.get('name', 'item')} x{item.get('quantity', 0)}"
        for item in inventory
    ) or "-"
    return f"{action_text}; res: {resources_text}; slots: {slot_text}; items: {item_text}"


def _targets_text(targets: list[dict[str, Any]]) -> str:
    if not targets:
        return "-"
    return ", ".join(
        str(
            target.get("name")
            or target.get("type")
            or f"{target.get('x', '?')},{target.get('y', '?')}"
        )
        for target in targets
    )


def _list_text(items: list[Any]) -> str:
    if not items:
        return "-"
    return "; ".join(str(item) for item in items)


def _bool_text(value: object) -> str:
    return "Y" if bool(value) else "N"


def _sort_key(value: object) -> tuple[int, object]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))
