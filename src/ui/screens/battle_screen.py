"""Main visual battle screen."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from combat import BattleReplay, CombatAction, CombatEnvironment, EndTurnAction, Team
from inference import ActionSelectionError
from ui.services import BattleSetupResult, ModelService
from ui.widgets import (
    BattleMapWidget,
    CombatLogWidget,
    CreatureStatusPanel,
    InitiativePanel,
)
from ui.widgets.screen import ScreenFrame


DEFAULT_REPLAY_DIR = Path("replays")


class BattleScreen(QWidget):
    """Show a tactical map and step a CombatEnvironment with GUI-selected AI."""

    def __init__(
        self,
        model_service: ModelService,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model_service = model_service
        self._setup_result: BattleSetupResult | None = None
        self._environment: CombatEnvironment | None = None
        self._replay: BattleReplay | None = None
        self._selected_creature_id: int | None = None
        self._finished_manually = False
        self._winner_logged = False

        self._summary_label = QLabel("Бой не запущен.")
        self._active_label = QLabel("")
        self._map_widget = BattleMapWidget()
        self._initiative_panel = InitiativePanel()
        self._status_panel = CreatureStatusPanel()
        self._log_widget = CombatLogWidget()
        self._timer = QTimer(self)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._next_step)

        self._next_button = QPushButton("Следующий шаг")
        self._auto_button = QPushButton("Автобой")
        self._pause_button = QPushButton("Пауза")
        self._finish_button = QPushButton("Завершить бой")
        self._save_replay_button = QPushButton("Сохранить реплей")

        self._build_layout()
        self._connect_signals()
        self.refresh()

    def set_battle(self, setup_result: BattleSetupResult) -> None:
        """Attach a new environment to the screen."""

        self._timer.stop()
        self._setup_result = setup_result
        self._environment = setup_result.environment
        self._selected_creature_id = self._environment.combat_state.active_actor_id
        self._finished_manually = False
        self._winner_logged = False
        self._replay = BattleReplay(
            metadata={
                "source": "gui",
                "summary": setup_result.summary,
                "map": setup_result.map_name,
                "difficulty": setup_result.difficulty,
                "controller_mode": setup_result.controller_mode,
                "seed": setup_result.seed,
            }
        )
        self.refresh()

    def refresh(self) -> None:
        if self._environment is None or self._setup_result is None:
            self._summary_label.setText("Бой не запущен.")
            self._active_label.setText("")
            self._map_widget.set_environment(None)
            self._initiative_panel.set_environment(None)
            self._status_panel.set_environment(None)
            self._log_widget.set_entries(())
            self._set_controls_enabled(False)
            return

        if self._environment.is_done() and not self._winner_logged:
            self._log_winner()

        self._summary_label.setText(self._setup_result.summary)
        self._active_label.setText(self._state_text())
        self._map_widget.set_environment(self._environment)
        self._map_widget.set_selected_creature_id(self._selected_creature_id)
        self._initiative_panel.set_environment(
            self._environment,
            self._selected_creature_id,
        )
        self._status_panel.set_environment(
            self._environment,
            self._selected_creature_id,
        )
        self._log_widget.set_entries(self._environment.action_log[-120:])
        self._set_controls_enabled(not self._finished_manually)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "Бой",
            "Тактический экран использует CombatEnvironment.step() и AI-инференс без дублирования правил боя.",
        )
        self._summary_label.setWordWrap(True)
        self._active_label.setObjectName("battleActiveLabel")
        self._active_label.setWordWrap(True)
        frame.content_layout.addWidget(self._summary_label)
        frame.content_layout.addWidget(self._active_label)
        frame.content_layout.addWidget(self._control_bar())

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._map_widget)
        splitter.addWidget(self._side_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        frame.content_layout.addWidget(splitter, stretch=1)
        frame.content_layout.addWidget(self._log_widget, stretch=1)
        layout.addWidget(frame)

    def _control_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("battleControlBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._finish_button.setObjectName("dangerButton")
        layout.addWidget(self._next_button)
        layout.addWidget(self._auto_button)
        layout.addWidget(self._pause_button)
        layout.addWidget(self._finish_button)
        layout.addWidget(self._save_replay_button)
        layout.addStretch(1)
        return bar

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._initiative_panel, stretch=1)
        layout.addWidget(self._status_panel, stretch=2)
        return panel

    def _connect_signals(self) -> None:
        self._next_button.clicked.connect(self._next_step)
        self._auto_button.clicked.connect(self._start_auto_battle)
        self._pause_button.clicked.connect(self._pause_auto_battle)
        self._finish_button.clicked.connect(self._finish_battle)
        self._save_replay_button.clicked.connect(self._save_replay)
        self._map_widget.creature_selected.connect(self._select_creature)
        self._initiative_panel.creature_selected.connect(self._select_creature)

    def _next_step(self) -> None:
        if self._environment is None or self._setup_result is None:
            return
        if self._finished_manually:
            self._pause_auto_battle()
            self.refresh()
            return
        if self._environment.is_done():
            self._pause_auto_battle()
            self.refresh()
            return

        actor_id = self._environment.combat_state.active_actor_id
        if actor_id is None:
            self._pause_auto_battle()
            self.refresh()
            return
        if not _actor_is_ai_controlled(
            self._environment,
            actor_id,
            self._setup_result.controller_mode,
        ):
            self._environment.action_log.append(
                "Текущий участник ожидает ручного управления."
            )
            self._pause_auto_battle()
            self.refresh()
            return

        before = self._replay.snapshot_state(self._environment.combat_state) if self._replay else None
        try:
            action = self._model_service.select_action(
                self._environment.combat_state,
                actor_id,
            )
        except ActionSelectionError as error:
            self._environment.action_log.append(f"AI action error: {error}")
            self._pause_auto_battle()
            self.refresh()
            return

        self._environment.action_log.append(f"Выбранное действие: {_describe_action(action)}")
        result = self._environment.step(action)
        if self._replay is not None and before is not None:
            self._replay.record_step(before, self._environment.combat_state, action, result)
        self._selected_creature_id = self._environment.combat_state.active_actor_id
        self.refresh()

    def _start_auto_battle(self) -> None:
        if self._environment is None or self._environment.is_done():
            self.refresh()
            return
        self._timer.start()
        self.refresh()

    def _pause_auto_battle(self) -> None:
        self._timer.stop()
        self.refresh()

    def _finish_battle(self) -> None:
        if self._environment is None:
            return
        self._timer.stop()
        self._finished_manually = True
        self._environment.action_log.append("Бой завершён вручную.")
        self.refresh()

    def _save_replay(self) -> None:
        if self._replay is None:
            QMessageBox.information(self, "Реплей", "Нет активного боя для сохранения.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._replay.save(DEFAULT_REPLAY_DIR / f"gui_battle_{timestamp}.json")
        QMessageBox.information(self, "Реплей сохранён", f"Реплей сохранён: {path}")

    def _select_creature(self, creature_id: int) -> None:
        self._selected_creature_id = creature_id
        self.refresh()

    def _set_controls_enabled(self, enabled: bool) -> None:
        active = enabled and self._environment is not None
        done = active and self._environment.is_done()
        self._next_button.setEnabled(active and not done)
        self._auto_button.setEnabled(active and not done and not self._timer.isActive())
        self._pause_button.setEnabled(active and self._timer.isActive())
        self._finish_button.setEnabled(active and not done)
        self._save_replay_button.setEnabled(self._replay is not None)

    def _log_winner(self) -> None:
        if self._environment is None:
            return
        winner = self._environment.get_winner()
        winner_text = winner.value if winner is not None else "none"
        self._environment.action_log.append(f"Победитель: {winner_text}.")
        self._winner_logged = True

    def _state_text(self) -> str:
        if self._environment is None:
            return ""
        if self._finished_manually:
            return "Бой завершён вручную."
        if self._environment.is_done():
            winner = self._environment.get_winner()
            winner_text = winner.value if winner is not None else "none"
            return f"Бой завершён. Победитель: {winner_text}."
        state = self._environment.combat_state
        actor = state.active_character
        actor_text = actor.name if actor is not None else "none"
        return f"Раунд {state.round_number}. Активный участник: {actor_text}."


def _actor_is_ai_controlled(
    environment: CombatEnvironment,
    actor_id: int,
    controller_mode: str,
) -> bool:
    actor = environment.combat_state.character_at(actor_id)
    if actor is None:
        return False
    if controller_mode == "ai_all":
        return True
    if controller_mode == "ai_players":
        return actor.team is Team.PLAYERS
    if controller_mode in {"ai_enemies", "manual_players_ai_enemies"}:
        return actor.team is Team.ENEMIES
    return False


def _describe_action(action: CombatAction) -> str:
    payload = action.__class__.__name__
    target_id = getattr(action, "target_id", None)
    if target_id is not None:
        payload += f" -> target {target_id}"
    destination = getattr(action, "destination", None)
    if destination is not None:
        payload += f" -> ({destination.x}, {destination.y})"
    spell = getattr(action, "spell", None)
    if spell is not None:
        payload += f" [{spell.name}]"
    item = getattr(action, "item", None)
    if item is not None:
        payload += f" [{item.name}]"
    if isinstance(action, EndTurnAction):
        payload = "EndTurnAction"
    return payload
