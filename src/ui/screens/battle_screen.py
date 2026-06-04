"""Main visual battle screen."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QAbstractAnimation, Qt, QTimer, QVariantAnimation
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from combat import (
    ActionResult,
    BattleReplay,
    CombatAction,
    CombatEnvironment,
    EndTurnAction,
    Position,
    Team,
)
from inference import ActionSelectionError
from ui.animations import (
    BattleAnimationFrame,
    animation_duration_ms,
    build_battle_animations,
    normalize_animation_speed,
)
from ui.services import (
    BattleSetupResult,
    ManualActionBuilder,
    ManualActionOption,
    ManualActionPlan,
    ManualTargetMode,
    ModelService,
)
from ui.settings import normalize_autobattle_delay
from ui.text import ru_label, ru_sentence
from ui.widgets import (
    ActionPanel,
    BattleMapWidget,
    CombatLogWidget,
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
        self._manual_action_builder = ManualActionBuilder()
        self._manual_plan: ManualActionPlan | None = None
        self._pending_manual_option: ManualActionOption | None = None
        self._finished_manually = False
        self._winner_logged = False
        self._animation_running = False

        self._summary_label = QLabel("Бой не запущен.")
        self._active_label = QLabel("")
        self._map_widget = BattleMapWidget()
        self._initiative_panel = InitiativePanel()
        self._action_panel = ActionPanel()
        self._combat_resources_label = QLabel("")
        self._combat_resources_label.setWordWrap(True)
        self._combat_resources_label.setTextFormat(Qt.TextFormat.RichText)
        self._combat_resources_label.setObjectName("battleResourcesLabel")
        self._log_widget = CombatLogWidget()
        self._timer = QTimer(self)
        self._timer.setInterval(self._autobattle_delay_ms())
        self._timer.timeout.connect(self._next_step)
        self._animation = QVariantAnimation(self)
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._animation.valueChanged.connect(self._on_animation_value_changed)
        self._animation.finished.connect(self._on_animation_finished)
        self._active_animations = ()

        self._next_button = QPushButton("Следующий шаг")
        self._auto_button = QPushButton("Автобой")
        self._pause_button = QPushButton("Пауза")
        self._finish_button = QPushButton("Завершить бой")
        self._save_replay_button = QPushButton("Сохранить реплей")
        self._animations_checkbox = QCheckBox("Анимации")
        self._animations_checkbox.setChecked(self._model_service.settings.animations_enabled)
        self._animation_speed_spin = QSpinBox()
        self._animation_speed_spin.setRange(300, 1500)
        self._animation_speed_spin.setSingleStep(100)
        self._animation_speed_spin.setSuffix(" мс")
        self._animation_speed_spin.setValue(self._animation_speed_ms())

        self._build_layout()
        self._connect_signals()
        self.refresh()

    def set_battle(self, setup_result: BattleSetupResult) -> None:
        """Attach a new environment to the screen."""

        self._timer.stop()
        self._stop_animation(clear=True)
        self._setup_result = setup_result
        self._environment = setup_result.environment
        self._selected_creature_id = self._environment.combat_state.active_actor_id
        self._pending_manual_option = None
        self._manual_plan = None
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
            self._map_widget.set_manual_click_mode(False)
            self._map_widget.set_manual_highlights()
            self._initiative_panel.set_environment(None)
            self._action_panel.set_viewer_mode("Бой не запущен.")
            self._combat_resources_label.setText("")
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
        self._log_widget.set_entries(self._environment.action_log[-120:])
        self._combat_resources_label.setText(_active_resources_text(self._environment))
        self._sync_manual_controls()
        self._set_controls_enabled(not self._finished_manually)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
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
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 300])
        frame.content_layout.addWidget(splitter, stretch=1)
        frame.content_layout.addWidget(self._bottom_panel(), stretch=0)
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
        layout.addSpacing(16)
        layout.addWidget(self._animations_checkbox)
        layout.addWidget(QLabel("Задержка"))
        layout.addWidget(self._animation_speed_spin)
        layout.addStretch(1)
        return bar

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._initiative_panel, stretch=1)
        return panel

    def _bottom_panel(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        action_column = QWidget()
        action_layout = QVBoxLayout(action_column)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        action_layout.addWidget(self._combat_resources_panel(), stretch=0)
        action_layout.addWidget(self._action_panel, stretch=1)

        layout.addWidget(action_column, stretch=3)
        layout.addWidget(self._log_widget, stretch=2)
        return panel

    def _combat_resources_panel(self) -> QFrame:
        group = QFrame()
        group.setObjectName("battleResourcesStrip")
        group.setMaximumHeight(46)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._combat_resources_label)
        return group

    def _connect_signals(self) -> None:
        self._next_button.clicked.connect(self._next_step)
        self._auto_button.clicked.connect(self._start_auto_battle)
        self._pause_button.clicked.connect(self._pause_auto_battle)
        self._finish_button.clicked.connect(self._finish_battle)
        self._save_replay_button.clicked.connect(self._save_replay)
        self._animations_checkbox.stateChanged.connect(self._save_animation_settings)
        self._animation_speed_spin.valueChanged.connect(self._save_animation_settings)
        self._action_panel.option_selected.connect(self._select_manual_option)
        self._map_widget.cell_clicked.connect(self._handle_manual_cell_clicked)
        self._map_widget.creature_selected.connect(self._select_creature)
        self._initiative_panel.creature_selected.connect(self._select_creature)

    def _next_step(self) -> None:
        if self._animation_running:
            return
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
            QMessageBox.warning(self, "Невозможно выполнить действие", str(error))
            self._pause_auto_battle()
            self.refresh()
            return

        self._environment.action_log.append(f"Выбранное действие: {_describe_action(action)}")
        result = self._environment.step(action)
        if self._replay is not None and before is not None:
            self._replay.record_step(before, self._environment.combat_state, action, result)
        self._selected_creature_id = self._environment.combat_state.active_actor_id
        self.refresh()
        self._play_step_animation(before, action, result)

    def _start_auto_battle(self) -> None:
        if self._environment is None or self._environment.is_done():
            self.refresh()
            return
        self._timer.setInterval(self._autobattle_delay_ms())
        self._timer.start()
        self.refresh()

    def _pause_auto_battle(self) -> None:
        self._timer.stop()
        self.refresh()

    def _finish_battle(self) -> None:
        if self._environment is None:
            return
        if not self._environment.is_done():
            answer = QMessageBox.question(
                self,
                "Завершить активный бой",
                "Завершить текущий бой без доигрывания?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._timer.stop()
        self._stop_animation(clear=True)
        self._finished_manually = True
        self._environment.action_log.append("Бой завершён вручную.")
        self.refresh()

    def _save_replay(self) -> None:
        if self._replay is None:
            QMessageBox.information(self, "Реплей", "Нет активного боя для сохранения.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_dir = Path(self._model_service.settings.replay_dir or DEFAULT_REPLAY_DIR)
        path = self._replay.save(replay_dir / f"gui_battle_{timestamp}.json")
        QMessageBox.information(self, "Реплей сохранён", f"Реплей сохранён: {path}")

    def _select_creature(self, creature_id: int) -> None:
        self._selected_creature_id = creature_id
        self.refresh()

    def _set_controls_enabled(self, enabled: bool) -> None:
        active = enabled and self._environment is not None
        done = active and self._environment.is_done()
        manual_current = active and self._active_actor_is_manual()
        can_step = active and not done and not self._animation_running and not manual_current
        self._next_button.setEnabled(can_step)
        self._auto_button.setEnabled(can_step and not self._timer.isActive())
        self._pause_button.setEnabled(active and self._timer.isActive())
        self._finish_button.setEnabled(active and not done)
        self._save_replay_button.setEnabled(self._replay is not None)
        self._animations_checkbox.setEnabled(True)
        self._animation_speed_spin.setEnabled(True)

    def _sync_manual_controls(self) -> None:
        if self._environment is None or self._setup_result is None:
            return
        actor_id = self._environment.combat_state.active_actor_id
        if actor_id is None or self._environment.is_done() or not self._active_actor_is_manual():
            self._manual_plan = None
            self._pending_manual_option = None
            self._action_panel.set_viewer_mode()
            self._map_widget.set_manual_click_mode(False)
            self._map_widget.set_manual_highlights()
            return

        self._manual_plan = self._manual_action_builder.build_plan(
            self._environment.combat_state,
            actor_id,
        )
        if self._pending_manual_option is not None:
            current_ids = {option.id for option in self._manual_plan.options}
            if self._pending_manual_option.id not in current_ids:
                self._pending_manual_option = None
        self._action_panel.set_plan(self._manual_plan, self._pending_manual_option)
        self._map_widget.set_manual_click_mode(
            self._pending_manual_option is not None
            and self._pending_manual_option.target_mode is not ManualTargetMode.NONE
        )
        movement, targets = self._manual_highlights()
        self._map_widget.set_manual_highlights(movement=movement, targets=targets)

    def _select_manual_option(self, option: ManualActionOption) -> None:
        if self._environment is None or not self._active_actor_is_manual():
            return
        actor_id = self._environment.combat_state.active_actor_id
        if actor_id is None:
            return
        if option.target_mode is ManualTargetMode.NONE:
            try:
                action = self._manual_action_builder.build_action(
                    self._environment.combat_state,
                    actor_id,
                    option,
                )
            except ValueError as error:
                self._environment.action_log.append(f"Manual action error: {error}")
                QMessageBox.warning(self, "Невозможно выполнить действие", str(error))
                self.refresh()
                return
            self._pending_manual_option = None
            self._execute_selected_action(action, prefix="Ручное действие")
            return

        self._pending_manual_option = option
        self._sync_manual_controls()

    def _handle_manual_cell_clicked(self, position: object) -> None:
        if (
            self._environment is None
            or self._pending_manual_option is None
            or not isinstance(position, Position)
        ):
            return
        actor_id = self._environment.combat_state.active_actor_id
        if actor_id is None or not self._active_actor_is_manual():
            return
        target_id = None
        if self._pending_manual_option.target_mode is ManualTargetMode.CREATURE:
            target_id = self._creature_id_at(
                position,
                allowed_target_ids=self._pending_manual_option.target_ids,
            )
        try:
            action = self._manual_action_builder.build_action(
                self._environment.combat_state,
                actor_id,
                self._pending_manual_option,
                target_id=target_id,
                target_cell=position,
            )
        except ValueError as error:
            self._environment.action_log.append(f"Manual action error: {error}")
            QMessageBox.warning(self, "Невозможно выполнить действие", str(error))
            self.refresh()
            return

        self._pending_manual_option = None
        self._execute_selected_action(action, prefix="Ручное действие")

    def _execute_selected_action(self, action: CombatAction, *, prefix: str) -> None:
        if self._environment is None:
            return
        before = self._replay.snapshot_state(self._environment.combat_state) if self._replay else None
        self._environment.action_log.append(f"{prefix}: {_describe_action(action)}")
        previous_auto_end_turn = self._environment.auto_end_turn_enabled
        self._environment.auto_end_turn_enabled = isinstance(action, EndTurnAction)
        try:
            result = self._environment.step(action)
        finally:
            self._environment.auto_end_turn_enabled = previous_auto_end_turn
        if self._replay is not None and before is not None:
            self._replay.record_step(before, self._environment.combat_state, action, result)
        self._selected_creature_id = self._environment.combat_state.active_actor_id
        self.refresh()
        self._play_step_animation(before, action, result)

    def _manual_highlights(self) -> tuple[set[Position] | None, set[Position] | None]:
        if self._environment is None or self._pending_manual_option is None:
            return None, None
        option = self._pending_manual_option
        if option.target_mode is ManualTargetMode.CELL:
            cells = set(option.target_cells)
            if option.metadata.get("kind") == "move":
                return cells, set()
            return set(), cells
        if option.target_mode is ManualTargetMode.CREATURE:
            targets = {
                self._environment.combat_state.characters[target_id].position
                for target_id in option.target_ids
                if 0 <= target_id < len(self._environment.combat_state.characters)
            }
            return set(), targets
        return None, None

    def _creature_id_at(
        self,
        position: Position,
        *,
        allowed_target_ids: tuple[int, ...] = (),
    ) -> int | None:
        if self._environment is None:
            return None
        allowed = set(allowed_target_ids)
        matching = [
            creature_id
            for creature_id, creature in enumerate(self._environment.combat_state.characters)
            if creature.position == position and (not allowed or creature_id in allowed)
        ]
        if not matching:
            return None
        alive = [
            creature_id
            for creature_id in matching
            if self._environment.combat_state.characters[creature_id].is_alive
        ]
        return alive[-1] if alive else matching[-1]

    def _active_actor_is_manual(self) -> bool:
        if self._environment is None or self._setup_result is None:
            return False
        actor_id = self._environment.combat_state.active_actor_id
        if actor_id is None:
            return False
        return _actor_is_manual_controlled(
            self._environment,
            actor_id,
            self._setup_result.controller_mode,
        )

    def _play_step_animation(
        self,
        before: dict[str, object] | None,
        action: CombatAction,
        result: ActionResult,
    ) -> None:
        if self._environment is None:
            return
        self._stop_animation(clear=True)
        if not self._animations_checkbox.isChecked():
            return
        animations = build_battle_animations(
            before,
            self._environment.combat_state,
            action,
            result,
        )
        if not animations:
            return
        self._active_animations = animations
        self._animation_running = True
        self._animation.setDuration(animation_duration_ms(self._animation_speed_ms()))
        self._map_widget.set_animation_frame(
            BattleAnimationFrame(animations=animations, progress=0.0)
        )
        self._set_controls_enabled(not self._finished_manually)
        self._animation.start()

    def _on_animation_value_changed(self, value: object) -> None:
        if not self._animation_running:
            return
        try:
            progress = float(value)
        except (TypeError, ValueError):
            progress = 1.0
        self._map_widget.set_animation_frame(
            BattleAnimationFrame(
                animations=self._active_animations,
                progress=progress,
            )
        )

    def _on_animation_finished(self) -> None:
        self._animation_running = False
        self._active_animations = ()
        self._map_widget.clear_animation()
        self._set_controls_enabled(not self._finished_manually)

    def _stop_animation(self, *, clear: bool) -> None:
        if self._animation.state() == QAbstractAnimation.State.Running:
            self._animation.stop()
        self._animation_running = False
        self._active_animations = ()
        if clear:
            self._map_widget.clear_animation()

    def _save_animation_settings(self) -> None:
        speed = self._animation_speed_ms()
        self._model_service.set_settings(
            animations_enabled=self._animations_checkbox.isChecked(),
            animation_speed=speed,
            autobattle_delay=speed,
        )
        self._timer.setInterval(self._autobattle_delay_ms())
        if not self._animations_checkbox.isChecked():
            self._stop_animation(clear=True)

    def _animation_speed_ms(self) -> int:
        if hasattr(self, "_animation_speed_spin"):
            return normalize_animation_speed(self._animation_speed_spin.value())
        return normalize_animation_speed(self._model_service.settings.animation_speed)

    def _autobattle_delay_ms(self) -> int:
        return normalize_autobattle_delay(self._model_service.settings.autobattle_delay)

    def _log_winner(self) -> None:
        if self._environment is None:
            return
        winner = self._environment.get_winner()
        winner_text = ru_label(winner.value) if winner is not None else "нет"
        self._environment.action_log.append(f"Победитель: {winner_text}.")
        self._winner_logged = True

    def _state_text(self) -> str:
        if self._environment is None:
            return ""
        if self._finished_manually:
            return "Бой завершён вручную."
        if self._environment.is_done():
            winner = self._environment.get_winner()
            winner_text = ru_label(winner.value) if winner is not None else "нет"
            return f"Бой завершён. Победитель: {winner_text}."
        state = self._environment.combat_state
        actor = state.active_character
        actor_text = ru_sentence(actor.name) if actor is not None else "none"
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


def _active_resources_text(environment: CombatEnvironment) -> str:
    state = environment.combat_state
    actor = state.active_character
    if actor is None:
        return "<span style='color:#6b7280;'>Нет активного участника</span>"

    economy = actor.action_economy
    markers = [
        _resource_marker("●", "Д", economy.action_available, "#2f9e44"),
        _resource_marker("■", "Б", economy.bonus_action_available, "#f08c00"),
        _resource_marker("◆", "Р", economy.reaction_available, "#1c7ed6"),
        _resource_marker(
            "▲",
            f"{economy.movement_remaining}/{actor.speed}",
            economy.movement_remaining > 0,
            "#0ca678",
        ),
    ]

    spell_slots = getattr(actor, "spell_slots", {}) or {}
    if spell_slots:
        remaining = getattr(actor, "spell_slots_remaining", {}) or {}
        for level in sorted(spell_slots):
            current = remaining.get(level, 0)
            maximum = spell_slots[level]
            markers.append(_resource_marker("◇", f"{level}:{current}/{maximum}", current > 0, "#7048e8"))

    resources = getattr(actor, "resources", {}) or {}
    if resources:
        for name, resource in resources.items():
            current = getattr(resource, "uses_remaining", resource)
            maximum = getattr(resource, "max_uses", current)
            label = f"{ru_sentence(name)} {current}/{maximum}"
            markers.append(_resource_marker("⬟", label, int(current) > 0, "#9c36b5"))

    prepared_action = getattr(actor, "prepared_action", None)
    prepared = ""
    if prepared_action:
        prepared = (
            "<span style='color:#334155; font-size:11px;'>"
            f"Подготовлено: {ru_sentence(prepared_action)}</span>"
        )
    return (
        "<div style='font-size:12px; line-height:1.35;'>"
        f"<div>{' '.join(markers)}</div>"
        f"{prepared}"
        "</div>"
    )


def _resource_marker(symbol: str, label: str, available: bool, color: str) -> str:
    marker_color = color if available else "#a8b0ba"
    label_color = "#253549" if available else "#7b8794"
    return (
        f"<span style='color:{marker_color}; font-size:16px; font-weight:800;'>{symbol}</span>"
        f"<span style='color:{label_color}; font-size:11px; font-weight:700;'>{label}</span>"
    )


def _actor_is_manual_controlled(
    environment: CombatEnvironment,
    actor_id: int,
    controller_mode: str,
) -> bool:
    actor = environment.combat_state.character_at(actor_id)
    if actor is None:
        return False
    if controller_mode == "manual_players_ai_enemies":
        return actor.team is Team.PLAYERS
    return False


def _describe_action(action: CombatAction) -> str:
    payload = ru_label(action.__class__.__name__)
    target_id = getattr(action, "target_id", None)
    if target_id is not None:
        payload += f" -> цель {target_id}"
    destination = getattr(action, "destination", None)
    if destination is not None:
        payload += f" -> клетка ({destination.x}, {destination.y})"
    spell = getattr(action, "spell", None)
    if spell is not None:
        payload += f" [{ru_sentence(spell.name)}]"
    item = getattr(action, "item", None)
    if item is not None:
        payload += f" [{ru_sentence(item.name)}]"
    if isinstance(action, EndTurnAction):
        payload = ru_label("EndTurnAction")
    return payload
