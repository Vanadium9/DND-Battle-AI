"""Replay list screen for saved BattleReplay JSON files."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from combat.replay import ReplaySummary, list_replay_summaries
from ui.widgets.screen import ScreenFrame


DEFAULT_REPLAY_DIR = Path("replays")


class ReplayListScreen(QWidget):
    """Show saved replay files and file management actions."""

    open_requested = Signal(object)

    def __init__(
        self,
        replay_dir: str | Path = DEFAULT_REPLAY_DIR,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.replay_dir = Path(replay_dir)
        self._summaries: list[ReplaySummary] = []
        self._table = QTableWidget(0, 6)
        self._open_button = QPushButton("Открыть")
        self._delete_button = QPushButton("Удалить")
        self._rename_button = QPushButton("Переименовать")
        self._empty_label = QLabel("Реплеи ещё не сохранены")
        self._status_label = QLabel()
        self._build_layout()
        self._connect_signals()
        self.refresh()

    def refresh(self) -> None:
        self._summaries = list_replay_summaries(self.replay_dir)
        self._empty_label.setVisible(not self._summaries)
        self._table.setVisible(bool(self._summaries))
        self._table.setRowCount(len(self._summaries))
        for row, summary in enumerate(self._summaries):
            values = (
                summary.modified_at,
                summary.display_name,
                summary.winner or "-",
                str(summary.round_count),
                ", ".join(summary.participants) or "-",
                str(summary.step_count),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, str(summary.path))
                if column in {2, 3, 5}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, column, item)
        self._table.resizeColumnsToContents()
        self._update_buttons()

    def selected_path(self) -> Path | None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._summaries):
            return None
        return self._summaries[row].path

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        frame = ScreenFrame(
            "Реплеи",
            "Просмотр сохранённых BattleReplay JSON из папки replays/.",
        )
        self._empty_label.setObjectName("emptyStateLabel")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame.content_layout.addWidget(self._empty_label)
        self._table.setHorizontalHeaderLabels(
            ("Дата", "Название боя", "Победитель", "Раунды", "Участники", "Шаги")
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        frame.content_layout.addWidget(self._table, stretch=1)

        button_row = QHBoxLayout()
        button_row.addWidget(self._open_button)
        button_row.addWidget(self._rename_button)
        button_row.addWidget(self._delete_button)
        button_row.addStretch(1)
        frame.content_layout.addLayout(button_row)
        self._status_label.setObjectName("inlineStatus")
        frame.content_layout.addWidget(self._status_label)
        layout.addWidget(frame)

    def _connect_signals(self) -> None:
        self._table.itemSelectionChanged.connect(self._update_buttons)
        self._table.itemDoubleClicked.connect(lambda _item: self._open_selected())
        self._open_button.clicked.connect(self._open_selected)
        self._delete_button.clicked.connect(self._delete_selected)
        self._rename_button.clicked.connect(self._rename_selected)

    def _open_selected(self) -> None:
        path = self.selected_path()
        if path is not None:
            self._status_label.setText(f"Открытие replay: {path.name}...")
            QApplication.processEvents()
            self.open_requested.emit(path)
            self._status_label.setText(f"Replay выбран: {path.name}")

    def _delete_selected(self) -> None:
        path = self.selected_path()
        if path is None:
            return
        answer = QMessageBox.question(
            self,
            "Удалить replay",
            f"Удалить replay {path.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            path.unlink()
        except OSError as error:
            QMessageBox.warning(self, "Ошибка удаления", str(error))
        self.refresh()

    def _rename_selected(self) -> None:
        path = self.selected_path()
        if path is None:
            return
        new_name, accepted = QInputDialog.getText(
            self,
            "Переименовать replay",
            "Новое имя файла:",
            text=path.stem,
        )
        if not accepted:
            return
        normalized = new_name.strip()
        if not normalized:
            return
        try:
            new_path = path.with_name(f"{normalized}.json")
        except ValueError:
            QMessageBox.warning(self, "Ошибка переименования", "Некорректное имя файла.")
            return
        if new_path == path:
            return
        if new_path.exists():
            QMessageBox.warning(self, "Ошибка переименования", "Файл уже существует.")
            return
        try:
            path.rename(new_path)
        except OSError as error:
            QMessageBox.warning(self, "Ошибка переименования", str(error))
        self.refresh()

    def _update_buttons(self) -> None:
        has_selection = self.selected_path() is not None
        self._open_button.setEnabled(has_selection)
        self._delete_button.setEnabled(has_selection)
        self._rename_button.setEnabled(has_selection)
