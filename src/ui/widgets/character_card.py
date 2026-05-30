"""Character card widget for the GUI character list."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from character import InternalCharacter


class CharacterCard(QFrame):
    """Compact card for one stored InternalCharacter."""

    view_requested = Signal(str)
    edit_requested = Signal(str)
    duplicate_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, character: InternalCharacter, parent=None) -> None:
        super().__init__(parent)
        self.character = character
        self.setObjectName("characterCard")
        self._build_layout()

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        title = QLabel(self.character.name)
        title.setObjectName("characterCardTitle")
        root.addWidget(title)

        details = QGridLayout()
        details.setHorizontalSpacing(18)
        details.setVerticalSpacing(6)
        rows = (
            ("Раса", self.character.race_name or "не указана"),
            ("Класс", self._class_text()),
            ("Уровень", str(self.character.level)),
            ("HP", str(self.character.hp)),
            ("AC", str(self.character.ac)),
            ("Оружие", self._main_weapon()),
            ("Роль", self.character.role or "не указана"),
        )
        for row_index, (label, value) in enumerate(rows):
            label_widget = QLabel(f"{label}:")
            label_widget.setObjectName("characterCardMetaLabel")
            value_widget = QLabel(value)
            value_widget.setObjectName("characterCardMetaValue")
            value_widget.setWordWrap(True)
            details.addWidget(label_widget, row_index, 0)
            details.addWidget(value_widget, row_index, 1)
        root.addLayout(details)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        view_button = QPushButton("Просмотр")
        edit_button = QPushButton("Редактировать")
        duplicate_button = QPushButton("Дублировать")
        delete_button = QPushButton("Удалить")
        delete_button.setObjectName("dangerButton")

        view_button.clicked.connect(lambda: self.view_requested.emit(self.character.id))
        edit_button.clicked.connect(lambda: self.edit_requested.emit(self.character.id))
        duplicate_button.clicked.connect(
            lambda: self.duplicate_requested.emit(self.character.id)
        )
        delete_button.clicked.connect(lambda: self.delete_requested.emit(self.character.id))

        button_row.addWidget(view_button)
        button_row.addWidget(edit_button)
        button_row.addWidget(duplicate_button)
        button_row.addWidget(delete_button)
        button_row.addStretch(1)
        root.addLayout(button_row)

    def _class_text(self) -> str:
        class_name = self.character.class_name or "не указан"
        if self.character.subclass_name:
            return f"{class_name} / {self.character.subclass_name}"
        return class_name

    def _main_weapon(self) -> str:
        if not self.character.weapons:
            return "нет"
        first_weapon = self.character.weapons[0]
        return str(first_weapon.get("name", "оружие"))
