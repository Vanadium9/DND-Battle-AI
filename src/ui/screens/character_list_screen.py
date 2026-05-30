"""Character list screen backed by CharacterRepository."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from character import CharacterRepository, InternalCharacter
from ui.widgets.character_card import CharacterCard
from ui.widgets.screen import ScreenFrame


class CharacterListScreen(QWidget):
    """Screen that lists, opens and manages stored GUI characters."""

    create_requested = Signal()
    edit_requested = Signal(str)

    def __init__(
        self,
        repository: CharacterRepository | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repository = repository or CharacterRepository()
        self._cards: list[CharacterCard] = []
        self._build_layout()
        self.refresh()

    def refresh(self) -> None:
        """Reload characters from the repository."""

        self._clear_cards()
        characters = self.repository.list_characters()
        self._empty_label.setVisible(not characters)
        for character in characters:
            self._add_character_card(character)
        self._card_layout.addStretch(1)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(0)

        frame = ScreenFrame(
            "Персонажи",
            "Сохранённые персонажи из локального CharacterRepository.",
        )
        root.addWidget(frame)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        create_button = QPushButton("Создать персонажа")
        refresh_button = QPushButton("Обновить")
        create_button.clicked.connect(self.create_requested.emit)
        refresh_button.clicked.connect(self.refresh)
        toolbar.addWidget(create_button)
        toolbar.addWidget(refresh_button)
        toolbar.addStretch(1)
        frame.content_layout.addLayout(toolbar)

        self._empty_label = QLabel("Персонажи ещё не созданы")
        self._empty_label.setObjectName("emptyStateLabel")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame.content_layout.addWidget(self._empty_label)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll_content = QWidget()
        self._card_layout = QVBoxLayout(self._scroll_content)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(12)
        self._scroll_area.setWidget(self._scroll_content)
        frame.content_layout.addWidget(self._scroll_area, stretch=1)

    def _add_character_card(self, character: InternalCharacter) -> None:
        card = CharacterCard(character)
        card.view_requested.connect(self._show_character_details)
        card.edit_requested.connect(self.edit_requested.emit)
        card.duplicate_requested.connect(self._duplicate_character)
        card.delete_requested.connect(self._confirm_delete_character)
        self._card_layout.addWidget(card)
        self._cards.append(card)

    def _clear_cards(self) -> None:
        while self._card_layout.count():
            item = self._card_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards = []

    def _show_character_details(self, character_id: str) -> None:
        character = self.repository.get_character(character_id)
        if character is None:
            QMessageBox.warning(self, "Персонаж не найден", "Запись уже удалена.")
            self.refresh()
            return
        CharacterDetailsDialog(character, self).exec()

    def _duplicate_character(self, character_id: str) -> None:
        self.repository.duplicate_character(character_id)
        self.refresh()

    def _confirm_delete_character(self, character_id: str) -> None:
        character = self.repository.get_character(character_id)
        if character is None:
            self.refresh()
            return
        answer = QMessageBox.question(
            self,
            "Удалить персонажа",
            f"Удалить персонажа «{character.name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.repository.delete_character(character_id)
        self.refresh()


class CharacterDetailsDialog(QDialog):
    """Read-only detailed character view."""

    def __init__(
        self,
        character: InternalCharacter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(character.name)
        self.resize(720, 640)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel(character.name)
        title.setObjectName("screenTitle")
        layout.addWidget(title)

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(format_character_details(character))
        layout.addWidget(text, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def format_character_details(character: InternalCharacter) -> str:
    """Return detailed character information for the read-only viewer."""

    sections = [
        (
            "Основное",
            [
                f"ID: {character.id}",
                f"Имя: {character.name}",
                f"Раса: {character.race_name or 'не указана'}",
                f"Класс: {_class_text(character)}",
                f"Уровень: {character.level}",
                f"Опыт: {character.experience}",
                f"Роль: {character.role or 'не указана'}",
                f"HP: {character.hp}",
                f"AC: {character.ac}",
                f"Скорость: {character.speed}",
                f"Бонус мастерства: {character.proficiency_bonus}",
            ],
        ),
        ("Характеристики", _format_mapping(character.stats)),
        ("Владения", _format_proficiencies(character)),
        ("Оружие", _format_mapping_list(character.weapons)),
        ("Броня", _format_mapping(character.armor)),
        ("Классовые способности", _format_sequence(character.class_features)),
        ("Расовые особенности", _format_mapping(character.race_traits)),
        ("Черты", _format_sequence(character.feats)),
        ("Заклинания", _format_spell_list(character.spells)),
        ("Подготовленные заклинания", _format_sequence(character.prepared_spells)),
        ("Ячейки заклинаний", _format_mapping(character.spell_slots)),
        ("Ресурсы", _format_mapping(character.resources)),
        ("Инвентарь", _format_mapping_list(character.inventory)),
        ("Сопротивления", _format_sequence(character.resistances)),
        ("Иммунитеты", _format_sequence(character.immunities)),
        ("Уязвимости", _format_sequence(character.vulnerabilities)),
    ]

    lines: list[str] = []
    for title, body_lines in sections:
        lines.append(title)
        lines.append("-" * len(title))
        lines.extend(body_lines or ["нет данных"])
        lines.append("")
    return "\n".join(lines).strip()


def _class_text(character: InternalCharacter) -> str:
    if character.subclass_name:
        return f"{character.class_name} / {character.subclass_name}"
    return character.class_name or "не указан"


def _format_mapping(values: dict[str, Any]) -> list[str]:
    if not values:
        return []
    return [f"{key}: {value}" for key, value in sorted(values.items())]


def _format_sequence(values: tuple[str, ...]) -> list[str]:
    return [f"- {value}" for value in values if value]


def _format_mapping_list(values: tuple[dict[str, Any], ...]) -> list[str]:
    result = []
    for value in values:
        name = value.get("name", "элемент")
        extras = [
            f"{key}={item_value}"
            for key, item_value in sorted(value.items())
            if key != "name"
        ]
        result.append(f"- {name}" + (f" ({', '.join(extras)})" if extras else ""))
    return result


def _format_spell_list(values: tuple[dict[str, Any] | str, ...]) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, dict):
            name = value.get("name", "заклинание")
            level = value.get("level")
            level_text = f", level={level}" if level is not None else ""
            result.append(f"- {name}{level_text}")
        else:
            result.append(f"- {value}")
    return result


def _format_proficiencies(character: InternalCharacter) -> list[str]:
    traits = character.race_traits
    lines = []
    for key, label in (
        ("skill_proficiencies", "Навыки"),
        ("weapon_proficiencies", "Оружие"),
        ("saving_throw_advantages", "Преимущества спасбросков"),
    ):
        values = traits.get(key)
        if isinstance(values, (list, tuple)) and values:
            lines.append(f"{label}: {', '.join(str(value) for value in values)}")
    if character.armor:
        lines.append(f"Броня: {character.armor.get('name', 'есть')}")
    return lines
