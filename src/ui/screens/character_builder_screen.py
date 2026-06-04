"""GUI character builder backed by the internal character repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from character import (
    CharacterRepository,
    CharacterValidationError,
    InternalCharacter,
)
from combat.abilities import ability_modifier
from combat.inventory import get_supported_item_definitions
from combat.spellcasting import get_supported_spell_definitions
from rules.classes import (
    build_class_features,
    get_class_definition,
    get_supported_class_definitions,
    get_supported_subclass_definitions,
    spell_slots_for_class_level,
)
from rules.feats import get_supported_feat_definitions
from rules.progression import get_proficiency_bonus
from rules.races import get_race_definition
from rules.registry import get_active_ruleset
from rules.spellcasting_progression import get_spellcasting_ability_for_class
from ui.widgets.inventory_editor import InventoryEditor
from ui.widgets.screen import ScreenFrame
from ui.widgets.spell_selector import SpellSelector
from ui.widgets.stat_editor import StatEditor


WEAPON_OPTIONS: tuple[dict[str, object], ...] = (
    {"name": "Longsword", "range": 1, "damage": "1d8", "damage_type": "slashing"},
    {"name": "Greatsword", "range": 1, "damage": "2d6", "damage_type": "slashing"},
    {"name": "Longbow", "range": 6, "damage": "1d8", "damage_type": "piercing"},
    {"name": "Mace", "range": 1, "damage": "1d6", "damage_type": "bludgeoning"},
    {"name": "Quarterstaff", "range": 1, "damage": "1d6", "damage_type": "bludgeoning"},
    {"name": "Dagger", "range": 4, "damage": "1d4", "damage_type": "piercing"},
    {
        "name": "Light Crossbow",
        "range": 6,
        "damage": "1d8",
        "damage_type": "piercing",
    },
)

ARMOR_OPTIONS: tuple[dict[str, object], ...] = (
    {"name": "None", "base_ac": 10, "dex_cap": None, "armor": False},
    {"name": "Leather Armor", "base_ac": 11, "dex_cap": None, "armor": True},
    {"name": "Scale Mail", "base_ac": 14, "dex_cap": 2, "armor": True},
    {"name": "Chain Mail", "base_ac": 16, "dex_cap": 0, "armor": True},
)

CLASS_WEAPON_LIMITS: dict[str, tuple[str, ...]] = {
    "Wizard": ("Dagger", "Quarterstaff", "Light Crossbow"),
    "Cleric": ("Mace", "Quarterstaff", "Dagger", "Light Crossbow"),
}

CLASS_ARMOR_LIMITS: dict[str, tuple[str, ...]] = {
    "Wizard": ("None",),
    "Cleric": ("None", "Leather Armor", "Scale Mail"),
}

FIGHTING_STYLES: tuple[str, ...] = (
    "Archery",
    "Defense",
    "Great Weapon Fighting",
)

RU_LABELS: dict[str, str] = {
    "Human": "Человек",
    "Dwarf": "Дварф",
    "Elf": "Эльф",
    "Halfling": "Полурослик",
    "Fighter": "Воин",
    "Cleric": "Жрец",
    "Wizard": "Волшебник",
    "Champion": "Чемпион",
    "Life Domain": "Домен Жизни",
    "School of Evocation": "Школа Воплощения",
    "Archery": "Стрельба",
    "Defense": "Защита",
    "Great Weapon Fighting": "Бой двуручным оружием",
    "Ability Score Improvement": "Улучшение характеристик",
    "Grappler": "Борец",
    "None": "Без брони",
    "Leather Armor": "Кожаная броня",
    "Scale Mail": "Чешуйчатый доспех",
    "Chain Mail": "Кольчуга",
    "Longsword": "Длинный меч",
    "Greatsword": "Двуручный меч",
    "Longbow": "Длинный лук",
    "Mace": "Булава",
    "Quarterstaff": "Боевой посох",
    "Dagger": "Кинжал",
    "Light Crossbow": "Лёгкий арбалет",
    "MELEE_DAMAGE": "Ближний урон",
    "RANGED_DAMAGE": "Дальний урон",
    "TANK": "Защитник",
    "SUPPORT": "Поддержка",
    "CASTER": "Заклинатель",
}

ROLE_OPTIONS: tuple[str, ...] = (
    "MELEE_DAMAGE",
    "RANGED_DAMAGE",
    "TANK",
    "SUPPORT",
    "CASTER",
)


@dataclass(frozen=True)
class CharacterBuilderData:
    """Pure builder input used by GUI and validation tests."""

    id: str = ""
    name: str = ""
    race_name: str = "Human"
    class_name: str = "Fighter"
    subclass_name: str | None = None
    level: int = 1
    experience: int = 0
    role: str = "MELEE_DAMAGE"
    stats: dict[str, int] = field(default_factory=lambda: {
        "str": 10,
        "dex": 10,
        "con": 10,
        "int": 10,
        "wis": 10,
        "cha": 10,
    })
    feat_name: str | None = None
    fighting_style: str | None = None
    prepared_spells: tuple[str, ...] = ()
    weapons: tuple[str, ...] = ()
    armor_name: str = "None"
    inventory: tuple[dict[str, object], ...] = ()


class CharacterBuilderScreen(QWidget):
    """Digital character sheet builder for InternalCharacter records."""

    saved = Signal()
    cancelled = Signal()

    def __init__(
        self,
        repository: CharacterRepository | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repository = repository or CharacterRepository()
        self._editing_id = ""
        self._build_layout()
        self.new_character()

    def new_character(self) -> None:
        """Reset the form for a new character."""

        self._editing_id = ""
        self.name_edit.setText("")
        self._select_combo_text(self.role_combo, "MELEE_DAMAGE")
        self.level_spin.setValue(1)
        self.experience_spin.setValue(0)
        self.stat_editor.set_racial_bonuses({})
        self.stat_editor.set_stats({
            "str": 8,
            "dex": 8,
            "con": 8,
            "int": 8,
            "wis": 8,
            "cha": 8,
        })
        self._refresh_rule_options()
        self.inventory_editor.set_inventory(())
        self._set_checked_names(self.weapon_list, ())
        self.update_review()

    def load_character(self, character_id: str) -> None:
        """Load an existing character into the builder."""

        character = self.repository.get_character(character_id)
        if character is None:
            QMessageBox.warning(self, "Персонаж не найден", "Запись уже удалена.")
            self.new_character()
            return
        self._editing_id = character.id
        self.name_edit.setText(character.name)
        self._select_combo_text(self.role_combo, character.role)
        self.level_spin.setValue(character.level)
        self.experience_spin.setValue(character.experience)
        self._select_combo_text(self.race_combo, character.race_name)
        self._select_combo_text(self.class_combo, character.class_name)
        self._refresh_subclasses()
        if character.subclass_name:
            self._select_combo_text(self.subclass_combo, character.subclass_name)
        self._refresh_racial_bonuses()
        self.stat_editor.set_stats(character.stats)
        self._refresh_class_options()
        if character.feats:
            self._select_combo_text(self.feat_combo, character.feats[0])
        self._select_combo_text(
            self.fighting_style_combo,
            character.race_traits.get("fighting_style", ""),
        )
        self.spell_selector.set_options(
            character.class_name,
            character.level,
            character.prepared_spells,
        )
        self._set_checked_names(
            self.weapon_list,
            tuple(str(weapon.get("name")) for weapon in character.weapons),
        )
        self._select_combo_text(
            self.armor_combo,
            str(character.armor.get("name", "None")),
        )
        self.inventory_editor.set_inventory(character.inventory)
        self.update_review()

    def update_review(self) -> None:
        """Recalculate derived values and validation feedback."""

        try:
            if self.stat_editor.points_remaining() < 0:
                raise ValueError("Превышен лимит point buy: доступно 27 очков характеристик.")
            character = build_internal_character_from_builder(self._collect_data())
            self.repository.validate_character(_character_for_preview_validation(character))
            errors: list[str] = []
        except (CharacterValidationError, ValueError) as exc:
            character = None
            errors = _error_lines(exc)

        if character is None:
            self.review_text.setPlainText("\n".join(errors))
            self.save_button.setEnabled(False)
            self._update_combat_summary(None)
            return

        lines = [
            "Персонаж готов к сохранению.",
            "",
            f"Бонус мастерства: {character.proficiency_bonus}",
            f"HP: {character.hp}",
            f"КД: {character.ac}",
            f"Скорость: {character.speed}",
            f"Сл спасброска заклинаний: {character.spell_save_dc}",
            f"Бонус атаки заклинанием: {character.spell_attack_bonus}",
            f"Ячейки заклинаний: {character.spell_slots or 'нет'}",
            f"Ресурсы класса: {character.resources or 'нет'}",
        ]
        self.review_text.setPlainText("\n".join(lines))
        self.save_button.setEnabled(True)
        self._update_combat_summary(character)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(0)
        frame = ScreenFrame(
            "Создать персонажа",
            "Редактируемый цифровой лист персонажа для внутреннего формата проекта.",
        )
        root.addWidget(frame)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        sheet = QWidget()
        self.sheet_grid = QGridLayout(sheet)
        self.sheet_grid.setContentsMargins(0, 0, 0, 0)
        self.sheet_grid.setHorizontalSpacing(10)
        self.sheet_grid.setVerticalSpacing(10)
        self.sheet_grid.setColumnStretch(0, 1)
        self.sheet_grid.setColumnStretch(1, 1)
        self.sheet_grid.setColumnStretch(2, 1)
        scroll.setWidget(sheet)
        frame.content_layout.addWidget(scroll, stretch=1)

        self._build_identity_section()
        self._build_stats_section()
        self._build_combat_section()
        self._build_class_section()
        self._build_spells_section()
        self._build_equipment_section()
        self._build_inventory_section()
        self._build_review_section()

        button_row = QHBoxLayout()
        self.save_button = QPushButton("Сохранить")
        cancel_button = QPushButton("Отмена")
        self.save_button.clicked.connect(self._save)
        cancel_button.clicked.connect(self.cancelled.emit)
        button_row.addStretch(1)
        button_row.addWidget(cancel_button)
        button_row.addWidget(self.save_button)
        frame.content_layout.addLayout(button_row)

    def _build_identity_section(self) -> None:
        group = QGroupBox("Персонаж")
        form = QFormLayout(group)
        self.name_edit = QLineEdit()
        self.role_combo = QComboBox()
        self.race_combo = QComboBox()
        self.class_combo = QComboBox()
        self.level_spin = QSpinBox()
        self.level_spin.setRange(1, 5)
        self.experience_spin = QSpinBox()
        self.experience_spin.setRange(0, 999999)

        for role in ROLE_OPTIONS:
            self.role_combo.addItem(_ru(role), role)
        form.addRow("Имя", self.name_edit)
        form.addRow("Раса", self.race_combo)
        form.addRow("Класс", self.class_combo)
        form.addRow("Уровень", self.level_spin)
        form.addRow("Опыт", self.experience_spin)
        form.addRow("Боевая роль", self.role_combo)
        self.sheet_grid.addWidget(group, 0, 0, 1, 3)

        self.name_edit.textChanged.connect(self.update_review)
        self.role_combo.currentTextChanged.connect(self.update_review)
        self.race_combo.currentTextChanged.connect(self._race_changed)
        self.class_combo.currentTextChanged.connect(self._class_changed)
        self.level_spin.valueChanged.connect(self._level_changed)
        self.experience_spin.valueChanged.connect(self.update_review)

    def _build_sheet_summary_section(self) -> None:
        self.sheet_summary = QLabel()
        self.sheet_summary.setObjectName("characterSheetSummary")
        self.sheet_summary.setWordWrap(True)
        self.sheet_summary.setMinimumWidth(220)
        self.sheet_grid.addWidget(self.sheet_summary, 0, 2)

    def _build_stats_section(self) -> None:
        group = QGroupBox("Характеристики")
        layout = QVBoxLayout(group)
        self.stat_editor = StatEditor()
        self.stat_editor.stats_changed.connect(self.update_review)
        layout.addWidget(self.stat_editor)
        self.sheet_grid.addWidget(group, 1, 0)

    def _build_combat_section(self) -> None:
        group = QGroupBox("Боевой лист")
        layout = QGridLayout(group)
        self.hp_value_label = self._metric_value_label()
        self.ac_value_label = self._metric_value_label()
        self.speed_value_label = self._metric_value_label()
        self.pb_value_label = self._metric_value_label()
        self.spell_dc_value_label = self._metric_value_label()
        self.spell_attack_value_label = self._metric_value_label()
        self.slots_value_label = self._metric_value_label()
        self.resources_value_label = self._metric_value_label()

        metrics = (
            ("HP", self.hp_value_label),
            ("КД", self.ac_value_label),
            ("Скорость", self.speed_value_label),
            ("Мастерство", self.pb_value_label),
            ("СЛ", self.spell_dc_value_label),
            ("Атака", self.spell_attack_value_label),
            ("Ячейки", self.slots_value_label),
            ("Ресурсы", self.resources_value_label),
        )
        for index, (label, value) in enumerate(metrics):
            row = index // 2
            col = (index % 2) * 2
            layout.addWidget(QLabel(label), row, col)
            layout.addWidget(value, row, col + 1)
        self.sheet_grid.addWidget(group, 1, 1)

    def _build_class_section(self) -> None:
        group = QGroupBox("Класс и развитие")
        form = QFormLayout(group)
        self.subclass_combo = QComboBox()
        self.feat_combo = QComboBox()
        self.fighting_style_combo = QComboBox()
        form.addRow("Подкласс", self.subclass_combo)
        form.addRow("Улучшение/черта", self.feat_combo)
        form.addRow("Боевой стиль", self.fighting_style_combo)
        self.sheet_grid.addWidget(group, 1, 2)
        self.subclass_combo.currentTextChanged.connect(self.update_review)
        self.feat_combo.currentTextChanged.connect(self.update_review)
        self.fighting_style_combo.currentTextChanged.connect(self.update_review)

    def _build_spells_section(self) -> None:
        group = QGroupBox("Заклинания")
        layout = QVBoxLayout(group)
        self.spell_selector = SpellSelector()
        self.spell_selector.selection_changed.connect(self.update_review)
        layout.addWidget(self.spell_selector)
        self.sheet_grid.addWidget(group, 2, 0)

    def _build_equipment_section(self) -> None:
        group = QGroupBox("Экипировка")
        layout = QVBoxLayout(group)
        self.armor_combo = QComboBox()
        for armor in ARMOR_OPTIONS:
            self.armor_combo.addItem(_ru(str(armor["name"])), str(armor["name"]))
        layout.addWidget(QLabel("Броня"))
        layout.addWidget(self.armor_combo)
        layout.addWidget(QLabel("Оружие"))
        self.weapon_list = QListWidget()
        self.weapon_list.setMinimumHeight(140)
        for weapon in WEAPON_OPTIONS:
            item = QListWidgetItem(_ru(str(weapon["name"])))
            item.setData(Qt.ItemDataRole.UserRole, str(weapon["name"]))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.weapon_list.addItem(item)
        self.weapon_list.itemChanged.connect(self.update_review)
        self.armor_combo.currentTextChanged.connect(self.update_review)
        self._refresh_equipment_options()
        layout.addWidget(self.weapon_list)
        self.sheet_grid.addWidget(group, 2, 1)

    def _build_inventory_section(self) -> None:
        group = QGroupBox("Инвентарь")
        layout = QVBoxLayout(group)
        self.inventory_editor = InventoryEditor()
        self.inventory_editor.inventory_changed.connect(self.update_review)
        layout.addWidget(self.inventory_editor)
        self.sheet_grid.addWidget(group, 2, 2)

    def _build_review_section(self) -> None:
        group = QGroupBox("Проверка листа")
        layout = QVBoxLayout(group)
        self.review_text = QTextEdit()
        self.review_text.setReadOnly(True)
        self.review_text.setMinimumHeight(120)
        layout.addWidget(self.review_text)
        self.sheet_grid.addWidget(group, 3, 0, 1, 3)

    @staticmethod
    def _metric_value_label() -> QLabel:
        label = QLabel("—")
        label.setObjectName("sheetMetricValue")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def _refresh_rule_options(self) -> None:
        ruleset = get_active_ruleset()
        race_blocked = self.race_combo.blockSignals(True)
        self.race_combo.clear()
        for race in ruleset.supported_races:
            self.race_combo.addItem(_ru(race), race)
        self.race_combo.blockSignals(race_blocked)

        class_blocked = self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for definition in get_supported_class_definitions(ruleset):
            self.class_combo.addItem(_ru(definition.name), definition.name)
        self.class_combo.blockSignals(class_blocked)
        self._refresh_subclasses()
        self._refresh_class_options()
        self._refresh_spells()
        self._refresh_racial_bonuses()
        self._refresh_stat_proficiencies()
        self._refresh_equipment_options()

    def _race_changed(self) -> None:
        self._refresh_racial_bonuses()
        self.update_review()

    def _class_changed(self) -> None:
        self._refresh_subclasses()
        self._refresh_class_options()
        self._refresh_spells()
        self._refresh_stat_proficiencies()
        self._refresh_equipment_options()
        self.update_review()

    def _level_changed(self) -> None:
        self._refresh_subclasses()
        self._refresh_class_options()
        self._refresh_spells()
        self._refresh_stat_proficiencies()
        self.update_review()

    def _refresh_subclasses(self) -> None:
        current = _combo_value(self.subclass_combo)
        self.subclass_combo.clear()
        self.subclass_combo.addItem("")
        for definition in get_supported_subclass_definitions(
            _combo_value(self.class_combo),
            level=self.level_spin.value(),
        ):
            self.subclass_combo.addItem(_ru(definition.name), definition.name)
        self._select_combo_text(self.subclass_combo, current)

    def _refresh_class_options(self) -> None:
        current_feat = _combo_value(self.feat_combo)
        self.feat_combo.clear()
        self.feat_combo.addItem("")
        if self.level_spin.value() >= 4:
            for definition in get_supported_feat_definitions():
                self.feat_combo.addItem(_ru(definition.name), definition.name)
        self._select_combo_text(self.feat_combo, current_feat)

        current_style = _combo_value(self.fighting_style_combo)
        self.fighting_style_combo.clear()
        self.fighting_style_combo.addItem("")
        if _combo_value(self.class_combo) == "Fighter":
            for style in FIGHTING_STYLES:
                self.fighting_style_combo.addItem(_ru(style), style)
        self._select_combo_text(self.fighting_style_combo, current_style)

    def _refresh_spells(self) -> None:
        selected = self.spell_selector.selected_spell_names()
        self.spell_selector.set_options(
            _combo_value(self.class_combo),
            self.level_spin.value(),
            selected,
        )

    def _refresh_equipment_options(self) -> None:
        if not hasattr(self, "armor_combo") or not hasattr(self, "weapon_list"):
            return
        class_name = _combo_value(self.class_combo)
        current_armor = _combo_value(self.armor_combo) if self.armor_combo.count() else "None"
        current_weapons = self._checked_names(self.weapon_list)
        self._populate_armor_options(class_name, current_armor)
        self._populate_weapon_options(class_name, current_weapons)

    def _populate_armor_options(self, class_name: str, current_armor: str) -> None:
        allowed = _allowed_armor_names(class_name)
        was_blocked = self.armor_combo.blockSignals(True)
        self.armor_combo.clear()
        for armor in ARMOR_OPTIONS:
            armor_name = str(armor["name"])
            if armor_name in allowed:
                self.armor_combo.addItem(_ru(armor_name), armor_name)
        self.armor_combo.blockSignals(was_blocked)
        self._select_combo_text(
            self.armor_combo,
            current_armor if current_armor in allowed else "None",
        )

    def _populate_weapon_options(self, class_name: str, selected_weapons: tuple[str, ...]) -> None:
        allowed = _allowed_weapon_names(class_name)
        selected = {name for name in selected_weapons if name in allowed}
        was_blocked = self.weapon_list.blockSignals(True)
        self.weapon_list.clear()
        for weapon in WEAPON_OPTIONS:
            weapon_name = str(weapon["name"])
            if weapon_name not in allowed:
                continue
            item = QListWidgetItem(_ru(weapon_name))
            item.setData(Qt.ItemDataRole.UserRole, weapon_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if weapon_name in selected else Qt.CheckState.Unchecked
            )
            self.weapon_list.addItem(item)
        self.weapon_list.blockSignals(was_blocked)

    def _refresh_racial_bonuses(self) -> None:
        race_name = _combo_value(self.race_combo)
        if not race_name:
            self.stat_editor.set_racial_bonuses({})
            return
        race = get_race_definition(race_name)
        self.stat_editor.set_racial_bonuses(dict(race.ability_score_bonuses))

    def _refresh_stat_proficiencies(self) -> None:
        class_definition = get_class_definition(_combo_value(self.class_combo))
        self.stat_editor.set_proficiency_context(
            proficiency_bonus=get_proficiency_bonus(self.level_spin.value()),
            saving_throw_proficiencies=(
                class_definition.saving_throw_proficiencies
                if class_definition is not None
                else ()
            ),
        )

    def _collect_data(self) -> CharacterBuilderData:
        return CharacterBuilderData(
            id=self._editing_id,
            name=self.name_edit.text().strip(),
            race_name=_combo_value(self.race_combo),
            class_name=_combo_value(self.class_combo),
            subclass_name=_combo_value(self.subclass_combo) or None,
            level=self.level_spin.value(),
            experience=self.experience_spin.value(),
            role=_combo_value(self.role_combo) or "MELEE_DAMAGE",
            stats=self.stat_editor.stats(),
            feat_name=_combo_value(self.feat_combo) or None,
            fighting_style=_combo_value(self.fighting_style_combo) or None,
            prepared_spells=self.spell_selector.selected_spell_names(),
            weapons=self._checked_names(self.weapon_list),
            armor_name=_combo_value(self.armor_combo),
            inventory=self.inventory_editor.inventory_items(),
        )

    def _save(self) -> None:
        try:
            character = build_internal_character_from_builder(self._collect_data())
            self.repository.save_character(character)
        except (CharacterValidationError, ValueError) as exc:
            QMessageBox.warning(self, "Невалидный персонаж", "\n".join(_error_lines(exc)))
            self.update_review()
            return
        self.saved.emit()

    @staticmethod
    def _select_combo_text(combo: QComboBox, text: str | None) -> None:
        if not text:
            return
        for index in range(combo.count()):
            if combo.itemData(index) == text or combo.itemText(index) == text:
                combo.setCurrentIndex(index)
                return
        display_index = combo.findText(_ru(text))
        if display_index >= 0:
            combo.setCurrentIndex(display_index)

    def _update_sheet_summary(self, character: InternalCharacter | None = None) -> None:
        if character is None:
            self.sheet_summary.setText(
                "Лист персонажа\n\n"
                "Заполните имя, расу, класс и характеристики. "
                "Итоговые HP, КД, скорость и ресурсы появятся здесь."
            )
            return
        self.sheet_summary.setText(
            "\n".join(
                (
                    f"{character.name or 'Без имени'}",
                    f"{_ru(character.race_name)} {_ru(character.class_name)}",
                    f"Уровень {character.level}",
                    "",
                    f"HP {character.hp}",
                    f"КД {character.ac}",
                    f"Скорость {character.speed}",
                    f"Бонус мастерства +{character.proficiency_bonus}",
                    "",
                    f"Роль: {_ru(character.role)}",
                    f"Оружие: {_join_ru(weapon.get('name') for weapon in character.weapons)}",
                    f"Броня: {_ru(str(character.armor.get('name', 'None')))}",
                )
            )
        )

    def _update_combat_summary(self, character: InternalCharacter | None = None) -> None:
        if character is None:
            for label in (
                self.hp_value_label,
                self.ac_value_label,
                self.speed_value_label,
                self.pb_value_label,
                self.spell_dc_value_label,
                self.spell_attack_value_label,
                self.slots_value_label,
                self.resources_value_label,
            ):
                label.setText("—")
            return

        self.hp_value_label.setText(str(character.hp))
        self.ac_value_label.setText(str(character.ac))
        self.speed_value_label.setText(str(character.speed))
        self.pb_value_label.setText(f"+{character.proficiency_bonus}")
        self.spell_dc_value_label.setText(str(character.spell_save_dc or "—"))
        self.spell_attack_value_label.setText(
            f"+{character.spell_attack_bonus}" if character.spell_attack_bonus else "—"
        )
        self.slots_value_label.setText(_format_mapping(character.spell_slots))
        self.resources_value_label.setText(_format_mapping(character.resources))

    @staticmethod
    def _checked_names(list_widget: QListWidget) -> tuple[str, ...]:
        names = []
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(str(item.data(Qt.ItemDataRole.UserRole) or item.text()))
        return tuple(names)

    @staticmethod
    def _set_checked_names(list_widget: QListWidget, names: tuple[str, ...]) -> None:
        selected = {_key(name) for name in names}
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            value = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
            state = Qt.CheckState.Checked if _key(value) in selected else Qt.CheckState.Unchecked
            item.setCheckState(state)


def build_internal_character_from_builder(data: CharacterBuilderData) -> InternalCharacter:
    """Build a validated InternalCharacter payload from builder data."""

    validate_builder_options(data)
    class_definition = get_class_definition(data.class_name)
    race = get_race_definition(data.race_name)
    stats = dict(data.stats)
    proficiency_bonus = get_proficiency_bonus(data.level)
    hp = calculate_hp(data.class_name, data.level, stats)
    armor = _armor_by_name(data.armor_name)
    ac = calculate_ac(armor, stats, data.fighting_style)
    spell_slots = spell_slots_for_class_level(data.class_name, data.level)
    spellcasting_ability = get_spellcasting_ability_for_class(data.class_name)
    spell_modifier = ability_modifier_from_stats(stats, spellcasting_ability)
    spell_save_dc = 8 + proficiency_bonus + spell_modifier if spellcasting_ability else 0
    spell_attack_bonus = proficiency_bonus + spell_modifier if spellcasting_ability else 0
    features = build_class_features(data.class_name, data.level, data.subclass_name)
    resources = _resources_for_features(features)
    selected_weapons = tuple(
        dict(weapon)
        for weapon in WEAPON_OPTIONS
        if weapon["name"] in set(data.weapons)
    )
    selected_spells = _spell_payloads(data.class_name, data.level, data.prepared_spells)

    class_features = tuple(
        feature.name
        for feature in features
        if feature.level <= data.level and feature.name not in _subclass_feature_names(
            data.class_name,
            data.subclass_name,
            data.level,
        )
    )
    subclass_features = tuple(
        feature.name
        for feature in features
        if feature.name in _subclass_feature_names(
            data.class_name,
            data.subclass_name,
            data.level,
        )
    )

    race_traits: dict[str, object] = {
        "ability_score_bonuses": dict(race.ability_score_bonuses),
        "size": race.size,
        "speed": race.speed,
        "darkvision_range": race.darkvision_range,
        "skill_proficiencies": list(race.skill_proficiencies),
        "weapon_proficiencies": list(race.weapon_proficiencies),
        "saving_throw_advantages": list(race.saving_throw_advantages),
        "damage_resistances": list(race.damage_resistances),
        "special_traits": list(race.special_traits),
    }
    if data.fighting_style:
        race_traits["fighting_style"] = data.fighting_style

    return InternalCharacter(
        id=data.id,
        name=data.name,
        class_name=data.class_name,
        subclass_name=data.subclass_name,
        level=data.level,
        experience=data.experience,
        race_name=data.race_name,
        role=data.role,
        stats=stats,
        hp=hp,
        ac=ac,
        speed=int(race.speed) * 10,
        proficiency_bonus=proficiency_bonus,
        weapons=selected_weapons,
        armor={key: value for key, value in armor.items() if key != "armor"},
        class_features=class_features,
        subclass_features=subclass_features,
        race_traits=race_traits,
        feats=(data.feat_name,) if data.feat_name else (),
        spells=selected_spells,
        prepared_spells=data.prepared_spells,
        spell_slots={str(level): count for level, count in spell_slots.items()},
        spell_save_dc=spell_save_dc,
        spell_attack_bonus=spell_attack_bonus,
        resources=resources,
        inventory=data.inventory,
        resistances=tuple(race.damage_resistances),
    )


def validate_builder_options(data: CharacterBuilderData) -> None:
    """Validate GUI builder choices before character construction."""

    ruleset = get_active_ruleset()
    if not ruleset.is_supported_content("race", data.race_name):
        raise ValueError(ruleset.get_unsupported_reason("race", data.race_name))
    if not ruleset.is_supported_content("class", data.class_name):
        raise ValueError(ruleset.get_unsupported_reason("class", data.class_name))
    if not ruleset.is_supported_content("level", data.level):
        raise ValueError(ruleset.get_unsupported_reason("level", data.level))
    allowed_subclasses = {
        definition.name
        for definition in get_supported_subclass_definitions(
            data.class_name,
            level=data.level,
        )
    }
    if data.subclass_name and data.subclass_name not in allowed_subclasses:
        raise ValueError(
            f"Подкласс недоступен на уровне {data.level}: {_ru(data.subclass_name)}"
        )
    allowed_spells = {
        definition.name
        for definition in get_supported_spell_definitions(data.class_name, data.level)
    }
    for spell_name in data.prepared_spells:
        if spell_name not in allowed_spells:
            raise ValueError(
                f"Заклинание «{_ru(spell_name)}» недоступно для "
                f"{_ru(data.class_name)} {data.level} уровня."
            )
    allowed_items = {definition.name for definition in get_supported_item_definitions()}
    for item in data.inventory:
        if str(item.get("name")) not in allowed_items:
            raise ValueError(f"Предмет не поддерживается: {_ru(item.get('name'))}")
    if data.feat_name:
        if int(data.level) < 4:
            raise ValueError("Улучшение характеристик или черта доступны только с 4 уровня.")
        supported_feats = {definition.name for definition in get_supported_feat_definitions()}
        if data.feat_name not in supported_feats:
            raise ValueError(f"Черта не поддерживается: {_ru(data.feat_name)}")
    allowed_weapons = {str(weapon["name"]) for weapon in WEAPON_OPTIONS}
    for weapon_name in data.weapons:
        if weapon_name not in allowed_weapons:
            raise ValueError(f"Оружие не поддерживается: {_ru(weapon_name)}")
    class_weapons = _allowed_weapon_names(data.class_name)
    for weapon_name in data.weapons:
        if weapon_name not in class_weapons:
            raise ValueError(f"Weapon is not available for {_ru(data.class_name)}: {_ru(weapon_name)}")
    if data.armor_name not in _allowed_armor_names(data.class_name):
        raise ValueError(f"Armor is not available for {_ru(data.class_name)}: {_ru(data.armor_name)}")
    if data.fighting_style and data.class_name != "Fighter":
        raise ValueError("Боевой стиль доступен только воину.")
    if data.fighting_style and data.fighting_style not in FIGHTING_STYLES:
        raise ValueError(f"Боевой стиль не поддерживается: {_ru(data.fighting_style)}")
    if not class_definition_allows_subclass_absence(data.class_name, data.level, data.subclass_name):
        raise ValueError(f"{_ru(data.class_name)} требует выбрать подкласс на уровне {data.level}.")


def calculate_hp(class_name: str, level: int, stats: dict[str, int]) -> int:
    class_definition = get_class_definition(class_name)
    hit_die = class_definition.hit_die if class_definition is not None else 8
    con_modifier = ability_modifier_from_stats(stats, "con")
    first_level = max(1, hit_die + con_modifier)
    average_gain = max(1, (hit_die // 2 + 1) + con_modifier)
    return first_level + max(0, int(level) - 1) * average_gain


def calculate_ac(
    armor: dict[str, object],
    stats: dict[str, int],
    fighting_style: str | None,
) -> int:
    dex_modifier = ability_modifier_from_stats(stats, "dex")
    dex_cap = armor.get("dex_cap")
    if dex_cap is not None:
        dex_modifier = min(dex_modifier, int(dex_cap))
    ac = int(armor["base_ac"]) + dex_modifier
    if fighting_style == "Defense" and bool(armor.get("armor", False)):
        ac += 1
    return ac


def ability_modifier_from_stats(stats: dict[str, int], ability: str | None) -> int:
    if ability is None:
        return 0
    return (int(stats.get(ability, 10)) - 10) // 2


def class_definition_allows_subclass_absence(
    class_name: str,
    level: int,
    subclass_name: str | None,
) -> bool:
    class_definition = get_class_definition(class_name)
    if class_definition is None or class_definition.subclass_level is None:
        return True
    return int(level) < class_definition.subclass_level or bool(subclass_name)


def _armor_by_name(name: str) -> dict[str, object]:
    for armor in ARMOR_OPTIONS:
        if armor["name"] == name:
            return dict(armor)
    return dict(ARMOR_OPTIONS[0])


def _allowed_weapon_names(class_name: str) -> set[str]:
    limited = CLASS_WEAPON_LIMITS.get(class_name)
    if limited is not None:
        return set(limited)
    return {str(weapon["name"]) for weapon in WEAPON_OPTIONS}


def _allowed_armor_names(class_name: str) -> set[str]:
    limited = CLASS_ARMOR_LIMITS.get(class_name)
    if limited is not None:
        return set(limited)
    return {str(armor["name"]) for armor in ARMOR_OPTIONS}


def _spell_payloads(
    class_name: str,
    level: int,
    spell_names: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    selected = {_key(name) for name in spell_names}
    return tuple(
        {
            "name": definition.name,
            "level": definition.spell_level,
            "action_cost": definition.action_cost,
        }
        for definition in get_supported_spell_definitions(class_name, level)
        if _key(definition.name) in selected
    )


def _resources_for_features(features: list[Any]) -> dict[str, int]:
    resources: dict[str, int] = {}
    for feature in features:
        resource_name = getattr(feature, "resource_name", None)
        if resource_name is None:
            continue
        resources[str(resource_name)] = 1
    return resources


def _subclass_feature_names(
    class_name: str,
    subclass_name: str | None,
    level: int,
) -> set[str]:
    if subclass_name is None:
        return set()
    from rules.subclasses import get_subclass_definition

    subclass = get_subclass_definition(class_name, subclass_name)
    if subclass is None:
        return set()
    return {feature.name for feature in subclass.features_for_level(level)}


def _error_lines(error: Exception) -> list[str]:
    if isinstance(error, CharacterValidationError):
        return [f"{issue.field}: {issue.message}" for issue in error.issues]
    return [str(error)]


def _character_for_preview_validation(character: InternalCharacter) -> InternalCharacter:
    if character.id:
        return character
    return character.with_id("preview-character")


def _combo_value(combo: QComboBox) -> str:
    data = combo.currentData()
    if data is not None:
        return str(data)
    return combo.currentText()


def _ru(value: object) -> str:
    return RU_LABELS.get(str(value), str(value))


def _join_ru(values: Any) -> str:
    labels = [_ru(value) for value in values if value]
    return ", ".join(labels) if labels else "нет"


def _format_mapping(values: dict[str, object] | None) -> str:
    if not values:
        return "нет"
    return ", ".join(f"{key}: {value}" for key, value in values.items())


def _key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
