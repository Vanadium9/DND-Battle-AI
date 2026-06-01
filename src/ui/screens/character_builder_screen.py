"""GUI character builder backed by the internal character repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
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

FIGHTING_STYLES: tuple[str, ...] = (
    "Archery",
    "Defense",
    "Great Weapon Fighting",
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
    """Tabbed GUI builder for new and existing InternalCharacter records."""

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
        self._applied_race_bonuses: dict[str, int] = {}
        self._build_layout()
        self.new_character()

    def new_character(self) -> None:
        """Reset the form for a new character."""

        self._editing_id = ""
        self.name_edit.setText("")
        self.role_edit.setText("MELEE_DAMAGE")
        self.level_spin.setValue(1)
        self.experience_spin.setValue(0)
        self.stat_editor.set_stats({
            "str": 10,
            "dex": 10,
            "con": 10,
            "int": 10,
            "wis": 10,
            "cha": 10,
        })
        self._applied_race_bonuses = {}
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
        self.role_edit.setText(character.role)
        self.level_spin.setValue(character.level)
        self.experience_spin.setValue(character.experience)
        self._select_combo_text(self.race_combo, character.race_name)
        self._select_combo_text(self.class_combo, character.class_name)
        self._refresh_subclasses()
        if character.subclass_name:
            self._select_combo_text(self.subclass_combo, character.subclass_name)
        self.stat_editor.set_stats(character.stats)
        self._applied_race_bonuses = {}
        self._refresh_class_options()
        if character.feats:
            self._select_combo_text(self.feat_combo, character.feats[0])
        self._select_combo_text(self.fighting_style_combo, character.race_traits.get("fighting_style", ""))
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
            character = build_internal_character_from_builder(self._collect_data())
            self.repository.validate_character(_character_for_preview_validation(character))
            errors: list[str] = []
        except (CharacterValidationError, ValueError) as exc:
            character = None
            errors = _error_lines(exc)

        if character is None:
            self.review_text.setPlainText("\n".join(errors))
            self.save_button.setEnabled(False)
            return

        lines = [
            "Персонаж валиден.",
            "",
            f"Proficiency bonus: {character.proficiency_bonus}",
            f"HP: {character.hp}",
            f"AC: {character.ac}",
            f"Speed: {character.speed}",
            f"Spell save DC: {character.spell_save_dc}",
            f"Spell attack bonus: {character.spell_attack_bonus}",
            f"Spell slots: {character.spell_slots or 'нет'}",
            f"Class resources: {character.resources or 'нет'}",
        ]
        self.review_text.setPlainText("\n".join(lines))
        self.save_button.setEnabled(True)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(0)
        frame = ScreenFrame(
            "Создать персонажа",
            "Пошаговый конструктор внутреннего формата персонажей.",
        )
        root.addWidget(frame)

        self.tabs = QTabWidget()
        frame.content_layout.addWidget(self.tabs, stretch=1)

        self._build_basic_tab()
        self._build_stats_tab()
        self._build_class_tab()
        self._build_spells_tab()
        self._build_equipment_tab()
        self._build_inventory_tab()
        self._build_review_tab()

        button_row = QHBoxLayout()
        self.save_button = QPushButton("Сохранить")
        cancel_button = QPushButton("Отмена")
        self.save_button.clicked.connect(self._save)
        cancel_button.clicked.connect(self.cancelled.emit)
        button_row.addStretch(1)
        button_row.addWidget(cancel_button)
        button_row.addWidget(self.save_button)
        frame.content_layout.addLayout(button_row)

        self.tabs.currentChanged.connect(lambda _: self.update_review())

    def _build_basic_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.name_edit = QLineEdit()
        self.role_edit = QLineEdit()
        self.race_combo = QComboBox()
        self.class_combo = QComboBox()
        self.level_spin = QSpinBox()
        self.level_spin.setRange(1, 5)
        self.experience_spin = QSpinBox()
        self.experience_spin.setRange(0, 999999)

        form.addRow("Имя", self.name_edit)
        form.addRow("Раса", self.race_combo)
        form.addRow("Класс", self.class_combo)
        form.addRow("Уровень", self.level_spin)
        form.addRow("Опыт", self.experience_spin)
        form.addRow("Роль", self.role_edit)
        self.tabs.addTab(tab, "Основное")

        self.name_edit.textChanged.connect(self.update_review)
        self.role_edit.textChanged.connect(self.update_review)
        self.race_combo.currentTextChanged.connect(self._race_changed)
        self.class_combo.currentTextChanged.connect(self._class_changed)
        self.level_spin.valueChanged.connect(self._level_changed)

    def _build_stats_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.stat_editor = StatEditor()
        apply_button = QPushButton("Применить racial bonuses")
        apply_button.clicked.connect(self._apply_racial_bonuses)
        self.stat_editor.stats_changed.connect(self.update_review)
        layout.addWidget(self.stat_editor)
        layout.addWidget(apply_button)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Характеристики")

    def _build_class_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.subclass_combo = QComboBox()
        self.feat_combo = QComboBox()
        self.fighting_style_combo = QComboBox()
        form.addRow("Подкласс", self.subclass_combo)
        form.addRow("ASI/feat", self.feat_combo)
        form.addRow("Fighting Style", self.fighting_style_combo)
        self.tabs.addTab(tab, "Класс")
        self.subclass_combo.currentTextChanged.connect(self.update_review)
        self.feat_combo.currentTextChanged.connect(self.update_review)
        self.fighting_style_combo.currentTextChanged.connect(self.update_review)

    def _build_spells_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.spell_selector = SpellSelector()
        self.spell_selector.selection_changed.connect(self.update_review)
        layout.addWidget(self.spell_selector)
        self.tabs.addTab(tab, "Заклинания")

    def _build_equipment_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.armor_combo = QComboBox()
        for armor in ARMOR_OPTIONS:
            self.armor_combo.addItem(str(armor["name"]))
        layout.addWidget(QLabel("Броня"))
        layout.addWidget(self.armor_combo)
        layout.addWidget(QLabel("Оружие"))
        self.weapon_list = QListWidget()
        for weapon in WEAPON_OPTIONS:
            item = QListWidgetItem(str(weapon["name"]))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.weapon_list.addItem(item)
        self.weapon_list.itemChanged.connect(self.update_review)
        self.armor_combo.currentTextChanged.connect(self.update_review)
        layout.addWidget(self.weapon_list)
        self.tabs.addTab(tab, "Экипировка")

    def _build_inventory_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.inventory_editor = InventoryEditor()
        self.inventory_editor.inventory_changed.connect(self.update_review)
        layout.addWidget(self.inventory_editor)
        self.tabs.addTab(tab, "Инвентарь")

    def _build_review_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.review_text = QTextEdit()
        self.review_text.setReadOnly(True)
        layout.addWidget(self.review_text)
        self.tabs.addTab(tab, "Проверка")

    def _refresh_rule_options(self) -> None:
        ruleset = get_active_ruleset()
        self.race_combo.clear()
        self.race_combo.addItems(list(ruleset.supported_races))
        self.class_combo.clear()
        self.class_combo.addItems(
            [definition.name for definition in get_supported_class_definitions(ruleset)]
        )
        self._refresh_subclasses()
        self._refresh_class_options()
        self._refresh_spells()

    def _race_changed(self) -> None:
        self._applied_race_bonuses = {}
        self.update_review()

    def _class_changed(self) -> None:
        self._refresh_subclasses()
        self._refresh_class_options()
        self._refresh_spells()
        self.update_review()

    def _level_changed(self) -> None:
        self._refresh_subclasses()
        self._refresh_class_options()
        self._refresh_spells()
        self.update_review()

    def _refresh_subclasses(self) -> None:
        current = self.subclass_combo.currentText()
        self.subclass_combo.clear()
        self.subclass_combo.addItem("")
        for definition in get_supported_subclass_definitions(
            self.class_combo.currentText(),
            level=self.level_spin.value(),
        ):
            self.subclass_combo.addItem(definition.name)
        self._select_combo_text(self.subclass_combo, current)

    def _refresh_class_options(self) -> None:
        current_feat = self.feat_combo.currentText()
        self.feat_combo.clear()
        self.feat_combo.addItem("")
        if self.level_spin.value() >= 4:
            for definition in get_supported_feat_definitions():
                self.feat_combo.addItem(definition.name)
        self._select_combo_text(self.feat_combo, current_feat)

        current_style = self.fighting_style_combo.currentText()
        self.fighting_style_combo.clear()
        self.fighting_style_combo.addItem("")
        if self.class_combo.currentText() == "Fighter":
            self.fighting_style_combo.addItems(list(FIGHTING_STYLES))
        self._select_combo_text(self.fighting_style_combo, current_style)

    def _refresh_spells(self) -> None:
        selected = self.spell_selector.selected_spell_names()
        self.spell_selector.set_options(
            self.class_combo.currentText(),
            self.level_spin.value(),
            selected,
        )

    def _apply_racial_bonuses(self) -> None:
        stats = self.stat_editor.stats()
        for ability, bonus in self._applied_race_bonuses.items():
            stats[ability] = stats.get(ability, 10) - bonus
        race = get_race_definition(self.race_combo.currentText())
        bonuses = dict(race.ability_score_bonuses)
        for ability, bonus in bonuses.items():
            stats[ability] = stats.get(ability, 10) + bonus
        self._applied_race_bonuses = bonuses
        self.stat_editor.set_stats(stats)
        self.update_review()

    def _collect_data(self) -> CharacterBuilderData:
        return CharacterBuilderData(
            id=self._editing_id,
            name=self.name_edit.text().strip(),
            race_name=self.race_combo.currentText(),
            class_name=self.class_combo.currentText(),
            subclass_name=self.subclass_combo.currentText() or None,
            level=self.level_spin.value(),
            experience=self.experience_spin.value(),
            role=self.role_edit.text().strip() or "combatant",
            stats=self.stat_editor.stats(),
            feat_name=self.feat_combo.currentText() or None,
            fighting_style=self.fighting_style_combo.currentText() or None,
            prepared_spells=self.spell_selector.selected_spell_names(),
            weapons=self._checked_names(self.weapon_list),
            armor_name=self.armor_combo.currentText(),
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
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _checked_names(list_widget: QListWidget) -> tuple[str, ...]:
        names = []
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(item.text())
        return tuple(names)

    @staticmethod
    def _set_checked_names(list_widget: QListWidget, names: tuple[str, ...]) -> None:
        selected = {_key(name) for name in names}
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            state = Qt.CheckState.Checked if _key(item.text()) in selected else Qt.CheckState.Unchecked
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
        raise ValueError(f"Unsupported subclass for level {data.level}: {data.subclass_name}")
    allowed_spells = {
        definition.name
        for definition in get_supported_spell_definitions(data.class_name, data.level)
    }
    for spell_name in data.prepared_spells:
        if spell_name not in allowed_spells:
            raise ValueError(
                f"Spell '{spell_name}' is not supported for {data.class_name} level {data.level}."
            )
    allowed_items = {definition.name for definition in get_supported_item_definitions()}
    for item in data.inventory:
        if str(item.get("name")) not in allowed_items:
            raise ValueError(f"Unsupported item: {item.get('name')}")
    if data.feat_name:
        if int(data.level) < 4:
            raise ValueError("ASI/feat can be selected only at level 4 or higher.")
        supported_feats = {definition.name for definition in get_supported_feat_definitions()}
        if data.feat_name not in supported_feats:
            raise ValueError(f"Unsupported feat: {data.feat_name}")
    allowed_weapons = {str(weapon["name"]) for weapon in WEAPON_OPTIONS}
    for weapon_name in data.weapons:
        if weapon_name not in allowed_weapons:
            raise ValueError(f"Unsupported weapon: {weapon_name}")
    if data.fighting_style and data.class_name != "Fighter":
        raise ValueError("Fighting Style is available only for Fighter.")
    if data.fighting_style and data.fighting_style not in FIGHTING_STYLES:
        raise ValueError(f"Unsupported fighting style: {data.fighting_style}")
    if not class_definition_allows_subclass_absence(data.class_name, data.level, data.subclass_name):
        raise ValueError(f"{data.class_name} requires a subclass at level {data.level}.")


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


def _key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())
