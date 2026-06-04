"""Inventory item editor widget."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QSpinBox, QVBoxLayout, QWidget

from combat.inventory import get_supported_item_definitions
from ui.text import ru_label


class InventoryEditor(QWidget):
    """Quantity editor for implemented supported inventory items."""

    inventory_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spin_boxes: dict[str, QSpinBox] = {}
        self._build_layout()

    def set_inventory(self, inventory: tuple[dict[str, object], ...]) -> None:
        """Replace visible inventory quantities."""

        quantities = {
            str(item.get("name")): int(item.get("quantity", 0))
            for item in inventory
        }
        changed = False
        for name, spin_box in self._spin_boxes.items():
            value = quantities.get(name, 0)
            if spin_box.value() == value:
                continue
            was_blocked = spin_box.blockSignals(True)
            spin_box.setValue(value)
            spin_box.blockSignals(was_blocked)
            changed = True
        if changed:
            self.inventory_changed.emit()

    def inventory_items(self) -> tuple[dict[str, object], ...]:
        """Return selected item payloads."""

        items = []
        definitions = {
            definition.name: definition
            for definition in get_supported_item_definitions()
        }
        for name, spin_box in self._spin_boxes.items():
            quantity = spin_box.value()
            if quantity <= 0:
                continue
            definition = definitions[name]
            items.append(
                {
                    "name": definition.name,
                    "item_type": definition.item_type,
                    "quantity": quantity,
                    "action_cost": str(getattr(definition.action_cost, "value", definition.action_cost)),
                    "target_type": str(getattr(definition.target_type, "value", definition.target_type)),
                    "range": definition.range,
                    "consumable": definition.consumable,
                    "implemented": definition.implemented,
                }
            )
        return tuple(items)

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        for row, definition in enumerate(get_supported_item_definitions()):
            label = QLabel(ru_label(definition.name))
            spin_box = QSpinBox()
            spin_box.setRange(0, 99)
            spin_box.valueChanged.connect(lambda _value: self.inventory_changed.emit())
            self._spin_boxes[definition.name] = spin_box
            grid.addWidget(label, row, 0)
            grid.addWidget(spin_box, row, 1)
        root.addLayout(grid)
        root.addStretch(1)
