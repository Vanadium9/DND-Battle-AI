"""Supported spell definition modules."""

from combat.spells.cleric_spells import (
    CLERIC_DEFAULT_CANTRIPS,
    CLERIC_DEFAULT_PREPARED_SPELLS,
    CLERIC_SPELLS,
)
from combat.spells.wizard_spells import (
    WIZARD_DEFAULT_CANTRIPS,
    WIZARD_DEFAULT_PREPARED_SPELLS,
    WIZARD_SPELLS,
)

__all__ = [
    "CLERIC_DEFAULT_CANTRIPS",
    "CLERIC_DEFAULT_PREPARED_SPELLS",
    "CLERIC_SPELLS",
    "WIZARD_DEFAULT_CANTRIPS",
    "WIZARD_DEFAULT_PREPARED_SPELLS",
    "WIZARD_SPELLS",
]
