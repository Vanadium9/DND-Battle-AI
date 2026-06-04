"""Condition and concentration runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
from typing import TYPE_CHECKING

from combat.checks import saving_throw_modifier

if TYPE_CHECKING:
    from combat.abilities import SpellAbility
    from combat.models import Character


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConcentrationSaveResult:
    """Result of a concentration save after taking damage."""

    spell_name: str
    damage: int
    dc: int
    d20_roll: int
    modifier: int
    total: int
    success: bool


def start_concentration(character: "Character", spell: "SpellAbility") -> None:
    """Start or replace a character's active concentration spell."""

    previous_spell = getattr(character, "active_concentration_spell", None)
    if previous_spell is not None:
        logger.info(
            "%s replaces concentration on %s with %s.",
            character.name,
            previous_spell.name,
            spell.name,
        )
    else:
        logger.info("%s starts concentration on %s.", character.name, spell.name)
    character.active_concentration_spell = spell


def end_concentration(character: "Character", reason: str = "") -> "SpellAbility | None":
    """End concentration and return the spell that was dropped."""

    active_spell = getattr(character, "active_concentration_spell", None)
    if active_spell is None:
        return None
    character.active_concentration_spell = None
    reason_text = f" ({reason})" if reason else ""
    logger.info(
        "%s loses concentration on %s%s.",
        character.name,
        active_spell.name,
        reason_text,
    )
    return active_spell


def handle_concentration_damage(
    character: "Character",
    damage: int,
) -> ConcentrationSaveResult | None:
    """Roll a CON save to maintain concentration after taking damage."""

    active_spell = getattr(character, "active_concentration_spell", None)
    normalized_damage = max(0, int(damage))
    if active_spell is None or normalized_damage <= 0:
        return None

    dc = int(max(10, normalized_damage / 2))
    d20_roll = random.randint(1, 20)
    modifier = saving_throw_modifier(character, "con")
    total = d20_roll + modifier
    success = total >= dc
    result = ConcentrationSaveResult(
        spell_name=active_spell.name,
        damage=normalized_damage,
        dc=dc,
        d20_roll=d20_roll,
        modifier=modifier,
        total=total,
        success=success,
    )
    logger.info(
        (
            "%s rolls concentration save for %s after %s damage: "
            "%s + %s = %s vs DC %s (%s)."
        ),
        character.name,
        active_spell.name,
        normalized_damage,
        d20_roll,
        modifier,
        total,
        dc,
        "success" if success else "failure",
    )
    if not success:
        end_concentration(
            character,
            reason=f"failed CON save {total} vs DC {dc}",
        )
    return result
