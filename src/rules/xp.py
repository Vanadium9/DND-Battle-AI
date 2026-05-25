"""Campaign XP rules for defeated monsters."""

from __future__ import annotations

from fractions import Fraction
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from combat.models import Character


CR_XP_TABLE: dict[str, int] = {
    "0": 10,
    "1/8": 25,
    "1/4": 50,
    "1/2": 100,
    "1": 200,
    "2": 450,
    "3": 700,
    "4": 1100,
    "5": 1800,
}


def get_xp_for_cr(cr: object) -> int:
    """Return monster XP for a supported challenge rating."""

    cr_key = _normalize_cr(cr)
    try:
        return CR_XP_TABLE[cr_key]
    except KeyError as exc:
        supported = ", ".join(CR_XP_TABLE)
        raise ValueError(
            f"Unsupported challenge rating {cr!r}. Supported: {supported}."
        ) from exc


def calculate_encounter_xp(monsters: Iterable[object]) -> int:
    """Return total campaign XP for defeated monsters."""

    total_xp = 0
    for monster in monsters:
        xp_value = int(getattr(monster, "xp_value", 0) or 0)
        if xp_value > 0:
            total_xp += xp_value
            continue

        challenge_rating = getattr(monster, "challenge_rating", None)
        if challenge_rating is None:
            continue
        total_xp += get_xp_for_cr(challenge_rating)
    return total_xp


def award_party_xp(
    party: Iterable["Character"],
    defeated_monsters: Iterable[object],
) -> int:
    """Award encounter XP as evenly as possible and return total encounter XP."""

    party_members = list(party)
    total_xp = calculate_encounter_xp(defeated_monsters)
    if total_xp <= 0 or not party_members:
        return total_xp

    base_share, remainder = divmod(total_xp, len(party_members))
    for index, character in enumerate(party_members):
        awarded_xp = base_share + (1 if index < remainder else 0)
        character.experience = (
            max(0, int(getattr(character, "experience", 0))) + awarded_xp
        )
    return total_xp


def _normalize_cr(cr: object) -> str:
    if isinstance(cr, Fraction):
        return _fraction_key(cr)
    if isinstance(cr, int):
        return str(cr)
    if isinstance(cr, float):
        return _fraction_key(Fraction(cr).limit_denominator(8))

    text = str(cr).strip()
    if not text:
        raise ValueError("Challenge rating cannot be empty.")
    return _fraction_key(Fraction(text))


def _fraction_key(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
