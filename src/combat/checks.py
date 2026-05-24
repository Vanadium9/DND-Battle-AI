"""Ability checks and contested checks for D&D-like combat actions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import random
from typing import Any

from combat.models import Character


ABILITY_NAMES = {"str", "dex", "con", "int", "wis", "cha"}
SKILL_TO_ABILITY = {
    "athletics": "str",
    "acrobatics": "dex",
    "stealth": "dex",
    "perception": "wis",
    "investigation": "int",
    "medicine": "wis",
}


@dataclass(frozen=True)
class AbilityCheckResult:
    """Detailed result of one ability or skill check."""

    character_name: str
    check_name: str
    ability: str
    rolls: tuple[int, ...]
    kept_roll: int
    ability_modifier: int
    proficiency_bonus: int
    total: int
    advantage_state: str = "normal"

    @property
    def log(self) -> str:
        rolls_text = "/".join(str(roll) for roll in self.rolls)
        proficiency_text = (
            f" + proficiency {self.proficiency_bonus}"
            if self.proficiency_bonus
            else ""
        )
        return (
            f"{self.check_name}: d20={rolls_text}, kept={self.kept_roll}, "
            f"{self.ability}_mod={self.ability_modifier}"
            f"{proficiency_text}, total={self.total}"
        )

    def __int__(self) -> int:
        return self.total

    def __str__(self) -> str:
        return str(self.total)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AbilityCheckResult):
            return self.total == other.total
        if isinstance(other, int):
            return self.total == other
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, AbilityCheckResult):
            return self.total < other.total
        if isinstance(other, int):
            return self.total < other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, AbilityCheckResult):
            return self.total <= other.total
        if isinstance(other, int):
            return self.total <= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, AbilityCheckResult):
            return self.total > other.total
        if isinstance(other, int):
            return self.total > other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, AbilityCheckResult):
            return self.total >= other.total
        if isinstance(other, int):
            return self.total >= other
        return NotImplemented


@dataclass(frozen=True)
class InitiativeCheckResult:
    """Detailed result of one initiative roll."""

    character_name: str
    roll: int
    dex_modifier: int
    total: int

    @property
    def log(self) -> str:
        return (
            f"initiative: d20={self.roll}, dex_mod={self.dex_modifier}, "
            f"total={self.total}"
        )


@dataclass(frozen=True)
class ContestedCheckResult:
    """Detailed result of a contested ability check."""

    actor_result: AbilityCheckResult
    target_result: AbilityCheckResult
    target_options: tuple[AbilityCheckResult, ...]

    @property
    def actor_wins(self) -> bool:
        return self.actor_result.total >= self.target_result.total

    @property
    def log(self) -> str:
        if len(self.target_options) <= 1:
            target_text = self.target_result.log
        else:
            options = "; ".join(result.log for result in self.target_options)
            target_text = f"best of [{options}]"
        return f"{self.actor_result.log} vs {target_text}"

    def __iter__(self):
        yield self.actor_result.total
        yield self.target_result.total


def ability_modifier(score: int) -> int:
    """Return a D&D-style ability modifier for a raw ability score."""

    return (int(score) - 10) // 2


def roll_d20(rng: random.Random | None = None) -> int:
    """Roll one d20."""

    random_source = rng or random
    return random_source.randint(1, 20)


def roll_initiative_check(
    character: Character,
    rng: random.Random | None = None,
) -> InitiativeCheckResult:
    """Roll initiative as d20 plus DEX modifier."""

    roll = roll_d20(rng)
    dex_modifier = ability_modifier(character.stats.dex)
    return InitiativeCheckResult(
        character_name=character.name,
        roll=roll,
        dex_modifier=dex_modifier,
        total=roll + dex_modifier,
    )


def roll_ability_check(
    character: Character,
    ability: str,
    proficiency: bool = False,
    advantage_state: str = "normal",
) -> AbilityCheckResult:
    """Roll an ability or skill check and return detailed roll information."""

    ability_key = ability.lower()
    ability_name = SKILL_TO_ABILITY.get(ability_key, ability_key)
    if ability_name not in ABILITY_NAMES:
        raise ValueError(f"Unknown ability or skill: {ability}")

    normalized_advantage = _normalize_advantage_state(advantage_state)
    if normalized_advantage == "normal" and character.advantage_on_next_check:
        normalized_advantage = "advantage"
        character.advantage_on_next_check = False

    rolls = _roll_with_advantage_state(normalized_advantage)
    kept_roll = _kept_roll(rolls, normalized_advantage)
    raw_score = getattr(character.stats, ability_name)
    stat_modifier = ability_modifier(raw_score)
    proficiency_bonus = character.proficiency_bonus if proficiency else 0
    total = kept_roll + stat_modifier + proficiency_bonus

    result = AbilityCheckResult(
        character_name=character.name,
        check_name=_display_check_name(ability_key),
        ability=ability_name,
        rolls=rolls,
        kept_roll=kept_roll,
        ability_modifier=stat_modifier,
        proficiency_bonus=proficiency_bonus,
        total=total,
        advantage_state=normalized_advantage,
    )
    from combat.features import on_ability_check

    return on_ability_check(character, result, ability=ability_key)


def roll_contested_check(
    actor: Character,
    target: Character,
    actor_check: Any,
    target_check_options: Iterable[Any],
) -> ContestedCheckResult:
    """Roll actor check against the target's best available check option."""

    actor_result = _roll_check_request(actor, actor_check)
    target_results = tuple(
        _roll_check_request(target, target_check)
        for target_check in target_check_options
    )
    if not target_results:
        raise ValueError("target_check_options must contain at least one check")

    target_result = max(target_results, key=lambda result: result.total)
    return ContestedCheckResult(
        actor_result=actor_result,
        target_result=target_result,
        target_options=target_results,
    )


def passive_perception(character: Character, proficiency: bool = True) -> int:
    """Return a simplified passive Perception DC."""

    proficiency_bonus = character.proficiency_bonus if proficiency else 0
    return 10 + ability_modifier(character.stats.wis) + proficiency_bonus


def _roll_check_request(character: Character, request: Any) -> AbilityCheckResult:
    if isinstance(request, str):
        return roll_ability_check(character, request, proficiency=True)
    if isinstance(request, tuple):
        ability = request[0]
        proficiency = bool(request[1]) if len(request) > 1 else True
        advantage_state = str(request[2]) if len(request) > 2 else "normal"
        return roll_ability_check(character, ability, proficiency, advantage_state)
    if isinstance(request, dict):
        return roll_ability_check(
            character,
            str(request["ability"]),
            bool(request.get("proficiency", True)),
            str(request.get("advantage_state", "normal")),
        )
    raise TypeError(f"Unsupported check request: {request!r}")


def _roll_with_advantage_state(advantage_state: str) -> tuple[int, ...]:
    if advantage_state == "normal":
        return (roll_d20(),)
    return (roll_d20(), roll_d20())


def _kept_roll(rolls: tuple[int, ...], advantage_state: str) -> int:
    if advantage_state == "advantage":
        return max(rolls)
    if advantage_state == "disadvantage":
        return min(rolls)
    return rolls[0]


def _normalize_advantage_state(advantage_state: str) -> str:
    normalized = advantage_state.lower()
    if normalized not in {"normal", "advantage", "disadvantage"}:
        raise ValueError(f"Unknown advantage_state: {advantage_state}")
    return normalized


def _display_check_name(ability: str) -> str:
    if ability in SKILL_TO_ABILITY:
        return ability.replace("_", " ").title()
    return ability.upper()
