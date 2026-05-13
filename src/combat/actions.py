"""Combat action models."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re

from combat.models import Character, CombatState, Position, WeaponAttack


@dataclass(frozen=True)
class ActionResult:
    """Result of an action execution."""

    success: bool
    description: str
    reward: float = 0.0


@dataclass
class CombatAction:
    """Base combat action."""

    actor_id: int

    def is_valid(self, combat_state: CombatState) -> bool:
        raise NotImplementedError

    def execute(self, combat_state: CombatState) -> ActionResult:
        raise NotImplementedError


@dataclass
class MoveAction(CombatAction):
    """Move an actor to a reachable grid cell."""

    destination: Position

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None or actor.is_dead or combat_state.grid_map is None:
            return False
        movement_cost = _distance(actor.position, self.destination, combat_state)
        if movement_cost > actor.action_economy.movement_remaining:
            return False
        return self.destination in combat_state.grid_map.movement_cells(
            actor.position,
            actor.action_economy.movement_remaining,
            combat_state.characters,
        )

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Move failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot move to {self.destination}.")

        previous_position = actor.position
        movement_cost = _distance(previous_position, self.destination, combat_state)
        actor.position = self.destination
        actor.action_economy.spend_movement(movement_cost)
        return ActionResult(
            True,
            (
                f"{actor.name} moves from {previous_position} to {self.destination}. "
                f"Movement spent: {movement_cost}, "
                f"movement remaining: {actor.action_economy.movement_remaining}."
            ),
        )


@dataclass
class AttackAction(CombatAction):
    """Attack a target with a weapon attack."""

    target_id: int
    weapon: WeaponAttack | None = None

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        weapon = self._resolve_weapon(actor)

        if actor is None or target is None or weapon is None:
            return False
        if (
            actor.is_dead
            or target.is_dead
            or not weapon.available
            or not actor.action_economy.action_available
        ):
            return False
        return _distance(actor.position, target.position, combat_state) <= weapon.range

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        weapon = self._resolve_weapon(actor)

        if actor is None:
            return ActionResult(False, f"Attack failed: actor {self.actor_id} not found.")
        if target is None:
            return ActionResult(
                False,
                f"{actor.name} cannot attack missing target {self.target_id}.",
            )
        if weapon is None:
            return ActionResult(False, f"{actor.name} has no available weapon attack.")
        if not self.is_valid(combat_state):
            return ActionResult(
                False,
                f"{actor.name} cannot attack {target.name} with {weapon.name}.",
            )

        actor.action_economy.spend_action()
        d20_roll = random.randint(1, 20)
        attack_total = d20_roll + weapon.attack_bonus
        if attack_total < target.ac:
            return ActionResult(
                True,
                (
                    f"{actor.name} attacks {target.name} with {weapon.name}: "
                    f"miss ({attack_total} vs AC {target.ac}). "
                    "Action spent: action_available=False."
                ),
            )

        damage = _roll_damage(weapon.damage)
        target.hp = max(0, target.hp - damage)
        return ActionResult(
            True,
            (
                f"{actor.name} attacks {target.name} with {weapon.name}: "
                f"hit ({attack_total} vs AC {target.ac}) for {damage} damage. "
                "Action spent: action_available=False."
            ),
        )

    def _resolve_weapon(self, actor: Character | None) -> WeaponAttack | None:
        if actor is None:
            return None
        if self.weapon is not None:
            if self.weapon in actor.abilities:
                return self.weapon
            return None
        for ability in actor.available_abilities:
            if isinstance(ability, WeaponAttack):
                return ability
        return None


@dataclass
class EndTurnAction(CombatAction):
    """End the actor's turn."""

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        return actor is not None and actor.is_alive and bool(combat_state.characters)

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(
                False,
                f"End turn failed: actor {self.actor_id} not found.",
            )
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot end turn.")

        next_actor = combat_state.advance_turn()
        if next_actor is None:
            return ActionResult(True, f"{actor.name} ends turn. Combat has no living actors.")

        return ActionResult(
            True,
            (
                f"{actor.name} ends turn. {next_actor.name} starts turn with "
                f"action_available={next_actor.action_economy.action_available}, "
                f"bonus_action_available="
                f"{next_actor.action_economy.bonus_action_available}, "
                f"reaction_available={next_actor.action_economy.reaction_available}, "
                f"movement_remaining={next_actor.action_economy.movement_remaining}."
            ),
        )


def _get_character(combat_state: CombatState, character_id: int) -> Character | None:
    if character_id < 0 or character_id >= len(combat_state.characters):
        return None
    return combat_state.characters[character_id]


def _distance(first: Position, second: Position, combat_state: CombatState) -> int:
    if combat_state.grid_map is not None:
        return combat_state.grid_map.manhattan_distance(first, second)
    return abs(first.x - second.x) + abs(first.y - second.y)


def _roll_damage(damage: int | str) -> int:
    if isinstance(damage, int):
        return max(0, damage)

    damage_text = damage.strip().lower()
    if damage_text.isdigit():
        return int(damage_text)

    match = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", damage_text)
    if match is None:
        raise ValueError(f"Unsupported damage value: {damage}")

    dice_count = int(match.group(1) or 1)
    die_size = int(match.group(2))
    modifier = int(match.group(3) or 0)
    total = sum(random.randint(1, die_size) for _ in range(dice_count)) + modifier
    return max(0, total)
