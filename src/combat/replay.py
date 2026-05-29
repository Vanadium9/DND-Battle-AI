"""Battle replay serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import Any

from combat.common_actions import (
    ActionResult,
    ActionSurgeAction,
    AttackAction,
    CastSpellAction,
    ChannelDivinityPreserveLifeAction,
    CombatAction,
    DashAction,
    DisengageAction,
    DodgeAction,
    EndTurnAction,
    GrappleAction,
    HelpAction,
    HideAction,
    ImprovisedAction,
    MoveAction,
    OpportunityAttackAction,
    ReadyAction,
    SearchAction,
    SecondWindAction,
    ShoveAction,
    StabilizeAction,
    UseObjectAction,
)
from combat.items import ItemActionCost, normalize_action_cost
from combat.models import Character, CombatState, Position


REPLAY_FORMAT = "BattleReplay"
REPLAY_VERSION = 1


StateSnapshot = dict[str, Any]


@dataclass
class BattleReplay:
    """Collect and save structured combat steps as JSON."""

    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def snapshot_state(self, state: CombatState) -> StateSnapshot:
        """Return a JSON-ready immutable snapshot of the current combat state."""

        return _state_snapshot(state)

    def record_step(
        self,
        before: CombatState | StateSnapshot,
        after: CombatState,
        action: CombatAction,
        result: ActionResult,
    ) -> dict[str, Any]:
        """Record one action step using state before and after execution."""

        before_snapshot = (
            before if isinstance(before, dict) else self.snapshot_state(before)
        )
        after_snapshot = self.snapshot_state(after)
        step = _build_step(before_snapshot, after_snapshot, after, action, result)
        self.steps.append(step)
        return step

    def to_dict(self) -> dict[str, Any]:
        """Return the full replay payload."""

        winner = self.steps[-1]["winner"] if self.steps else None
        xp_total = 0
        if self.steps:
            xp_total = int(self.steps[-1]["xp_gained"].get("total", 0))
        return {
            "format": REPLAY_FORMAT,
            "version": REPLAY_VERSION,
            "metadata": dict(self.metadata),
            "steps": list(self.steps),
            "winner": winner,
            "xp_gained": xp_total,
        }

    def save(self, path: str | Path) -> Path:
        """Save the replay JSON and return the resolved path."""

        replay_path = Path(path)
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return replay_path


def _build_step(
    before: StateSnapshot,
    after: StateSnapshot,
    after_state: CombatState,
    action: CombatAction,
    result: ActionResult,
) -> dict[str, Any]:
    actor = _character_by_id(before, action.actor_id)
    return {
        "round": before["round_number"],
        "turn_index": before["current_turn_index"],
        "initiative_order": _initiative_order(before),
        "actor": _actor_payload(actor, action.actor_id),
        "actor_team": actor["team"] if actor is not None else None,
        "positions": _positions(after),
        "hp_values": _hp_values(after),
        "conditions": _conditions(after),
        "resources": _resources(after),
        "action": _action_payload(action),
        "action_category": _action_category(action),
        "targets": _targets(before, action),
        "dice_rolls": _dice_rolls(result.description),
        "damage": _damage_delta(before, after),
        "healing": _healing_delta(before, after),
        "spell_slots_spent": _spell_slots_spent(before, after),
        "items_spent": _items_spent(before, after),
        "deaths": _deaths(before, after),
        "reward": float(result.reward),
        "reward_breakdown": dict(result.reward_breakdown),
        "success": bool(result.success),
        "description": result.description,
        "map_metadata": after["map_metadata"],
        "cover_line_of_sight": _cover_line_of_sight(before, after_state, action),
        "winner": _winner(after),
        "xp_gained": _xp_gained(before, after),
    }


def _state_snapshot(state: CombatState) -> StateSnapshot:
    characters = [
        _character_snapshot(character_id, character)
        for character_id, character in enumerate(state.characters)
    ]
    return {
        "round_number": int(state.round_number),
        "turn_index": int(state.turn_index),
        "current_turn_index": int(state.current_turn_index),
        "initiative_order": list(state.initiative_order),
        "characters": characters,
        "map_metadata": _map_metadata(state),
    }


def _character_snapshot(character_id: int, character: Character) -> dict[str, Any]:
    action_economy = character.action_economy
    return {
        "id": character_id,
        "name": character.name,
        "team": character.team.value,
        "position": _position_payload(character.position),
        "hp": int(character.hp),
        "max_hp": int(character.max_hp),
        "ac": int(character.ac),
        "alive": bool(character.is_alive),
        "conditions": [
            {
                "name": condition.name,
                "duration_rounds": condition.duration_rounds,
                "description": condition.description,
            }
            for condition in character.conditions
        ],
        "condition_flags": {
            "prone": bool(character.prone),
            "grappled": bool(character.grappled),
            "hidden": bool(character.hidden),
            "dodging": bool(character.dodging_until_start_of_next_turn),
            "disengaged": bool(character.disengaged_until_end_of_turn),
            "stable": bool(character.stable),
            "active_concentration": (
                character.active_concentration_spell.name
                if character.active_concentration_spell is not None
                else None
            ),
        },
        "resources": {
            name: int(resource.uses_remaining or 0)
            for name, resource in character.resources.items()
        },
        "spell_slots": {
            str(level): int(count)
            for level, count in character.spell_slots.items()
        },
        "spell_slots_remaining": {
            str(level): int(count)
            for level, count in character.spell_slots_remaining.items()
        },
        "inventory": [
            {
                "name": getattr(item, "name", str(item)),
                "quantity": int(getattr(item, "quantity", 0)),
            }
            for item in getattr(character, "inventory", ())
        ],
        "action_economy": {
            "action_available": bool(action_economy.action_available),
            "bonus_action_available": bool(action_economy.bonus_action_available),
            "reaction_available": bool(action_economy.reaction_available),
            "movement_remaining": int(action_economy.movement_remaining),
            "free_object_interaction_available": bool(
                action_economy.free_object_interaction_available
            ),
            "reaction_used_this_round": bool(action_economy.reaction_used_this_round),
            "prepared_action": action_economy.prepared_action,
            "trigger_description": action_economy.trigger_description,
        },
        "experience": int(getattr(character, "experience", 0)),
    }


def _map_metadata(state: CombatState) -> dict[str, Any] | None:
    grid_map = state.grid_map
    if grid_map is None:
        return None

    terrain_snapshot = [
        [grid_map.terrain_at(Position(x, y)).value for x in range(grid_map.width)]
        for y in range(grid_map.height)
    ]
    terrain_counts: dict[str, int] = {}
    for row in terrain_snapshot:
        for terrain in row:
            terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1

    return {
        "width": grid_map.width,
        "height": grid_map.height,
        "terrain_counts": terrain_counts,
        "terrain_snapshot": terrain_snapshot,
    }


def _initiative_order(snapshot: StateSnapshot) -> list[dict[str, Any]]:
    order = snapshot["initiative_order"] or [
        character["id"] for character in snapshot["characters"]
    ]
    result = []
    for character_id in order:
        character = _character_by_id(snapshot, int(character_id))
        result.append(
            {
                "id": int(character_id),
                "name": character["name"] if character is not None else None,
            }
        )
    return result


def _actor_payload(character: dict[str, Any] | None, actor_id: int) -> dict[str, Any]:
    if character is None:
        return {"id": int(actor_id), "name": None}
    return {
        "id": character["id"],
        "name": character["name"],
    }


def _positions(snapshot: StateSnapshot) -> dict[str, dict[str, Any]]:
    return {
        str(character["id"]): {
            "name": character["name"],
            **character["position"],
        }
        for character in snapshot["characters"]
    }


def _hp_values(snapshot: StateSnapshot) -> dict[str, dict[str, Any]]:
    return {
        str(character["id"]): {
            "name": character["name"],
            "hp": character["hp"],
            "max_hp": character["max_hp"],
            "alive": character["alive"],
        }
        for character in snapshot["characters"]
    }


def _conditions(snapshot: StateSnapshot) -> dict[str, dict[str, Any]]:
    return {
        str(character["id"]): {
            "name": character["name"],
            "conditions": character["conditions"],
            "flags": character["condition_flags"],
        }
        for character in snapshot["characters"]
    }


def _resources(snapshot: StateSnapshot) -> dict[str, dict[str, Any]]:
    return {
        str(character["id"]): {
            "name": character["name"],
            "class_resources": character["resources"],
            "spell_slots": character["spell_slots"],
            "spell_slots_remaining": character["spell_slots_remaining"],
            "inventory": character["inventory"],
            "action_economy": character["action_economy"],
        }
        for character in snapshot["characters"]
    }


def _action_payload(action: CombatAction) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": action.__class__.__name__,
        "actor_id": int(action.actor_id),
    }
    if isinstance(action, MoveAction):
        payload["destination"] = _position_payload(action.destination)
    if isinstance(action, AttackAction):
        payload["target_id"] = action.target_id
        payload["weapon"] = action.weapon.name if action.weapon is not None else None
    if isinstance(action, CastSpellAction):
        payload["spell"] = action.spell.name if action.spell is not None else None
        payload["target_id"] = action.target_id
        payload["target_cell"] = _optional_position_payload(action.target_cell)
        payload["direction"] = _value(action.direction)
        payload["cast_level"] = action.cast_level
    if isinstance(action, UseObjectAction):
        payload["object_name"] = action.object_name
        payload["item"] = action.item.name if action.item is not None else None
        payload["target_id"] = action.target_id
        payload["target_cell"] = _optional_position_payload(action.target_cell)
        payload["direction"] = _value(action.direction)
    if isinstance(action, (HelpAction, GrappleAction, ShoveAction, StabilizeAction)):
        payload["target_id"] = action.target_id
    if isinstance(action, ShoveAction):
        payload["shove_effect"] = action.shove_effect
    if isinstance(action, HideAction):
        payload["dc"] = action.dc
        payload["observer_id"] = action.observer_id
    if isinstance(action, SearchAction):
        payload["skill"] = action.skill
        payload["dc"] = action.dc
    if isinstance(action, ReadyAction):
        payload["prepared_action"] = action.prepared_action
        payload["trigger_description"] = action.trigger_description
    if isinstance(action, ImprovisedAction):
        payload["description"] = action.description
    return payload


def _action_category(action: CombatAction) -> str:
    if isinstance(action, MoveAction):
        return "movement"
    if isinstance(action, EndTurnAction):
        return "end_turn"
    if isinstance(action, OpportunityAttackAction):
        return "reaction"
    if isinstance(action, SecondWindAction):
        return "bonus_action"
    if isinstance(action, ActionSurgeAction):
        return "class_feature"
    if isinstance(action, ChannelDivinityPreserveLifeAction):
        return "main_action"
    if isinstance(action, CastSpellAction) and action.spell is not None:
        return _action_cost_category(action.spell.action_cost)
    if isinstance(action, UseObjectAction) and action.item is not None:
        item_cost = normalize_action_cost(action.item.action_cost)
        return _item_cost_category(item_cost)
    if isinstance(
        action,
        (
            AttackAction,
            DashAction,
            DisengageAction,
            DodgeAction,
            HelpAction,
            HideAction,
            SearchAction,
            UseObjectAction,
            ReadyAction,
            GrappleAction,
            ShoveAction,
            StabilizeAction,
            ImprovisedAction,
        ),
    ):
        return "main_action"
    return "unknown"


def _targets(snapshot: StateSnapshot, action: CombatAction) -> list[dict[str, Any]]:
    targets = []
    target_id = getattr(action, "target_id", None)
    if target_id is not None:
        target = _character_by_id(snapshot, int(target_id))
        targets.append(
            {
                "type": "creature",
                "id": int(target_id),
                "name": target["name"] if target is not None else None,
                "team": target["team"] if target is not None else None,
            }
        )
    destination = getattr(action, "destination", None)
    if isinstance(destination, Position):
        targets.append({"type": "movement_cell", **_position_payload(destination)})
    target_cell = getattr(action, "target_cell", None)
    if isinstance(target_cell, Position):
        targets.append({"type": "target_cell", **_position_payload(target_cell)})
    direction = getattr(action, "direction", None)
    if direction is not None:
        targets.append({"type": "direction", "direction": _value(direction)})
    return targets


def _dice_rolls(description: str) -> list[dict[str, Any]]:
    rolls: list[dict[str, Any]] = []
    for match in re.finditer(r"\bd20=([0-9/]+)", description):
        raw_rolls = match.group(1)
        rolls.append(
            {
                "die": "d20",
                "rolls": [int(value) for value in raw_rolls.split("/")],
            }
        )
    for match in re.finditer(r"\brolled\s+(\d+)", description, flags=re.IGNORECASE):
        rolls.append({"type": "effect_total", "total": int(match.group(1))})
    return rolls


def _damage_delta(before: StateSnapshot, after: StateSnapshot) -> list[dict[str, Any]]:
    damage = []
    for before_character, after_character in _paired_characters(before, after):
        amount = max(0, before_character["hp"] - after_character["hp"])
        if amount <= 0:
            continue
        damage.append(
            {
                "target_id": after_character["id"],
                "target": after_character["name"],
                "amount": amount,
                "hp_before": before_character["hp"],
                "hp_after": after_character["hp"],
            }
        )
    return damage


def _healing_delta(before: StateSnapshot, after: StateSnapshot) -> list[dict[str, Any]]:
    healing = []
    for before_character, after_character in _paired_characters(before, after):
        amount = max(0, after_character["hp"] - before_character["hp"])
        if amount <= 0:
            continue
        healing.append(
            {
                "target_id": after_character["id"],
                "target": after_character["name"],
                "amount": amount,
                "hp_before": before_character["hp"],
                "hp_after": after_character["hp"],
            }
        )
    return healing


def _spell_slots_spent(before: StateSnapshot, after: StateSnapshot) -> list[dict[str, Any]]:
    spent = []
    for before_character, after_character in _paired_characters(before, after):
        levels = set(before_character["spell_slots_remaining"]) | set(
            after_character["spell_slots_remaining"]
        )
        for level in sorted(levels, key=int):
            before_count = int(before_character["spell_slots_remaining"].get(level, 0))
            after_count = int(after_character["spell_slots_remaining"].get(level, 0))
            amount = max(0, before_count - after_count)
            if amount <= 0:
                continue
            spent.append(
                {
                    "character_id": after_character["id"],
                    "character": after_character["name"],
                    "level": int(level),
                    "spent": amount,
                }
            )
    return spent


def _items_spent(before: StateSnapshot, after: StateSnapshot) -> list[dict[str, Any]]:
    spent = []
    for before_character, after_character in _paired_characters(before, after):
        before_items = _inventory_quantities(before_character)
        after_items = _inventory_quantities(after_character)
        for item_name in sorted(set(before_items) | set(after_items)):
            amount = max(0, before_items.get(item_name, 0) - after_items.get(item_name, 0))
            if amount <= 0:
                continue
            spent.append(
                {
                    "character_id": after_character["id"],
                    "character": after_character["name"],
                    "item": item_name,
                    "spent": amount,
                }
            )
    return spent


def _deaths(before: StateSnapshot, after: StateSnapshot) -> list[dict[str, Any]]:
    deaths = []
    for before_character, after_character in _paired_characters(before, after):
        if before_character["alive"] and not after_character["alive"]:
            deaths.append(
                {
                    "id": after_character["id"],
                    "name": after_character["name"],
                    "team": after_character["team"],
                }
            )
    return deaths


def _xp_gained(before: StateSnapshot, after: StateSnapshot) -> dict[str, Any]:
    by_character = {}
    total = 0
    for before_character, after_character in _paired_characters(before, after):
        amount = max(
            0,
            int(after_character["experience"]) - int(before_character["experience"]),
        )
        if amount <= 0:
            continue
        total += amount
        by_character[str(after_character["id"])] = {
            "name": after_character["name"],
            "amount": amount,
        }
    return {
        "total": total,
        "by_character": by_character,
    }


def _cover_line_of_sight(
    snapshot: StateSnapshot,
    state: CombatState,
    action: CombatAction,
) -> dict[str, Any] | None:
    grid_map = state.grid_map
    if grid_map is None:
        return None

    actor = _character_by_id(snapshot, action.actor_id)
    if actor is None:
        return None
    target_position = _target_position(snapshot, action)
    if target_position is None:
        return None

    actor_position = _position_from_payload(actor["position"])
    return {
        "actor_position": _position_payload(actor_position),
        "target_position": _position_payload(target_position),
        "line_of_sight": bool(grid_map.line_of_sight(actor_position, target_position)),
        "cover": grid_map.get_cover_between(actor_position, target_position).value,
    }


def _target_position(snapshot: StateSnapshot, action: CombatAction) -> Position | None:
    target_cell = getattr(action, "target_cell", None)
    if isinstance(target_cell, Position):
        return target_cell
    target_id = getattr(action, "target_id", None)
    if target_id is not None:
        target = _character_by_id(snapshot, int(target_id))
        if target is not None:
            return _position_from_payload(target["position"])
    return None


def _winner(snapshot: StateSnapshot) -> str | None:
    living_teams = {
        character["team"]
        for character in snapshot["characters"]
        if bool(character["alive"])
    }
    if len(living_teams) != 1:
        return None
    return next(iter(living_teams))


def _paired_characters(
    before: StateSnapshot,
    after: StateSnapshot,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    after_by_id = {
        int(character["id"]): character
        for character in after["characters"]
    }
    pairs = []
    for before_character in before["characters"]:
        after_character = after_by_id.get(int(before_character["id"]))
        if after_character is not None:
            pairs.append((before_character, after_character))
    return pairs


def _inventory_quantities(character: dict[str, Any]) -> dict[str, int]:
    quantities: dict[str, int] = {}
    for item in character["inventory"]:
        item_name = str(item["name"])
        quantities[item_name] = quantities.get(item_name, 0) + int(item["quantity"])
    return quantities


def _character_by_id(
    snapshot: StateSnapshot,
    character_id: int,
) -> dict[str, Any] | None:
    for character in snapshot["characters"]:
        if int(character["id"]) == int(character_id):
            return character
    return None


def _position_payload(position: Position) -> dict[str, int]:
    return {
        "x": int(position.x),
        "y": int(position.y),
    }


def _optional_position_payload(position: Position | None) -> dict[str, int] | None:
    if position is None:
        return None
    return _position_payload(position)


def _position_from_payload(payload: dict[str, Any]) -> Position:
    return Position(int(payload["x"]), int(payload["y"]))


def _action_cost_category(action_cost: str | None) -> str:
    if action_cost == "reaction":
        return "reaction"
    if action_cost == "bonus_action":
        return "bonus_action"
    return "main_action"


def _item_cost_category(action_cost: ItemActionCost) -> str:
    if action_cost is ItemActionCost.REACTION:
        return "reaction"
    if action_cost is ItemActionCost.BONUS_ACTION:
        return "bonus_action"
    if action_cost is ItemActionCost.FREE_INTERACTION:
        return "free_interaction"
    return "main_action"


def _value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value
