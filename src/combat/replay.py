"""Battle replay serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import re
from typing import Any

from combat.action_economy import ActionEconomy
from combat.class_features import Resource
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
from combat.map import GridMap
from combat.items import ItemActionCost, normalize_action_cost
from combat.models import Character, CombatState, Condition, Position, Stats, Team
from combat.terrain import coerce_terrain_type


REPLAY_FORMAT = "BattleReplay"
REPLAY_VERSION = 1


StateSnapshot = dict[str, Any]


@dataclass(frozen=True)
class ReplaySummary:
    """Compact metadata displayed by GUI replay lists."""

    path: Path
    display_name: str
    modified_at: str
    winner: str | None
    round_count: int
    participants: tuple[str, ...]
    step_count: int


class ReplayLoadError(ValueError):
    """Raised when a replay JSON cannot be loaded or validated."""


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


def load_replay_file(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a BattleReplay JSON payload."""

    replay_path = Path(path)
    try:
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayLoadError(f"Cannot load replay {replay_path}: {error}") from error
    validate_replay_payload(payload)
    return payload


def validate_replay_payload(payload: dict[str, Any]) -> None:
    """Validate the structural fields needed by replay viewers."""

    if not isinstance(payload, dict):
        raise ReplayLoadError("Replay payload must be a JSON object.")
    if payload.get("format") != REPLAY_FORMAT:
        raise ReplayLoadError(f"Unsupported replay format: {payload.get('format')!r}")
    if not isinstance(payload.get("steps"), list):
        raise ReplayLoadError("Replay JSON must contain a steps list.")


def list_replay_summaries(directory: str | Path = "replays") -> list[ReplaySummary]:
    """Return summaries for all replay JSON files in a directory."""

    replay_dir = Path(directory)
    if not replay_dir.exists():
        return []
    summaries: list[ReplaySummary] = []
    for path in sorted(
        replay_dir.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        try:
            payload = load_replay_file(path)
        except ReplayLoadError:
            continue
        summaries.append(replay_summary(path, payload))
    return summaries


def replay_summary(path: str | Path, payload: dict[str, Any]) -> ReplaySummary:
    """Build display metadata for one replay payload."""

    replay_path = Path(path)
    steps = payload.get("steps") or []
    metadata = payload.get("metadata") or {}
    display_name = str(metadata.get("summary") or metadata.get("name") or replay_path.stem)
    winner = payload.get("winner")
    if winner is None and steps:
        winner = steps[-1].get("winner")
    rounds = [
        int(step.get("round", 0))
        for step in steps
        if isinstance(step, dict) and _is_int_like(step.get("round", 0))
    ]
    participants = _participants_from_steps(steps)
    modified_at = ""
    try:
        modified_at = datetime.fromtimestamp(replay_path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )
    except OSError:
        modified_at = ""
    return ReplaySummary(
        path=replay_path,
        display_name=display_name,
        modified_at=modified_at,
        winner=str(winner) if winner is not None else None,
        round_count=max(rounds, default=0),
        participants=participants,
        step_count=len(steps),
    )


def replay_step_to_state(step: dict[str, Any]) -> CombatState:
    """Convert one saved replay step into a minimal CombatState for GUI drawing."""

    characters = _characters_from_step(step)
    state = CombatState(
        characters=characters,
        grid_map=_grid_map_from_step(step),
        round_number=int(step.get("round", 1) or 1),
        initiative_order=_initiative_ids(step),
        current_turn_index=0,
        turn_index=0,
    )
    actor_id = int((step.get("actor") or {}).get("id", 0) or 0)
    if state.initiative_order and actor_id in state.initiative_order:
        state.current_turn_index = state.initiative_order.index(actor_id)
        state.turn_index = actor_id
    elif state.characters:
        state.turn_index = min(max(actor_id, 0), len(state.characters) - 1)
    return state


def _participants_from_steps(steps: list[Any]) -> tuple[str, ...]:
    if not steps:
        return ()
    first_step = steps[0] if isinstance(steps[0], dict) else {}
    hp_values = first_step.get("hp_values") or {}
    names = [
        str(payload.get("name"))
        for _, payload in sorted(
            hp_values.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
        )
        if isinstance(payload, dict) and payload.get("name")
    ]
    if names:
        return tuple(names)
    positions = first_step.get("positions") or {}
    return tuple(
        str(payload.get("name"))
        for _, payload in sorted(
            positions.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
        )
        if isinstance(payload, dict) and payload.get("name")
    )


def _characters_from_step(step: dict[str, Any]) -> list[Character]:
    ids = _character_ids_from_step(step)
    teams = _team_map_from_step(step)
    characters: list[Character] = []
    for character_id in ids:
        key = str(character_id)
        position_payload = (step.get("positions") or {}).get(key, {})
        hp_payload = (step.get("hp_values") or {}).get(key, {})
        resource_payload = (step.get("resources") or {}).get(key, {})
        condition_payload = (step.get("conditions") or {}).get(key, {})
        name = str(
            hp_payload.get("name")
            or position_payload.get("name")
            or resource_payload.get("name")
            or f"Creature {character_id}"
        )
        hp = int(hp_payload.get("hp", 0) or 0)
        max_hp = int(hp_payload.get("max_hp", max(1, hp)) or max(1, hp))
        movement = int(
            (resource_payload.get("action_economy") or {}).get("movement_remaining", 0)
            or 0
        )
        character = Character(
            name=name,
            hp=hp,
            max_hp=max(1, max_hp),
            ac=int(hp_payload.get("ac", position_payload.get("ac", 10)) or 10),
            position=_position_from_payload_safe(position_payload),
            speed=max(1, movement),
            stats=Stats(),
            team=teams.get(
                character_id,
                Team.PLAYERS if character_id == 0 else Team.ENEMIES,
            ),
        )
        character.conditions = [
            Condition(
                name=str(condition.get("name", "condition")),
                duration_rounds=condition.get("duration_rounds"),
                description=str(condition.get("description", "")),
            )
            for condition in condition_payload.get("conditions", [])
            if isinstance(condition, dict)
        ]
        _apply_condition_flags(character, condition_payload.get("flags") or {})
        character.resources = {
            str(name): Resource(str(name), max(0, int(value or 0)), int(value or 0))
            for name, value in (resource_payload.get("class_resources") or {}).items()
        }
        character.spell_slots = {
            int(level): int(value or 0)
            for level, value in (resource_payload.get("spell_slots") or {}).items()
            if _is_int_like(level)
        }
        character.spell_slots_remaining = {
            int(level): int(value or 0)
            for level, value in (resource_payload.get("spell_slots_remaining") or {}).items()
            if _is_int_like(level)
        }
        _apply_action_economy(
            character.action_economy,
            resource_payload.get("action_economy") or {},
        )
        if not bool(hp_payload.get("alive", hp > 0)):
            character.hp = min(character.hp, 0)
        characters.append(character)
    return characters


def _grid_map_from_step(step: dict[str, Any]) -> GridMap | None:
    metadata = step.get("map_metadata") or {}
    width = int(metadata.get("width", 0) or 0)
    height = int(metadata.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        positions = step.get("positions") or {}
        if not positions:
            return None
        width = max(int(position.get("x", 0)) for position in positions.values()) + 1
        height = max(int(position.get("y", 0)) for position in positions.values()) + 1
    terrain_snapshot = metadata.get("terrain_snapshot")
    terrain_grid = None
    if isinstance(terrain_snapshot, list):
        terrain_grid = tuple(
            tuple(coerce_terrain_type(cell) for cell in row)
            for row in terrain_snapshot
            if isinstance(row, list)
        )
    return GridMap(width=width, height=height, terrain_grid=terrain_grid)


def _character_ids_from_step(step: dict[str, Any]) -> list[int]:
    ids: set[int] = set()
    for key in (step.get("positions") or {}):
        if _is_int_like(key):
            ids.add(int(key))
    for key in (step.get("hp_values") or {}):
        if _is_int_like(key):
            ids.add(int(key))
    for item in step.get("initiative_order") or []:
        if isinstance(item, dict) and _is_int_like(item.get("id")):
            ids.add(int(item["id"]))
    return sorted(ids)


def _team_map_from_step(step: dict[str, Any]) -> dict[int, Team]:
    teams: dict[int, Team] = {}
    actor = step.get("actor") or {}
    if _is_int_like(actor.get("id")) and step.get("actor_team") is not None:
        teams[int(actor["id"])] = _team_from_value(step.get("actor_team"))
    for target in step.get("targets") or []:
        if (
            isinstance(target, dict)
            and _is_int_like(target.get("id"))
            and target.get("team") is not None
        ):
            teams[int(target["id"])] = _team_from_value(target.get("team"))
    for death in step.get("deaths") or []:
        if (
            isinstance(death, dict)
            and _is_int_like(death.get("id"))
            and death.get("team") is not None
        ):
            teams[int(death["id"])] = _team_from_value(death.get("team"))
    for character_id, payload in (step.get("positions") or {}).items():
        if (
            _is_int_like(character_id)
            and isinstance(payload, dict)
            and payload.get("team") is not None
        ):
            teams[int(character_id)] = _team_from_value(payload.get("team"))
    for character_id, payload in (step.get("hp_values") or {}).items():
        if (
            _is_int_like(character_id)
            and isinstance(payload, dict)
            and payload.get("team") is not None
        ):
            teams[int(character_id)] = _team_from_value(payload.get("team"))
    return teams


def _initiative_ids(step: dict[str, Any]) -> list[int]:
    ids = []
    for item in step.get("initiative_order") or []:
        if isinstance(item, dict) and _is_int_like(item.get("id")):
            ids.append(int(item["id"]))
        elif _is_int_like(item):
            ids.append(int(item))
    return ids


def _position_from_payload_safe(payload: dict[str, Any]) -> Position:
    return Position(int(payload.get("x", 0) or 0), int(payload.get("y", 0) or 0))


def _team_from_value(value: object) -> Team:
    normalized = str(value).strip().casefold()
    if normalized in {Team.PLAYERS.value, "player", "ally", "allies"}:
        return Team.PLAYERS
    return Team.ENEMIES


def _apply_condition_flags(character: Character, flags: dict[str, Any]) -> None:
    character.prone = bool(flags.get("prone", False))
    character.grappled = bool(flags.get("grappled", False))
    character.hidden = bool(flags.get("hidden", False))
    character.dodging_until_start_of_next_turn = bool(flags.get("dodging", False))
    character.disengaged_until_end_of_turn = bool(flags.get("disengaged", False))
    character.stable = bool(flags.get("stable", False))


def _apply_action_economy(action_economy: ActionEconomy, payload: dict[str, Any]) -> None:
    action_economy.action_available = bool(payload.get("action_available", True))
    action_economy.bonus_action_available = bool(payload.get("bonus_action_available", True))
    action_economy.reaction_available = bool(payload.get("reaction_available", True))
    action_economy.movement_remaining = int(payload.get("movement_remaining", 0) or 0)
    action_economy.free_object_interaction_available = bool(
        payload.get("free_object_interaction_available", True)
    )
    action_economy.reaction_used_this_round = bool(
        payload.get("reaction_used_this_round", False)
    )
    action_economy.prepared_action = payload.get("prepared_action")
    action_economy.trigger_description = payload.get("trigger_description")


def _is_int_like(value: object) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


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
            "team": character["team"],
            **character["position"],
        }
        for character in snapshot["characters"]
    }


def _hp_values(snapshot: StateSnapshot) -> dict[str, dict[str, Any]]:
    return {
        str(character["id"]): {
            "name": character["name"],
            "team": character["team"],
            "hp": character["hp"],
            "max_hp": character["max_hp"],
            "ac": character["ac"],
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
