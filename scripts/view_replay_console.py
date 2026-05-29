from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


TERRAIN_CHARS = {
    "normal": ".",
    "difficult_terrain": "~",
    "blocked": "#",
    "low_cover": "=",
    "high_cover": "^",
}


def main() -> None:
    args = parse_args()
    replay = load_replay(args.replay)
    run_viewer(replay, autoplay_delay=args.autoplay_delay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a BattleReplay JSON in console.")
    parser.add_argument("replay", type=Path, help="Path to BattleReplay JSON.")
    parser.add_argument(
        "--autoplay-delay",
        type=float,
        default=1.0,
        help="Seconds between steps in autoplay mode.",
    )
    args = parser.parse_args()
    if args.autoplay_delay < 0:
        parser.error("--autoplay-delay must be non-negative")
    return args


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a BattleReplay JSON payload."""

    replay_path = Path(path)
    data = json.loads(replay_path.read_text(encoding="utf-8"))
    if data.get("format") != "BattleReplay":
        raise ValueError(f"Unsupported replay format: {data.get('format')!r}")
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Replay JSON must contain a steps list")
    return data


def run_viewer(replay: dict[str, Any], *, autoplay_delay: float = 1.0) -> None:
    """Interactive console loop for replay navigation."""

    steps = replay.get("steps", [])
    if not steps:
        print("Replay contains no steps.")
        return

    step_index = 0
    while True:
        print(render_step(replay, step_index))
        command = _read_command()
        if command in {"quit", "q"}:
            return
        if command in {"next", "n", ""}:
            step_index = min(step_index + 1, len(steps) - 1)
            continue
        if command in {"prev", "p"}:
            step_index = max(step_index - 1, 0)
            continue
        if command in {"autoplay", "a"}:
            step_index = _autoplay(replay, step_index, autoplay_delay)
            continue
        print("Unknown command. Use: next, prev, autoplay, quit.")


def render_step(replay: dict[str, Any], step_index: int) -> str:
    """Render one replay step as console text."""

    steps = replay.get("steps", [])
    if step_index < 0 or step_index >= len(steps):
        raise IndexError(f"Replay step index out of range: {step_index}")

    step = steps[step_index]
    lines = [
        f"BattleReplay step {step_index + 1}/{len(steps)}",
        (
            f"Round {step.get('round', '?')} | "
            f"turn_index {step.get('turn_index', '?')} | "
            f"winner {_format_optional(step.get('winner'))}"
        ),
        f"Initiative: {_format_initiative(step)}",
        "",
        "Map:",
        render_ascii_map(step),
        "",
        "Creatures:",
        *_format_creatures(step),
        "",
        "Last action:",
        *_format_action(step),
    ]
    return "\n".join(lines)


def render_ascii_map(step: dict[str, Any]) -> str:
    """Render terrain and creature positions as an ASCII tactical map."""

    map_metadata = step.get("map_metadata") or {}
    terrain_snapshot = map_metadata.get("terrain_snapshot")
    width = int(map_metadata.get("width") or _infer_map_width(step))
    height = int(map_metadata.get("height") or _infer_map_height(step))
    if width <= 0 or height <= 0:
        return "(no map)"

    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            terrain = _terrain_at(terrain_snapshot, x, y)
            row.append(TERRAIN_CHARS.get(terrain, "?"))
        grid.append(row)

    hp_values = step.get("hp_values") or {}
    for entity_id, position in sorted(
        (step.get("positions") or {}).items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
    ):
        hp = hp_values.get(str(entity_id), {})
        if hp and not bool(hp.get("alive", True)):
            continue
        x = int(position.get("x", -1))
        y = int(position.get("y", -1))
        if 0 <= x < width and 0 <= y < height:
            token = _entity_token(entity_id)
            grid[y][x] = "*" if grid[y][x].isalnum() else token

    lines = ["".join(row) for row in grid]
    lines.append("Legend: .=normal ~=difficult #=blocked ==low_cover ^=high_cover")
    lines.append(f"Entities: {_format_entity_legend(step)}")
    return "\n".join(lines)


def _format_creatures(step: dict[str, Any]) -> list[str]:
    hp_values = step.get("hp_values") or {}
    resources = step.get("resources") or {}
    lines = []
    for entity_id, hp in sorted(
        hp_values.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
    ):
        resource = resources.get(str(entity_id), {})
        lines.append(
            (
                f"[{entity_id}] {hp.get('name', 'unknown')} "
                f"HP {hp.get('hp', '?')}/{hp.get('max_hp', '?')} "
                f"{_alive_text(hp)} | {_format_resource_summary(resource)}"
            )
        )
    return lines or ["(no creatures)"]


def _format_action(step: dict[str, Any]) -> list[str]:
    action = step.get("action") or {}
    actor = step.get("actor") or {}
    lines = [
        (
            f"{actor.get('name', 'unknown')} -> {action.get('type', 'unknown')} "
            f"({step.get('action_category', 'unknown')})"
        ),
        f"Description: {step.get('description', '')}",
        f"Targets: {_format_targets(step.get('targets') or [])}",
        f"Dice: {_format_list(step.get('dice_rolls') or [])}",
        f"Damage: {_format_list(step.get('damage') or [])}",
        f"Healing: {_format_list(step.get('healing') or [])}",
        f"Spell slots spent: {_format_list(step.get('spell_slots_spent') or [])}",
        f"Items spent: {_format_list(step.get('items_spent') or [])}",
        f"Deaths: {_format_list(step.get('deaths') or [])}",
        f"Reward: {step.get('reward', 0.0)} {_format_reward(step)}",
        f"Cover/LoS: {_format_optional(step.get('cover_line_of_sight'))}",
        f"XP gained: {step.get('xp_gained', {}).get('total', 0)}",
    ]
    return lines


def _format_initiative(step: dict[str, Any]) -> str:
    order = step.get("initiative_order") or []
    if not order:
        return "(none)"
    return " -> ".join(
        f"{item.get('id', '?')}:{item.get('name') or '?'}"
        if isinstance(item, dict)
        else str(item)
        for item in order
    )


def _format_resource_summary(resource: dict[str, Any]) -> str:
    action_economy = resource.get("action_economy") or {}
    class_resources = resource.get("class_resources") or {}
    spell_slots = resource.get("spell_slots") or {}
    spell_slots_remaining = resource.get("spell_slots_remaining") or {}
    inventory = resource.get("inventory") or []

    action_text = (
        "AE "
        f"a={_bool_char(action_economy.get('action_available'))} "
        f"b={_bool_char(action_economy.get('bonus_action_available'))} "
        f"r={_bool_char(action_economy.get('reaction_available'))} "
        f"move={action_economy.get('movement_remaining', 0)}"
    )
    resource_text = _format_key_values(class_resources) or "none"
    slot_text = _format_spell_slots(spell_slots, spell_slots_remaining) or "none"
    item_text = ", ".join(
        f"{item.get('name', 'item')} x{item.get('quantity', 0)}"
        for item in inventory
    ) or "none"
    return (
        f"{action_text} | resources: {resource_text} | "
        f"slots: {slot_text} | items: {item_text}"
    )


def _format_spell_slots(
    slots: dict[str, Any],
    slots_remaining: dict[str, Any],
) -> str:
    parts = []
    for level in sorted(set(slots) | set(slots_remaining), key=int):
        parts.append(
            f"L{level} {slots_remaining.get(level, 0)}/{slots.get(level, 0)}"
        )
    return ", ".join(parts)


def _format_key_values(values: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _format_targets(targets: list[dict[str, Any]]) -> str:
    if not targets:
        return "none"
    parts = []
    for target in targets:
        if target.get("type") == "creature":
            parts.append(f"{target.get('id')}:{target.get('name')}")
        elif "x" in target and "y" in target:
            parts.append(f"{target.get('type')}({target.get('x')},{target.get('y')})")
        else:
            parts.append(str(target))
    return ", ".join(parts)


def _format_reward(step: dict[str, Any]) -> str:
    breakdown = step.get("reward_breakdown") or {}
    if not breakdown:
        return ""
    return f"breakdown={_format_key_values(breakdown)}"


def _format_list(items: list[Any]) -> str:
    return "none" if not items else json.dumps(items, ensure_ascii=False)


def _format_optional(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _format_entity_legend(step: dict[str, Any]) -> str:
    positions = step.get("positions") or {}
    if not positions:
        return "none"
    parts = []
    for entity_id, position in sorted(
        positions.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
    ):
        parts.append(
            (
                f"{_entity_token(entity_id)}=[{entity_id}]"
                f"{position.get('name', 'unknown')}"
            )
        )
    return ", ".join(parts)


def _entity_token(entity_id: object) -> str:
    text = str(entity_id)
    if text.isdigit():
        value = int(text)
        if value < 10:
            return str(value)
        return chr(ord("A") + (value - 10) % 26)
    return text[:1].upper() if text else "?"


def _terrain_at(
    terrain_snapshot: list[list[str]] | None,
    x: int,
    y: int,
) -> str:
    if terrain_snapshot is None:
        return "normal"
    if y >= len(terrain_snapshot) or x >= len(terrain_snapshot[y]):
        return "normal"
    return str(terrain_snapshot[y][x])


def _infer_map_width(step: dict[str, Any]) -> int:
    positions = step.get("positions") or {}
    if not positions:
        return 0
    return max(int(position.get("x", 0)) for position in positions.values()) + 1


def _infer_map_height(step: dict[str, Any]) -> int:
    positions = step.get("positions") or {}
    if not positions:
        return 0
    return max(int(position.get("y", 0)) for position in positions.values()) + 1


def _alive_text(hp: dict[str, Any]) -> str:
    return "alive" if bool(hp.get("alive", False)) else "dead"


def _bool_char(value: Any) -> str:
    return "Y" if bool(value) else "N"


def _read_command() -> str:
    try:
        return input("Command [next/prev/autoplay/quit]: ").strip().lower()
    except EOFError:
        return "quit"


def _autoplay(
    replay: dict[str, Any],
    step_index: int,
    autoplay_delay: float,
) -> int:
    steps = replay.get("steps", [])
    for next_index in range(step_index + 1, len(steps)):
        time.sleep(autoplay_delay)
        print(render_step(replay, next_index))
    return len(steps) - 1


if __name__ == "__main__":
    main()
