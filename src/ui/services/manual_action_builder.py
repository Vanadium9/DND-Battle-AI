"""Build legal manual GUI actions from the same masks used by policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from agents import ActionCategory, MainActionType, build_action_masks, decode_action
from combat import (
    ActionSurgeAction,
    CastSpellAction,
    ChannelDivinityPreserveLifeAction,
    CombatAction,
    CombatState,
    EndTurnAction,
    Position,
    SecondWindAction,
    UseObjectAction,
)
from combat.aoe import (
    AOE_DIRECTIONS,
    AoEShape,
    AoETargeting,
    direction_from_positions,
    positions_for_aoe,
)
from combat.class_features import implemented_feature_active_actions
from combat.items import (
    CombatItem,
    ItemActionCost,
    item_has_quantity,
    normalize_action_cost,
    resolve_item,
    supported_item_aoe_shape,
)
from combat.models import Character, SpellAbility
from combat.spellcasting import (
    available_castable_spells,
    can_cast_spell,
    spell_aoe_shape,
    spell_requires_direction,
    spell_requires_target_cell,
)


class ManualTargetMode(str, Enum):
    """What the GUI needs from the map after an option is selected."""

    NONE = "none"
    CREATURE = "creature"
    CELL = "cell"


@dataclass(frozen=True)
class ManualActionOption:
    """One legal manual action option rendered by ActionPanel."""

    id: str
    group: str
    label: str
    target_mode: ManualTargetMode = ManualTargetMode.NONE
    target_ids: tuple[int, ...] = ()
    target_cells: tuple[Position, ...] = ()
    category: ActionCategory | None = None
    main_action_type: MainActionType | None = None
    option_index: int = 0
    slot_level: int | None = None
    spell: SpellAbility | None = None
    item: CombatItem | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManualActionPlan:
    """Grouped legal options for the current manual actor."""

    actor_id: int
    actor_name: str
    groups: dict[str, tuple[ManualActionOption, ...]]

    @property
    def options(self) -> tuple[ManualActionOption, ...]:
        return tuple(option for group in self.groups.values() for option in group)


class ManualActionBuilder:
    """Create and validate GUI manual actions from action masks."""

    GROUP_MOVEMENT = "Movement"
    GROUP_MAIN = "Main Action"
    GROUP_BONUS = "Bonus Action"
    GROUP_REACTION = "Reaction"
    GROUP_END_TURN = "End Turn"

    def build_plan(self, state: CombatState, actor_id: int) -> ManualActionPlan:
        actor = state.character_at(actor_id)
        if actor is None:
            return ManualActionPlan(actor_id, "unknown", {})

        masks = build_action_masks(state, actor_id)
        groups: dict[str, list[ManualActionOption]] = {
            self.GROUP_MOVEMENT: [],
            self.GROUP_MAIN: [],
            self.GROUP_BONUS: [],
            self.GROUP_REACTION: [],
            self.GROUP_END_TURN: [],
        }

        self._add_movement(groups, state, actor_id, masks)
        self._add_main_actions(groups, state, actor_id, actor, masks)
        self._add_spell_actions(groups, state, actor_id, actor, masks)
        self._add_item_actions(groups, state, actor_id, actor, masks)
        self._add_class_features(groups, state, actor_id, actor, masks)
        self._add_end_turn(groups, actor_id, masks)

        return ManualActionPlan(
            actor_id=actor_id,
            actor_name=actor.name,
            groups={
                group: tuple(options)
                for group, options in groups.items()
                if options
            },
        )

    def build_action(
        self,
        state: CombatState,
        actor_id: int,
        option: ManualActionOption,
        *,
        target_id: int | None = None,
        target_cell: Position | None = None,
    ) -> CombatAction:
        """Return a concrete action for a selected legal option."""

        if option.target_mode is ManualTargetMode.CREATURE:
            if target_id is None or target_id not in option.target_ids:
                raise ValueError("Выбранная цель недоступна для этого действия.")
        if option.target_mode is ManualTargetMode.CELL:
            if target_cell is None or target_cell not in option.target_cells:
                raise ValueError("Выбранная клетка недоступна для этого действия.")

        kind = option.metadata.get("kind")
        if kind == "move":
            assert target_cell is not None
            return self._decode_masked(
                state,
                actor_id,
                ActionCategory.MOVEMENT,
                move_index=_position_index(state, target_cell),
            )
        if kind == "simple_main":
            assert option.main_action_type is not None
            return self._decode_masked(
                state,
                actor_id,
                ActionCategory.MAIN_ACTION,
                main_action_type=option.main_action_type,
                option_index=option.option_index,
                target_index=target_id or 0,
            )
        if kind == "spell":
            action = self._spell_action(actor_id, option, target_id, target_cell)
            self._validate_action(action, state)
            return action
        if kind == "item":
            action = self._item_action(actor_id, option, target_id, target_cell)
            self._validate_action(action, state)
            return action
        if kind == "second_wind":
            action = SecondWindAction(actor_id=actor_id)
            self._validate_action(action, state)
            return action
        if kind == "action_surge":
            action = ActionSurgeAction(actor_id=actor_id)
            self._validate_action(action, state)
            return action
        if kind == "preserve_life":
            action = ChannelDivinityPreserveLifeAction(
                actor_id=actor_id,
                target_id=target_id,
            )
            self._validate_action(action, state)
            return action
        if kind == "end_turn":
            return EndTurnAction(actor_id=actor_id)

        raise ValueError(f"Неизвестный ручной вариант действия: {option.id}")

    def _add_movement(
        self,
        groups: dict[str, list[ManualActionOption]],
        state: CombatState,
        actor_id: int,
        masks: dict[str, torch.Tensor],
    ) -> None:
        if not _mask_allowed(masks["action_category"], ActionCategory.MOVEMENT):
            return
        cells = tuple(
            _position_from_index(state, index)
            for index, allowed in enumerate(masks["move_index"])
            if bool(allowed)
        )
        if not cells:
            return
        groups[self.GROUP_MOVEMENT].append(
            ManualActionOption(
                id="move",
                group=self.GROUP_MOVEMENT,
                label=f"Move ({len(cells)} cells)",
                target_mode=ManualTargetMode.CELL,
                target_cells=cells,
                category=ActionCategory.MOVEMENT,
                metadata={"kind": "move"},
            )
        )

    def _add_main_actions(
        self,
        groups: dict[str, list[ManualActionOption]],
        state: CombatState,
        actor_id: int,
        actor: Character,
        masks: dict[str, torch.Tensor],
    ) -> None:
        main_mask = masks["main_action_type"]
        option_mask = masks["option_index"]

        if _mask_allowed(main_mask, MainActionType.ATTACK):
            for weapon_index, weapon in enumerate(actor.weapons):
                if not _mask_allowed(option_mask, weapon_index):
                    continue
                targets = tuple(
                    target_id
                    for target_id in range(len(state.characters))
                    if self._can_decode(
                        state,
                        actor_id,
                        ActionCategory.MAIN_ACTION,
                        MainActionType.ATTACK,
                        target_index=target_id,
                        option_index=weapon_index,
                    )
                )
                if targets:
                    groups[self.GROUP_MAIN].append(
                        ManualActionOption(
                            id=f"attack:{weapon_index}",
                            group=self.GROUP_MAIN,
                            label=f"Attack: {weapon.name}",
                            target_mode=ManualTargetMode.CREATURE,
                            target_ids=targets,
                            category=ActionCategory.MAIN_ACTION,
                            main_action_type=MainActionType.ATTACK,
                            option_index=weapon_index,
                            metadata={"kind": "simple_main"},
                        )
                    )

        self._add_targetless_main(groups, state, actor_id, main_mask)
        self._add_targeted_main(groups, state, actor_id, main_mask)

    def _add_targetless_main(
        self,
        groups: dict[str, list[ManualActionOption]],
        state: CombatState,
        actor_id: int,
        main_mask: torch.Tensor,
    ) -> None:
        actions = (
            (MainActionType.DASH, "Dash", 0),
            (MainActionType.DISENGAGE, "Disengage", 0),
            (MainActionType.DODGE, "Dodge", 0),
            (MainActionType.HIDE, "Hide", 0),
            (MainActionType.READY, "Ready", 0),
            (MainActionType.IMPROVISED, "Improvised Action", 0),
            (MainActionType.SEARCH, "Search: Perception", 0),
            (MainActionType.SEARCH, "Search: Investigation", 1),
        )
        for main_action, label, option_index in actions:
            if not _mask_allowed(main_mask, main_action):
                continue
            if not self._can_decode(
                state,
                actor_id,
                ActionCategory.MAIN_ACTION,
                main_action,
                option_index=option_index,
            ):
                continue
            groups[self.GROUP_MAIN].append(
                ManualActionOption(
                    id=f"main:{main_action.name}:{option_index}",
                    group=self.GROUP_MAIN,
                    label=label,
                    category=ActionCategory.MAIN_ACTION,
                    main_action_type=main_action,
                    option_index=option_index,
                    metadata={"kind": "simple_main"},
                )
            )

    def _add_targeted_main(
        self,
        groups: dict[str, list[ManualActionOption]],
        state: CombatState,
        actor_id: int,
        main_mask: torch.Tensor,
    ) -> None:
        actions = (
            (MainActionType.HELP, "Help", 0),
            (MainActionType.GRAPPLE, "Grapple", 0),
            (MainActionType.SHOVE, "Shove Prone", 0),
            (MainActionType.SHOVE, "Shove Push", 1),
            (MainActionType.STABILIZE, "Stabilize", 0),
        )
        for main_action, label, option_index in actions:
            if not _mask_allowed(main_mask, main_action):
                continue
            targets = tuple(
                target_id
                for target_id in range(len(state.characters))
                if self._can_decode(
                    state,
                    actor_id,
                    ActionCategory.MAIN_ACTION,
                    main_action,
                    target_index=target_id,
                    option_index=option_index,
                )
            )
            if not targets:
                continue
            groups[self.GROUP_MAIN].append(
                ManualActionOption(
                    id=f"main:{main_action.name}:{option_index}",
                    group=self.GROUP_MAIN,
                    label=label,
                    target_mode=ManualTargetMode.CREATURE,
                    target_ids=targets,
                    category=ActionCategory.MAIN_ACTION,
                    main_action_type=main_action,
                    option_index=option_index,
                    metadata={"kind": "simple_main"},
                )
            )

    def _add_spell_actions(
        self,
        groups: dict[str, list[ManualActionOption]],
        state: CombatState,
        actor_id: int,
        actor: Character,
        masks: dict[str, torch.Tensor],
    ) -> None:
        spell_specs = (
            ("action", self.GROUP_MAIN, ActionCategory.MAIN_ACTION),
            ("bonus_action", self.GROUP_BONUS, ActionCategory.BONUS_ACTION),
            ("reaction", self.GROUP_REACTION, ActionCategory.REACTION),
        )
        for action_cost, group, category in spell_specs:
            if not _mask_allowed(masks["action_category"], category):
                continue
            for option_indexes, spell in self._spell_option_indexes(actor, action_cost):
                allowed_indexes = [
                    index for index in option_indexes if _mask_allowed(masks["option_index"], index)
                ]
                if not allowed_indexes:
                    continue
                option_index = allowed_indexes[0]
                for slot_level in _legal_slot_levels(actor, spell):
                    option = self._spell_option(
                        state,
                        actor_id,
                        spell,
                        option_index,
                        slot_level,
                        group,
                        category,
                    )
                    if option is not None:
                        groups[group].append(option)

    def _add_item_actions(
        self,
        groups: dict[str, list[ManualActionOption]],
        state: CombatState,
        actor_id: int,
        actor: Character,
        masks: dict[str, torch.Tensor],
    ) -> None:
        for item_index, item in enumerate(_available_items(actor)):
            if not _mask_allowed(masks["option_index"], item_index):
                continue
            group = _group_for_item(item)
            option = self._item_option(state, actor_id, item, item_index, group)
            if option is not None:
                groups[group].append(option)

    def _add_class_features(
        self,
        groups: dict[str, list[ManualActionOption]],
        state: CombatState,
        actor_id: int,
        actor: Character,
        masks: dict[str, torch.Tensor],
    ) -> None:
        if _mask_allowed(masks["action_category"], ActionCategory.BONUS_ACTION):
            if "second_wind" in implemented_feature_active_actions(actor, "bonus_action"):
                action = SecondWindAction(actor_id=actor_id)
                if action.is_valid(state):
                    groups[self.GROUP_BONUS].append(
                        ManualActionOption(
                            id="feature:second_wind",
                            group=self.GROUP_BONUS,
                            label="Second Wind",
                            metadata={"kind": "second_wind"},
                        )
                    )

        if not _mask_allowed(masks["action_category"], ActionCategory.CLASS_FEATURE):
            return

        action_surge = ActionSurgeAction(actor_id=actor_id)
        if action_surge.is_valid(state):
            groups[self.GROUP_MAIN].append(
                ManualActionOption(
                    id="feature:action_surge",
                    group=self.GROUP_MAIN,
                    label="Class Feature: Action Surge",
                    category=ActionCategory.CLASS_FEATURE,
                    metadata={"kind": "action_surge"},
                )
            )

        preserve_life_targets = tuple(
            target_id
            for target_id in range(len(state.characters))
            if ChannelDivinityPreserveLifeAction(
                actor_id=actor_id,
                target_id=target_id,
            ).is_valid(state)
        )
        if preserve_life_targets:
            groups[self.GROUP_MAIN].append(
                ManualActionOption(
                    id="feature:preserve_life",
                    group=self.GROUP_MAIN,
                    label="Class Feature: Preserve Life",
                    target_mode=ManualTargetMode.CREATURE,
                    target_ids=preserve_life_targets,
                    category=ActionCategory.CLASS_FEATURE,
                    metadata={"kind": "preserve_life"},
                )
            )

    def _add_end_turn(
        self,
        groups: dict[str, list[ManualActionOption]],
        actor_id: int,
        masks: dict[str, torch.Tensor],
    ) -> None:
        if not _mask_allowed(masks["action_category"], ActionCategory.END_TURN):
            return
        groups[self.GROUP_END_TURN].append(
            ManualActionOption(
                id="end_turn",
                group=self.GROUP_END_TURN,
                label="End Turn",
                category=ActionCategory.END_TURN,
                metadata={"kind": "end_turn"},
            )
        )

    def _spell_option(
        self,
        state: CombatState,
        actor_id: int,
        spell: SpellAbility,
        option_index: int,
        slot_level: int | None,
        group: str,
        category: ActionCategory,
    ) -> ManualActionOption | None:
        base_label = spell.name
        if slot_level is not None:
            base_label = f"{base_label} (slot {slot_level})"

        if spell_requires_target_cell(spell):
            cells = tuple(
                position
                for position in _grid_positions(state)
                if CastSpellAction(
                    actor_id=actor_id,
                    spell=spell,
                    target_cell=position,
                    cast_level=slot_level,
                ).is_valid(state)
            )
            if not cells:
                return None
            return ManualActionOption(
                id=f"spell:{group}:{option_index}:{slot_level}:cell",
                group=group,
                label=f"Spell: {base_label}",
                target_mode=ManualTargetMode.CELL,
                target_cells=cells,
                category=category,
                main_action_type=MainActionType.CAST_SPELL,
                option_index=option_index,
                slot_level=slot_level,
                spell=spell,
                metadata={"kind": "spell"},
            )

        if spell_requires_direction(spell):
            cells = self._directional_spell_cells(state, actor_id, spell, slot_level)
            if not cells:
                return None
            return ManualActionOption(
                id=f"spell:{group}:{option_index}:{slot_level}:direction",
                group=group,
                label=f"Spell: {base_label}",
                target_mode=ManualTargetMode.CELL,
                target_cells=cells,
                category=category,
                main_action_type=MainActionType.CAST_SPELL,
                option_index=option_index,
                slot_level=slot_level,
                spell=spell,
                metadata={
                    "kind": "spell",
                    "directional": True,
                    "actor_position": state.characters[actor_id].position,
                },
            )

        if spell.damage is not None or spell.healing is not None:
            targets = tuple(
                target_id
                for target_id in range(len(state.characters))
                if CastSpellAction(
                    actor_id=actor_id,
                    spell=spell,
                    target_id=target_id,
                    cast_level=slot_level,
                ).is_valid(state)
            )
            if not targets:
                return None
            return ManualActionOption(
                id=f"spell:{group}:{option_index}:{slot_level}:target",
                group=group,
                label=f"Spell: {base_label}",
                target_mode=ManualTargetMode.CREATURE,
                target_ids=targets,
                category=category,
                main_action_type=MainActionType.CAST_SPELL,
                option_index=option_index,
                slot_level=slot_level,
                spell=spell,
                metadata={"kind": "spell"},
            )

        action = CastSpellAction(
            actor_id=actor_id,
            spell=spell,
            cast_level=slot_level,
        )
        if not action.is_valid(state):
            return None
        return ManualActionOption(
            id=f"spell:{group}:{option_index}:{slot_level}:self",
            group=group,
            label=f"Spell: {base_label}",
            category=category,
            main_action_type=MainActionType.CAST_SPELL,
            option_index=option_index,
            slot_level=slot_level,
            spell=spell,
            metadata={"kind": "spell"},
        )

    def _item_option(
        self,
        state: CombatState,
        actor_id: int,
        item: CombatItem,
        option_index: int,
        group: str,
    ) -> ManualActionOption | None:
        shape = supported_item_aoe_shape(item)
        if shape is AoEShape.RADIUS:
            cells = tuple(
                position
                for position in _grid_positions(state)
                if UseObjectAction(
                    actor_id=actor_id,
                    object_name=item.name,
                    item=item,
                    target_cell=position,
                ).is_valid(state)
            )
            if not cells:
                return None
            return ManualActionOption(
                id=f"item:{option_index}:cell",
                group=group,
                label=f"Item: {item.name}",
                target_mode=ManualTargetMode.CELL,
                target_cells=cells,
                main_action_type=MainActionType.USE_OBJECT,
                option_index=option_index,
                item=item,
                metadata={"kind": "item"},
            )
        if shape in {AoEShape.CONE, AoEShape.LINE}:
            cells = self._directional_item_cells(state, actor_id, item)
            if not cells:
                return None
            return ManualActionOption(
                id=f"item:{option_index}:direction",
                group=group,
                label=f"Item: {item.name}",
                target_mode=ManualTargetMode.CELL,
                target_cells=cells,
                main_action_type=MainActionType.USE_OBJECT,
                option_index=option_index,
                item=item,
                metadata={
                    "kind": "item",
                    "directional": True,
                    "actor_position": state.characters[actor_id].position,
                },
            )

        targets = tuple(
            target_id
            for target_id in range(len(state.characters))
            if UseObjectAction(
                actor_id=actor_id,
                object_name=item.name,
                item=item,
                target_id=target_id,
            ).is_valid(state)
        )
        if targets:
            return ManualActionOption(
                id=f"item:{option_index}:target",
                group=group,
                label=f"Item: {item.name}",
                target_mode=ManualTargetMode.CREATURE,
                target_ids=targets,
                main_action_type=MainActionType.USE_OBJECT,
                option_index=option_index,
                item=item,
                metadata={"kind": "item"},
            )

        action = UseObjectAction(actor_id=actor_id, object_name=item.name, item=item)
        if not action.is_valid(state):
            return None
        return ManualActionOption(
            id=f"item:{option_index}:self",
            group=group,
            label=f"Item: {item.name}",
            main_action_type=MainActionType.USE_OBJECT,
            option_index=option_index,
            item=item,
            metadata={"kind": "item"},
        )

    def _directional_spell_cells(
        self,
        state: CombatState,
        actor_id: int,
        spell: SpellAbility,
        slot_level: int | None,
    ) -> tuple[Position, ...]:
        actor = state.character_at(actor_id)
        if actor is None:
            return ()
        cells: set[Position] = set()
        shape = spell_aoe_shape(spell)
        if shape not in {AoEShape.CONE, AoEShape.LINE}:
            return ()
        for direction in AOE_DIRECTIONS:
            action = CastSpellAction(
                actor_id=actor_id,
                spell=spell,
                direction=direction,
                cast_level=slot_level,
            )
            if not action.is_valid(state):
                continue
            cells.update(
                _in_bounds(
                    state,
                    positions_for_aoe(
                        AoETargeting(
                            shape=shape,
                            origin=actor.position,
                            size=spell.area_size,
                            direction=direction,
                        )
                    ),
                )
            )
        return tuple(sorted(cells, key=lambda position: (position.y, position.x)))

    def _directional_item_cells(
        self,
        state: CombatState,
        actor_id: int,
        item: CombatItem,
    ) -> tuple[Position, ...]:
        actor = state.character_at(actor_id)
        if actor is None:
            return ()
        cells: set[Position] = set()
        shape = supported_item_aoe_shape(item)
        if shape not in {AoEShape.CONE, AoEShape.LINE}:
            return ()
        for direction in AOE_DIRECTIONS:
            action = UseObjectAction(
                actor_id=actor_id,
                object_name=item.name,
                item=item,
                direction=direction,
            )
            if not action.is_valid(state):
                continue
            cells.update(
                _in_bounds(
                    state,
                    positions_for_aoe(
                        AoETargeting(
                            shape=shape,
                            origin=actor.position,
                            size=item.area_size,
                            direction=direction,
                        )
                    ),
                )
            )
        return tuple(sorted(cells, key=lambda position: (position.y, position.x)))

    def _spell_action(
        self,
        actor_id: int,
        option: ManualActionOption,
        target_id: int | None,
        target_cell: Position | None,
    ) -> CastSpellAction:
        spell = option.spell
        if spell is None:
            raise ValueError("Вариант заклинания потерял ссылку на spell.")
        if option.metadata.get("directional"):
            actor_position = option.metadata.get("actor_position")
            if not isinstance(actor_position, Position):
                raise ValueError("Не удалось определить направление заклинания.")
            assert target_cell is not None
            direction = direction_from_positions(actor_position, target_cell)
            return CastSpellAction(
                actor_id=actor_id,
                spell=spell,
                direction=direction,
                cast_level=option.slot_level,
            )
        if option.target_mode is ManualTargetMode.CELL:
            return CastSpellAction(
                actor_id=actor_id,
                spell=spell,
                target_cell=target_cell,
                cast_level=option.slot_level,
            )
        if option.target_mode is ManualTargetMode.CREATURE:
            return CastSpellAction(
                actor_id=actor_id,
                spell=spell,
                target_id=target_id,
                cast_level=option.slot_level,
            )
        return CastSpellAction(
            actor_id=actor_id,
            spell=spell,
            cast_level=option.slot_level,
        )

    def _item_action(
        self,
        actor_id: int,
        option: ManualActionOption,
        target_id: int | None,
        target_cell: Position | None,
    ) -> UseObjectAction:
        item = option.item
        if item is None:
            raise ValueError("Вариант предмета потерял ссылку на item.")
        if option.metadata.get("directional"):
            actor_position = option.metadata.get("actor_position")
            if not isinstance(actor_position, Position):
                raise ValueError("Не удалось определить направление предмета.")
            assert target_cell is not None
            direction = direction_from_positions(actor_position, target_cell)
            return UseObjectAction(
                actor_id=actor_id,
                object_name=item.name,
                item=item,
                direction=direction,
            )
        if option.target_mode is ManualTargetMode.CELL:
            return UseObjectAction(
                actor_id=actor_id,
                object_name=item.name,
                item=item,
                target_cell=target_cell,
            )
        if option.target_mode is ManualTargetMode.CREATURE:
            return UseObjectAction(
                actor_id=actor_id,
                object_name=item.name,
                item=item,
                target_id=target_id,
            )
        return UseObjectAction(actor_id=actor_id, object_name=item.name, item=item)

    def _spell_option_indexes(
        self,
        actor: Character,
        action_cost: str,
    ) -> list[tuple[tuple[int, ...], SpellAbility]]:
        options: list[tuple[tuple[int, ...], SpellAbility]] = []
        option_index = 0
        for spell in available_castable_spells(actor):
            if spell.action_cost != action_cost:
                continue
            if spell_requires_direction(spell):
                indexes = tuple(range(option_index, option_index + len(AOE_DIRECTIONS)))
                options.append((indexes, spell))
                option_index += len(AOE_DIRECTIONS)
            else:
                options.append(((option_index,), spell))
                option_index += 1
        return options

    def _decode_masked(
        self,
        state: CombatState,
        actor_id: int,
        category: ActionCategory,
        *,
        main_action_type: MainActionType = MainActionType.ATTACK,
        target_index: int = 0,
        move_index: int = 0,
        option_index: int = 0,
    ) -> CombatAction:
        return decode_action(
            category,
            main_action_type,
            target_index,
            move_index,
            option_index,
            state,
            actor_id,
        )

    def _can_decode(
        self,
        state: CombatState,
        actor_id: int,
        category: ActionCategory,
        main_action_type: MainActionType,
        *,
        target_index: int = 0,
        option_index: int = 0,
    ) -> bool:
        try:
            action = self._decode_masked(
                state,
                actor_id,
                category,
                main_action_type=main_action_type,
                target_index=target_index,
                option_index=option_index,
            )
        except (ValueError, IndexError):
            return False
        return action.is_valid(state)

    @staticmethod
    def _validate_action(action: CombatAction, state: CombatState) -> None:
        if not action.is_valid(state):
            raise ValueError("Выбранное действие больше не является легальным.")


def _mask_allowed(mask: torch.Tensor, index: int | ActionCategory | MainActionType) -> bool:
    idx = int(index)
    return 0 <= idx < len(mask) and bool(mask[idx])


def _grid_positions(state: CombatState) -> tuple[Position, ...]:
    grid_map = state.grid_map
    if grid_map is None:
        return ()
    return tuple(
        Position(x, y)
        for y in range(grid_map.height)
        for x in range(grid_map.width)
    )


def _position_index(state: CombatState, position: Position) -> int:
    if state.grid_map is None:
        raise ValueError("Нет карты для выбора клетки.")
    return position.y * state.grid_map.width + position.x


def _position_from_index(state: CombatState, index: int) -> Position:
    if state.grid_map is None:
        raise ValueError("Нет карты для выбора клетки.")
    return Position(index % state.grid_map.width, index // state.grid_map.width)


def _legal_slot_levels(actor: Character, spell: SpellAbility) -> tuple[int | None, ...]:
    if spell.spell_level <= 0:
        return (None,)
    remaining = getattr(actor, "spell_slots_remaining", {})
    levels = tuple(
        level
        for level in sorted(int(level) for level in remaining)
        if level >= int(spell.spell_level)
        and int(remaining.get(level, 0)) > 0
        and can_cast_spell(actor, spell, level)
    )
    return levels


def _available_items(actor: Character) -> tuple[CombatItem, ...]:
    inventory = getattr(actor, "inventory", None)
    if not isinstance(inventory, (list, tuple)):
        return ()
    items = []
    for candidate in inventory:
        item = resolve_item(actor, candidate)
        if item is not None and item.implemented and item_has_quantity(item):
            items.append(item)
    return tuple(items)


def _group_for_item(item: CombatItem) -> str:
    cost = normalize_action_cost(item.action_cost)
    if cost is ItemActionCost.BONUS_ACTION:
        return ManualActionBuilder.GROUP_BONUS
    if cost is ItemActionCost.REACTION:
        return ManualActionBuilder.GROUP_REACTION
    return ManualActionBuilder.GROUP_MAIN


def _in_bounds(state: CombatState, positions: set[Position]) -> set[Position]:
    if state.grid_map is None:
        return positions
    return {position for position in positions if state.grid_map.in_bounds(position)}
