"""Common D&D 5e-style combat actions available to creatures."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import re

from combat.aoe import (
    AoEDirection,
    AoEShape,
    AoETargeting,
    affected_creatures,
    coerce_aoe_direction,
    coerce_aoe_shape,
    direction_from_positions,
    log_affected_targets,
    positions_for_aoe,
)
from combat.abilities import SpellAbility, WeaponAttack, ability_modifier
from combat.checks import (
    ContestedCheckResult,
    passive_perception,
    roll_ability_check,
    roll_contested_check,
)
from combat.class_features import (
    archery_attack_bonus,
    can_use_feature_action,
    character_has_class_feature,
    critical_hit_threshold,
    should_use_great_weapon_fighting,
    spend_feature_resource,
    weapon_attack_count_for_attack_action,
)
from combat.cover import CoverType, apply_cover_to_ac, apply_cover_to_dex_save
from combat.conditions import handle_concentration_damage
from combat.damage import apply_damage_modifiers
from combat.features import (
    attack_roll_advantage_state,
    on_attack_roll,
    on_damage_roll,
)
from combat.items import (
    CombatItem,
    consume_item,
    item_damage,
    item_damage_type,
    item_has_quantity,
    item_healing,
    item_save_ability,
    item_save_half_damage,
    item_stabilizes,
    normalize_action_cost,
    normalize_target_type,
    resolve_item,
    supported_item_aoe_shape,
    ItemActionCost,
    ItemTargetType,
)
from combat.models import Character, CombatState, Condition, Position
from combat.race_traits import use_halfling_lucky
from combat.spellcasting import (
    available_castable_spells,
    begin_spell_concentration,
    can_cast_spell,
    can_target_spell as can_target_spell_with_rules,
    spell_cast_level,
    spell_has_aoe,
    spell_aoe_shape,
    spend_spell_resources,
)


COMMON_ACTION_MOVE = "move"
COMMON_ACTION_ATTACK = "attack"
COMMON_ACTION_CAST_SPELL = "cast_spell"
COMMON_ACTION_DASH = "dash"
COMMON_ACTION_DISENGAGE = "disengage"
COMMON_ACTION_DODGE = "dodge"
COMMON_ACTION_HELP = "help"
COMMON_ACTION_HIDE = "hide"
COMMON_ACTION_SEARCH = "search"
COMMON_ACTION_USE_OBJECT = "use_object"
COMMON_ACTION_READY = "ready"
COMMON_ACTION_GRAPPLE = "grapple"
COMMON_ACTION_SHOVE = "shove"
COMMON_ACTION_STABILIZE = "stabilize"
COMMON_ACTION_IMPROVISED = "improvised_action"
COMMON_ACTION_OPPORTUNITY_ATTACK = "opportunity_attack"
COMMON_ACTION_END_TURN = "end_turn"


@dataclass(frozen=True)
class ActionResult:
    """Result of an action execution."""

    success: bool
    description: str
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)


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
        if COMMON_ACTION_MOVE not in actor.common_actions:
            return False
        movement_cost = _movement_cost(actor, self.destination, combat_state)
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
        movement_cost = _movement_cost(actor, self.destination, combat_state)
        path_cost = _movement_path_cost(actor, self.destination, combat_state)
        if path_cost is None:
            return ActionResult(False, f"{actor.name} cannot move to {self.destination}.")
        if actor.prone and not actor.action_economy.stand_up(actor.speed):
            return ActionResult(False, f"{actor.name} cannot stand up from prone.")
        actor.position = self.destination
        actor.action_economy.spend_movement(path_cost)
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
    """Attack a target with one of the actor's weapon attacks."""

    target_id: int
    weapon: WeaponAttack | None = None

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        weapon = self._resolve_weapon(actor)

        if actor is None or target is None or weapon is None:
            return False
        if (
            not actor.can_take_turn
            or target.is_dead
            or COMMON_ACTION_ATTACK not in actor.common_actions
            or not weapon.available
            or not actor.action_economy.action_available
        ):
            return False
        if _distance(actor.position, target.position, combat_state) > weapon.range:
            return False
        return _can_weapon_target_from_map(combat_state, actor, target, weapon)

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
        attack_count = weapon_attack_count_for_attack_action(actor)
        attack_summaries: list[str] = []
        total_damage = 0

        for attack_number in range(1, attack_count + 1):
            if target.is_dead:
                attack_summaries.append(f"attack {attack_number}: target already defeated")
                break

            d20_roll = _attack_roll(actor, target, combat_state)
            attack_modifier = _weapon_attack_modifier(actor, weapon)
            cover = _cover_between(combat_state, actor.position, target.position)
            effective_ac = apply_cover_to_ac(target.ac, cover)
            attack_total = d20_roll + attack_modifier
            critical = d20_roll >= critical_hit_threshold(actor)
            hit = d20_roll == 20 or attack_total >= effective_ac
            if not hit:
                attack_summaries.append(
                    (
                        f"attack {attack_number}: miss ({attack_total} vs AC "
                        f"{effective_ac}; d20={d20_roll}, modifier={attack_modifier})"
                    )
                )
                continue

            raw_damage = on_damage_roll(
                actor,
                _roll_weapon_damage(actor, weapon, critical=critical),
                target=target,
                weapon=weapon,
                combat_state=combat_state,
            )
            damage = apply_damage_modifiers(target, raw_damage, weapon.damage_type)
            _apply_damage_to_character(target, damage)
            total_damage += damage
            outcome_text = "critical hit" if critical else "hit"
            attack_summaries.append(
                (
                    f"attack {attack_number}: {outcome_text} "
                    f"({attack_total} vs AC {effective_ac}; d20={d20_roll}, "
                    f"modifier={attack_modifier}) for {damage} damage"
                )
            )

        return ActionResult(
            True,
            (
                f"{actor.name} attacks {target.name} with {weapon.name}: "
                f"{'; '.join(attack_summaries)}. Total damage: {total_damage}. "
                "Action spent: action_available=False."
            ),
        )

    def _resolve_weapon(self, actor: Character | None) -> WeaponAttack | None:
        if actor is None:
            return None
        if self.weapon is not None:
            if self.weapon in actor.weapons:
                return self.weapon
            return None
        for weapon in actor.available_weapons:
            return weapon
        return None


@dataclass
class OpportunityAttackAction(CombatAction):
    """Make a melee weapon attack as a reaction against a creature leaving reach."""

    target_id: int
    weapon: WeaponAttack | None = None

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        weapon = self._resolve_weapon(actor)
        if actor is None or target is None or weapon is None:
            return False
        if (
            not actor.can_take_turn
            or target.is_dead
            or target.team == actor.team
            or target.disengaged_until_end_of_turn
            or COMMON_ACTION_OPPORTUNITY_ATTACK not in actor.common_actions
            or not actor.action_economy.reaction_available
            or not weapon.available
        ):
            return False
        return _distance(actor.position, target.position, combat_state) <= 1

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        weapon = self._resolve_weapon(actor)
        if actor is None:
            return ActionResult(
                False,
                f"Opportunity attack failed: actor {self.actor_id} not found.",
            )
        if target is None:
            return ActionResult(
                False,
                f"{actor.name} cannot opportunity attack missing target {self.target_id}.",
            )
        if weapon is None:
            return ActionResult(False, f"{actor.name} has no melee weapon for opportunity attack.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot opportunity attack {target.name}.")

        actor.action_economy.spend_reaction()
        d20_roll = _attack_roll(actor, target, combat_state)
        attack_modifier = _weapon_attack_modifier(actor, weapon)
        cover = _cover_between(combat_state, actor.position, target.position)
        effective_ac = apply_cover_to_ac(target.ac, cover)
        attack_total = d20_roll + attack_modifier
        critical = d20_roll >= critical_hit_threshold(actor)
        hit = d20_roll == 20 or attack_total >= effective_ac
        if not hit:
            return ActionResult(
                True,
                (
                    f"{actor.name} opportunity attacks {target.name} with {weapon.name}: "
                    f"miss ({attack_total} vs AC {effective_ac}; "
                    f"d20={d20_roll}, modifier={attack_modifier}). "
                    "Reaction spent: reaction_available=False."
                ),
            )

        raw_damage = on_damage_roll(
            actor,
            _roll_weapon_damage(actor, weapon, critical=critical),
            target=target,
            weapon=weapon,
            combat_state=combat_state,
        )
        damage = apply_damage_modifiers(target, raw_damage, weapon.damage_type)
        _apply_damage_to_character(target, damage)
        return ActionResult(
            True,
            (
                f"{actor.name} opportunity attacks {target.name} with {weapon.name}: "
                f"{'critical hit' if critical else 'hit'} ({attack_total} vs AC {effective_ac}; "
                f"d20={d20_roll}, modifier={attack_modifier}) for {damage} damage. "
                "Reaction spent: reaction_available=False."
            ),
        )

    def _resolve_weapon(self, actor: Character | None) -> WeaponAttack | None:
        if actor is None:
            return None
        if self.weapon is not None:
            if self.weapon in actor.weapons and self.weapon.range <= 1:
                return self.weapon
            return None
        for weapon in actor.available_weapons:
            if weapon.range <= 1:
                return weapon
        return None


@dataclass
class CastSpellAction(CombatAction):
    """Cast a simple SpellAbility if the actor has one."""

    spell: SpellAbility | None = None
    target_id: int | None = None
    target_cell: Position | None = None
    direction: AoEDirection | str | int | None = None
    cast_level: int | None = None

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        spell = self._resolve_spell(actor)
        if actor is None or not actor.can_take_turn or spell is None:
            return False
        if COMMON_ACTION_CAST_SPELL not in actor.common_actions:
            return False
        if not spell.available or not _can_spend_spell_action(actor, spell):
            return False
        if not can_cast_spell(actor, spell, self.cast_level):
            return False
        if spell_has_aoe(spell):
            return self._aoe_targeting_for_spell(combat_state, actor, spell) is not None
        target = self._target_for_spell(combat_state, actor, spell)
        return spell.damage is None and spell.healing is None or target is not None

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        spell = self._resolve_spell(actor)
        if actor is None:
            return ActionResult(False, f"Cast spell failed: actor {self.actor_id} not found.")
        if spell is None:
            return ActionResult(False, f"{actor.name} has no available spell.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot cast {spell.name}.")

        _spend_spell_action(actor, spell)
        target = self._target_for_spell(combat_state, actor, spell)
        casting_level = spell_cast_level(spell, self.cast_level)
        if spell.ac_bonus > 0:
            _apply_temporary_ac_spell(actor, spell)
            spend_spell_resources(actor, spell, casting_level)
            begin_spell_concentration(actor, spell)
            spent_text = _spell_action_spent_text(spell)
            return ActionResult(
                True,
                (
                    f"{actor.name} casts {spell.name} at level {casting_level} "
                    f"and gains +{spell.ac_bonus} AC until the start of their next turn. "
                    f"{spent_text}"
                ),
            )
        if spell.damage is not None:
            if spell_has_aoe(spell):
                targeting = self._aoe_targeting_for_spell(combat_state, actor, spell)
                if targeting is None:
                    return ActionResult(False, f"{actor.name} cannot place {spell.name}.")
                return _execute_area_damage_spell(
                    combat_state,
                    actor,
                    spell,
                    casting_level,
                    targeting,
                )
            if target is None:
                return ActionResult(False, f"{actor.name} cannot target {spell.name}.")
            raw_damage = _roll_spell_effect(
                spell.damage,
                spell.upcast_damage_per_level,
                spell.spell_level,
                casting_level,
            )
            saved = _target_saves_against_spell(actor, target, spell, combat_state)
            if saved:
                raw_damage = raw_damage // 2 if spell.save_half_damage else 0
            raw_damage = on_damage_roll(
                actor,
                raw_damage,
                target=target,
                spell=spell,
                combat_state=combat_state,
            )
            damage = apply_damage_modifiers(
                target,
                raw_damage,
                spell.damage_type,
            )
            _apply_damage_to_character(target, damage)
            spend_spell_resources(actor, spell, casting_level)
            begin_spell_concentration(actor, spell)
            save_text = " after save" if saved else ""
            spent_text = _spell_action_spent_text(spell)
            return ActionResult(
                True,
                (
                    f"{actor.name} casts {spell.name} on {target.name} "
                    f"at level {casting_level} for {damage} damage{save_text}. "
                    f"{spent_text}"
                ),
            )
        if spell.healing is not None and target is not None:
            healing = _roll_spell_effect(
                spell.healing,
                spell.upcast_healing_per_level,
                spell.spell_level,
                casting_level,
            )
            before_hp = target.hp
            target.hp = min(target.max_hp, target.hp + healing)
            if target.hp > 0:
                target.stable = True
            spend_spell_resources(actor, spell, casting_level)
            begin_spell_concentration(actor, spell)
            spent_text = _spell_action_spent_text(spell)
            return ActionResult(
                True,
                (
                    f"{actor.name} casts {spell.name} on {target.name} "
                    f"at level {casting_level} and heals {target.hp - before_hp} HP. "
                    f"{spent_text}"
                ),
            )
        spend_spell_resources(actor, spell, casting_level)
        begin_spell_concentration(actor, spell)
        spent_text = _spell_action_spent_text(spell)
        return ActionResult(
            True,
            (
                f"{actor.name} casts {spell.name} at level {casting_level}. "
                f"{spent_text}"
            ),
        )

    def _resolve_spell(self, actor: Character | None) -> SpellAbility | None:
        if actor is None:
            return None
        if self.spell is not None:
            if (
                self.spell in actor.abilities
                or self.spell in actor.cantrips
                or self.spell in actor.prepared_spells
            ) and can_cast_spell(actor, self.spell, self.cast_level):
                return self.spell
            return None
        for spell in available_castable_spells(actor):
            return spell
        return None

    def _aoe_targeting_for_spell(
        self,
        combat_state: CombatState,
        actor: Character,
        spell: SpellAbility,
    ) -> AoETargeting | None:
        shape = spell_aoe_shape(spell)
        if shape is None:
            return None
        if shape is AoEShape.RADIUS:
            target_cell = self.target_cell
            if target_cell is None:
                target = self._target_for_spell(combat_state, actor, spell)
                target_cell = target.position if target is not None else None
            if target_cell is None:
                return None
            if not _can_place_target_cell_aoe(combat_state, actor, target_cell, spell.range):
                return None
            targeting = AoETargeting(
                shape=shape,
                origin=actor.position,
                size=spell.area_size,
                target_cell=target_cell,
            )
            if not _aoe_has_affected_creature(combat_state, targeting):
                return None
            return targeting

        direction = coerce_aoe_direction(self.direction)
        if direction is None:
            target = self._target_for_spell(combat_state, actor, spell)
            if target is not None:
                direction = direction_from_positions(actor.position, target.position)
        if direction is None:
            return None
        targeting = AoETargeting(
            shape=shape,
            origin=actor.position,
            size=spell.area_size,
            direction=direction,
        )
        if not _aoe_has_affected_creature(combat_state, targeting):
            return None
        return targeting

    def _target_for_spell(
        self,
        combat_state: CombatState,
        actor: Character,
        spell: SpellAbility,
    ) -> Character | None:
        if self.target_id is not None:
            target = _get_character(combat_state, self.target_id)
            if target is not None and _can_target_spell(combat_state, actor, target, spell):
                return target
            return None
        for target in combat_state.characters:
            if _can_target_spell(combat_state, actor, target, spell):
                return target
        if spell.healing is not None and _can_target_spell(combat_state, actor, actor, spell):
            return actor
        return None


@dataclass
class SecondWindAction(CombatAction):
    """Use the Fighter Second Wind bonus action."""

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        return (
            actor is not None
            and actor.can_take_turn
            and actor.action_economy.bonus_action_available
            and can_use_feature_action(actor, "second_wind", "bonus_action")
        )

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Second Wind failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot use Second Wind.")

        healing = random.randint(1, 10) + max(1, int(getattr(actor, "level", 1)))
        before_hp = actor.hp
        actor.hp = min(actor.max_hp, actor.hp + healing)
        actor.stable = actor.hp > 0
        actor.action_economy.spend_bonus_action()
        spend_feature_resource(actor, "second_wind")
        return ActionResult(
            True,
            (
                f"{actor.name} uses Second Wind and heals {actor.hp - before_hp} HP "
                f"(rolled {healing}). "
                "Bonus action spent: bonus_action_available=False."
            ),
        )


@dataclass
class ActionSurgeAction(CombatAction):
    """Use the Fighter Action Surge resource to restore the main action."""

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        return (
            actor is not None
            and actor.can_take_turn
            and can_use_feature_action(actor, "action_surge")
        )

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Action Surge failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot use Action Surge.")

        if not spend_feature_resource(actor, "action_surge"):
            return ActionResult(False, f"{actor.name} has no Action Surge uses remaining.")
        actor.action_economy.action_available = True
        return ActionResult(
            True,
            f"{actor.name} uses Action Surge. action_available=True.",
        )


@dataclass
class ChannelDivinityPreserveLifeAction(CombatAction):
    """Use Life Domain Channel Divinity to heal a wounded ally."""

    target_id: int | None = None
    range: int = 6

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None or not actor.can_take_turn:
            return False
        if not actor.action_economy.action_available:
            return False
        if not can_use_feature_action(actor, "preserve_life", "action"):
            return False
        return self._resolve_target(combat_state, actor) is not None

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Preserve Life failed: actor {self.actor_id} not found.")
        target = self._resolve_target(combat_state, actor)
        if target is None or not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot use Preserve Life.")

        actor.action_economy.spend_action()
        spend_feature_resource(actor, "preserve_life")
        healing_pool = max(1, int(getattr(actor, "level", 1))) * 5
        half_hp_cap = max(1, target.max_hp // 2)
        allowed_healing = max(0, half_hp_cap - target.hp)
        healing = min(healing_pool, allowed_healing)
        before_hp = target.hp
        target.hp = min(target.max_hp, target.hp + healing)
        if target.hp > 0:
            target.stable = True
        return ActionResult(
            True,
            (
                f"{actor.name} uses Channel Divinity: Preserve Life on "
                f"{target.name} and heals {target.hp - before_hp} HP. "
                "Action spent: action_available=False."
            ),
        )

    def _resolve_target(
        self,
        combat_state: CombatState,
        actor: Character,
    ) -> Character | None:
        if self.target_id is not None:
            target = _get_character(combat_state, self.target_id)
            if target is not None and _can_preserve_life_target(
                combat_state,
                actor,
                target,
                self.range,
            ):
                return target
            return None
        candidates = [
            target
            for target in combat_state.characters
            if _can_preserve_life_target(combat_state, actor, target, self.range)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda target: target.hp / max(1, target.max_hp))


@dataclass
class DashAction(CombatAction):
    """Spend an action to gain extra movement equal to speed."""

    def is_valid(self, combat_state: CombatState) -> bool:
        return _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_DASH)

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Dash failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Dash.")
        actor.action_economy.spend_action()
        actor.action_economy.dash(actor.speed)
        return ActionResult(
            True,
            (
                f"{actor.name} uses Dash. Movement remaining: "
                f"{actor.action_economy.movement_remaining}. "
                "Action spent: action_available=False."
            ),
        )


@dataclass
class DisengageAction(CombatAction):
    """Spend an action to avoid opportunity attacks this turn."""

    def is_valid(self, combat_state: CombatState) -> bool:
        return _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_DISENGAGE)

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Disengage failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Disengage.")
        actor.action_economy.spend_action()
        actor.action_economy.disengaged_until_end_of_turn = True
        return ActionResult(True, f"{actor.name} Disengages. Action spent: action_available=False.")


@dataclass
class DodgeAction(CombatAction):
    """Spend an action to impose disadvantage on incoming attacks."""

    def is_valid(self, combat_state: CombatState) -> bool:
        return _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_DODGE)

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Dodge failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Dodge.")
        actor.action_economy.spend_action()
        actor.action_economy.dodging_until_start_of_next_turn = True
        return ActionResult(True, f"{actor.name} Dodges. Action spent: action_available=False.")


@dataclass
class HelpAction(CombatAction):
    """Spend an action to help an ally or distract a target."""

    target_id: int | None = None

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        if not _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_HELP):
            return False
        return self._resolve_target(combat_state, actor) is not None

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        target = self._resolve_target(combat_state, actor)
        if actor is None:
            return ActionResult(False, f"Help failed: actor {self.actor_id} not found.")
        if target is None or not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Help now.")
        actor.action_economy.spend_action()
        if target.team == actor.team:
            actor.helped_target_id = combat_state.characters.index(target)
            target.advantage_on_next_check = True
            detail = f"{target.name} gains advantage on the next ability check."
        else:
            actor.help_against_target_id = combat_state.characters.index(target)
            detail = f"the next allied attack against {target.name} gains advantage."
        return ActionResult(
            True,
            f"{actor.name} uses Help; {detail} Action spent: action_available=False.",
        )

    def _resolve_target(
        self,
        combat_state: CombatState,
        actor: Character | None,
    ) -> Character | None:
        if actor is None:
            return None
        if self.target_id is not None:
            target = _get_character(combat_state, self.target_id)
            if target is not None and target is not actor and not target.is_dead:
                return target
            return None
        for target in combat_state.characters:
            if target is not actor and not target.is_dead:
                return target
        return None


@dataclass
class HideAction(CombatAction):
    """Spend an action to attempt a Stealth check."""

    dc: int | None = 10
    observer_id: int | None = None

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        return (
            _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_HIDE)
            and actor is not None
            and _can_hide_from_enemies(combat_state, actor)
        )

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Hide failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Hide.")
        actor.action_economy.spend_action()
        check = roll_ability_check(actor, "stealth", proficiency=True)
        dc, dc_description = self._resolve_dc(combat_state, actor)
        actor.action_economy.hidden = check.total >= dc
        outcome = "succeeds" if actor.action_economy.hidden else "fails"
        return ActionResult(
            True,
            (
                f"{actor.name} attempts to Hide and {outcome} "
                f"({check.total} vs {dc_description}; {check.log}). "
                "Action spent: action_available=False."
            ),
        )

    def _resolve_dc(
        self,
        combat_state: CombatState,
        actor: Character,
    ) -> tuple[int, str]:
        if self.dc is not None:
            return self.dc, f"DC {self.dc}"

        observer = (
            _get_character(combat_state, self.observer_id)
            if self.observer_id is not None
            else None
        )
        if observer is not None:
            dc = passive_perception(observer)
            return dc, f"{observer.name} passive Perception {dc}"

        observer_dcs = [
            (passive_perception(candidate), candidate.name)
            for candidate in combat_state.characters
            if candidate is not actor and candidate.team != actor.team and candidate.is_alive
        ]
        if not observer_dcs:
            return 10, "DC 10"
        dc, observer_name = max(observer_dcs, key=lambda item: item[0])
        return dc, f"{observer_name} passive Perception {dc}"


@dataclass
class SearchAction(CombatAction):
    """Spend an action to make a Perception or Investigation check."""

    skill: str = "perception"
    dc: int = 10

    def is_valid(self, combat_state: CombatState) -> bool:
        return _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_SEARCH)

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Search failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Search.")
        actor.action_economy.spend_action()
        skill = "investigation" if self.skill.lower() == "investigation" else "perception"
        check = roll_ability_check(actor, skill, proficiency=True)
        outcome = "succeeds" if check.total >= self.dc else "fails"
        discovered = _discover_hidden_targets(combat_state, actor, check.total)
        discovered_text = (
            f" Revealed hidden targets: {', '.join(discovered)}."
            if discovered
            else ""
        )
        return ActionResult(
            True,
            (
                f"{actor.name} Searches with {self.skill} and {outcome} "
                f"({check.total} vs DC {self.dc}; {check.log}). "
                f"{discovered_text}"
                "Action spent: action_available=False."
            ),
        )


@dataclass
class UseObjectAction(CombatAction):
    """Spend an action to use a simple object if one is provided."""

    object_name: str = "object"
    item: CombatItem | None = None
    target_id: int | None = None
    target_cell: Position | None = None
    direction: AoEDirection | str | int | None = None

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return False
        item = self._resolve_item(actor)
        if item is None:
            return _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_USE_OBJECT)
        if (
            COMMON_ACTION_USE_OBJECT not in actor.common_actions
            or not item.implemented
            or not item_has_quantity(item)
            or not _can_spend_item_action(actor, item)
        ):
            return False
        if item.has_aoe:
            return self._aoe_targeting_for_item(combat_state, actor, item) is not None
        return self._target_for_item(combat_state, actor, item) is not None

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Use object failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Use an Object.")
        item = self._resolve_item(actor)
        if item is None:
            actor.action_economy.spend_action()
            actor.action_economy.spend_free_object_interaction()
            return ActionResult(
                True,
                (
                    f"{actor.name} uses {self.object_name}. No item effect is implemented. "
                    "Action spent: action_available=False."
                ),
            )

        _spend_item_action(actor, item)
        if item is not None and item.has_aoe:
            targeting = self._aoe_targeting_for_item(combat_state, actor, item)
            if targeting is None:
                return ActionResult(False, f"{actor.name} cannot place {item.name}.")
            result = _execute_area_item_effect(combat_state, actor, item, targeting)
            consume_item(item)
            return result
        target = self._target_for_item(combat_state, actor, item)
        if target is None:
            return ActionResult(False, f"{actor.name} cannot target {item.name}.")
        result = _execute_single_target_item_effect(combat_state, actor, target, item)
        consume_item(item)
        return ActionResult(
            result.success,
            f"{result.description} {_item_action_spent_text(item)}",
            result.reward,
        )

    def _resolve_item(self, actor: Character) -> CombatItem | None:
        return resolve_item(actor, self.item or self.object_name)

    def _aoe_targeting_for_item(
        self,
        combat_state: CombatState,
        actor: Character,
        item: CombatItem,
    ) -> AoETargeting | None:
        shape = supported_item_aoe_shape(item)
        if shape is None:
            return None
        if shape is AoEShape.RADIUS:
            if self.target_cell is None:
                return None
            if not _can_place_target_cell_aoe(combat_state, actor, self.target_cell, item.range):
                return None
            targeting = AoETargeting(
                shape=shape,
                origin=actor.position,
                size=item.area_size,
                target_cell=self.target_cell,
            )
        else:
            direction = coerce_aoe_direction(self.direction)
            if direction is None:
                return None
            targeting = AoETargeting(
                shape=shape,
                origin=actor.position,
                size=item.area_size,
                direction=direction,
            )
        if not _aoe_has_affected_creature(combat_state, targeting):
            return None
        return targeting

    def _target_for_item(
        self,
        combat_state: CombatState,
        actor: Character,
        item: CombatItem,
    ) -> Character | None:
        target_type = normalize_target_type(item.target_type)
        if target_type is ItemTargetType.POINT:
            return None
        if target_type is ItemTargetType.SELF:
            return actor if _can_item_target_character(combat_state, actor, actor, item) else None
        if self.target_id is not None:
            target = _get_character(combat_state, self.target_id)
            if target is not None and _can_item_target_character(
                combat_state,
                actor,
                target,
                item,
            ):
                return target
            return None
        for target in combat_state.characters:
            if _can_item_target_character(combat_state, actor, target, item):
                return target
        return None


@dataclass
class ReadyAction(CombatAction):
    """Spend an action to prepare a simple future action description."""

    prepared_action: str = "prepared action"
    trigger_description: str = "unspecified trigger"

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        return (
            _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_READY)
            and actor is not None
            and actor.action_economy.reaction_available
        )

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Ready failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Ready an action.")
        actor.action_economy.reserve_ready_reaction(
            self.prepared_action,
            self.trigger_description,
        )
        return ActionResult(
            True,
            (
                f"{actor.name} readies {self.prepared_action} for trigger: "
                f"{self.trigger_description}. Action spent: action_available=False."
            ),
        )


@dataclass
class GrappleAction(CombatAction):
    """Special melee attack: Athletics vs Athletics or Acrobatics."""

    target_id: int

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        return _is_valid_special_melee_attack(
            combat_state,
            self.actor_id,
            actor,
            target,
            COMMON_ACTION_GRAPPLE,
        )

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        if actor is None or target is None:
            return ActionResult(False, "Grapple failed: actor or target not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Grapple {target.name}.")
        actor.action_economy.spend_action()
        contest = _roll_athletics_contest(actor, target)
        if contest.actor_wins:
            target.action_economy.apply_grappled()
            target.grappled_by = self.actor_id
            actor.grappling_target_id = self.target_id
            outcome = "succeeds"
        else:
            outcome = "fails"
        return ActionResult(
            True,
            (
                f"{actor.name} attempts to Grapple {target.name} and {outcome} "
                f"({contest.actor_result.total} vs {contest.target_result.total}; "
                f"{contest.log}). "
                "Action spent: action_available=False."
            ),
        )


@dataclass
class ShoveAction(CombatAction):
    """Special melee attack: shove prone or push one cell."""

    target_id: int
    shove_effect: str = "prone"

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        return _is_valid_special_melee_attack(
            combat_state,
            self.actor_id,
            actor,
            target,
            COMMON_ACTION_SHOVE,
        )

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        if actor is None or target is None:
            return ActionResult(False, "Shove failed: actor or target not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Shove {target.name}.")
        actor.action_economy.spend_action()
        contest = _roll_athletics_contest(actor, target)
        if contest.actor_wins:
            result_detail = _apply_shove_success(combat_state, actor, target, self.shove_effect)
            outcome = f"succeeds; {result_detail}"
        else:
            outcome = "fails"
        return ActionResult(
            True,
            (
                f"{actor.name} attempts to Shove {target.name} and {outcome} "
                f"({contest.actor_result.total} vs {contest.target_result.total}; "
                f"{contest.log}). "
                "Action spent: action_available=False."
            ),
        )


@dataclass
class StabilizeAction(CombatAction):
    """Spend an action to make a DC 10 Medicine check on a 0 HP creature."""

    target_id: int
    dc: int = 10

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        if not _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_STABILIZE):
            return False
        if actor is None or target is None:
            return False
        return target.hp <= 0 and _distance(actor.position, target.position, combat_state) <= 1

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        target = _get_character(combat_state, self.target_id)
        if actor is None or target is None:
            return ActionResult(False, "Stabilize failed: actor or target not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Stabilize {target.name}.")
        actor.action_economy.spend_action()
        check = roll_ability_check(actor, "medicine", proficiency=True)
        target.stable = check.total >= self.dc
        outcome = "stabilizes" if target.stable else "fails to stabilize"
        return ActionResult(
            True,
            (
                f"{actor.name} {outcome} {target.name} "
                f"({check.total} vs DC {self.dc}; {check.log}). "
                "Action spent: action_available=False."
            ),
        )


@dataclass
class ImprovisedAction(CombatAction):
    """Placeholder for a custom improvised action."""

    description: str = "improvised action"

    def is_valid(self, combat_state: CombatState) -> bool:
        return _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_IMPROVISED)

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Improvised action failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot improvise an action.")
        actor.action_economy.spend_action()
        return ActionResult(
            True,
            (
                f"{actor.name} attempts an improvised action: {self.description}. "
                "Action spent: action_available=False."
            ),
        )


@dataclass
class EndTurnAction(CombatAction):
    """End the actor's turn."""

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        return (
            actor is not None
            and actor.can_take_turn
            and bool(combat_state.characters)
            and COMMON_ACTION_END_TURN in actor.common_actions
        )

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(
                False,
                f"End turn failed: actor {self.actor_id} not found.",
            )
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot end turn.")

        actor.reset_end_of_turn_state()
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


def _can_spend_action(
    combat_state: CombatState,
    actor_id: int,
    action_name: str,
) -> bool:
    actor = _get_character(combat_state, actor_id)
    return (
        actor is not None
        and actor.can_take_turn
        and action_name in actor.common_actions
        and actor.action_economy.action_available
    )


def _can_target_spell(
    combat_state: CombatState,
    actor: Character,
    target: Character,
    spell: SpellAbility,
) -> bool:
    single_target_valid = can_target_spell_with_rules(
        actor,
        target,
        spell,
        distance=_distance(actor.position, target.position, combat_state),
        has_line_of_sight=_has_line_of_sight(
            combat_state,
            actor.position,
            target.position,
        ),
        has_full_cover=(
            _cover_between(combat_state, actor.position, target.position)
            is CoverType.FULL_COVER
        ),
    )
    if not single_target_valid:
        return False
    if spell_has_aoe(spell):
        return _can_area_spell_target(combat_state, actor, target, spell)
    return True


def _can_spend_spell_action(actor: Character, spell: SpellAbility) -> bool:
    if spell.action_cost == "reaction":
        return actor.action_economy.reaction_available
    if spell.action_cost == "bonus_action":
        return actor.action_economy.bonus_action_available
    return actor.action_economy.action_available


def _can_spend_item_action(actor: Character, item: CombatItem) -> bool:
    action_cost = normalize_action_cost(item.action_cost)
    if action_cost is ItemActionCost.REACTION:
        return actor.action_economy.reaction_available
    if action_cost is ItemActionCost.BONUS_ACTION:
        return actor.action_economy.bonus_action_available
    if action_cost is ItemActionCost.FREE_INTERACTION:
        return actor.action_economy.free_object_interaction_available
    return actor.action_economy.action_available


def _spend_item_action(actor: Character, item: CombatItem) -> None:
    action_cost = normalize_action_cost(item.action_cost)
    if action_cost is ItemActionCost.REACTION:
        actor.action_economy.spend_reaction()
    elif action_cost is ItemActionCost.BONUS_ACTION:
        actor.action_economy.spend_bonus_action()
    elif action_cost is ItemActionCost.FREE_INTERACTION:
        actor.action_economy.spend_free_object_interaction()
    else:
        actor.action_economy.spend_action()


def _item_action_spent_text(item: CombatItem) -> str:
    action_cost = normalize_action_cost(item.action_cost)
    if action_cost is ItemActionCost.REACTION:
        return "Reaction spent: reaction_available=False."
    if action_cost is ItemActionCost.BONUS_ACTION:
        return "Bonus action spent: bonus_action_available=False."
    if action_cost is ItemActionCost.FREE_INTERACTION:
        return "Free interaction spent: free_object_interaction_available=False."
    return "Action spent: action_available=False."


def _spend_spell_action(actor: Character, spell: SpellAbility) -> None:
    if spell.action_cost == "reaction":
        actor.action_economy.spend_reaction()
    elif spell.action_cost == "bonus_action":
        actor.action_economy.spend_bonus_action()
    else:
        actor.action_economy.spend_action()


def _spell_action_spent_text(spell: SpellAbility) -> str:
    if spell.action_cost == "reaction":
        return "Reaction spent: reaction_available=False."
    if spell.action_cost == "bonus_action":
        return "Bonus action spent: bonus_action_available=False."
    return "Action spent: action_available=False."


def _can_item_target_character(
    combat_state: CombatState,
    actor: Character,
    target: Character,
    item: CombatItem,
) -> bool:
    target_type = normalize_target_type(item.target_type)
    if target_type is ItemTargetType.SELF and target is not actor:
        return False
    if target_type is ItemTargetType.ALLY and target.team != actor.team:
        return False
    if target_type is ItemTargetType.ENEMY and target.team == actor.team:
        return False
    if target_type is ItemTargetType.POINT:
        return False
    if item_stabilizes(item):
        if target.hp > 0:
            return False
    elif item_healing(item) is not None:
        if target.is_dead or target.hp >= target.max_hp:
            return False
    elif _item_damage(item) is not None:
        if target.is_dead:
            return False
    if _distance(actor.position, target.position, combat_state) > item.range:
        return False
    if item.thrown or item.range > 1:
        if not _has_line_of_sight(combat_state, actor.position, target.position):
            return False
        if _cover_between(combat_state, actor.position, target.position) is CoverType.FULL_COVER:
            return False
    return True


def _target_saves_against_spell(
    actor: Character,
    target: Character,
    spell: SpellAbility,
    combat_state: CombatState | None = None,
    source_position: Position | None = None,
) -> bool:
    if spell.save_ability is None:
        return False
    save_roll = random.randint(1, 20) + ability_modifier(target.stats, spell.save_ability)
    if combat_state is not None and spell.save_ability.casefold() == "dex":
        cover_origin = source_position or actor.position
        save_roll = apply_cover_to_dex_save(
            save_roll,
            _cover_between(combat_state, cover_origin, target.position),
        )
    save_dc = spell.save_dc or getattr(actor, "spell_save_dc", 10)
    return save_roll >= save_dc


def _execute_area_damage_spell(
    combat_state: CombatState,
    actor: Character,
    spell: SpellAbility,
    casting_level: int,
    targeting: AoETargeting,
) -> ActionResult:
    base_damage = _roll_spell_effect(
        spell.damage or 0,
        spell.upcast_damage_per_level,
        spell.spell_level,
        casting_level,
    )
    affected_targets = _area_targets(combat_state, targeting)
    log_affected_targets(spell.name, targeting, affected_targets)
    summaries: list[str] = []
    total_damage = 0

    for affected in affected_targets:
        if _is_sculpted_ally(actor, affected, spell):
            summaries.append(f"{affected.name} protected by Sculpt Spells")
            continue

        raw_damage = base_damage
        saved = _target_saves_against_spell(
            actor,
            affected,
            spell,
            combat_state,
            source_position=targeting.source_position,
        )
        if saved:
            raw_damage = raw_damage // 2 if spell.save_half_damage else 0
        raw_damage = on_damage_roll(
            actor,
            raw_damage,
            target=affected,
            spell=spell,
            combat_state=combat_state,
        )
        damage = apply_damage_modifiers(affected, raw_damage, spell.damage_type)
        _apply_damage_to_character(affected, damage)
        total_damage += damage
        save_text = " after save" if saved else ""
        summaries.append(f"{affected.name}: {damage} damage{save_text}")

    spend_spell_resources(actor, spell, casting_level)
    begin_spell_concentration(actor, spell)
    spent_text = _spell_action_spent_text(spell)
    return ActionResult(
        True,
        (
            f"{actor.name} casts {spell.name} at level {casting_level}; "
            f"{'; '.join(summaries)}. Total damage: {total_damage}. "
            f"{spent_text}"
        ),
    )


def _execute_area_item_effect(
    combat_state: CombatState,
    actor: Character,
    item: CombatItem,
    targeting: AoETargeting,
) -> ActionResult:
    affected_targets = _area_targets(combat_state, targeting)
    log_affected_targets(item.name, targeting, affected_targets)
    summaries: list[str] = []
    total_damage = 0

    for affected in affected_targets:
        raw_damage = _roll_damage(_item_damage(item) or 0)
        saved = _target_saves_against_item(
            actor,
            affected,
            item,
            combat_state,
            source_position=targeting.source_position,
        )
        if saved:
            raw_damage = raw_damage // 2 if item_save_half_damage(item) else 0
        damage = apply_damage_modifiers(affected, raw_damage, _item_damage_type(item))
        _apply_damage_to_character(affected, damage)
        total_damage += damage
        save_text = " after save" if saved else ""
        summaries.append(f"{affected.name}: {damage} damage{save_text}")

    return ActionResult(
        True,
        (
            f"{actor.name} uses {item.name}; {'; '.join(summaries)}. "
            f"Total damage: {total_damage}. {_item_action_spent_text(item)}"
        ),
    )


def _execute_single_target_item_effect(
    combat_state: CombatState,
    actor: Character,
    target: Character,
    item: CombatItem,
) -> ActionResult:
    healing_value = item_healing(item)
    if healing_value is not None:
        healing = _roll_damage(healing_value)
        before_hp = target.hp
        target.hp = min(target.max_hp, target.hp + healing)
        if target.hp > 0:
            target.stable = True
        return ActionResult(
            True,
            (
                f"{actor.name} uses {item.name} on {target.name} and heals "
                f"{target.hp - before_hp} HP."
            ),
        )

    if item_stabilizes(item):
        if target.hp > 0:
            return ActionResult(True, f"{actor.name} uses {item.name} on {target.name}.")
        target.stable = True
        return ActionResult(
            True,
            f"{actor.name} uses {item.name} and stabilizes {target.name}.",
        )

    damage_value = _item_damage(item)
    if damage_value is not None:
        raw_damage = _roll_damage(damage_value)
        saved = _target_saves_against_item(
            actor,
            target,
            item,
            combat_state,
            source_position=actor.position,
        )
        if saved:
            raw_damage = raw_damage // 2 if item_save_half_damage(item) else 0
        damage = apply_damage_modifiers(target, raw_damage, _item_damage_type(item))
        _apply_damage_to_character(target, damage)
        save_text = " after save" if saved else ""
        return ActionResult(
            True,
            (
                f"{actor.name} uses {item.name} on {target.name} for "
                f"{damage} damage{save_text}."
            ),
        )

    return ActionResult(True, f"{actor.name} uses {item.name}.")


def _item_damage_type(item: CombatItem) -> object:
    return item_damage_type(item)


def _item_damage(item: CombatItem) -> int | str | None:
    return item_damage(item)


def _apply_temporary_ac_spell(actor: Character, spell: SpellAbility) -> None:
    previous_bonus = int(getattr(actor, "_temporary_spell_ac_bonus", 0))
    if previous_bonus > 0:
        actor.ac = max(0, actor.ac - previous_bonus)
    actor.ac += spell.ac_bonus
    actor._temporary_spell_ac_bonus = spell.ac_bonus
    actor._temporary_spell_ac_source = spell.name
    actor.conditions = [
        condition for condition in actor.conditions if condition.name != spell.name
    ]
    actor.conditions.append(
        Condition(
            name=spell.name,
            duration_rounds=1,
            description=f"+{spell.ac_bonus} AC until the start of the next turn.",
        )
    )


def _apply_damage_to_character(target: Character, damage: int) -> None:
    normalized_damage = max(0, int(damage))
    target.hp = max(0, target.hp - normalized_damage)
    if normalized_damage > 0:
        handle_concentration_damage(target, normalized_damage)
    if target.hp > 0 and normalized_damage > 0:
        target.stable = False


def _target_saves_against_item(
    actor: Character,
    target: Character,
    item: CombatItem,
    combat_state: CombatState,
    source_position: Position,
) -> bool:
    save_ability = item_save_ability(item)
    if save_ability is None:
        return False
    save_roll = random.randint(1, 20) + ability_modifier(target.stats, save_ability)
    if save_ability.casefold() == "dex":
        save_roll = apply_cover_to_dex_save(
            save_roll,
            _cover_between(combat_state, source_position, target.position),
        )
    save_dc = 10 + ability_modifier(actor.stats, "dex")
    return save_roll >= save_dc


def _consume_item(actor: Character, item: CombatItem) -> None:
    consume_item(item)


def _can_area_spell_target(
    combat_state: CombatState,
    actor: Character,
    target: Character,
    spell: SpellAbility,
) -> bool:
    targeting = _spell_targeting_from_creature(actor, target, spell)
    return targeting is not None and _aoe_has_affected_creature(combat_state, targeting)


def _spell_targeting_from_creature(
    actor: Character,
    target: Character,
    spell: SpellAbility,
) -> AoETargeting | None:
    shape = spell_aoe_shape(spell)
    if shape is None:
        return None
    if shape is AoEShape.RADIUS:
        return AoETargeting(
            shape=shape,
            origin=actor.position,
            size=spell.area_size,
            target_cell=target.position,
        )
    return AoETargeting(
        shape=shape,
        origin=actor.position,
        size=spell.area_size,
        direction=direction_from_positions(actor.position, target.position),
    )


def _area_targets(
    combat_state: CombatState,
    targeting: AoETargeting,
) -> list[Character]:
    positions = positions_for_aoe(targeting)
    if combat_state.grid_map is not None:
        positions = {
            position
            for position in positions
            if combat_state.grid_map.in_bounds(position)
        }
    return affected_creatures(combat_state.characters, positions)


def _aoe_has_affected_creature(
    combat_state: CombatState,
    targeting: AoETargeting,
) -> bool:
    return bool(_area_targets(combat_state, targeting))


def _can_place_target_cell_aoe(
    combat_state: CombatState,
    actor: Character,
    target_cell: Position,
    range_limit: int,
) -> bool:
    if _distance(actor.position, target_cell, combat_state) > range_limit:
        return False
    if not _has_line_of_sight(combat_state, actor.position, target_cell):
        return False
    return _cover_between(combat_state, actor.position, target_cell) is not CoverType.FULL_COVER


def _has_sculpt_spells(actor: Character, spell: SpellAbility) -> bool:
    return (
        (spell.school or "").casefold() == "evocation"
        and spell_has_aoe(spell)
        and character_has_class_feature(actor, "Sculpt Spells")
    )


def _is_sculpted_ally(
    actor: Character,
    target: Character,
    spell: SpellAbility,
) -> bool:
    return target.team == actor.team and _has_sculpt_spells(actor, spell)


def _can_preserve_life_target(
    combat_state: CombatState,
    actor: Character,
    target: Character,
    range_limit: int,
) -> bool:
    return (
        target.team == actor.team
        and target.is_alive
        and target.hp < max(1, target.max_hp // 2)
        and _distance(actor.position, target.position, combat_state) <= range_limit
        and _has_line_of_sight(combat_state, actor.position, target.position)
    )


def _is_valid_special_melee_attack(
    combat_state: CombatState,
    actor_id: int,
    actor: Character | None,
    target: Character | None,
    action_name: str,
) -> bool:
    if actor is None or target is None:
        return False
    if not _can_spend_action(combat_state, actor_id, action_name):
        return False
    if COMMON_ACTION_ATTACK not in actor.common_actions:
        return False
    return (
        target.team != actor.team
        and target.hp > 0
        and _distance(actor.position, target.position, combat_state) <= 1
    )


def _distance(first: Position, second: Position, combat_state: CombatState) -> int:
    if combat_state.grid_map is not None:
        return combat_state.grid_map.manhattan_distance(first, second)
    return abs(first.x - second.x) + abs(first.y - second.y)


def _cover_between(
    combat_state: CombatState,
    attacker_position: Position,
    target_position: Position,
) -> CoverType:
    grid_map = combat_state.grid_map
    if grid_map is None:
        return CoverType.NO_COVER
    return grid_map.get_cover_between(attacker_position, target_position)


def _has_line_of_sight(
    combat_state: CombatState,
    origin: Position,
    target: Position,
) -> bool:
    grid_map = combat_state.grid_map
    if grid_map is None:
        return True
    return grid_map.line_of_sight(origin, target)


def _can_weapon_target_from_map(
    combat_state: CombatState,
    actor: Character,
    target: Character,
    weapon: WeaponAttack,
) -> bool:
    if weapon.range <= 1:
        return True
    return (
        _has_line_of_sight(combat_state, actor.position, target.position)
        and _cover_between(combat_state, actor.position, target.position)
        is not CoverType.FULL_COVER
    )


def _movement_cost(
    actor: Character,
    destination: Position,
    combat_state: CombatState,
) -> int:
    path_cost = _movement_path_cost(actor, destination, combat_state)
    if path_cost is None:
        return 10**9
    if actor.prone and destination != actor.position:
        return path_cost + max(1, max(0, actor.speed) // 2)
    return path_cost


def _movement_path_cost(
    actor: Character,
    destination: Position,
    combat_state: CombatState,
) -> int | None:
    if combat_state.grid_map is not None:
        return combat_state.grid_map.path_movement_cost(
            actor.position,
            destination,
            combat_state.characters,
        )
    return _distance(actor.position, destination, combat_state)


def _can_hide_from_enemies(combat_state: CombatState, actor: Character) -> bool:
    enemies = [
        character
        for character in combat_state.characters
        if character is not actor and character.team != actor.team and character.is_alive
    ]
    if not enemies:
        return True
    return all(
        not _has_line_of_sight(combat_state, enemy.position, actor.position)
        or _cover_between(combat_state, enemy.position, actor.position)
        is not CoverType.NO_COVER
        for enemy in enemies
    )


def _discover_hidden_targets(
    combat_state: CombatState,
    actor: Character,
    check_total: int,
) -> list[str]:
    if check_total < 10:
        return []
    discovered: list[str] = []
    for target in combat_state.characters:
        if target is actor or target.team == actor.team or not target.hidden:
            continue
        if _has_line_of_sight(combat_state, actor.position, target.position):
            target.hidden = False
            discovered.append(target.name)
    return discovered


def _weapon_attack_modifier(actor: Character, weapon: WeaponAttack) -> int:
    return weapon.attack_modifier(actor) + archery_attack_bonus(actor, weapon)


def _roll_weapon_damage(
    actor: Character,
    weapon: WeaponAttack,
    *,
    critical: bool = False,
) -> int:
    dice_multiplier = 2 if critical else 1
    damage = _roll_damage_dice(
        weapon.damage,
        dice_multiplier=dice_multiplier,
        reroll_low=should_use_great_weapon_fighting(actor, weapon),
    )
    return max(0, damage + weapon.damage_modifier(actor))


def _roll_damage(damage: int | str) -> int:
    return _roll_damage_dice(damage)


def _roll_spell_effect(
    base_effect: int | str,
    upcast_effect_per_level: int | str | None,
    spell_level: int,
    cast_level: int,
) -> int:
    total = _roll_damage(base_effect)
    extra_levels = max(0, int(cast_level) - max(1, int(spell_level)))
    if upcast_effect_per_level is not None:
        for _ in range(extra_levels):
            total += _roll_damage(upcast_effect_per_level)
    return max(0, total)


def _roll_damage_dice(
    damage: int | str,
    *,
    dice_multiplier: int = 1,
    reroll_low: bool = False,
) -> int:
    if isinstance(damage, int):
        return max(0, damage * max(1, dice_multiplier))

    damage_text = damage.strip().lower()
    if damage_text.isdigit():
        return int(damage_text) * max(1, dice_multiplier)

    match = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", damage_text)
    if match is None:
        raise ValueError(f"Unsupported damage value: {damage}")

    dice_count = int(match.group(1) or 1) * max(1, dice_multiplier)
    die_size = int(match.group(2))
    modifier = int(match.group(3) or 0)
    total = sum(_roll_damage_die(die_size, reroll_low=reroll_low) for _ in range(dice_count))
    total += modifier
    return max(0, total)


def _roll_damage_die(die_size: int, *, reroll_low: bool = False) -> int:
    roll = random.randint(1, die_size)
    if reroll_low and roll <= 2:
        return random.randint(1, die_size)
    return roll


def _attack_roll(
    actor: Character,
    target: Character,
    combat_state: CombatState,
) -> int:
    has_advantage = actor.hidden or _consume_help_advantage(actor, target, combat_state)
    has_disadvantage = target.dodging_until_start_of_next_turn
    has_advantage, has_disadvantage = attack_roll_advantage_state(
        actor,
        target,
        combat_state,
        has_advantage=has_advantage,
        has_disadvantage=has_disadvantage,
    )
    first_roll = _roll_d20_with_racial_traits(actor)
    if has_advantage == has_disadvantage:
        roll = first_roll
    else:
        second_roll = _roll_d20_with_racial_traits(actor)
        roll = max(first_roll, second_roll) if has_advantage else min(first_roll, second_roll)
    actor.hidden = False
    return on_attack_roll(
        actor,
        roll,
        target=target,
        combat_state=combat_state,
        has_advantage=has_advantage,
        has_disadvantage=has_disadvantage,
    )


def _roll_d20_with_racial_traits(actor: Character) -> int:
    roll = random.randint(1, 20)
    traits = getattr(actor, "race_traits", None)
    if roll != 1 or traits is None or not traits.halfling_lucky_enabled:
        return roll
    return use_halfling_lucky(actor, roll, random.randint(1, 20))


def _consume_help_advantage(
    actor: Character,
    target: Character,
    combat_state: CombatState,
) -> bool:
    target_id = combat_state.characters.index(target)
    for helper in combat_state.characters:
        if helper is actor or helper.team != actor.team:
            continue
        if helper.help_against_target_id == target_id:
            helper.help_against_target_id = None
            return True
    return False


def _roll_athletics_contest(
    actor: Character,
    target: Character,
) -> ContestedCheckResult:
    return roll_contested_check(
        actor,
        target,
        "athletics",
        ("athletics", "acrobatics"),
    )


def _apply_shove_success(
    combat_state: CombatState,
    actor: Character,
    target: Character,
    shove_effect: str,
) -> str:
    if shove_effect == "push":
        pushed_position = _pushed_position(actor.position, target.position)
        if (
            combat_state.grid_map is not None
            and combat_state.grid_map.in_bounds(pushed_position)
            and not combat_state.grid_map.is_occupied(
                pushed_position,
                combat_state.characters,
            )
        ):
            target.position = pushed_position
            return f"{target.name} is pushed to {pushed_position}"
        target.prone = True
        return f"{target.name} falls prone because push movement is blocked"
    target.prone = True
    return f"{target.name} is knocked prone"


def _pushed_position(actor_position: Position, target_position: Position) -> Position:
    dx = target_position.x - actor_position.x
    dy = target_position.y - actor_position.y
    if abs(dx) >= abs(dy):
        step_x = 1 if dx >= 0 else -1
        return Position(target_position.x + step_x, target_position.y)
    step_y = 1 if dy >= 0 else -1
    return Position(target_position.x, target_position.y + step_y)
