"""Common D&D 5e-style combat actions available to creatures."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re

from combat.abilities import SpellAbility, WeaponAttack
from combat.checks import (
    ContestedCheckResult,
    passive_perception,
    roll_ability_check,
    roll_contested_check,
)
from combat.cover import CoverType, apply_cover_to_ac
from combat.features import (
    attack_roll_advantage_state,
    on_attack_roll,
    on_damage_roll,
)
from combat.models import Character, CombatState, Position
from combat.race_traits import apply_damage_resistance, use_halfling_lucky


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
        d20_roll = _attack_roll(actor, target, combat_state)
        attack_modifier = weapon.attack_modifier(actor)
        cover = _cover_between(combat_state, actor.position, target.position)
        effective_ac = apply_cover_to_ac(target.ac, cover)
        attack_total = d20_roll + attack_modifier
        if attack_total < effective_ac:
            return ActionResult(
                True,
                (
                    f"{actor.name} attacks {target.name} with {weapon.name}: "
                    f"miss ({attack_total} vs AC {effective_ac}; "
                    f"d20={d20_roll}, modifier={attack_modifier}). "
                    "Action spent: action_available=False."
                ),
            )

        raw_damage = on_damage_roll(
            actor,
            max(0, _roll_damage(weapon.damage) + weapon.damage_modifier(actor)),
            target=target,
            weapon=weapon,
            combat_state=combat_state,
        )
        damage = apply_damage_resistance(target, raw_damage, weapon.damage_type)
        target.hp = max(0, target.hp - damage)
        if target.hp > 0:
            target.stable = False
        return ActionResult(
            True,
            (
                f"{actor.name} attacks {target.name} with {weapon.name}: "
                f"hit ({attack_total} vs AC {effective_ac}; "
                f"d20={d20_roll}, modifier={attack_modifier}) for {damage} damage. "
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
        attack_modifier = weapon.attack_modifier(actor)
        cover = _cover_between(combat_state, actor.position, target.position)
        effective_ac = apply_cover_to_ac(target.ac, cover)
        attack_total = d20_roll + attack_modifier
        if attack_total < effective_ac:
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
            max(0, _roll_damage(weapon.damage) + weapon.damage_modifier(actor)),
            target=target,
            weapon=weapon,
            combat_state=combat_state,
        )
        damage = apply_damage_resistance(target, raw_damage, weapon.damage_type)
        target.hp = max(0, target.hp - damage)
        if target.hp > 0:
            target.stable = False
        return ActionResult(
            True,
            (
                f"{actor.name} opportunity attacks {target.name} with {weapon.name}: "
                f"hit ({attack_total} vs AC {effective_ac}; "
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

    def is_valid(self, combat_state: CombatState) -> bool:
        actor = _get_character(combat_state, self.actor_id)
        spell = self._resolve_spell(actor)
        if actor is None or not actor.can_take_turn or spell is None:
            return False
        if COMMON_ACTION_CAST_SPELL not in actor.common_actions:
            return False
        if not actor.action_economy.action_available or not spell.available:
            return False
        target = self._target_for_spell(combat_state, actor, spell)
        return spell.damage is None or target is not None

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        spell = self._resolve_spell(actor)
        if actor is None:
            return ActionResult(False, f"Cast spell failed: actor {self.actor_id} not found.")
        if spell is None:
            return ActionResult(False, f"{actor.name} has no available spell.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot cast {spell.name}.")

        actor.action_economy.spend_action()
        target = self._target_for_spell(combat_state, actor, spell)
        if spell.damage is not None and target is not None:
            raw_damage = on_damage_roll(
                actor,
                _roll_damage(spell.damage),
                target=target,
                spell=spell,
                combat_state=combat_state,
            )
            damage = apply_damage_resistance(
                target,
                raw_damage,
                spell.damage_type,
            )
            target.hp = max(0, target.hp - damage)
            return ActionResult(
                True,
                (
                    f"{actor.name} casts {spell.name} on {target.name} "
                    f"for {damage} damage. Action spent: action_available=False."
                ),
            )
        return ActionResult(
            True,
            f"{actor.name} casts {spell.name}. Action spent: action_available=False.",
        )

    def _resolve_spell(self, actor: Character | None) -> SpellAbility | None:
        if actor is None:
            return None
        if self.spell is not None:
            if self.spell in actor.abilities:
                return self.spell
            return None
        for ability in actor.available_abilities:
            if isinstance(ability, SpellAbility):
                return ability
        return None

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
        return None


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

    def is_valid(self, combat_state: CombatState) -> bool:
        return _can_spend_action(combat_state, self.actor_id, COMMON_ACTION_USE_OBJECT)

    def execute(self, combat_state: CombatState) -> ActionResult:
        actor = _get_character(combat_state, self.actor_id)
        if actor is None:
            return ActionResult(False, f"Use object failed: actor {self.actor_id} not found.")
        if not self.is_valid(combat_state):
            return ActionResult(False, f"{actor.name} cannot Use an Object.")
        actor.action_economy.spend_action()
        actor.action_economy.spend_free_object_interaction()
        return ActionResult(
            True,
            (
                f"{actor.name} uses {self.object_name}. No item effect is implemented. "
                "Action spent: action_available=False."
            ),
        )


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
    if target is actor or target.team == actor.team or target.hp <= 0:
        return False
    return (
        _distance(actor.position, target.position, combat_state) <= spell.range
        and _has_line_of_sight(combat_state, actor.position, target.position)
        and _cover_between(combat_state, actor.position, target.position)
        is not CoverType.FULL_COVER
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
