"""Tactical combat environment."""

from __future__ import annotations

import copy
from typing import Sequence

from combat.actions import (
    COMMON_ACTION_ATTACK,
    COMMON_ACTION_MOVE,
    ActionResult,
    AttackAction,
    CastSpellAction,
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
    ReadyAction,
    SearchAction,
    ShoveAction,
    StabilizeAction,
    UseObjectAction,
)
from combat.initiative import (
    apply_fixed_turn_order,
    apply_initiative_result,
    roll_initiative_order,
)
from combat.map import GridMap
from combat.models import (
    Character,
    CombatState,
    Position,
    Stats,
    Team,
    WeaponAttack,
)
from combat.rewards import (
    CombatReward,
    CombatRewardSnapshot,
    RewardConfig,
    calculate_combat_reward,
    format_reward_breakdown,
    snapshot_combat_state,
)
from rules.xp import award_party_xp


class CombatEnvironment:
    """A small turn-based combat environment."""

    def __init__(
        self,
        characters: Sequence[Character] | None = None,
        grid_map: GridMap | None = None,
        use_initiative: bool = True,
        initiative_seed: int | None = None,
        seed: int | None = None,
        log_to_console: bool = True,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._initial_characters = list(characters) if characters is not None else None
        self._initial_grid_map = grid_map
        self.use_initiative = use_initiative
        self.initiative_seed = initiative_seed if initiative_seed is not None else seed
        self.log_to_console = log_to_console
        self.reward_config = reward_config or RewardConfig()
        self.combat_state = CombatState()
        self.initiative_order: list[int] = []
        self.current_turn_index = 0
        self.round_number = 1
        self.action_log: list[str] = []
        self.xp_awarded = False
        self.last_awarded_xp = 0
        self.auto_end_turn_enabled = True
        self.reset()

    def reset(self) -> CombatState:
        characters = copy.deepcopy(self._initial_characters)
        if characters is None:
            characters = self._default_characters()

        grid_map = copy.deepcopy(self._initial_grid_map) or GridMap(width=5, height=5)
        self.combat_state = CombatState(characters=characters, grid_map=grid_map)
        self.combat_state.reset_combat_resources()
        self.action_log = []
        self.xp_awarded = False
        self.last_awarded_xp = 0
        self._initialize_turn_order()
        self._sync_turn_metadata()
        self._skip_unavailable_active_actor()
        if not self.is_done():
            self._begin_active_turn()
            self._record_round_start()
            self._record_active_actor()
        return self.combat_state

    def step(self, action: CombatAction) -> ActionResult:
        if self.is_done():
            return self._record_result(
                ActionResult(False, f"Combat is already done. Winner: {self.get_winner()}.")
            )

        self._skip_unavailable_active_actor()
        if self.is_done():
            return self._record_result(
                ActionResult(False, f"Combat is done. Winner: {self.get_winner()}.")
            )

        active_actor_id = self._active_actor_id()
        active_actor = self.combat_state.characters[active_actor_id]
        reward_before = snapshot_combat_state(self.combat_state)
        if action.actor_id != active_actor_id:
            return self._record_rewarded_result(
                ActionResult(
                    False,
                    (
                        f"{action.__class__.__name__} rejected: actor {action.actor_id} "
                        f"is not active actor {active_actor_id} ({active_actor.name})."
                    ),
                ),
                reward_before,
                active_actor.team,
                action,
            )

        if not action.is_valid(self.combat_state):
            return self._record_rewarded_result(
                ActionResult(
                    False,
                    f"{action.__class__.__name__} is not valid for {active_actor.name}.",
                ),
                reward_before,
                active_actor.team,
                action,
            )

        before_round = self.combat_state.round_number
        if isinstance(action, EndTurnAction):
            result = self._end_active_turn(action.actor_id)
        else:
            result = action.execute(self.combat_state)
        self._record_result(result)
        if isinstance(action, EndTurnAction) and result.success:
            self._record_turn_transition(before_round)
        if (
            self.auto_end_turn_enabled
            and not isinstance(action, EndTurnAction)
            and result.success
        ):
            self._auto_end_turn_if_actor_has_no_actions(action.actor_id)
        if result.success:
            self._skip_unavailable_active_actor()
        if result.success and self.is_done():
            self._award_xp_if_combat_complete()
            self.combat_state.reset_combat_resources()
        self._sync_turn_metadata()
        rewarded_result = self._with_reward(result, reward_before, active_actor.team, action)
        self._record_reward_breakdown(rewarded_result)
        return rewarded_result

    def get_observation(self, actor_id: int) -> dict[str, object]:
        actor = self.combat_state.character_at(actor_id)
        return {
            "actor_id": actor_id,
            "active_actor_id": self.combat_state.active_actor_id,
            "initiative_order": list(self.initiative_order),
            "current_turn_index": self.current_turn_index,
            "round_number": self.round_number,
            "is_done": self.is_done(),
            "winner": self.get_winner().value if self.get_winner() is not None else None,
            "actor": self._character_observation(actor) if actor is not None else None,
            "characters": [
                self._character_observation(character)
                for character in self.combat_state.characters
            ],
            "available_actions": [
                action.__class__.__name__
                for action in self.get_available_actions(actor_id)
            ],
        }

    def get_available_actions(self, actor_id: int) -> list[CombatAction]:
        actor = self.combat_state.character_at(actor_id)
        if (
            actor is None
            or not actor.can_take_turn
            or self.is_done()
            or not self._is_active_actor(actor_id)
        ):
            return []

        actions: list[CombatAction] = []
        actions.extend(self._available_move_actions(actor_id, actor))
        actions.extend(self._available_attack_actions(actor_id, actor))
        actions.extend(self._available_other_common_actions(actor_id, actor))
        end_turn = EndTurnAction(actor_id=actor_id)
        if end_turn.is_valid(self.combat_state):
            actions.append(end_turn)
        return actions

    def is_done(self) -> bool:
        living_teams = {
            character.team
            for character in self.combat_state.characters
            if character.is_alive
        }
        return len(living_teams) <= 1

    def get_winner(self) -> Team | None:
        living_teams = {
            character.team
            for character in self.combat_state.characters
            if character.is_alive
        }
        if len(living_teams) != 1:
            return None
        return next(iter(living_teams))

    def _award_xp_if_combat_complete(self) -> None:
        if self.xp_awarded or not self.is_done():
            return

        winner = self.get_winner()
        if winner is not Team.PLAYERS:
            self.xp_awarded = True
            return

        party = [
            character
            for character in self.combat_state.characters
            if character.team is Team.PLAYERS
        ]
        defeated_monsters = [
            character
            for character in self.combat_state.characters
            if character.team is not Team.PLAYERS and character.is_dead
        ]
        total_xp = award_party_xp(party, defeated_monsters)
        self.last_awarded_xp = total_xp
        self.xp_awarded = True
        if total_xp > 0:
            self._record_result(
                ActionResult(
                    True,
                    f"Awarded {total_xp} XP to player party after victory.",
                )
            )

    def _available_move_actions(
        self,
        actor_id: int,
        actor: Character,
    ) -> list[MoveAction]:
        if (
            self.combat_state.grid_map is None
            or COMMON_ACTION_MOVE not in actor.common_actions
            or actor.action_economy.grappled
        ):
            return []

        movement_remaining = actor.action_economy.movement_remaining
        if actor.prone:
            movement_remaining -= max(1, max(0, actor.speed) // 2)
        if movement_remaining <= 0:
            return []

        movement_cells = self.combat_state.grid_map.movement_cells(
            actor.position,
            movement_remaining,
            self.combat_state.characters,
        )
        return [
            MoveAction(actor_id=actor_id, destination=position)
            for position in sorted(movement_cells, key=lambda item: (item.x, item.y))
            if position != actor.position
        ]

    def _available_attack_actions(
        self,
        actor_id: int,
        actor: Character,
    ) -> list[AttackAction]:
        if (
            not actor.action_economy.action_available
            or COMMON_ACTION_ATTACK not in actor.common_actions
        ):
            return []

        actions: list[AttackAction] = []
        for weapon in actor.available_weapons:
            for target_id, target in enumerate(self.combat_state.characters):
                if target.team == actor.team or target.is_dead:
                    continue
                action = AttackAction(
                    actor_id=actor_id,
                    target_id=target_id,
                    weapon=weapon,
                )
                if action.is_valid(self.combat_state):
                    actions.append(action)
        return actions

    def _available_other_common_actions(
        self,
        actor_id: int,
        actor: Character,
    ) -> list[CombatAction]:
        if not actor.action_economy.action_available:
            return []

        candidates: list[CombatAction] = [
            CastSpellAction(actor_id=actor_id),
            DashAction(actor_id=actor_id),
            DisengageAction(actor_id=actor_id),
            DodgeAction(actor_id=actor_id),
            HideAction(actor_id=actor_id),
            SearchAction(actor_id=actor_id),
            UseObjectAction(actor_id=actor_id),
            ReadyAction(actor_id=actor_id),
            ImprovisedAction(actor_id=actor_id),
        ]
        for target_id, target in enumerate(self.combat_state.characters):
            if target_id == actor_id:
                continue
            candidates.append(HelpAction(actor_id=actor_id, target_id=target_id))
            candidates.append(StabilizeAction(actor_id=actor_id, target_id=target_id))
            if target.team != actor.team:
                candidates.append(GrappleAction(actor_id=actor_id, target_id=target_id))
                candidates.append(ShoveAction(actor_id=actor_id, target_id=target_id))

        return [action for action in candidates if action.is_valid(self.combat_state)]

    def _auto_end_turn_if_actor_has_no_actions(self, actor_id: int) -> None:
        if self.is_done() or not self._is_active_actor(actor_id):
            return

        non_end_turn_actions = [
            action
            for action in self.get_available_actions(actor_id)
            if not isinstance(action, EndTurnAction)
        ]
        if not non_end_turn_actions:
            before_round = self.combat_state.round_number
            result = self._end_active_turn(actor_id)
            if result.success:
                self._record_result(result)
                self._record_turn_transition(before_round)

    def _skip_unavailable_active_actor(self) -> None:
        while (
            self.combat_state.characters
            and not self.is_done()
            and self.combat_state.active_character is not None
            and not self.combat_state.active_character.can_take_turn
        ):
            skipped_actor = self.combat_state.active_character
            before_round = self.combat_state.round_number
            next_actor = self.combat_state.advance_turn()
            self._sync_turn_metadata()
            if next_actor is None:
                return
            reason = "dead" if skipped_actor.is_dead else "incapacitated"
            self._record_result(
                ActionResult(
                    True,
                    (
                        f"{skipped_actor.name} is {reason} and skips turn. "
                        f"{next_actor.name} starts turn."
                    ),
                )
            )
            self._record_skipped_turns(exclude={skipped_actor.name})
            self._record_turn_transition(before_round)

    def _is_active_actor(self, actor_id: int) -> bool:
        return actor_id == self.combat_state.active_actor_id

    def _begin_active_turn(self) -> Character | None:
        if not self.combat_state.characters:
            return None
        actor_id = self.combat_state.active_actor_id
        if actor_id is None:
            return None
        return self.combat_state.reset_turn_resources(actor_id)

    def _end_active_turn(self, actor_id: int) -> ActionResult:
        end_turn = EndTurnAction(actor_id=actor_id)
        if not end_turn.is_valid(self.combat_state):
            return ActionResult(False, f"Actor {actor_id} cannot end turn.")
        return end_turn.execute(self.combat_state)

    def _initialize_turn_order(self) -> None:
        if self.use_initiative:
            initiative = roll_initiative_order(
                self.combat_state.characters,
                seed=self.initiative_seed,
            )
            apply_initiative_result(self.combat_state, initiative)
            for roll in initiative.rolls:
                self._record_result(ActionResult(True, roll.log))
            self._record_result(ActionResult(True, initiative.order_log))
            return

        apply_fixed_turn_order(self.combat_state)

    def _active_actor_id(self) -> int:
        actor_id = self.combat_state.active_actor_id
        if actor_id is None:
            raise ValueError("combat has no active actor")
        return actor_id

    def _sync_turn_metadata(self) -> None:
        self.initiative_order = list(self.combat_state.initiative_order)
        self.current_turn_index = self.combat_state.current_turn_index
        self.round_number = self.combat_state.round_number

    def _record_turn_transition(self, before_round: int) -> None:
        self._record_skipped_turns()
        if self.combat_state.round_number != before_round:
            self._record_round_start()
        self._record_active_actor()
        self._sync_turn_metadata()

    def _record_round_start(self) -> None:
        self._record_result(
            ActionResult(True, f"Round {self.combat_state.round_number} begins.")
        )

    def _record_active_actor(self) -> None:
        actor = self.combat_state.active_character
        if actor is not None:
            self._record_result(ActionResult(True, f"Active actor: {actor.name}."))

    def _record_skipped_turns(self, exclude: set[str] | None = None) -> None:
        excluded_names = exclude or set()
        for actor_id in self.combat_state.skipped_turn_actor_ids:
            actor = self.combat_state.character_at(actor_id)
            if actor is None or actor.name in excluded_names:
                continue
            reason = "dead" if actor.is_dead else "incapacitated"
            self._record_result(
                ActionResult(True, f"{actor.name} is {reason} and skips turn.")
            )

    def _record_result(self, result: ActionResult) -> ActionResult:
        self.action_log.append(result.description)
        if self.log_to_console:
            print(result.description)
        return result

    def _record_rewarded_result(
        self,
        result: ActionResult,
        before: CombatRewardSnapshot,
        actor_team: Team,
        action: CombatAction | None = None,
    ) -> ActionResult:
        rewarded_result = self._with_reward(result, before, actor_team, action)
        self._record_result(rewarded_result)
        self._record_reward_breakdown(rewarded_result)
        return rewarded_result

    def _with_reward(
        self,
        result: ActionResult,
        before: CombatRewardSnapshot,
        actor_team: Team,
        action: CombatAction | None = None,
    ) -> ActionResult:
        reward = calculate_combat_reward(
            before,
            snapshot_combat_state(self.combat_state),
            actor_team,
            action_success=result.success,
            config=self.reward_config,
            action=action,
            action_result=result,
        )
        return ActionResult(
            success=result.success,
            description=result.description,
            reward=reward.total,
            reward_breakdown=reward.breakdown,
        )

    def _record_reward_breakdown(self, result: ActionResult) -> None:
        if not result.reward_breakdown:
            return
        self._record_result(
            ActionResult(
                True,
                format_reward_breakdown(
                    CombatReward(
                        total=result.reward,
                        breakdown=result.reward_breakdown,
                    )
                ),
            )
        )

    @staticmethod
    def _character_observation(character: Character) -> dict[str, object]:
        return {
            "name": character.name,
            "hp": character.hp,
            "max_hp": character.max_hp,
            "ac": character.ac,
            "position": {
                "x": character.position.x,
                "y": character.position.y,
            },
            "speed": character.speed,
            "team": character.team.value,
            "class_name": character.class_name,
            "subclass_name": character.subclass_name,
            "race_name": character.race_name,
            "race_traits": (
                {
                    "size": character.race_traits.size,
                    "speed": character.race_traits.speed,
                    "darkvision_range": character.race_traits.darkvision_range,
                    "skill_proficiencies": list(character.race_traits.skill_proficiencies),
                    "weapon_proficiencies": list(character.race_traits.weapon_proficiencies),
                    "saving_throw_advantages": list(
                        character.race_traits.saving_throw_advantages
                    ),
                    "damage_resistances": list(character.race_traits.damage_resistances),
                    "special_traits": list(character.race_traits.special_traits),
                }
                if character.race_traits is not None
                else None
            ),
            "level": character.level,
            "experience": character.experience,
            "proficiency_bonus": character.proficiency_bonus,
            "challenge_rating": character.challenge_rating,
            "xp_value": character.xp_value,
            "role": character.role,
            "alive": character.is_alive,
            "action_available": character.action_economy.action_available,
            "bonus_action_available": character.action_economy.bonus_action_available,
            "reaction_available": character.action_economy.reaction_available,
            "movement_remaining": character.action_economy.movement_remaining,
            "free_object_interaction_available": (
                character.action_economy.free_object_interaction_available
            ),
            "disengaged_until_end_of_turn": character.disengaged_until_end_of_turn,
            "dodging_until_start_of_next_turn": character.dodging_until_start_of_next_turn,
            "hidden": character.hidden,
            "prone": character.prone,
            "grappled": character.grappled,
            "grappling_target_id": character.grappling_target_id,
            "helped_target_id": character.helped_target_id,
            "help_against_target_id": character.help_against_target_id,
            "prepared_action": character.prepared_action,
            "trigger_description": character.trigger_description,
            "grappled_by": character.grappled_by,
            "reaction_used_this_round": character.action_economy.reaction_used_this_round,
            "stable": character.stable,
            "weapons": [weapon.name for weapon in character.weapons],
            "common_actions": list(character.common_actions),
            "class_features": [feature.name for feature in character.class_features],
            "implemented_class_features": [
                feature.name
                for feature in character.class_features
                if getattr(feature, "implemented", False)
            ],
            "not_implemented_class_features": [
                feature.name
                for feature in character.class_features
                if not getattr(feature, "implemented", False)
            ],
            "feats": [
                getattr(feat, "name", str(feat))
                for feat in getattr(character, "feats", ())
            ],
            "ability_score_improvements": [
                dict(getattr(asi, "bonuses", asi))
                for asi in getattr(character, "ability_score_improvements", ())
                if isinstance(getattr(asi, "bonuses", asi), dict)
            ],
            "resources": {
                name: resource.uses_remaining
                for name, resource in character.resources.items()
            },
        }

    @staticmethod
    def _default_characters() -> list[Character]:
        hero_weapon = WeaponAttack(
            name="Training Sword",
            range=1,
            damage=3,
            attack_bonus=4,
        )
        enemy_weapon = WeaponAttack(
            name="Training Claws",
            range=1,
            damage=2,
            attack_bonus=3,
        )
        return [
            Character(
                name="Hero",
                hp=10,
                max_hp=10,
                ac=14,
                position=Position(0, 0),
                speed=3,
                stats=Stats(dex=14),
                team=Team.PLAYERS,
                weapons=[hero_weapon],
            ),
            Character(
                name="Enemy",
                hp=8,
                max_hp=8,
                ac=12,
                position=Position(1, 0),
                speed=3,
                stats=Stats(dex=12),
                team=Team.ENEMIES,
                weapons=[enemy_weapon],
            ),
        ]
