"""Tactical combat environment."""

from __future__ import annotations

import copy
from typing import Sequence

from combat.actions import (
    ActionResult,
    AttackAction,
    CombatAction,
    EndTurnAction,
    MoveAction,
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
    CombatRewardSnapshot,
    RewardConfig,
    calculate_combat_reward,
    snapshot_combat_state,
)


class CombatEnvironment:
    """A small turn-based combat environment."""

    def __init__(
        self,
        characters: Sequence[Character] | None = None,
        grid_map: GridMap | None = None,
        use_initiative: bool = False,
        log_to_console: bool = True,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self._initial_characters = list(characters) if characters is not None else None
        self._initial_grid_map = grid_map
        self.use_initiative = use_initiative
        self.log_to_console = log_to_console
        self.reward_config = reward_config or RewardConfig()
        self.combat_state = CombatState()
        self.action_log: list[str] = []
        self.reset()

    def reset(self) -> CombatState:
        characters = copy.deepcopy(self._initial_characters)
        if characters is None:
            characters = self._default_characters()

        if self.use_initiative:
            characters.sort(key=lambda character: character.stats.dex, reverse=True)

        grid_map = copy.deepcopy(self._initial_grid_map) or GridMap(width=5, height=5)
        self.combat_state = CombatState(characters=characters, grid_map=grid_map)
        self.action_log = []
        self._skip_dead_active_actor()
        if not self.is_done():
            self.combat_state.reset_turn_resources()
        return self.combat_state

    def step(self, action: CombatAction) -> ActionResult:
        if self.is_done():
            return self._record_result(
                ActionResult(False, f"Combat is already done. Winner: {self.get_winner()}.")
            )

        self._skip_dead_active_actor()
        if self.is_done():
            return self._record_result(
                ActionResult(False, f"Combat is done. Winner: {self.get_winner()}.")
            )

        active_actor_id = self.combat_state.turn_index % len(self.combat_state.characters)
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
            )

        if not action.is_valid(self.combat_state):
            return self._record_rewarded_result(
                ActionResult(
                    False,
                    f"{action.__class__.__name__} is not valid for {active_actor.name}.",
                ),
                reward_before,
                active_actor.team,
            )

        result = action.execute(self.combat_state)
        self._record_result(result)
        if not isinstance(action, EndTurnAction) and result.success:
            self._auto_end_turn_if_actor_has_no_actions(action.actor_id)
        return self._with_reward(result, reward_before, active_actor.team)

    def get_observation(self, actor_id: int) -> dict[str, object]:
        actor = self.combat_state.character_at(actor_id)
        return {
            "actor_id": actor_id,
            "active_actor_id": self.combat_state.turn_index
            if self.combat_state.characters
            else None,
            "round_number": self.combat_state.round_number,
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
            or actor.is_dead
            or self.is_done()
            or not self._is_active_actor(actor_id)
        ):
            return []

        actions: list[CombatAction] = []
        actions.extend(self._available_move_actions(actor_id, actor))
        actions.extend(self._available_attack_actions(actor_id, actor))
        actions.append(EndTurnAction(actor_id=actor_id))
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

    def _available_move_actions(
        self,
        actor_id: int,
        actor: Character,
    ) -> list[MoveAction]:
        if self.combat_state.grid_map is None:
            return []

        movement_cells = self.combat_state.grid_map.movement_cells(
            actor.position,
            actor.action_economy.movement_remaining,
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
        if not actor.action_economy.action_available:
            return []

        actions: list[AttackAction] = []
        weapons = [
            ability
            for ability in actor.available_abilities
            if isinstance(ability, WeaponAttack)
        ]
        for weapon in weapons:
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

    def _auto_end_turn_if_actor_has_no_actions(self, actor_id: int) -> None:
        if self.is_done() or not self._is_active_actor(actor_id):
            return

        non_end_turn_actions = [
            action
            for action in self.get_available_actions(actor_id)
            if not isinstance(action, EndTurnAction)
        ]
        if not non_end_turn_actions:
            end_turn = EndTurnAction(actor_id=actor_id)
            if end_turn.is_valid(self.combat_state):
                self._record_result(end_turn.execute(self.combat_state))

    def _skip_dead_active_actor(self) -> None:
        while (
            self.combat_state.characters
            and not self.is_done()
            and self.combat_state.active_character is not None
            and self.combat_state.active_character.is_dead
        ):
            skipped_actor = self.combat_state.active_character
            next_actor = self.combat_state.advance_turn()
            if next_actor is None:
                return
            self._record_result(
                ActionResult(
                    True,
                    f"{skipped_actor.name} is dead and skips turn. {next_actor.name} starts turn.",
                )
            )

    def _is_active_actor(self, actor_id: int) -> bool:
        return (
            bool(self.combat_state.characters)
            and actor_id == self.combat_state.turn_index % len(self.combat_state.characters)
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
    ) -> ActionResult:
        return self._record_result(self._with_reward(result, before, actor_team))

    def _with_reward(
        self,
        result: ActionResult,
        before: CombatRewardSnapshot,
        actor_team: Team,
    ) -> ActionResult:
        reward = calculate_combat_reward(
            before,
            snapshot_combat_state(self.combat_state),
            actor_team,
            action_success=result.success,
            config=self.reward_config,
        )
        return ActionResult(
            success=result.success,
            description=result.description,
            reward=reward.total,
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
            "alive": character.is_alive,
            "action_available": character.action_economy.action_available,
            "bonus_action_available": character.action_economy.bonus_action_available,
            "reaction_available": character.action_economy.reaction_available,
            "movement_remaining": character.action_economy.movement_remaining,
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
                abilities=[hero_weapon],
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
                abilities=[enemy_weapon],
            ),
        ]
