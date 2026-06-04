"""Policy routing helpers for multi-agent PPO training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any

import torch

from agents.rule_based import (
    AggressiveMeleeAgent,
    RangedKitingAgent,
    RuleBasedAgent,
    SimpleCasterAgent,
    SimpleHealerAgent,
)
from agents.action_space import (
    ACTION_CATEGORY_COUNT,
    MAIN_ACTION_TYPE_COUNT,
    MIN_OPTION_COUNT,
    ActionCategory,
    MainActionType,
)
from agents.entity_observation import COMBAT_ROLE_NAMES, combat_role_name
from combat.models import Character, Team


class CombatRole(Enum):
    """Supported coarse combat roles for policy routing and observations."""

    MELEE_DAMAGE = "MELEE_DAMAGE"
    RANGED_DAMAGE = "RANGED_DAMAGE"
    TANK = "TANK"
    SUPPORT = "SUPPORT"
    CASTER = "CASTER"
    BRUTE_ENEMY = "BRUTE_ENEMY"
    SKIRMISHER_ENEMY = "SKIRMISHER_ENEMY"


@dataclass
class RandomPolicy:
    """Mask-aware random baseline policy."""

    seed: int | None = None
    target_count: int = 8
    move_count: int = 64
    option_count: int = MIN_OPTION_COUNT
    action_category_count: int = ACTION_CATEGORY_COUNT
    main_action_type_count: int = MAIN_ACTION_TYPE_COUNT
    generator: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.generator = random.Random(self.seed)

    def act(
        self,
        observation: Any,
        masks: Mapping[str, torch.Tensor],
        deterministic: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Select a random legal hierarchical action under the provided masks."""

        action_category = _select_from_mask(
            masks.get("action_category"),
            self.action_category_count,
            deterministic,
            self.generator,
            fallback=int(ActionCategory.END_TURN),
        )
        main_action_type = _select_from_mask(
            masks.get("main_action_type"),
            self.main_action_type_count,
            deterministic,
            self.generator,
            fallback=int(MainActionType.ATTACK),
        )
        target_index = _select_from_mask(
            masks.get("target_index"),
            self.target_count,
            deterministic,
            self.generator,
        )
        move_index = _select_from_mask(
            masks.get("move_index"),
            self.move_count,
            deterministic,
            self.generator,
        )
        option_index = _select_from_mask(
            masks.get("option_index"),
            self.option_count,
            deterministic,
            self.generator,
        )
        return _action_output(
            action_category=action_category,
            main_action_type=main_action_type,
            target_index=target_index,
            move_index=move_index,
            option_index=option_index,
            value=0.0,
        )


class RuleBasedEnemyPolicy(RandomPolicy):
    """Simple enemy baseline: attack if possible, otherwise end turn."""

    def act(
        self,
        observation: Any,
        masks: Mapping[str, torch.Tensor],
        deterministic: bool = True,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        category_mask = _mask_or_default(masks.get("action_category"), self.action_category_count)
        main_mask = _mask_or_default(masks.get("main_action_type"), self.main_action_type_count)
        if (
            _mask_allows(category_mask, int(ActionCategory.MAIN_ACTION))
            and _mask_allows(main_mask, int(MainActionType.ATTACK))
        ):
            return _action_output(
                action_category=int(ActionCategory.MAIN_ACTION),
                main_action_type=int(MainActionType.ATTACK),
                target_index=_first_allowed_index(masks.get("target_index"), self.target_count),
                move_index=0,
                option_index=0,
                value=0.0,
            )
        if _mask_allows(category_mask, int(ActionCategory.END_TURN)):
            return _action_output(
                action_category=int(ActionCategory.END_TURN),
                main_action_type=0,
                target_index=0,
                move_index=0,
                option_index=0,
                value=0.0,
            )
        return super().act(observation, masks, deterministic=deterministic)


@dataclass
class AggressiveCombatPolicy:
    """Role-aware aggressive baseline for training the opposite side."""

    seed: int | None = None
    target_count: int = 8
    move_count: int = 64
    option_count: int = MIN_OPTION_COUNT
    action_category_count: int = ACTION_CATEGORY_COUNT
    main_action_type_count: int = MAIN_ACTION_TYPE_COUNT
    _melee: AggressiveMeleeAgent = field(init=False)
    _ranged: RangedKitingAgent = field(init=False)
    _caster: SimpleCasterAgent = field(init=False)
    _healer: SimpleHealerAgent = field(init=False)
    _fallback: RuleBasedAgent = field(init=False)

    def __post_init__(self) -> None:
        self._melee = AggressiveMeleeAgent(seed=self.seed)
        self._ranged = RangedKitingAgent(seed=self.seed)
        self._caster = SimpleCasterAgent(seed=self.seed)
        self._healer = SimpleHealerAgent(seed=self.seed)
        self._fallback = RuleBasedAgent(seed=self.seed)

    def act(
        self,
        observation: Any,
        masks: Mapping[str, torch.Tensor],
        deterministic: bool = True,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        actor = kwargs.get("actor")
        policy = self._select_policy(actor)
        return policy.act(
            observation,
            masks,
            deterministic=deterministic,
            **kwargs,
        )

    def _select_policy(self, actor: Any | None) -> RuleBasedAgent:
        if actor is None:
            return self._fallback
        role = combat_role_name(actor)
        if role == CombatRole.SUPPORT.value:
            return self._healer
        if role == CombatRole.CASTER.value:
            return self._caster
        if role in {CombatRole.RANGED_DAMAGE.value, CombatRole.SKIRMISHER_ENEMY.value}:
            if _has_ranged_weapon(actor):
                return self._ranged
        return self._melee


def random_policy(seed: int | None = None) -> RandomPolicy:
    """Create a random baseline policy."""

    return RandomPolicy(seed=seed)


def rule_based_enemy_policy() -> RuleBasedEnemyPolicy:
    """Create the default rule-based enemy baseline policy."""

    return RuleBasedEnemyPolicy()


def aggressive_combat_policy(seed: int | None = None) -> AggressiveCombatPolicy:
    """Create a role-aware aggressive baseline policy."""

    return AggressiveCombatPolicy(seed=seed)


@dataclass
class MultiAgentPolicyRouter:
    """Resolve which policy controls a combat participant."""

    shared_policy: Any | None = None
    player_policy: Any | None = None
    enemy_policy: Any | None = None
    role_policies: Mapping[CombatRole | str, Any] | None = None
    rule_based_enemy_policy: Any | None = None
    random_policy: Any | None = None

    def policy_for(self, actor: Character) -> Any:
        """Return the policy assigned to the actor's team or role."""

        role_policy = self._role_policy(actor)
        if role_policy is not None:
            return role_policy
        if actor.team is Team.PLAYERS and self.player_policy is not None:
            return self.player_policy
        if actor.team is Team.ENEMIES:
            if self.enemy_policy is not None:
                return self.enemy_policy
            if self.rule_based_enemy_policy is not None:
                return self.rule_based_enemy_policy
        if self.shared_policy is not None:
            return self.shared_policy
        if self.random_policy is not None:
            return self.random_policy
        raise ValueError("No policy configured for actor")

    def policies(self) -> tuple[Any, ...]:
        """Return unique configured policy objects."""

        policies: list[Any] = []
        for policy in (
            self.shared_policy,
            self.player_policy,
            self.enemy_policy,
            self.rule_based_enemy_policy,
            self.random_policy,
        ):
            if policy is not None and not any(policy is existing for existing in policies):
                policies.append(policy)
        for policy in (self.role_policies or {}).values():
            if policy is not None and not any(policy is existing for existing in policies):
                policies.append(policy)
        return tuple(policies)

    def _role_policy(self, actor: Character) -> Any | None:
        if not self.role_policies:
            return None
        role_key = normalize_role_key(combat_role_name(actor))
        for configured_role, policy in self.role_policies.items():
            if normalize_role_key(configured_role) == role_key:
                return policy
        return None


def normalize_role_key(role: CombatRole | str) -> str:
    """Normalize role enum/string keys for role policy lookup."""

    if isinstance(role, CombatRole):
        return role.value
    text = str(role).strip()
    normalized = text.upper().replace(" ", "_").replace("-", "_")
    if normalized not in COMBAT_ROLE_NAMES:
        return normalized
    return normalized


def role_id_for_actor(actor: Character) -> int:
    """Return a stable one-based role id for a character."""

    role_name = combat_role_name(actor)
    return COMBAT_ROLE_NAMES.index(role_name) + 1


def role_embedding_for_actor(actor: Character) -> torch.Tensor:
    """Return a one-hot role embedding for diagnostics or external policies."""

    role_id = role_id_for_actor(actor)
    embedding = torch.zeros(len(COMBAT_ROLE_NAMES), dtype=torch.float32)
    embedding[role_id - 1] = 1.0
    return embedding


def _action_output(
    *,
    action_category: int,
    main_action_type: int,
    target_index: int,
    move_index: int,
    option_index: int,
    value: float,
) -> dict[str, torch.Tensor]:
    return {
        "action_category": torch.tensor(action_category, dtype=torch.long),
        "main_action_type": torch.tensor(main_action_type, dtype=torch.long),
        "target_index": torch.tensor(target_index, dtype=torch.long),
        "move_index": torch.tensor(move_index, dtype=torch.long),
        "option_index": torch.tensor(option_index, dtype=torch.long),
        "log_prob": torch.tensor(0.0, dtype=torch.float32),
        "entropy": torch.tensor(0.0, dtype=torch.float32),
        "value": torch.tensor(value, dtype=torch.float32),
    }


def _select_from_mask(
    mask: torch.Tensor | None,
    size: int,
    deterministic: bool,
    generator: random.Random,
    fallback: int = 0,
) -> int:
    prepared = _mask_or_default(mask, size)
    allowed = torch.nonzero(prepared, as_tuple=False).reshape(-1).tolist()
    if not allowed:
        return int(fallback)
    if deterministic:
        return int(allowed[0])
    return int(generator.choice(allowed))


def _mask_or_default(mask: torch.Tensor | None, size: int) -> torch.Tensor:
    if mask is None:
        return torch.ones(size, dtype=torch.bool)
    prepared = mask.detach().cpu().bool()
    if prepared.ndim == 2:
        prepared = prepared[0]
    if prepared.ndim != 1:
        raise ValueError("policy masks must be 1D or batched 2D tensors")
    if prepared.shape[0] >= size:
        return prepared[:size]
    padding = torch.zeros(size - prepared.shape[0], dtype=torch.bool)
    return torch.cat((prepared, padding), dim=0)


def _mask_allows(mask: torch.Tensor, index: int) -> bool:
    return 0 <= index < mask.shape[0] and bool(mask[index])


def _first_allowed_index(mask: torch.Tensor | None, size: int) -> int:
    prepared = _mask_or_default(mask, size)
    allowed = torch.nonzero(prepared, as_tuple=False).reshape(-1)
    if allowed.numel() == 0:
        return 0
    return int(allowed[0].item())


def _has_ranged_weapon(actor: Any) -> bool:
    return any(int(getattr(weapon, "range", 1)) > 1 for weapon in getattr(actor, "weapons", ()))


__all__ = [
    "AggressiveCombatPolicy",
    "CombatRole",
    "MultiAgentPolicyRouter",
    "RandomPolicy",
    "RuleBasedEnemyPolicy",
    "random_policy",
    "aggressive_combat_policy",
    "role_embedding_for_actor",
    "role_id_for_actor",
    "rule_based_enemy_policy",
]
